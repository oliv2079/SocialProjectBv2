"""Helpers for compacting municipality GeoJSON for web visualizations."""

from __future__ import annotations

from math import hypot
from typing import Any

MIN_RING_AREA = 1e-10


def rounded_geojson_coordinates(value: Any, precision: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [rounded_geojson_coordinates(item, precision) for item in value]
    if isinstance(value, dict):
        return {
            key: rounded_geojson_coordinates(item, precision)
            for key, item in value.items()
        }
    return value


def perpendicular_distance(point: list[float], start: list[float], end: list[float]) -> float:
    x, y = point[:2]
    x1, y1 = start[:2]
    x2, y2 = end[:2]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return hypot(x - x1, y - y1)
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / hypot(dx, dy)


def simplify_line(points: list[list[float]], tolerance: float) -> list[list[float]]:
    if len(points) <= 2:
        return points

    max_distance = 0.0
    max_index = 0
    start = points[0]
    end = points[-1]
    for index in range(1, len(points) - 1):
        distance = perpendicular_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = index

    if max_distance <= tolerance:
        return [start, end]

    left = simplify_line(points[: max_index + 1], tolerance)
    right = simplify_line(points[max_index:], tolerance)
    return left[:-1] + right


def simplify_ring(ring: list[list[float]], tolerance: float) -> list[list[float]]:
    if len(ring) <= 4:
        return ring

    closed = ring[0] == ring[-1]
    line = ring[:-1] if closed else ring
    simplified = simplify_line(line, tolerance)

    if len(simplified) < 3:
        simplified = line[:3]

    if closed:
        simplified = simplified + [simplified[0]]
    return simplified


def close_ring(ring: list[list[float]]) -> list[list[float]]:
    if ring and ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring


def ring_area(ring: list[list[float]]) -> float:
    area = 0.0
    for start, end in zip(ring, ring[1:]):
        area += start[0] * end[1] - end[0] * start[1]
    return area / 2


def has_polygon_area(ring: list[list[float]]) -> bool:
    if len(ring) < 4:
        return False
    if len({tuple(point[:2]) for point in ring}) < 3:
        return False
    return abs(ring_area(ring)) > MIN_RING_AREA


def orient_ring(ring: list[list[float]], *, clockwise: bool) -> list[list[float]]:
    is_clockwise = ring_area(ring) < 0
    if is_clockwise != clockwise:
        return list(reversed(ring))
    return ring


def clean_ring(
    ring: list[list[float]], *, tolerance: float, precision: int
) -> list[list[float]] | None:
    rounded = rounded_geojson_coordinates(
        simplify_ring(ring, tolerance),
        precision=precision,
    )
    closed = close_ring(rounded)
    if not has_polygon_area(closed):
        return None
    return closed


def clean_polygon(
    polygon: list[list[list[float]]], *, tolerance: float, precision: int
) -> list[list[list[float]]]:
    if not polygon:
        return []

    exterior = clean_ring(polygon[0], tolerance=tolerance, precision=precision)
    if exterior is None:
        return []

    holes = [
        ring
        for ring in (
            clean_ring(hole, tolerance=tolerance, precision=precision)
            for hole in polygon[1:]
        )
        if ring is not None
    ]

    # Keep RFC 7946 winding after removing degenerate rings. Plotly's
    # choropleth renderer can interpret reversed rings as world-sized fills.
    return [
        orient_ring(exterior, clockwise=False),
        *[orient_ring(hole, clockwise=True) for hole in holes],
    ]


def simplify_geometry(
    geometry: dict[str, Any], *, tolerance: float, precision: int
) -> dict[str, Any]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return geometry

    if geometry_type == "Polygon":
        geometry = geometry.copy()
        geometry["coordinates"] = clean_polygon(
            coordinates,
            tolerance=tolerance,
            precision=precision,
        )
        return geometry

    if geometry_type == "MultiPolygon":
        geometry = geometry.copy()
        geometry["coordinates"] = [
            cleaned
            for polygon in coordinates
            if (
                cleaned := clean_polygon(
                    polygon,
                    tolerance=tolerance,
                    precision=precision,
                )
            )
        ]
        return geometry

    return geometry


def compact_geojson(
    geojson: dict[str, Any], precision: int = 3, tolerance: float = 0.004
) -> dict[str, Any]:
    features = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        geometry = simplify_geometry(
            feature.get("geometry", {}),
            tolerance=tolerance,
            precision=precision,
        )
        if not geometry.get("coordinates"):
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kode": properties.get("kode"),
                    "navn": properties.get("navn"),
                },
                "geometry": geometry,
            }
        )
    return {"type": "FeatureCollection", "features": features}
