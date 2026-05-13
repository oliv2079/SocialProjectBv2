"""Download official source data for the Denmark inequality project.

The script uses StatBank's public API and saves raw semicolon-separated CSVs
with both official codes and labels. It also downloads municipality geometry
from the Danish address/data supply API for the Plotly choropleth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

try:
    from scripts.geojson_utils import compact_geojson
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from geojson_utils import compact_geojson


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
METADATA_DIR = RAW_DIR / "metadata"
GEOJSON_PATH = RAW_DIR / "denmark_municipalities.geojson"

STATBANK_API = "https://api.statbank.dk/v1"
GEOJSON_URL = "https://api.dataforsyningen.dk/kommuner?format=geojson"

HEADERS = {
    "User-Agent": "Mozilla/5.0 Denmark inequality course project",
    "Content-Type": "application/json",
}

TABLES = {
    "IFOR41": "Income distribution on equivalised disposable income",
    "IFOR32": "Average equivalised disposable income by decile",
    "IFOR12P": "Persons in risk-of-poverty families",
    "AUP02": "Unemployed in percent of the labour force",
    "HFUDD11": "Educational attainment, ages 15-69",
    "HISBK": "Life expectancy for newborn babies",
    "BOL101": "Dwellings by region, resident type, tenure and time",
    "BOLRD": "Dwellings with registered population by tenure",
}

OMRAADE = "OMR\u00c5DE"
KON = "K\u00d8N"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_tableinfo(table: str) -> dict[str, Any]:
    url = f"{STATBANK_API}/tableinfo/{table}?lang=en"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def values_for(info: dict[str, Any], variable_code: str) -> list[str]:
    for variable in info["variables"]:
        if variable["id"] == variable_code:
            return [value["id"] for value in variable["values"]]
    available = ", ".join(variable["id"] for variable in info["variables"])
    raise KeyError(f"{info['id']} has no variable {variable_code}. Available: {available}")


def keep_available(
    info: dict[str, Any], variable_code: str, requested: list[str]
) -> list[str]:
    available = set(values_for(info, variable_code))
    kept = [value for value in requested if value in available]
    if not kept:
        raise ValueError(
            f"No requested values for {info['id']}.{variable_code} were available."
        )
    return kept


def municipality_codes(ifor41_info: dict[str, Any]) -> list[str]:
    """Use IFOR41 as the authoritative list of Denmark plus 98 municipalities."""
    return values_for(ifor41_info, "KOMMUNEDK")


def variables_payload(
    info: dict[str, Any], table: str, municipality_values: list[str]
) -> list[dict[str, Any]]:
    """Return a lean, course-aligned StatBank variable selection."""
    all_time = values_for(info, "Tid")

    if table == "IFOR41":
        return [
            {"code": "ULLIG", "values": ["70", "71", "72", "73"]},
            {"code": "KOMMUNEDK", "values": municipality_values},
            {"code": "Tid", "values": all_time},
        ]

    if table == "IFOR32":
        return [
            {"code": "DECILGEN", "values": values_for(info, "DECILGEN")},
            {"code": "KOMMUNEDK", "values": municipality_values},
            {"code": "Tid", "values": all_time},
        ]

    if table == "IFOR12P":
        return [
            {"code": "KOMMUNEDK", "values": municipality_values},
            {"code": "INDKN", "values": ["50", "60"]},
            {"code": "Tid", "values": all_time},
        ]

    if table == "AUP02":
        return [
            {"code": OMRAADE, "values": keep_available(info, OMRAADE, municipality_values)},
            {"code": "ALDER", "values": ["TOT"]},
            {"code": KON, "values": ["TOT"]},
            {"code": "Tid", "values": all_time},
        ]

    if table == "HFUDD11":
        return [
            {"code": "BOPOMR", "values": keep_available(info, "BOPOMR", municipality_values)},
            {"code": "HERKOMST", "values": ["TOT"]},
            {"code": "HFUDD", "values": ["TOT", "H40", "H50", "H60", "H70", "H80"]},
            {"code": "ALDER", "values": ["TOT"]},
            {"code": KON, "values": ["TOT"]},
            {"code": "Tid", "values": all_time},
        ]

    if table == "HISBK":
        return [
            {"code": OMRAADE, "values": keep_available(info, OMRAADE, municipality_values)},
            {"code": KON, "values": ["TOT"]},
            {"code": "Tid", "values": all_time},
        ]

    if table == "BOL101":
        return [
            {"code": OMRAADE, "values": keep_available(info, OMRAADE, municipality_values)},
            {"code": "BEBO", "values": ["1000"]},
            {"code": "ANVENDELSE", "values": values_for(info, "ANVENDELSE")},
            {"code": "UDLFORH", "values": ["EJ", "LEJ"]},
            {"code": "EJER", "values": values_for(info, "EJER")},
            {"code": "OPF\u00d8RELSES\u00c5R", "values": values_for(info, "OPF\u00d8RELSES\u00c5R")},
            {"code": "Tid", "values": all_time},
        ]

    if table == "BOLRD":
        return [
            {"code": variable["id"], "values": values_for(info, variable["id"])}
            for variable in info["variables"]
        ]

    raise KeyError(f"No downloader configuration exists for {table}.")


def download_table(
    table: str,
    info: dict[str, Any],
    municipality_values: list[str],
    skip_existing: bool,
) -> dict[str, Any]:
    csv_path = RAW_DIR / f"{table}.csv"
    metadata_path = METADATA_DIR / f"{table}.json"

    metadata_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if skip_existing and csv_path.exists():
        return {"table": table, "status": "skipped", "path": str(csv_path)}

    payload = {
        "table": table,
        "format": "BULK",
        "lang": "en",
        "valuePresentation": "CodeAndValue",
        "variables": variables_payload(info, table, municipality_values),
    }

    response = requests.post(
        f"{STATBANK_API}/data",
        headers=HEADERS,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    csv_path.write_text(response.text, encoding="utf-8")
    return {
        "table": table,
        "status": "downloaded",
        "path": str(csv_path),
        "bytes": len(response.content),
    }


def download_geojson(skip_existing: bool) -> dict[str, Any]:
    if skip_existing and GEOJSON_PATH.exists():
        return {"file": "denmark_municipalities.geojson", "status": "skipped"}

    response = requests.get(GEOJSON_URL, headers=HEADERS, timeout=120)
    response.raise_for_status()
    compact = compact_geojson(response.json(), precision=3, tolerance=0.004)
    GEOJSON_PATH.write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "file": "denmark_municipalities.geojson",
        "status": "downloaded",
        "path": str(GEOJSON_PATH),
        "source_bytes": len(response.content),
        "saved_bytes": GEOJSON_PATH.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download StatBank data for the Denmark inequality project."
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(TABLES),
        choices=list(TABLES),
        help="Subset of StatBank tables to download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not replace CSV or GeoJSON files that already exist.",
    )
    parser.add_argument(
        "--no-geojson",
        action="store_true",
        help="Skip downloading Danish municipality boundaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    report: dict[str, Any] = {"tables": [], "geometry": None}
    try:
        ifor41_info = fetch_tableinfo("IFOR41")
        municipalities = municipality_codes(ifor41_info)

        metadata_cache = {"IFOR41": ifor41_info}
        for table in args.tables:
            info = metadata_cache.get(table) or fetch_tableinfo(table)
            result = download_table(table, info, municipalities, args.skip_existing)
            report["tables"].append(result)
            print(f"{table}: {result['status']} -> {result.get('path', '')}")

        if not args.no_geojson:
            geo_result = download_geojson(args.skip_existing)
            report["geometry"] = geo_result
            print(f"GeoJSON: {geo_result['status']} -> {geo_result.get('path', '')}")

        (RAW_DIR / "download_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return 0

    except Exception as exc:  # pragma: no cover - CLI failure path
        report["error"] = str(exc)
        (RAW_DIR / "download_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
