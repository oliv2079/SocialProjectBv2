"""Clean StatBank data and generate project outputs.

Run after `scripts/download_statbank_data.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

try:
    from scripts.geojson_utils import compact_geojson
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from geojson_utils import compact_geojson


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "images"
VIS_DIR = ROOT / "visualizations"

OMRAADE = "OMR\u00c5DE"
KON = "K\u00d8N"
VALUE_COL = "INDHOLD"

EXPECTED_PANEL_COLUMNS = [
    "Year",
    "MunicipalityCode",
    "Municipality",
    "Gini",
    "P90P10",
    "S80S20",
    "Poverty50",
    "Poverty60",
    "UnemploymentRate",
    "TertiaryShare",
    "LifeExpectancy",
]
PANEL_KEYS = ["MunicipalityCode", "Municipality", "Year"]


def ensure_dirs() -> None:
    for directory in [PROCESSED_DIR, IMAGES_DIR, VIS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def raw_path(table: str) -> Path:
    return RAW_DIR / f"{table}.csv"


def require_raw(table: str) -> Path:
    path = raw_path(table)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_statbank_data.py` first "
            "or place a matching StatBank CSV in data/raw/."
        )
    return path


def read_statbank_csv(table: str) -> pd.DataFrame:
    df = pd.read_csv(require_raw(table), sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = [column.strip() for column in df.columns]
    return df


def split_code_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.fillna("").astype(str).str.strip()
    parts = text.str.extract(r"^(?P<code>\S+)(?:\s+(?P<label>.*))?$")
    code = parts["code"].fillna("").str.strip()
    label = parts["label"].fillna("").str.strip()
    label = label.mask(label.eq(""), code)
    return code, label


def parse_number(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def parse_year_from_time_code(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"^(\d{4})")[0], errors="coerce")


def parse_period_end_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})$")[0], errors="coerce")


def add_geo_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MunicipalityCode4"] = df["MunicipalityCode"].astype(str).str.zfill(4)
    return df


def pivot_measures(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index=PANEL_KEYS,
            columns="Measure",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(["MunicipalityCode", "Year"])
    )


def clean_inequality() -> pd.DataFrame:
    df = read_statbank_csv("IFOR41")
    df["IndicatorCode"], _ = split_code_label(df["ULLIG"])
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df["KOMMUNEDK"])
    df["TimeCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_year_from_time_code(df["TimeCode"]).astype("Int64")
    df["Value"] = parse_number(df[VALUE_COL])

    measure_map = {
        "70": "Gini",
        "71": "Hoover",
        "72": "S80S20",
        "73": "P90P10",
    }
    df = df[df["IndicatorCode"].isin(measure_map)].copy()
    df["Measure"] = df["IndicatorCode"].map(measure_map)
    return pivot_measures(df)


def clean_deciles() -> pd.DataFrame:
    df = read_statbank_csv("IFOR32")
    df["DecileCode"], df["Decile"] = split_code_label(df["DECILGEN"])
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df["KOMMUNEDK"])
    df["TimeCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_year_from_time_code(df["TimeCode"]).astype("Int64")
    df["AvgDisposableIncome"] = parse_number(df[VALUE_COL])
    df["DecileNumber"] = pd.to_numeric(
        df["DecileCode"].str.extract(r"^(\d+)")[0], errors="coerce"
    ).astype("Int64")
    columns = [
        "Year",
        "MunicipalityCode",
        "Municipality",
        "DecileCode",
        "DecileNumber",
        "Decile",
        "AvgDisposableIncome",
    ]
    return df[columns].sort_values(["MunicipalityCode", "Year", "DecileNumber"])


def clean_poverty() -> pd.DataFrame:
    df = read_statbank_csv("IFOR12P")
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df["KOMMUNEDK"])
    df["ThresholdCode"], _ = split_code_label(df["INDKN"])
    df["TimeCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_year_from_time_code(df["TimeCode"]).astype("Int64")
    df["Value"] = parse_number(df[VALUE_COL])
    df["Measure"] = "Poverty" + df["ThresholdCode"].astype(str)
    return pivot_measures(df)


def clean_unemployment() -> pd.DataFrame:
    df = read_statbank_csv("AUP02")
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df[OMRAADE])
    df["AgeCode"], _ = split_code_label(df["ALDER"])
    df["SexCode"], _ = split_code_label(df[KON])
    df["TimeCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_year_from_time_code(df["TimeCode"]).astype("Int64")
    df["UnemploymentRate"] = parse_number(df[VALUE_COL])
    df = df[(df["AgeCode"] == "TOT") & (df["SexCode"] == "TOT")].copy()
    annual = (
        df.groupby(["MunicipalityCode", "Municipality", "Year"], as_index=False)[
            "UnemploymentRate"
        ]
        .mean()
        .sort_values(["MunicipalityCode", "Year"])
    )
    return annual


def clean_education() -> pd.DataFrame:
    df = read_statbank_csv("HFUDD11")
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df["BOPOMR"])
    df["AncestryCode"], _ = split_code_label(df["HERKOMST"])
    df["EducationCode"], _ = split_code_label(df["HFUDD"])
    df["AgeCode"], _ = split_code_label(df["ALDER"])
    df["SexCode"], _ = split_code_label(df[KON])
    df["TimeCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_year_from_time_code(df["TimeCode"]).astype("Int64")
    df["Persons"] = parse_number(df[VALUE_COL])
    df = df[
        (df["AncestryCode"] == "TOT")
        & (df["AgeCode"] == "TOT")
        & (df["SexCode"] == "TOT")
    ].copy()

    denominator = (
        df[df["EducationCode"] == "TOT"]
        .groupby(PANEL_KEYS, as_index=False)["Persons"]
        .sum()
        .rename(columns={"Persons": "EducationPopulation"})
    )
    tertiary_codes = {"H40", "H50", "H60", "H70", "H80"}
    tertiary = (
        df[df["EducationCode"].isin(tertiary_codes)]
        .groupby(PANEL_KEYS, as_index=False)["Persons"]
        .sum()
        .rename(columns={"Persons": "TertiaryPersons"})
    )
    merged = denominator.merge(tertiary, on=PANEL_KEYS, how="left")
    merged["TertiaryShare"] = (
        merged["TertiaryPersons"] / merged["EducationPopulation"] * 100
    )
    return merged[PANEL_KEYS + ["TertiaryShare"]].sort_values(
        ["MunicipalityCode", "Year"]
    )


def clean_life_expectancy() -> pd.DataFrame:
    df = read_statbank_csv("HISBK")
    df["MunicipalityCode"], df["Municipality"] = split_code_label(df[OMRAADE])
    df["SexCode"], _ = split_code_label(df[KON])
    df["PeriodCode"], _ = split_code_label(df["TID"])
    df["Year"] = parse_period_end_year(df["PeriodCode"]).astype("Int64")
    df["LifeExpectancy"] = parse_number(df[VALUE_COL])
    df = df[df["SexCode"] == "TOT"].copy()
    return df[PANEL_KEYS + ["LifeExpectancy"]].sort_values(["MunicipalityCode", "Year"])


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    inequality = clean_inequality()
    poverty = clean_poverty()
    unemployment = clean_unemployment()
    education = clean_education()
    life = clean_life_expectancy()
    deciles = clean_deciles()

    panel = inequality.merge(poverty, on=PANEL_KEYS, how="left")
    panel = panel.merge(unemployment, on=PANEL_KEYS, how="left")
    panel = panel.merge(education, on=PANEL_KEYS, how="left")
    panel = panel.merge(life, on=PANEL_KEYS, how="left")

    for column in EXPECTED_PANEL_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan

    panel = panel[EXPECTED_PANEL_COLUMNS + [c for c in panel.columns if c == "Hoover"]]
    panel = panel.sort_values(["MunicipalityCode", "Year"])

    panel.to_csv(PROCESSED_DIR / "municipality_year.csv", index=False)
    deciles.to_csv(PROCESSED_DIR / "decile_distribution.csv", index=False)

    missingness = (
        panel[EXPECTED_PANEL_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .rename("MissingPercent")
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    missingness.to_csv(PROCESSED_DIR / "missingness_summary.csv", index=False)
    return panel, deciles


def latest_year_with(panel: pd.DataFrame, required: list[str]) -> int:
    usable = municipal_rows(panel).dropna(subset=required)
    if usable.empty:
        raise ValueError(f"No usable municipality-year rows for {required}.")
    return int(usable["Year"].max())


def municipal_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["MunicipalityCode"] != "000"].copy()


def benchmark_years_for_metric(
    panel: pd.DataFrame, metric: str, preferred_years: tuple[int, ...] = (2000, 2010)
) -> list[int]:
    usable = municipal_rows(panel).dropna(subset=[metric])
    if usable.empty:
        return []

    available_years = sorted(int(year) for year in usable["Year"].dropna().unique())
    selected = [available_years[0]]
    selected.extend(year for year in preferred_years if year in available_years)
    selected.append(available_years[-1])
    return sorted(dict.fromkeys(selected))


def municipal_distribution_stats(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    municipal = municipal_rows(panel)
    for metric in ["Gini", "Poverty60"]:
        for year in benchmark_years_for_metric(panel, metric):
            series = municipal.loc[municipal["Year"] == year, metric].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "year": int(year),
                    "count": int(series.count()),
                    "min": float(series.min()),
                    "q1": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q3": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                }
            )
    return rows


def latest_rank_summary(panel: pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
    year = latest_year_with(panel, ["Gini", "Poverty60", "UnemploymentRate"])
    data = municipal_rows(panel)
    data = data[
        (data["Year"] == year)
        & data["Gini"].notna()
        & data["Poverty60"].notna()
        & data["UnemploymentRate"].notna()
    ].copy()

    columns = ["Municipality", "Gini", "Poverty60", "UnemploymentRate"]
    return {
        "year": int(year),
        "top_gini": data.nlargest(top_n, "Gini")[columns].to_dict(orient="records"),
        "top_poverty60": data.nlargest(top_n, "Poverty60")[columns].to_dict(
            orient="records"
        ),
        "top_unemployment": data.nlargest(top_n, "UnemploymentRate")[columns].to_dict(
            orient="records"
        ),
    }


def plot_trend(panel: pd.DataFrame) -> Path:
    national = panel[panel["MunicipalityCode"] == "000"].sort_values("Year")
    metrics = ["Gini", "P90P10", "Poverty60"]
    colors = {"Gini": "#1f6f8b", "P90P10": "#8f3f71", "Poverty60": "#c77b30"}

    fig, ax = plt.subplots(figsize=(10, 5.8))
    plotted = False
    for metric in metrics:
        series = national[["Year", metric]].dropna()
        if series.empty:
            continue
        base = series[metric].iloc[0]
        if not np.isfinite(base) or base == 0:
            continue
        ax.plot(
            series["Year"],
            series[metric] / base * 100,
            marker="o",
            linewidth=2.2,
            markersize=3.5,
            label=f"{metric} (first available = 100)",
            color=colors[metric],
        )
        plotted = True

    if not plotted:
        raise ValueError("Could not plot trend because national series were empty.")

    ax.axhline(100, color="#777777", linewidth=1, alpha=0.6)
    ax.set_title("Denmark inequality and poverty trend", fontsize=15, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = IMAGES_DIR / "dk_inequality_trend.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_deciles(deciles: pd.DataFrame) -> Path:
    national = deciles[deciles["MunicipalityCode"] == "000"].dropna(
        subset=["AvgDisposableIncome", "DecileNumber"]
    )
    if national.empty:
        raise ValueError("No national decile observations available.")
    latest_year = int(national["Year"].max())
    latest = national[national["Year"] == latest_year].sort_values("DecileNumber")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        latest["DecileNumber"].astype(int).astype(str),
        latest["AvgDisposableIncome"],
        color="#5f8d4e",
    )
    ax.set_title(f"Disposable income by decile, Denmark, {latest_year}", fontsize=15)
    ax.set_xlabel("Decile")
    ax.set_ylabel("Average equivalised disposable income")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = IMAGES_DIR / "dk_decile_distribution.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_municipal_distribution(panel: pd.DataFrame) -> Path:
    municipal = municipal_rows(panel)
    metrics = [
        ("Gini", "Municipal Gini coefficient"),
        ("Poverty60", "Risk-of-poverty rate"),
    ]
    colors = {"Gini": "#1f6f8b", "Poverty60": "#b65f2a"}
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        years = benchmark_years_for_metric(panel, metric)
        if not years:
            ax.set_axis_off()
            continue

        groups = [
            municipal.loc[municipal["Year"] == year, metric].dropna().to_numpy()
            for year in years
        ]
        positions = np.arange(1, len(years) + 1)
        ax.boxplot(
            groups,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": "#ffffff", "edgecolor": colors[metric], "linewidth": 1.5},
            medianprops={"color": "#222222", "linewidth": 1.8},
            whiskerprops={"color": colors[metric], "linewidth": 1.1},
            capprops={"color": colors[metric], "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": colors[metric],
                "markeredgecolor": "none",
                "alpha": 0.25,
            },
        )
        for position, values in zip(positions, groups):
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=15,
                color=colors[metric],
                alpha=0.28,
                edgecolors="none",
            )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(year) for year in years])
        ax.set_xlabel("Benchmark year")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.22)

    fig.suptitle("Municipal distributions over time", fontsize=16)
    path = IMAGES_DIR / "dk_municipal_distribution.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def load_geojson() -> dict[str, Any]:
    path = RAW_DIR / "denmark_municipalities.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_statbank_data.py`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def iter_geojson_positions(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            yield value[0], value[1]
            return
        for item in value:
            yield from iter_geojson_positions(item)


def geojson_bounds(geojson: dict[str, Any]) -> dict[str, float]:
    positions = [
        point
        for feature in geojson.get("features", [])
        for point in iter_geojson_positions(
            feature.get("geometry", {}).get("coordinates", [])
        )
    ]
    if not positions:
        raise ValueError("GeoJSON contains no coordinate positions.")

    longitudes, latitudes = zip(*positions)
    return {
        "west": min(longitudes),
        "east": max(longitudes),
        "south": min(latitudes),
        "north": max(latitudes),
    }


def write_plotly_html(fig: go.Figure, path: Path) -> Path:
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_width="100%",
        default_height="100%",
        config={"responsive": True},
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
    }}
    body > div,
    .plotly-graph-div,
    .js-plotly-plot,
    .plot-container,
    .svg-container {{
      width: 100% !important;
      height: 100% !important;
    }}
  </style>
</head>
<body>
{plot_html}
  <script>
    function resizePlot() {{
      const plot = document.querySelector(".plotly-graph-div");
      if (plot && window.Plotly) {{
        plot.style.width = "100%";
        plot.style.height = "100%";
        window.Plotly.Plots.resize(plot);
      }}
    }}
    window.addEventListener("load", resizePlot);
    window.addEventListener("resize", resizePlot);
    window.requestAnimationFrame(resizePlot);
  </script>
</body>
</html>""",
        encoding="utf-8",
    )
    return path


def plot_map(panel: pd.DataFrame) -> Path:
    geojson = compact_geojson(load_geojson(), precision=3, tolerance=0.004)
    year = latest_year_with(panel, ["Gini"])
    data = municipal_rows(panel)
    data = data[(data["Year"] == year) & data["Gini"].notna()].copy()
    data = add_geo_code(data)

    bounds = geojson_bounds(geojson)
    center = {
        "lon": (bounds["west"] + bounds["east"]) / 2,
        "lat": (bounds["south"] + bounds["north"]) / 2,
    }

    fig = px.choropleth_map(
        data,
        geojson=geojson,
        locations="MunicipalityCode4",
        featureidkey="properties.kode",
        color="Gini",
        hover_name="Municipality",
        hover_data={
            "MunicipalityCode4": False,
            "Gini": ":.2f",
            "Poverty60": ":.2f",
            "UnemploymentRate": ":.2f",
        },
        opacity=0.96,
        color_continuous_scale="YlGnBu",
        title=f"Municipal income inequality in Denmark, {year}",
        center=center,
        zoom=5.45,
        map_style="white-bg",
    )
    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.9)",
        marker_line_width=0.7,
    )
    fig.update_layout(
        margin={"r": 0, "t": 44, "l": 0, "b": 0},
        autosize=True,
        coloraxis_colorbar={"title": "Gini"},
        font={"family": "Arial, sans-serif"},
    )
    fig.update_maps(
        bounds={
            "west": bounds["west"] - 0.35,
            "east": bounds["east"] + 0.35,
            "south": bounds["south"] - 0.2,
            "north": bounds["north"] + 0.2,
        }
    )
    path = VIS_DIR / "dk_inequality_map.html"
    return write_plotly_html(fig, path)


def plot_rank_comparison(panel: pd.DataFrame, top_n: int = 12) -> Path:
    year = latest_year_with(panel, ["Gini", "Poverty60", "UnemploymentRate"])
    data = municipal_rows(panel)
    data = data[
        (data["Year"] == year)
        & data["Gini"].notna()
        & data["Poverty60"].notna()
        & data["UnemploymentRate"].notna()
    ].copy()

    top_gini = data.nlargest(top_n, "Gini").sort_values("Gini")
    top_poverty = data.nlargest(top_n, "Poverty60").sort_values("Poverty60")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.18,
        subplot_titles=(
            "Highest inequality",
            "Highest risk of poverty",
        ),
    )
    marker = {
        "coloraxis": "coloraxis",
        "line": {"color": "rgba(255,255,255,0.9)", "width": 0.8},
    }
    fig.add_trace(
        go.Bar(
            x=top_gini["Gini"],
            y=top_gini["Municipality"],
            orientation="h",
            marker={**marker, "color": top_gini["UnemploymentRate"]},
            customdata=top_gini[["Poverty60", "UnemploymentRate"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gini: %{x:.2f}<br>"
                "Poverty60: %{customdata[0]:.1f}%<br>"
                "Unemployment: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
            name="Gini",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=top_poverty["Poverty60"],
            y=top_poverty["Municipality"],
            orientation="h",
            marker={**marker, "color": top_poverty["UnemploymentRate"]},
            customdata=top_poverty[["Gini", "UnemploymentRate"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Poverty60: %{x:.1f}%<br>"
                "Gini: %{customdata[0]:.2f}<br>"
                "Unemployment: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
            name="Poverty60",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Gini coefficient", row=1, col=1)
    fig.update_xaxes(title_text="Risk-of-poverty rate", ticksuffix="%", row=1, col=2)
    fig.update_layout(
        title=f"Where inequality and hardship split, {year}",
        margin={"r": 16, "t": 70, "l": 12, "b": 48},
        autosize=True,
        font={"family": "Arial, sans-serif"},
        showlegend=False,
        coloraxis={
            "colorscale": "Tealrose",
            "colorbar": {"title": "Unemployment"},
        },
    )
    path = VIS_DIR / "dk_inequality_poverty_rank.html"
    return write_plotly_html(fig, path)


def regression_summary(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    usable = df.dropna(subset=[x, y])
    if len(usable) < 3:
        return {
            "x": x,
            "y": y,
            "n": len(usable),
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
        }
    result = stats.linregress(usable[x], usable[y])
    return {
        "x": x,
        "y": y,
        "n": int(len(usable)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "r2": float(result.rvalue**2),
        "p_value": float(result.pvalue),
    }


def calculate_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Gini", "Poverty60"),
        ("Poverty60", "UnemploymentRate"),
        ("Poverty60", "LifeExpectancy"),
        ("Gini", "UnemploymentRate"),
        ("Gini", "TertiaryShare"),
        ("Gini", "LifeExpectancy"),
    ]
    rows = []
    municipal = municipal_rows(panel)
    for x, y in pairs:
        try:
            year = latest_year_with(panel, [x, y])
        except ValueError:
            rows.append({"Year": np.nan, **regression_summary(municipal, x, y)})
            continue
        cross_section = municipal[municipal["Year"] == year]
        rows.append({"Year": year, **regression_summary(cross_section, x, y)})
    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED_DIR / "regression_summary.csv", index=False)
    return results


def calculate_correlation_over_time(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Gini", "Poverty60", "Gini vs Poverty60"),
        ("Poverty60", "UnemploymentRate", "Poverty60 vs UnemploymentRate"),
    ]
    rows: list[dict[str, Any]] = []
    municipal = municipal_rows(panel)

    for x, y, relationship in pairs:
        years = sorted(
            int(year)
            for year in municipal.dropna(subset=[x, y])["Year"].dropna().unique()
        )
        for year in years:
            cross_section = municipal[municipal["Year"] == year]
            rows.append(
                {
                    "Year": year,
                    "relationship": relationship,
                    **regression_summary(cross_section, x, y),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED_DIR / "correlation_over_time.csv", index=False)
    return results


def plot_correlation_over_time(correlations: pd.DataFrame) -> Path:
    data = correlations.dropna(subset=["r2"]).copy()
    if data.empty:
        raise ValueError("No over-time correlation rows available to plot.")

    colors = {
        "Gini vs Poverty60": "#1f6f8b",
        "Poverty60 vs UnemploymentRate": "#b65f2a",
    }

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for relationship, group in data.groupby("relationship", sort=False):
        group = group.sort_values("Year")
        ax.plot(
            group["Year"],
            group["r2"],
            marker="o",
            linewidth=2.2,
            markersize=3.8,
            label=relationship,
            color=colors.get(relationship, "#5b6770"),
        )

    ax.set_title("The main relationships over time", fontsize=15, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cross-sectional R2")
    ax.set_ylim(0, min(1.0, max(0.65, data["r2"].max() + 0.08)))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    path = IMAGES_DIR / "dk_correlation_robustness.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter(panel: pd.DataFrame) -> Path:
    year = latest_year_with(panel, ["Poverty60", "UnemploymentRate"])
    data = municipal_rows(panel)
    data = data[
        (data["Year"] == year)
        & data["Poverty60"].notna()
        & data["UnemploymentRate"].notna()
    ].copy()

    fig = px.scatter(
        data,
        x="Poverty60",
        y="UnemploymentRate",
        color="Gini" if data["Gini"].notna().any() else None,
        hover_name="Municipality",
        hover_data={
            "MunicipalityCode": False,
            "Gini": ":.2f",
            "Poverty60": ":.2f",
            "UnemploymentRate": ":.2f",
            "TertiaryShare": ":.2f",
            "LifeExpectancy": ":.2f",
        },
        title=f"Municipal poverty and unemployment, {year}",
        labels={
            "Gini": "Gini coefficient",
            "Poverty60": "Risk-of-poverty rate, 60 percent threshold",
            "UnemploymentRate": "Unemployment rate",
        },
        color_continuous_scale="Tealrose",
    )

    summary = regression_summary(data, "Poverty60", "UnemploymentRate")
    if np.isfinite(summary["slope"]):
        x_line = np.linspace(data["Poverty60"].min(), data["Poverty60"].max(), 100)
        y_line = summary["intercept"] + summary["slope"] * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Linear fit, R2={summary['r2']:.2f}",
                line={"color": "#222222", "width": 2},
            )
        )

    fig.update_layout(
        margin={"r": 16, "t": 58, "l": 12, "b": 12},
        autosize=True,
        font={"family": "Arial, sans-serif"},
        legend={"orientation": "h", "y": -0.2},
    )
    path = VIS_DIR / "dk_poverty_unemployment_scatter.html"
    return write_plotly_html(fig, path)


def write_summary(
    panel: pd.DataFrame, regressions: pd.DataFrame, correlations: pd.DataFrame
) -> Path:
    latest_gini_year = latest_year_with(panel, ["Gini"])
    latest_poverty_year = latest_year_with(panel, ["Gini", "Poverty60"])
    latest_life_year = int(panel["Year"].max())
    national = panel[panel["MunicipalityCode"] == "000"].copy()
    latest_national = national[national["Year"] == latest_gini_year].iloc[0].to_dict()
    municipal = municipal_rows(panel)
    life_missing = municipal[
        (municipal["Year"] == latest_life_year) & municipal["LifeExpectancy"].isna()
    ][["MunicipalityCode", "Municipality"]]
    negative_s80s20 = municipal[municipal["S80S20"] < 0][
        ["Year", "MunicipalityCode", "Municipality", "S80S20"]
    ]
    payload = {
        "latest_gini_year": latest_gini_year,
        "latest_poverty_year": latest_poverty_year,
        "national_latest": {
            key: value
            for key, value in latest_national.items()
            if key in ["Year", "Gini", "P90P10", "S80S20", "Poverty60"]
        },
        "regressions": regressions.to_dict(orient="records"),
        "correlation_over_time": correlations.to_dict(orient="records"),
        "benchmark_distribution_stats": municipal_distribution_stats(panel),
        "latest_rank_summary": latest_rank_summary(panel),
        "data_caveats": {
            "panel_end_year": int(panel["Year"].max()),
            "panel_end_note": (
                "The merged panel ends in 2024 because the income inequality "
                "and poverty source tables end in 2024 in the current extract."
            ),
            "life_expectancy_missing_latest_year": life_missing.to_dict(
                orient="records"
            ),
            "negative_s80s20_source_rows": negative_s80s20.to_dict(orient="records"),
        },
    }
    path = PROCESSED_DIR / "summary_stats.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def make_figures(
    panel: pd.DataFrame,
    deciles: pd.DataFrame,
    regressions: pd.DataFrame,
    correlations: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "trend": plot_trend(panel),
        "deciles": plot_deciles(deciles),
        "distribution": plot_municipal_distribution(panel),
        "correlation_robustness": plot_correlation_over_time(correlations),
        "map": plot_map(panel),
        "rank_comparison": plot_rank_comparison(panel),
        "scatter": plot_scatter(panel),
        "summary": write_summary(panel, regressions, correlations),
    }
    return outputs


def run_pipeline(skip_figures: bool = False) -> dict[str, Any]:
    ensure_dirs()
    panel, deciles = build_panel()
    regressions = calculate_regressions(panel)
    correlations = calculate_correlation_over_time(panel)
    outputs: dict[str, Any] = {
        "panel_rows": len(panel),
        "decile_rows": len(deciles),
        "processed_files": [
            PROCESSED_DIR / "municipality_year.csv",
            PROCESSED_DIR / "decile_distribution.csv",
            PROCESSED_DIR / "missingness_summary.csv",
            PROCESSED_DIR / "regression_summary.csv",
            PROCESSED_DIR / "correlation_over_time.csv",
        ],
    }
    if not skip_figures:
        outputs["figures"] = make_figures(panel, deciles, regressions, correlations)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed CSVs and figures for the Denmark inequality project."
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only write processed data files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_pipeline(skip_figures=args.skip_figures)
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1

    print("Build complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
