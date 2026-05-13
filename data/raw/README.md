# Raw Data

Run this command from the project root:

```bash
python scripts/download_statbank_data.py
```

The downloader writes StatBank CSVs with code-and-label values:

- `IFOR41.csv`
- `IFOR32.csv`
- `IFOR12P.csv`
- `AUP02.csv`
- `HFUDD11.csv`
- `HISBK.csv`
- `BOL101.csv`
- `BOLRD.csv`

It also writes a compact `denmark_municipalities.geojson` for the Plotly
choropleth and metadata JSON files under `data/raw/metadata/`. The geometry is
trimmed to the map join fields, simplified, and rounded so it remains practical
for GitHub Pages.

`BOL101.csv` is the municipal housing tenure source. `BOLRD.csv` has national
tenure and time only, so the build uses it as a quality-control cross-check for
the national BOL101 owner/tenant totals.
