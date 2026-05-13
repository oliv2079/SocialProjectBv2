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

It also writes a compact `denmark_municipalities.geojson` for the Plotly
choropleth and metadata JSON files under `data/raw/metadata/`. The geometry is
trimmed to the map join fields, simplified, and rounded so it remains practical
for GitHub Pages.
