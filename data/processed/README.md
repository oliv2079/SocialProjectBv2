# Processed Data

Run:

```bash
python scripts/build_project.py
```

Expected outputs:

- `municipality_year.csv`
- `decile_distribution.csv`
- `missingness_summary.csv`
- `regression_summary.csv`
- `correlation_over_time.csv`
- `housing_tenure_crosscheck.csv`
- `summary_stats.json`

`summary_stats.json` also records the benchmark municipal distribution stats,
latest-year ranked inequality/poverty lists, housing tenure summary, BOL101 vs.
BOLRD national housing-tenure cross-check, over-time relationship checks, and
key data caveats used in the website and notebook.
