# Municipal Inequality and Hardship in Denmark

This is a course-aligned narrative data project on inequality across Danish
municipalities. We follow the assignment workflow: clean the data with pandas,
validate the schemas, create static charts with matplotlib and interactive
charts with Plotly, run simple regressions and correlations, and publish a
single-page static website.

## Research Question

How unequal are Danish municipalities, how has that changed over time, and why
are the municipalities with the highest income inequality not always the same as
the municipalities with the strongest poverty and unemployment pressure?

## Project Structure

- `scripts/download_statbank_data.py`: downloads official raw data.
- `scripts/build_project.py`: creates processed CSVs, figures, summary JSON, and
  Plotly HTML.
- `notebooks/denmark_inequality_project.ipynb`: course-style explainer notebook.
- `notebooks/denmark_inequality_project.html`: exported explainer notebook for
  the website.
- `data/raw/`: downloaded StatBank CSVs and municipality GeoJSON.
- `data/processed/`: cleaned municipality-year outputs.
- `images/`: static matplotlib figures.
- `visualizations/`: exported Plotly HTML files.
- `index.html`: single-page magazine-style website.

## Reproduce

Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Download official data:

```bash
python scripts/download_statbank_data.py
```

Build processed data and figures:

```bash
python scripts/build_project.py
```

Run the notebook top-to-bottom:

```bash
python -m nbconvert --to notebook --execute notebooks/denmark_inequality_project.ipynb --inplace
```

Export the explainer notebook for the website:

```bash
python -m nbconvert --to html notebooks/denmark_inequality_project.ipynb --output denmark_inequality_project.html --output-dir notebooks
```

Preview the website:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Data Sources

Core StatBank tables:

- `IFOR41`: income distribution on equivalised disposable income.
- `IFOR32`: average equivalised disposable income by decile.
- `IFOR12P`: persons in risk-of-poverty families.
- `AUP02`: unemployed in percent of the labour force.
- `HFUDD11`: educational attainment, ages 15-69.
- `HISBK`: life expectancy for newborn babies.
- `BOL101`: dwellings by region, resident type, tenure, ownership, use, year of
  construction, and time.
- `BOLRD`: national dwellings with registered population by tenure, used as a
  quality-control cross-check for BOL101 national tenure totals.

Municipality boundaries are downloaded from the public Dataforsyningen API.

## Method

The main panel is municipality-year level. Monthly unemployment is aggregated
to annual means. Life expectancy uses the end year of the published rolling
period. Tertiary education share is calculated as short-cycle higher education,
vocational bachelor, bachelor, master, and PhD categories divided by the total
education population. Housing tenure is calculated from BOL101 as owner- and
tenant-occupied dwellings with registered population, matched only by the same
municipality-year. BOLRD is not municipality-level in StatBank, so it is used
only to validate national owner/tenant totals from the BOL101 pipeline.

The analysis uses the tools expected in the course: distribution summaries,
Pearson-style correlations, simple fitted lines, slope, intercept, and `R2`.
These results are descriptive; we do not draw causal conclusions. The website
uses seven focused visualizations: a national trend, a municipal distribution
view, an interactive municipal inequality map, an interactive ranked comparison
of inequality and poverty, an interactive housing-tenure bridge, an interactive
poverty-unemployment scatter, and an over-time `R2` robustness figure. The main
insight is that municipal inequality and municipal hardship are related, but
they often point to different places: the latest Gini-poverty cross-section is
nearly flat, while poverty-unemployment is strong and renter-heavy housing
composition helps explain where hardship concentrates.

## Course Alignment

- **Project Assignment B deliverables**: `index.html` is the public website and
  `notebooks/denmark_inequality_project.html` is the linked explainer notebook.
- **Notebook-first workflow**: the notebook contains motivation, dataset,
  cleaning/preprocessing, basic stats, data analysis, genre, visualization,
  discussion, contributions, and references sections.
- **Week 1-2 data work**: raw StatBank CSVs are loaded into pandas, code-label
  fields are split, schemas are normalized, and missingness and source notes are
  exported.
- **Week 3-4 statistics**: the analysis uses distributions, Pearson-style
  correlations, simple linear fits, `R2`, and non-causal interpretation.
- **Week 5-6 visualization**: the site includes a Plotly choropleth map, an
  interactive ranked comparison, a housing bridge view, and a separate
  interactive Plotly scatterplot with hover details and a fitted line. A static
  robustness figure checks whether the key relationships persist over time.
- **Week 7-8 website/story**: the website follows the Segel and Heer magazine
  genre, uses a guided six-visual sequence, includes captions, and targets a
  general reader rather than a technical audience.

## Data Caveats

- The main merged panel ends in 2024 because the income inequality and poverty
  source tables end in 2024 in the current local extract.
- Life expectancy is missing in the 2024 merged panel for four small island
  municipalities: Ærø, Fanø, Samsø, and Læsø.
- StatBank reports one implausible negative S80/S20 value for Rudersdal in
  1994. We flag it and do not base any argument on it.
- BOL101 housing tenure is available for 2010-2020 and 2023-2026 in the current
  extract. The 2021 and 2022 housing years are closed by Statistics Denmark due
  to BBR data errors, and 2026 housing observations are not carried back into
  the 2024 income panel.
- BOLRD validates the national tenure totals only; it does not replace BOL101
  for municipal tenure joins because it has tenure and time, but no municipality
  dimension in the StatBank extract.
- Open municipal wealth, housing-cost, and household consumption data are
  useful context, but they are not included because the current
  municipality-year panel is already enough to support the main descriptive
  finding.

## Reproducibility and Submission Notes

- Assignment A was submitted separately.
- This is a solo project submission.
- The checked-in folder structure matches the relative paths used by the
  homepage, notebook export, scripts, images, data, and interactive
  visualizations.
- To reproduce the submitted outputs from checked-in raw data, run
  `python scripts/build_project.py`, then execute and export the notebook with
  the `nbconvert` commands above.
