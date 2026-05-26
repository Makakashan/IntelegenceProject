# Offshore Wind Intelligence

Python toolkit for offshore wind project analysis: budget normalization, data quality validation,
CAPEX/OPEX/LCOE calculation, ML-based CAPEX benchmarking, mapping, and new-project classification.

The project is designed as an end-to-end offshore wind decision-support prototype. It keeps the
domain formulas transparent while adding source verification, confidence scoring, model validation,
and reproducible reports.

## What It Does

- normalizes project budgets into `EUR 2026`;
- validates budget plausibility with `EUR/MW` bounds;
- ranks budget source quality with `A/B/C/D` verification levels;
- applies manual curated overrides from `input/budget_overrides.csv`;
- handles combined budgets by allocating them proportionally by capacity;
- computes CAPEX, OPEX, CRF, annual production, and LCOE;
- classifies projects against validated historic projects;
- estimates a new project's budget with validated nearest neighbours;
- reports budget range, confidence score, confidence reasons, and outlier type;
- compares ML models against a median baseline.

## Project Structure

```text
.
├── input/
│   ├── european_offshore_wind_capex.csv
│   ├── wind_dataset.csv
│   └── budget_overrides.csv
├── examples/
│   └── new_project_500mw.json
├── output/
├── src/
│   ├── budget/
│   └── lcoe/
├── tests/
├── wind_budget.py
├── wind_farm_classifier.py
├── model_training.py
├── windAlgorithm.ipynb
├── METHODOLOGY.md
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
uv sync --group dev
cp .env.example .env
```

Set a Serper API key in `.env` if you want to run web budget enrichment:

```env
SERPER_API_KEY=your_serper_api_key_here
```

The analysis and classifier can run from the existing cleaned dataset without rerunning web search.

## Data

The main dataset is:

```text
input/wind_dataset.csv
```

Current budget quality fields include:

- `total_budget_raw` - raw selected budget text;
- `original_budget_raw` - original dataset budget before enrichment/selection;
- `total_budget_EUR_2026` - normalized budget in EUR 2026;
- `budget_EUR_per_MW` - normalized budget per installed MW;
- `budget_quality_status` - `valid`, `missing`, or `rejected`;
- `budget_source_type` - `declared_csv`, `internet_lookup`, `manual_override`, or `none`;
- `budget_verification_level` - `A`, `B`, `C`, or `D`;
- `budget_confidence` - numeric confidence score.

Verification levels:

- `A`: official developer, regulator, government, EIB, or equivalent primary source;
- `B`: specialist industry source or curated declared dataset value;
- `C`: generic web lookup that passed plausibility checks but requires manual review;
- `D`: rejected or missing.

LCOE, budget estimation, and ML training use only `A/B` budget rows.

Manual budget corrections are stored in:

```text
input/budget_overrides.csv
```

Schema:

```text
wind_farm_name,budget_raw,budget_year,source_url,verification_level,notes
```

## Budget Enrichment

Run web budget enrichment:

```bash
uv run python wind_budget.py --input input/wind_dataset.csv --output-dir output --workers 5
```

The script writes:

```text
output/wind_dataset.csv
output/wind_dataset_budget_lookup_2026EUR.json
```

Do not blindly replace the input dataset with the output. Web lookup can find sector-wide or
unrelated values. The quality layer prefers manual overrides and declared project-level budgets over
weaker web lookup values.

## Classify A New Project

Run the included 500 MW example:

```bash
uv run python wind_farm_classifier.py \
  --dataset input/wind_dataset.csv \
  --json examples/new_project_500mw.json
```

The classifier estimates budget from validated similar projects and writes:

```text
output/classification_result.json
output/classification_preview.html
output/classification_history/
output/history/
```

The report includes:

- point budget estimate;
- budget range;
- confidence label and reasons;
- CAPEX/OPEX/LCOE metrics;
- outlier type;
- nearest historical project;
- ML model comparison and prediction.

## ML Training

Run model comparison:

```bash
uv run python model_training.py --dataset input/wind_dataset.csv
```

The training report compares:

- median baseline;
- Ridge regression;
- Random Forest;
- Histogram Gradient Boosting.

Target:

```text
log(budget_EUR_per_MW)
```

Metrics:

- holdout MAE in `EUR/MW`;
- MAPE;
- R2 on log target;
- cross-validation MAE on log target;
- errors by country and foundation;
- feature importance where available.

The report is saved to:

```text
output/model_training_report.json
```

## Notebook

```bash
uv run --with jupyter jupyter lab
```

Open:

```text
windAlgorithm.ipynb
```

## Tests

```bash
uv run pytest
```

The test suite covers:

- budget parsing and currency conversion;
- blocked social sources;
- source verification ranking;
- combined budget allocation;
- protection against bad web lookup replacing good declared budgets;
- validated-reference budget estimation;
- outlier classification;
- classifier CLI smoke test.

## Methodology

See:

```text
METHODOLOGY.md
```

It documents data sources, budget normalization, source verification, combined-budget handling,
CAPEX/OPEX/LCOE formulas, ML validation, confidence scoring, and limitations.

## Current Status

The cleaned dataset currently contains:

```text
182 projects
127 A/B verified projects usable for LCOE and ML
```

The project is suitable as a strong educational decision-support prototype. It is not a bankable
investment model; project-level financial decisions still require source-by-source manual review.
