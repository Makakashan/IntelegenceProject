# Offshore Wind Intelligence

Python toolkit for offshore wind project analysis: budget enrichment, CAPEX/OPEX/LCOE calculation, mapping, and project classification.

## Project Structure

```text
.
├── .env.example
├── pyproject.toml
├── uv.lock
├── wind_budget.py
├── wind_farm_classifier.py
├── windAlgorithm.ipynb
├── input/
├── output/
└── src/
    ├── budget/
    └── lcoe/
```

## Setup

```bash
uv sync --group dev
cp .env.example .env
```

Set your Serper API key in `.env`:

```env
SERPER_API_KEY=your_serper_api_key_here
```

## CSV Convention

Use two CSV names:

- `input/<name>_for_clean.csv` for a raw file that still needs budget enrichment;
- `input/<name>_dataset.csv` for the clean dataset used by the notebook and classifier.

The raw CSV should contain at least a `wind_farm_name` column. After enrichment, move or copy the generated `output/<name>_dataset.csv` into `input/` if you want it to be the default analysis dataset.

The project defaults to:

```text
input/wind_for_clean.csv
input/wind_dataset.csv
```

## Run

Budget enrichment:

```bash
uv run python wind_budget.py --input input/<name>_for_clean.csv --output-dir output --workers 5
```

Project classifier:

```bash
uv run python wind_farm_classifier.py --dataset input/<name>_dataset.csv
```

Notebook:

```bash
uv run --with jupyter jupyter lab
```

Open `windAlgorithm.ipynb`.

## Outputs

Generated files are written to `output/`.

Typical outputs:

- `<name>_dataset.csv` with normalized budget values;
- JSON budget lookup details;
- `classification_result.json` from the classifier;
- notebook exports such as LCOE results CSV and HTML maps.
