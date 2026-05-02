# Wind Farm Budget Enricher

Authors: Vatset Maksym, Krauchyk Maksim

## Overview

This repository contains two related scripts to enrich a CSV of wind farm names with estimated total project budgets (converted to EUR):

- `wind_budget_csv.py` — lightweight CSV enricher that searches for budget strings and converts them to EUR using approximate FX rates.
- `wind_budget_inflation.py` — more advanced enricher that converts historical budgets to a target year (default 2026) by applying historical FX rates (ECB via Frankfurter API) and CPI inflation (World Bank CPI).

Both scripts use Serper (a Google Search API) to fetch snippets and attempt to extract budget amounts from public sources. All network requests and metadata are logged to `logs/requests_<DATE>.jsonl`.

## Key concepts

- The scripts look for budget phrases (e.g. "£9 billion", "$3.6bn") in search result snippets.
- `wind_budget_csv.py` converts found amounts to EUR using a configurable approximate FX table.
- `wind_budget_inflation.py` does a three-step conversion:
   1. parse the raw budget string (number + unit)
   2. convert from original currency to EUR using the ECB historical rate for the budget year (via Frankfurter)
   3. inflate EUR from the budget year to the target year using World Bank CPI

## Requirements

- Python 3.8+
- Packages: `requests`, `pandas`

You can install them with:

```bash
pip install requests pandas
```

Or run with `uv` (if you use the `uv` runner shipped in the project environment) which handles dependencies automatically:

```bash
uv run wind_budget_inflation.py
uv run wind_budget_csv.py
```

## Serper API key

Both scripts require a Serper API key to perform Google-like searches. Get a free key (2500 requests) at:

https://serper.dev

Then export it into your environment:

```bash
export SERPER_API_KEY="your_key_here"
```

If the environment variable is not present the scripts will prompt for the key on startup.

## Usage

Basic usage (CSV must contain a `wind_farm_name` column):

- wind_budget_csv.py (simple FX conversion):

```bash
python wind_budget_csv.py --input windTurbineData.csv --workers 5
```

- wind_budget_inflation.py (historical FX + CPI inflation → target EUR year, default 2026):

```bash
python wind_budget_inflation.py --input windTurbineData.csv --workers 5
```

Options:

- `--input, -i` — path to input CSV (default: `windTurbineData.csv`).
- `--workers, -w` — number of parallel threads (default: 5). Do not exceed Serper rate limits.

Example outputs produced by the scripts:

- `<input_stem>_enriched.csv` (CSV with added budget columns)
- `<input_stem>_budgets.json` (JSON list of lookup results)
- `logs/requests_<DATE>.jsonl` (JSONL log of API calls and metadata)

## Columns added to your CSV

Common columns added by the scripts include:

- `total_budget_raw` — budget string extracted from search snippets (or "Not found").
- `total_budget_EUR` / `total_budget_EUR_fmt` — the converted budget formatted as EUR (short format like €1.23B or €4.5M).
- `budget_year` (inflation script) — inferred year of the budget figure.
- `budget_currency` (inflation script) — ISO currency code inferred from the budget string.
- `total_budget_EUR_2026` (inflation script) — converted budget in EUR for the target year (numeric).
- `budget_source` / `source` — first discovered source URL.
- `budget_lookup_date` / `lookup_date` — date when the lookup was performed.

## Logging

All API calls and metadata are appended to `logs/requests_<DATE>.jsonl`. There is also a file logger `logs/requests_<DATE>.log` with human-readable log messages.

## Notes and limitations

- Budget extraction uses regex heuristics and is best-effort. It may miss values or pick unrelated numbers.
- FX data: the inflation script uses Frankfurter (ECB-derived) rates when available and falls back to an internal table for older or missing years.
- CPI series: the script tries World Bank CPI for the Eurozone and uses an internal fallback if unavailable.
- Serper API rate limits: avoid setting `--workers` too high. The code includes a small inter-request delay per thread.

## Customization

- Adjust `TARGET_YEAR` in `wind_budget_inflation.py` if you want a different target (not 2026).
- Edit the `FX` table in `wind_budget_csv.py` to fine-tune approximate conversion rates.

## License & Authors

Authors: Vatset Maksym, Krauchyk Maksim

(Feel free to request any further rewording, additions or a Russian version of this README.)
