# Offshore Wind LCOE Calculator

A project for analyzing offshore wind farms. It includes:

- `wind_budget.py` — a script that enriches the source CSV with budgets converted to EUR 2026.
- `windAlgorythm.ipynb` — a Jupyter Notebook for LCOE calculation, charts, and mapping.
- `windTurbineData.csv` — the source dataset.

## Running the Notebook

This project is meant to be used through Jupyter Notebook. To open Jupyter Lab, run:

```bash
uv run --with jupyter jupyter lab
```

Then open `windAlgorythm.ipynb`.

If you want fresh data before working in the notebook, run the budget enrichment script first:

```bash
uv run python wind_budget.py --input windTurbineData.csv --workers 5
```

## Environment setup

If dependencies are not installed yet:

```bash
uv sync --group dev
```

## Input format

The input file `windTurbineData.csv` must be a UTF-8 CSV. For the enrichment script, only `wind_farm_name` is required, but the notebook works best with the full dataset.

### Column reference

| Column                   | Meaning                                | Type / example                                | Required                           |
| ------------------------ | -------------------------------------- | --------------------------------------------- | ---------------------------------- |
| `wind_farm_name`         | Wind farm name                         | `Hornsea 1`                                   | Yes                                |
| `country`                | Country or market                      | `UK`                                          | Yes                                |
| `commissioning_year`     | Year the project entered operation     | `2020`                                        | Recommended                        |
| `installed_capacity_MW`  | Installed capacity in MW               | `1218`                                        | Yes                                |
| `turbine_model`          | Turbine model                          | `Siemens Gamesa SG 14-236 DD`                 | Recommended                        |
| `foundation_type`        | Foundation type                        | `Monopile`, `Jacket`                          | Yes                                |
| `water_depth_m`          | Water depth in meters                  | `35`                                          | Recommended                        |
| `distance_from_shore_km` | Distance from shore in km              | `69`                                          | Recommended                        |
| `total_budget_EUR`       | Raw budget text, not a final EUR value | `GBP 4.2 billion`, `USD 3.64 billion`, `null` | Yes for enrichment, `null` allowed |
| `lat`                    | Latitude                               | `52.7`                                        | Recommended                        |
| `lon`                    | Longitude                              | `2.86`                                        | Recommended                        |
| `mean_hub_wind_speed`    | Mean hub-height wind speed in m/s      | `9.2`                                         | Recommended                        |
| `grid_connection_model`  | Grid connection model                  | `TSO_provided`                                | Recommended                        |
| `capacity_factor`        | Capacity factor                        | `0.55`, empty                                 | Optional                           |
| `project_lifetime_years` | Project lifetime in years              | `25`                                          | Recommended                        |
| `data source`            | Source of the original record          | `windfarminfo.com`                            | Recommended                        |

### Data notes

- Use a dot for decimals, for example `9.2`, not `9,2`.
- `total_budget_EUR` may contain a currency and a unit, for example `GBP 780-900 million`.
- Empty values are allowed for fields you do not have yet.
- After running `wind_budget.py`, the file `output/windTurbineData_enriched_2026EUR.csv` will be created and used by the notebook.

## Output files

After enrichment, the following files are created:

- `output/windTurbineData_enriched_2026EUR.csv` — CSV with calculated fields added.
- `output/windTurbineData_budgets_2026EUR.json` — JSON with budget lookup details.
- `logs/requests_<DATE>.jsonl` and `logs/requests_<DATE>.log` — request and processing logs.

## Workflow summary

1. The script searches for budget information by wind farm name.
2. It detects the currency and budget year.
3. It converts the amount to EUR using the exchange rate for that year.
4. It adjusts the amount to 2026 using inflation.
5. The notebook uses the enriched file for analysis and visualizations.

## Note

Budget lookup and validation use the Serper API, so a valid API key is required to run the script.
