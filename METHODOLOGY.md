# Methodology

## Data Sources

The core project dataset is `input/wind_dataset.csv`. The baseline project attributes come from
`input/european_offshore_wind_capex.csv` and wind farm metadata from `windfarminfo.com`.
Budget enrichment can be run with `wind_budget.py`, but web-derived values are not accepted blindly.

Budget sources are ranked:

- `A`: official developer, government, regulator, EIB, or equivalent primary source.
- `B`: specialist industry database or curated declared dataset value, including windfarminfo-based values.
- `C`: generic web lookup that passed numeric plausibility checks but still needs manual verification.
- `D`: missing, blocked, or rejected source.

LCOE, budget estimation, and ML training use only `A` and `B` budget records.

## Budget Normalization

Raw budget text is parsed into `total_budget_EUR_2026` using:

- detected currency and unit (`million`, `billion`, `mn`, `bn`);
- historical FX fallback tables;
- Eurozone CPI inflation to the target year 2026;
- project capacity to compute `budget_EUR_per_MW`.

Combined budgets such as `EUR 2.2 billion (combined 1+2)` are allocated across projects with the
same raw combined budget in proportion to installed capacity.

Manual overrides can be placed in `input/budget_overrides.csv` with:

```text
wind_farm_name,budget_raw,budget_year,source_url,verification_level,notes
```

Overrides are used only when they pass plausibility checks and have a stronger verification level
than the automatic lookup.

## CAPEX, OPEX, And LCOE

CAPEX is based on normalized budget:

```text
CAPEX_total = total_budget_EUR_2026 * grid_connection_multiplier
CAPEX_unit = CAPEX_total / installed_capacity_MW
```

The grid multiplier is `1.15` for TSO-provided connection and `1.00` otherwise.

Capacity factor is taken from the dataset when present. Missing values are estimated from wind speed
with a clipped linear relation. OPEX is rule-based by foundation type and distance from shore.

LCOE is computed as:

```text
LCOE = (CAPEX_total * CRF + annual_OPEX) / annual_production_MWh
```

where CRF uses a 5 percent discount rate and the project lifetime.

## Classification And Outliers

Projects are classified by LCOE distribution against validated historic projects. Outliers are
separated into:

- `Data outlier`: rejected or missing data.
- `Economic outlier`: LCOE outside expected range.
- `Model uncertainty`: low-confidence or weakly verified budget evidence.

This avoids hiding outliers behind an average percentile label.

## Budget Estimate For New Projects

The classifier estimates new-project budget using validated nearest neighbours:

- only `A/B` verified records are used;
- candidates are matched by capacity, shore distance, wind speed, commissioning year, and foundation;
- the point estimate is a weighted median EUR/MW;
- the report includes an interquartile budget range and a High/Medium/Low confidence label.

Confidence combines source quality, number of neighbours, normalized distance, EUR/MW spread, and
foundation match.

## ML Validation

`model_training.py` compares:

- median baseline;
- Ridge regression;
- Random Forest;
- Histogram Gradient Boosting.

The target is `log(budget_EUR_per_MW)`. The report includes holdout MAE, MAPE, R2 on the log target,
cross-validation MAE on the log target, feature importance where available, and errors by country and
foundation type.

If the ML model does not beat the median baseline, the project explicitly treats ML as supporting
evidence and keeps the robust neighbour estimator as the primary estimate.

## Limitations

The dataset is suitable for educational decision-support prototyping. It is not a bankable
investment model. Offshore wind budgets may include different scope boundaries across sources
(transmission assets, financing costs, port works, or multi-project packages), so high-confidence
results require project-level source review.
