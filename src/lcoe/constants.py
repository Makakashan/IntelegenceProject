from __future__ import annotations

from pathlib import Path

CF_SLOPE: float = 0.0960
CF_INTERCEPT: float = -0.4125
DISCOUNT_RATE: float = 0.05

FLOATING_KEYWORDS = (
    "floating",
    "spar",
    "semi-submersible",
    "tlp",
    "damping",
    "sath",
)

PROJECT_FIELDS = (
    "total_budget_EUR_2026",
    "installed_capacity_MW",
    "grid_connection_model",
    "foundation_type",
    "distance_from_shore_km",
    "mean_hub_wind_speed",
    "capacity_factor",
    "project_lifetime_years",
    "commissioning_year",
)

DEFAULT_GRID_CONNECTION_MODEL = "developer_provided"
DEFAULT_PROJECT_LIFETIME_YEARS = 25.0
DEFAULT_COMMISSIONING_YEAR = 2026
DEFAULT_DATASET_CANDIDATES = (
    Path("input") / "wind_dataset.csv",
    Path("output") / "wind_dataset.csv",
    Path("wind_dataset.csv"),
)
OUTPUT_DIR = Path("output")
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_result.json"
CLASSIFICATION_HISTORY_DIR = OUTPUT_DIR / "classification_history"
CLASSIFICATION_PREVIEW_PATH = OUTPUT_DIR / "classification_preview.html"
OUTPUT_HISTORY_DIR = OUTPUT_DIR / "history"

CAPEX_COLUMNS = ("CAPEX_total_EUR", "CAPEX_unit_EUR_per_MW")
LCOE_COLUMNS = ("CRF", "annual_CAPEX_EUR", "LCOE_EUR_per_MWh")
CLASSIFICATION_BANDS = (
    (0.20, "Good"),
    (0.40, "Above average"),
    (0.60, "Average"),
    (0.80, "Below average"),
    (1.00, "Bad"),
)
NEAREST_PROJECT_FEATURES = (
    "installed_capacity_MW",
    "distance_from_shore_km",
    "mean_hub_wind_speed",
    "commissioning_year",
)

COMPUTED_METRIC_KEYS = (
    "CAPEX_total_EUR",
    "CAPEX_unit_EUR_per_MW",
    "capacity_factor_est",
    "annual_production_MWh",
    "OPEX_unit_kEUR_per_MW_year",
    "annual_OPEX_total_EUR",
    "CRF",
    "annual_CAPEX_EUR",
    "LCOE_EUR_per_MWh",
)

FIELD_DESCRIPTIONS = {
    "project_parameters.total_budget_EUR_2026": "Total budget (2026, EUR)",
    "project_parameters.installed_capacity_MW": "Installed capacity (MW)",
    "project_parameters.grid_connection_model": "Grid connection model",
    "project_parameters.foundation_type": "Foundation type",
    "project_parameters.distance_from_shore_km": "Distance to shore (km)",
    "project_parameters.mean_hub_wind_speed": "Mean wind speed (m/s)",
    "project_parameters.capacity_factor": "Specified capacity factor",
    "project_parameters.project_lifetime_years": "Project lifetime (years)",
    "project_parameters.commissioning_year": "Commissioning year",
    "budget_estimate.estimated_total_budget_EUR_2026": "Estimated total budget (2026, EUR)",
    "budget_estimate.method": "Budget estimate method",
    "budget_estimate.reference_projects": "Reference projects",
    "budget_estimate.unit_budget_EUR_per_MW": "Estimated budget per MW, EUR",
    "budget_estimate.confidence_score": "Budget estimate confidence score",
    "budget_estimate.confidence_label": "Budget estimate confidence",
    "budget_estimate.confidence_reasons": "Budget estimate confidence reasons",
    "budget_estimate.estimated_budget_range_EUR_2026.low": "Estimated budget range low (2026, EUR)",
    "budget_estimate.estimated_budget_range_EUR_2026.high": "Estimated budget range high (2026, EUR)",
    "computed_metrics.CAPEX_total_EUR": "Total capital expenditure (CAPEX), EUR",
    "computed_metrics.CAPEX_unit_EUR_per_MW": "CAPEX per 1 MW, EUR",
    "computed_metrics.capacity_factor_est": "Estimated capacity factor",
    "computed_metrics.annual_production_MWh": "Annual production (MWh)",
    "computed_metrics.OPEX_unit_kEUR_per_MW_year": "OPEX per 1 MW (kEUR/year)",
    "computed_metrics.annual_OPEX_total_EUR": "Annual operating expenses (OPEX), EUR",
    "computed_metrics.CRF": "Capital recovery factor (CRF)",
    "computed_metrics.annual_CAPEX_EUR": "Annual CAPEX, EUR",
    "computed_metrics.LCOE_EUR_per_MWh": "Levelized cost of energy (LCOE), EUR/MWh",
    "validation.status": "Threshold status",
    "validation.outlier_type": "Outlier type",
    "validation.model_uncertainty": "Model uncertainty",
    "validation.classification_algorithmic": "Algorithmic classification",
    "validation.nearest_project_name": "Nearest project (historical)",
    "validation.nearest_project_distance": "Distance to nearest (in normalized feature space)",
    "validation.classification_nearest": "Nearest-project classification",
    "ml_model.model": "Best ML model",
    "ml_model.conclusion": "ML validation conclusion",
    "ml_model.model_comparison": "ML model comparison",
    "ml_model.error_by_country": "ML errors by country",
    "ml_model.error_by_foundation": "ML errors by foundation",
    "ml_model.feature_importance": "ML feature importance",
    "ml_model.baseline_MAE_EUR_per_MW": "Baseline MAE (EUR/MW)",
    "ml_model.best_test_MAE_EUR_per_MW": "Best model MAE (EUR/MW)",
    "ml_model.best_test_MAPE_percent": "Best model MAPE",
    "ml_model.best_test_R2_log_target": "Best model R2 (log target)",
    "ml_model.beats_baseline": "ML beats baseline",
}
