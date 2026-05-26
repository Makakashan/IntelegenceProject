from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.budget.quality import apply_budget_quality_columns

from .constants import (
    CAPEX_COLUMNS,
    CF_INTERCEPT,
    CF_SLOPE,
    CLASSIFICATION_BANDS,
    DEFAULT_COMMISSIONING_YEAR,
    DEFAULT_GRID_CONNECTION_MODEL,
    DEFAULT_PROJECT_LIFETIME_YEARS,
    DISCOUNT_RATE,
    FLOATING_KEYWORDS,
    LCOE_COLUMNS,
    NEAREST_PROJECT_FEATURES,
    PROJECT_FIELDS,
)
from .utils import nan_metrics, read_project_csv, to_float_or_nan

ACCEPTED_BUDGET_VERIFICATION_LEVELS = {"A", "B"}


def is_floating_foundation(foundation_type: str | None) -> bool:
    if not isinstance(foundation_type, str):
        return False
    foundation = foundation_type.lower()
    return any(keyword in foundation for keyword in FLOATING_KEYWORDS)


def estimate_capacity_factor(row: pd.Series) -> float:
    capacity_factor = to_float_or_nan(row.get("capacity_factor"))
    if np.isfinite(capacity_factor) and capacity_factor > 0:
        return capacity_factor

    wind_speed = to_float_or_nan(row.get("mean_hub_wind_speed"))
    if not np.isfinite(wind_speed):
        return 0.40

    estimate = CF_SLOPE * wind_speed + CF_INTERCEPT
    return float(np.clip(estimate, 0.25, 0.65))


def calculate_capex(row: pd.Series) -> pd.Series:
    base_capex = to_float_or_nan(row.get("total_budget_EUR_2026"))
    capacity = to_float_or_nan(row.get("installed_capacity_MW"))
    if not np.isfinite(base_capex) or not np.isfinite(capacity):
        return nan_metrics(CAPEX_COLUMNS)
    if base_capex <= 0 or capacity <= 0:
        return nan_metrics(CAPEX_COLUMNS)

    cable_multiplier = 1.15 if row.get("grid_connection_model") == "TSO_provided" else 1.0
    total_capex = base_capex * cable_multiplier
    return pd.Series(
        {
            "CAPEX_total_EUR": total_capex,
            "CAPEX_unit_EUR_per_MW": total_capex / capacity,
        }
    )


def calculate_opex(row: pd.Series) -> float:
    base = 85.0 if bool(row.get("is_floating")) else 65.0
    distance = to_float_or_nan(row.get("distance_from_shore_km"))
    if not np.isfinite(distance):
        return base + 10.0
    if distance <= 50:
        return base + 5.0
    if distance <= 100:
        return base + 15.0
    return base + 25.0


def calculate_crcf(n_years: float) -> float:
    years = to_float_or_nan(n_years)
    if not np.isfinite(years) or years <= 0:
        return np.nan
    return DISCOUNT_RATE / (1.0 - (1.0 + DISCOUNT_RATE) ** (-years))


def calculate_lcoe(row: pd.Series) -> pd.Series:
    capex = to_float_or_nan(row.get("CAPEX_total_EUR"))
    production = to_float_or_nan(row.get("annual_production_MWh"))
    opex = to_float_or_nan(row.get("annual_OPEX_total_EUR"))
    lifetime = to_float_or_nan(row.get("project_lifetime_years"))

    if not all(np.isfinite(value) for value in (capex, production, opex, lifetime)):
        return nan_metrics(LCOE_COLUMNS)
    if capex <= 0 or production <= 0:
        return nan_metrics(LCOE_COLUMNS)

    crf = calculate_crcf(lifetime)
    if not np.isfinite(crf):
        return nan_metrics(LCOE_COLUMNS)

    annual_capex = capex * crf
    lcoe = (annual_capex + opex) / production
    return pd.Series(
        {
            "CRF": crf,
            "annual_CAPEX_EUR": annual_capex,
            "LCOE_EUR_per_MWh": lcoe,
        }
    )


def expected_lcoe_range(row: pd.Series) -> tuple[float, float]:
    try:
        commissioning_year = int(row.get("commissioning_year"))
    except (TypeError, ValueError):
        commissioning_year = DEFAULT_COMMISSIONING_YEAR

    if bool(row.get("is_floating")):
        return 60.0, 200.0
    if commissioning_year < 2015:
        return 50.0, 250.0
    return 30.0, 120.0


def validate_project(row: pd.Series) -> str:
    lcoe = to_float_or_nan(row.get("LCOE_EUR_per_MWh"))
    if not np.isfinite(lcoe):
        return "No data"

    low, high = expected_lcoe_range(row)
    if lcoe < low:
        return "Below expected"
    if lcoe > high:
        return "Above expected (outlier)"
    return "Valid"


def classify_outlier_type(row: pd.Series) -> str:
    if row.get("budget_quality_status") not in (None, "valid"):
        return "Data outlier"
    if row.get("budget_verification_level") == "C":
        return "Model uncertainty"
    if row.get("validation_status") == "Above expected (outlier)":
        return "Economic outlier"
    if row.get("validation_status") == "Below expected":
        return "Economic outlier"
    return "None"


def classify_project_quality(row: pd.Series) -> str:
    if row.get("validation_status") == "Above expected (outlier)":
        return "Outlier"

    lcoe = to_float_or_nan(row.get("LCOE_EUR_per_MWh"))
    if not np.isfinite(lcoe):
        return "No data"

    low, high = expected_lcoe_range(row)
    score = (lcoe - low) / (high - low)
    return classify_percentile(score)


def classify_percentile(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "No data"
    for limit, label in CLASSIFICATION_BANDS:
        if percentile <= limit:
            return label
    return CLASSIFICATION_BANDS[-1][1]


def lcoe_percentile(lcoe: float, reference_lcoe: pd.Series) -> float:
    value = to_float_or_nan(lcoe)
    reference = pd.to_numeric(reference_lcoe, errors="coerce").dropna()
    if not np.isfinite(value) or reference.empty:
        return np.nan
    return float((reference <= value).mean())


def classify_lcoe_by_distribution(lcoe: float, reference_lcoe: pd.Series) -> str:
    return classify_percentile(lcoe_percentile(lcoe, reference_lcoe))


def apply_data_driven_classification(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    outlier_mask = df_result["validation_status"].eq("Above expected (outlier)")
    reference_lcoe = df_result.loc[~outlier_mask, "LCOE_EUR_per_MWh"]
    if pd.to_numeric(reference_lcoe, errors="coerce").dropna().empty:
        df_result["classification"] = df_result.apply(classify_project_quality, axis=1)
        return df_result

    ranks = pd.to_numeric(reference_lcoe, errors="coerce").rank(pct=True, method="average")
    df_result.loc[~outlier_mask, "classification"] = ranks.apply(classify_percentile)
    df_result.loc[outlier_mask, "classification"] = "Outlier"
    return df_result


def classify_project_against_dataset(
    project_result: dict[str, Any],
    reference_df: pd.DataFrame,
) -> str:
    if project_result.get("validation_status") == "Above expected (outlier)":
        return "Outlier"

    if reference_df.empty or "LCOE_EUR_per_MWh" not in reference_df.columns:
        return str(project_result.get("classification", "No data"))

    if "validation_status" in reference_df.columns:
        reference_df = reference_df[reference_df["validation_status"] != "Above expected (outlier)"]

    label = classify_lcoe_by_distribution(
        to_float_or_nan(project_result.get("LCOE_EUR_per_MWh")),
        reference_df["LCOE_EUR_per_MWh"],
    )
    if label == "No data":
        return str(project_result.get("classification", "No data"))
    return label


def prepare_project_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = apply_budget_quality_columns(df, replace_total_budget=True)

    if "capacity_factor" in df.columns:
        df["capacity_factor"] = pd.to_numeric(df["capacity_factor"], errors="coerce")
    else:
        df["capacity_factor"] = np.nan

    if "project_lifetime_years" in df.columns:
        df["project_lifetime_years"] = df["project_lifetime_years"].fillna(
            DEFAULT_PROJECT_LIFETIME_YEARS
        )
    else:
        df["project_lifetime_years"] = DEFAULT_PROJECT_LIFETIME_YEARS

    if "commissioning_year" in df.columns:
        df["commissioning_year"] = df["commissioning_year"].fillna(DEFAULT_COMMISSIONING_YEAR)
    else:
        df["commissioning_year"] = DEFAULT_COMMISSIONING_YEAR

    if "grid_connection_model" in df.columns:
        df["grid_connection_model"] = df["grid_connection_model"].fillna(
            DEFAULT_GRID_CONNECTION_MODEL
        )
    else:
        df["grid_connection_model"] = DEFAULT_GRID_CONNECTION_MODEL

    df["is_floating"] = df["foundation_type"].apply(is_floating_foundation)
    return df


def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df.copy()
    capacity = pd.to_numeric(df_model["installed_capacity_MW"], errors="coerce")

    df_model[list(CAPEX_COLUMNS)] = df_model.apply(calculate_capex, axis=1)
    df_model["capacity_factor_est"] = df_model.apply(estimate_capacity_factor, axis=1)
    df_model["productivity_brutto_h"] = 8760.0 * df_model["capacity_factor_est"]
    df_model["productivity_netto_h"] = df_model["productivity_brutto_h"] * 0.85
    df_model["annual_production_MWh"] = df_model["productivity_netto_h"] * capacity
    df_model["OPEX_unit_kEUR_per_MW_year"] = df_model.apply(calculate_opex, axis=1)
    df_model["annual_OPEX_total_EUR"] = (
        df_model["OPEX_unit_kEUR_per_MW_year"] * 1000.0 * capacity
    )
    df_model[list(LCOE_COLUMNS)] = df_model.apply(calculate_lcoe, axis=1)
    df_model["validation_status"] = df_model.apply(validate_project, axis=1)
    df_model["outlier_type"] = df_model.apply(classify_outlier_type, axis=1)
    df_model["classification"] = df_model.apply(classify_project_quality, axis=1)
    return df_model


def load_and_analyse(csv_path: str | Path) -> pd.DataFrame:
    df = prepare_project_frame(read_project_csv(csv_path))
    mask = df["total_budget_EUR_2026"].notna() & df["installed_capacity_MW"].notna()
    if "budget_verification_level" in df.columns:
        mask = mask & df["budget_verification_level"].isin(ACCEPTED_BUDGET_VERIFICATION_LEVELS)
    df_model = add_calculated_columns(df.loc[mask])
    df_valid = df_model[df_model["LCOE_EUR_per_MWh"].notna()].copy()
    return apply_data_driven_classification(df_valid)


def classify_new_project(project_data: dict[str, Any]) -> dict[str, Any]:
    row_data = {field: project_data.get(field) for field in PROJECT_FIELDS}
    df_project = prepare_project_frame(pd.DataFrame([row_data]))
    return add_calculated_columns(df_project).iloc[0].to_dict()


def find_nearest_project(
    df: pd.DataFrame,
    new_data: dict[str, Any],
    features: Sequence[str] | None = None,
) -> pd.Series:
    if df.empty:
        raise ValueError("Historic dataset is empty; cannot compute similarity.")

    feature_names: Sequence[str] = features if features is not None else NEAREST_PROJECT_FEATURES
    usable_features = [
        feature
        for feature in feature_names
        if feature in df.columns and np.isfinite(to_float_or_nan(new_data.get(feature)))
    ]
    if not usable_features:
        row = df.iloc[0].copy()
        row["similarity_distance"] = np.nan
        return row

    x = df[usable_features].copy().astype(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, 1.0)
    x_norm = (x - means) / stds
    new_vec = np.array(
        [
            (to_float_or_nan(new_data[feature]) - means[feature]) / stds[feature]
            for feature in usable_features
        ],
        dtype=float,
    )
    distances = np.linalg.norm(x_norm.values - new_vec, axis=1)
    nearest_idx = int(np.argmin(distances))
    row = df.iloc[nearest_idx].copy()
    row["similarity_distance"] = float(distances[nearest_idx])
    return row
