from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any, Callable, TypeVar
from datetime import datetime

import numpy as np
import pandas as pd

from src.lcoe.calculations import (
    ACCEPTED_BUDGET_VERIFICATION_LEVELS,
    classify_new_project,
    classify_project_against_dataset,
    find_nearest_project,
    load_and_analyse,
)
from src.lcoe.constants import (
    CLASSIFICATION_HISTORY_DIR,
    CLASSIFICATION_PREVIEW_PATH,
    CLASSIFICATION_REPORT_PATH,
    COMPUTED_METRIC_KEYS,
    DEFAULT_COMMISSIONING_YEAR,
    DEFAULT_DATASET_CANDIDATES,
    DEFAULT_GRID_CONNECTION_MODEL,
    DEFAULT_PROJECT_LIFETIME_YEARS,
    FIELD_DESCRIPTIONS,
    OUTPUT_HISTORY_DIR,
)
from src.lcoe.utils import flatten_dict
from src.lcoe.ml import build_capex_ml_report

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse offshore wind projects and classify a new proposal."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to the historic project dataset. If omitted, uses "
            "input/wind_dataset.csv when present."
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to a JSON file containing new project data. If omitted, prompts are used.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_DATASET_CANDIDATES[-1]


def load_historic_projects(dataset_path: Path) -> pd.DataFrame:
    try:
        df_valid = load_and_analyse(dataset_path)
        print(f"Loaded {len(df_valid)} projects from '{dataset_path}'.")
        return df_valid
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return pd.DataFrame()


def prompt_value(
    prompt: str,
    default: T | None,
    converter: Callable[[str], T],
    error_message: str,
    required: bool = False,
) -> T | None:
    suffix = f" [default: {default}]" if default is not None and not required else ""
    while True:
        value = input(f"{prompt}{suffix}: ")
        if not value.strip():
            if not required:
                return default
            print("This value is required.")
            continue
        try:
            return converter(value)
        except ValueError:
            print(error_message)


def prompt_float(
    prompt: str,
    default: float | None = None,
    required: bool = False,
) -> float | None:
    return prompt_value(prompt, default, float, "Invalid number. Try again.", required)


def prompt_int(
    prompt: str,
    default: int | None = None,
    required: bool = False,
) -> int | None:
    return prompt_value(prompt, default, int, "Invalid integer. Try again.", required)


def prompt_choice(prompt: str, choices: set[int]) -> int:
    while True:
        choice = prompt_int(prompt, required=True)
        if choice is None:
            continue
        if choice in choices:
            return choice
        print(f"Invalid choice. Choose one of: {', '.join(str(item) for item in sorted(choices))}.")


def prompt_new_project() -> dict[str, Any]:
    print("Enter data for the new project (leave blank for the default value):")

    print("Select the grid connection model:")
    print("  1 - Developer-funded connection")
    print("  2 - System operator-funded connection (TSO)")
    grid_choice = prompt_choice("Enter a number (1/2)", {1, 2})
    grid_model = "TSO_provided" if grid_choice == 2 else DEFAULT_GRID_CONNECTION_MODEL

    print("Select the foundation type:")
    print("  1 - Fixed (monopile/jacket)")
    print("  2 - Floating (semi-submersible, etc.)")
    foundation_choice = prompt_choice("Enter a number (1/2)", {1, 2})
    foundation_type = "Semi-submersible" if foundation_choice == 2 else "Monopile"

    return {
        "installed_capacity_MW": prompt_float("Installed capacity, MW", required=True),
        "grid_connection_model": grid_model,
        "foundation_type": foundation_type,
        "distance_from_shore_km": prompt_float("Distance to shore, km", required=True),
        "mean_hub_wind_speed": prompt_float("Mean wind speed at hub height, m/s", required=True),
        "capacity_factor": prompt_float("Capacity factor", required=True),
        "project_lifetime_years": prompt_float(
            "Project lifetime, years",
            DEFAULT_PROJECT_LIFETIME_YEARS,
            required=True,
        ),
        "commissioning_year": prompt_int(
            "Commissioning year",
            DEFAULT_COMMISSIONING_YEAR,
            required=True,
        ),
    }


def load_project_data(json_path: str | None) -> dict[str, Any]:
    if not json_path:
        return prompt_new_project()
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_nearest_or_none(
    df_valid: pd.DataFrame,
    new_project_data: dict[str, Any],
) -> pd.Series | None:
    if df_valid.empty:
        return None
    try:
        return find_nearest_project(df_valid, new_project_data)
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def _is_floating_value(value: Any) -> bool:
    text = str(value or "").lower()
    return any(keyword in text for keyword in ("floating", "spar", "semi-submersible", "tlp"))


def estimate_budget_from_dataset(
    df_valid: pd.DataFrame,
    project_data: dict[str, Any],
    nearest_count: int = 5,
) -> dict[str, Any]:
    capacity = float(project_data.get("installed_capacity_MW", np.nan))
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Installed capacity is required to estimate project budget.")
    candidates = df_valid.copy()
    starting_candidate_count = len(candidates)
    if "budget_EUR_per_MW" in candidates.columns:
        candidates["unit_budget_EUR_per_MW"] = pd.to_numeric(
            candidates["budget_EUR_per_MW"], errors="coerce"
        )
    else:
        candidates["unit_budget_EUR_per_MW"] = (
            pd.to_numeric(candidates["total_budget_EUR_2026"], errors="coerce")
            / pd.to_numeric(candidates["installed_capacity_MW"], errors="coerce")
        )
    candidates["budget_confidence"] = pd.to_numeric(
        candidates.get("budget_confidence", pd.Series(0.5, index=candidates.index)),
        errors="coerce",
    ).fillna(0.5)
    if "budget_quality_status" in candidates.columns:
        candidates = candidates[candidates["budget_quality_status"].eq("valid")].copy()
    if "budget_verification_level" in candidates.columns:
        candidates = candidates[
            candidates["budget_verification_level"].isin(ACCEPTED_BUDGET_VERIFICATION_LEVELS)
        ].copy()
    candidates = candidates[
        candidates["unit_budget_EUR_per_MW"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    candidates = candidates[
        candidates["unit_budget_EUR_per_MW"].between(500_000.0, 35_000_000.0)
    ].copy()
    invalid_candidate_count = starting_candidate_count - len(candidates)

    if candidates.empty:
        raise ValueError("Cannot estimate budget: dataset has no valid budget/capacity rows.")

    project_is_floating = _is_floating_value(project_data.get("foundation_type"))
    same_foundation = candidates[
        candidates["foundation_type"].apply(_is_floating_value) == project_is_floating
    ].copy()
    if len(same_foundation) >= nearest_count:
        candidates = same_foundation

    features = [
        "installed_capacity_MW",
        "distance_from_shore_km",
        "mean_hub_wind_speed",
        "commissioning_year",
    ]
    usable_features = []
    for feature in features:
        if feature not in candidates.columns:
            continue
        try:
            value = float(project_data.get(feature, np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            usable_features.append(feature)

    if usable_features:
        x = candidates[usable_features].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True))
        means = x.mean(axis=0)
        stds = x.std(axis=0).replace(0, 1.0).fillna(1.0)
        project_vector = np.array(
            [(float(project_data[feature]) - means[feature]) / stds[feature] for feature in usable_features],
            dtype=float,
        )
        distances = np.linalg.norm(((x - means) / stds).to_numpy() - project_vector, axis=1)
        candidates["budget_estimate_distance"] = distances
        reference_pool = candidates.nsmallest(max(nearest_count * 3, nearest_count), "budget_estimate_distance")
    else:
        candidates["budget_estimate_distance"] = 1.0
        reference_pool = candidates.head(max(nearest_count * 3, nearest_count))

    reference = trim_unit_budget_outliers(reference_pool).head(nearest_count)
    unit_budget = weighted_median(
        reference["unit_budget_EUR_per_MW"].to_numpy(dtype=float),
        budget_reference_weights(reference).to_numpy(dtype=float),
    )
    estimated_budget = unit_budget * capacity
    confidence = budget_estimate_confidence(reference, project_is_floating)
    budget_range = budget_estimate_range(reference, capacity)
    reference_projects = [
        str(name) for name in reference.get("wind_farm_name", pd.Series(dtype=str)).dropna().head(nearest_count)
    ]

    return {
        "estimated_total_budget_EUR_2026": round(estimated_budget, 0),
        "unit_budget_EUR_per_MW": round(unit_budget, 2),
        "reference_projects": reference_projects,
        "reference_unit_budgets_EUR_per_MW": [
            round(float(value), 2) for value in reference["unit_budget_EUR_per_MW"].tolist()
        ],
        "estimated_budget_range_EUR_2026": budget_range,
        "confidence_score": confidence["score"],
        "confidence_label": confidence["label"],
        "confidence_reasons": confidence["reasons"],
        "reference_verification_levels": [
            str(value) for value in reference.get("budget_verification_level", pd.Series(dtype=str)).fillna("C").tolist()
        ],
        "excluded_invalid_candidate_count": int(invalid_candidate_count),
        "method": f"weighted median EUR/MW from {len(reference)} validated nearest historic projects",
    }


def trim_unit_budget_outliers(reference: pd.DataFrame) -> pd.DataFrame:
    if len(reference) < 4:
        return reference.copy()
    units = pd.to_numeric(reference["unit_budget_EUR_per_MW"], errors="coerce")
    low = float(units.quantile(0.10))
    high = float(units.quantile(0.90))
    trimmed = reference[units.between(low, high)].copy()
    return trimmed if len(trimmed) >= 2 else reference.copy()


def budget_reference_weights(reference: pd.DataFrame) -> pd.Series:
    distances = pd.to_numeric(reference["budget_estimate_distance"], errors="coerce").fillna(1.0)
    confidence = pd.to_numeric(reference["budget_confidence"], errors="coerce").fillna(0.5)
    return confidence / (1.0 + distances)


def budget_estimate_range(reference: pd.DataFrame, capacity_mw: float) -> dict[str, float | None]:
    units = pd.to_numeric(reference["unit_budget_EUR_per_MW"], errors="coerce").dropna()
    if units.empty:
        return {"low": None, "high": None}
    if len(units) == 1:
        low = high = float(units.iloc[0])
    else:
        low = float(units.quantile(0.25))
        high = float(units.quantile(0.75))
    return {
        "low": round(low * capacity_mw, 0),
        "high": round(high * capacity_mw, 0),
    }


def budget_estimate_confidence(reference: pd.DataFrame, project_is_floating: bool) -> dict[str, Any]:
    if reference.empty:
        return {"score": 0.0, "label": "Low", "reasons": ["no validated reference projects"]}

    count_score = min(1.0, len(reference) / 5.0)
    levels = reference.get("budget_verification_level", pd.Series("C", index=reference.index)).fillna("C")
    level_scores = levels.map({"A": 1.0, "B": 0.8, "C": 0.45, "D": 0.0}).fillna(0.45)
    source_score = float(level_scores.mean())
    distances = pd.to_numeric(reference.get("budget_estimate_distance", pd.Series(1.0, index=reference.index)), errors="coerce").fillna(1.0)
    distance_score = float(1.0 / (1.0 + distances.median()))
    units = pd.to_numeric(reference["unit_budget_EUR_per_MW"], errors="coerce").dropna()
    if units.empty or float(units.median()) <= 0:
        spread_score = 0.0
    else:
        spread_score = float(max(0.0, 1.0 - ((units.quantile(0.75) - units.quantile(0.25)) / units.median())))

    same_foundation_share = 1.0
    if "foundation_type" in reference.columns:
        same_foundation_share = float(
            (reference["foundation_type"].apply(_is_floating_value) == project_is_floating).mean()
        )

    score = 0.30 * source_score + 0.25 * count_score + 0.20 * distance_score + 0.15 * spread_score + 0.10 * same_foundation_share
    if score >= 0.75:
        label = "High"
    elif score >= 0.55:
        label = "Medium"
    else:
        label = "Low"

    reasons = [
        f"{len(reference)} validated neighbours",
        f"source levels: {', '.join(str(level) for level in levels.tolist())}",
        f"median normalized distance {float(distances.median()):.2f}",
        f"EUR/MW spread score {spread_score:.2f}",
    ]
    if same_foundation_share >= 1.0:
        reasons.append("foundation type matched")
    else:
        reasons.append(f"foundation match share {same_foundation_share:.0%}")
    return {"score": round(score, 2), "label": label, "reasons": reasons}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float(np.nanmedian(values))
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff)])


def add_estimated_budget(
    project_data: dict[str, Any],
    df_valid: pd.DataFrame,
) -> dict[str, Any]:
    estimate = estimate_budget_from_dataset(df_valid, project_data)
    project_data = project_data.copy()
    project_data["total_budget_EUR_2026"] = estimate["estimated_total_budget_EUR_2026"]
    project_data["budget_EUR_per_MW"] = estimate["unit_budget_EUR_per_MW"]
    project_data["budget_quality_status"] = "valid"
    project_data["budget_quality_reason"] = "model estimate from validated reference projects"
    project_data["budget_confidence"] = estimate["confidence_score"]
    project_data["budget_source_type"] = "model_estimate"
    project_data["budget_verification_level"] = "B" if estimate["confidence_label"] in {"High", "Medium"} else "C"
    project_data["budget_verification_reason"] = "derived from A/B verified neighbour projects"
    project_data["budget_estimate"] = estimate
    return project_data


def apply_estimate_quality_to_computed(
    computed: dict[str, Any],
    project_data: dict[str, Any],
) -> dict[str, Any]:
    computed = computed.copy()
    for key in (
        "budget_EUR_per_MW",
        "budget_quality_status",
        "budget_quality_reason",
        "budget_confidence",
        "budget_source_type",
        "budget_verification_level",
        "budget_verification_reason",
    ):
        if key in project_data:
            computed[key] = project_data[key]
    return computed


def build_report(
    new_project_data: dict[str, Any],
    computed: dict[str, Any],
    nearest: pd.Series | None,
    ml_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget_estimate = new_project_data.get("budget_estimate", {})
    return {
        "project_parameters": {
            key: value
            for key, value in new_project_data.items()
            if value is not None and key != "budget_estimate"
        },
        "budget_estimate": budget_estimate,
        "computed_metrics": {key: computed.get(key) for key in COMPUTED_METRIC_KEYS},
        "data_quality": {
            "budget_quality_status": computed.get("budget_quality_status"),
            "budget_quality_reason": computed.get("budget_quality_reason"),
            "budget_confidence": computed.get("budget_confidence"),
            "budget_source_type": computed.get("budget_source_type"),
            "budget_verification_level": computed.get("budget_verification_level"),
            "budget_verification_reason": computed.get("budget_verification_reason"),
            "budget_EUR_per_MW": computed.get("budget_EUR_per_MW"),
        },
        "ml_model": ml_report or {},
        "validation": {
            "status": computed.get("validation_status"),
            "outlier_type": computed.get("outlier_type"),
            "model_uncertainty": "Low budget confidence"
            if budget_estimate.get("confidence_label") == "Low"
            else None,
            "classification_algorithmic": computed.get("classification"),
            "nearest_project_name": nearest.get("wind_farm_name") if nearest is not None else None,
            "nearest_project_distance": nearest.get("similarity_distance")
            if nearest is not None
            else None,
            "classification_nearest": nearest.get("classification") if nearest is not None else None,
        },
    }


def round_report_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {child_key: round_report_value(child_key, child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [round_report_value(key, item) for item in value]
    if not isinstance(value, (float, np.floating)):
        return value
    if not math.isfinite(float(value)):
        return None

    if key in {
        "total_budget_EUR_2026",
        "estimated_total_budget_EUR_2026",
        "CAPEX_total_EUR",
        "annual_CAPEX_EUR",
        "annual_OPEX_total_EUR",
        "annual_production_MWh",
    }:
        return int(round(float(value), 0))
    if key in {"CAPEX_unit_EUR_per_MW", "unit_budget_EUR_per_MW"}:
        return int(round(float(value), 0))
    if key in {
        "LCOE_EUR_per_MWh",
        "OPEX_unit_kEUR_per_MW_year",
        "distance_from_shore_km",
        "mean_hub_wind_speed",
        "installed_capacity_MW",
        "project_lifetime_years",
    }:
        return round(float(value), 2)
    if key in {"capacity_factor", "capacity_factor_est"}:
        return round(float(value), 2)
    if key in {"CRF", "nearest_project_distance"}:
        return round(float(value), 3)
    return round(float(value), 2)


def round_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: round_report_value(key, value) for key, value in report.items()}


def timestamped_report_path(base_path: str | Path = CLASSIFICATION_REPORT_PATH) -> Path:
    base = Path(base_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CLASSIFICATION_HISTORY_DIR / f"{base.stem}_{stamp}{base.suffix}"


def save_report(
    report: dict[str, Any],
    json_filename: str | Path = CLASSIFICATION_REPORT_PATH,
) -> None:
    try:
        report = round_report(report)
        output_path = Path(json_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, ensure_ascii=False))
        history_path = timestamped_report_path(output_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON report saved to file '{output_path}'.")
        print(f"Historical copy saved to file '{history_path}'.")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Could not write JSON to file: {exc}")


def report_table(report: dict[str, Any]) -> pd.DataFrame:
    flat_items = []
    for key, value in sorted(flatten_dict(report), key=lambda item: item[0]):
        if isinstance(value, dict):
            continue
        flat_items.append((key, format_report_value(key, value)))
    df_table = pd.DataFrame(flat_items, columns=["Field", "Value"])
    df_table["Field"] = df_table["Field"].apply(lambda key: FIELD_DESCRIPTIONS.get(key, key))
    return df_table


def format_report_value(key: str, value: Any) -> str:
    field_name = key.rsplit(".", 1)[-1]
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return format_report_list(field_name, value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)

    numeric = float(value)
    if not math.isfinite(numeric):
        return "n/a"
    if field_name in {
        "total_budget_EUR_2026",
        "estimated_total_budget_EUR_2026",
        "CAPEX_total_EUR",
        "annual_CAPEX_EUR",
        "annual_OPEX_total_EUR",
        "annual_production_MWh",
        "CAPEX_unit_EUR_per_MW",
        "unit_budget_EUR_per_MW",
    }:
        return f"{numeric:,.0f}"
    if field_name in {"capacity_factor", "capacity_factor_est"}:
        return f"{numeric:.2f}"
    if field_name in {"CRF", "nearest_project_distance"}:
        return f"{numeric:.3f}"
    if field_name == "commissioning_year":
        return f"{numeric:.0f}"
    return f"{numeric:,.2f}"


def format_report_list(field_name: str, value: list[Any]) -> str:
    if not value:
        return "n/a"
    if not all(isinstance(item, dict) for item in value):
        return ", ".join(str(item) for item in value)

    if field_name in {"error_by_country", "error_by_foundation"}:
        return "; ".join(
            format_error_group(item)
            for item in value
            if isinstance(item, dict)
        )
    if field_name == "model_comparison":
        return "; ".join(
            format_model_comparison_row(item)
            for item in value
            if isinstance(item, dict)
        )
    if field_name == "feature_importance":
        return "; ".join(
            format_feature_importance_row(item)
            for item in value
            if isinstance(item, dict)
        )
    return "; ".join(", ".join(f"{key}: {format_compact_value(child)}" for key, child in item.items()) for item in value)


def format_error_group(item: dict[str, Any]) -> str:
    group = item.get("group", "unknown")
    rows = item.get("rows")
    mae = format_millions(item.get("MAE_EUR_per_MW"), suffix=" EUR/MW")
    mape = format_percent(item.get("MAPE_percent"))
    return f"{group}: MAE {mae}, MAPE {mape} (n={rows})"


def format_model_comparison_row(item: dict[str, Any]) -> str:
    model = str(item.get("model", "model")).replace("_", " ")
    mae = format_millions(item.get("test_MAE_EUR_per_MW"), suffix=" EUR/MW")
    mape = format_percent(item.get("test_MAPE_percent"))
    r2 = format_compact_value(item.get("test_R2_log_target"))
    cv = format_compact_value(item.get("cv_MAE_log_target"))
    return f"{model}: MAE {mae}, MAPE {mape}, R2 {r2}, CV log-MAE {cv}"


def format_feature_importance_row(item: dict[str, Any]) -> str:
    feature = str(item.get("feature", "feature")).replace("_", " ")
    importance = item.get("importance")
    try:
        pct = float(importance) * 100.0
    except (TypeError, ValueError):
        return f"{feature}: n/a"
    return f"{feature}: {pct:.1f}%"


def format_millions(value: Any, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M{suffix}"
    return f"{numeric:,.0f}{suffix}"


def format_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2f}%"


def format_compact_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "n/a"
        return f"{numeric:.2f}"
    return str(value)


def print_report_table(report: dict[str, Any]) -> None:
    df_table = report_table(report)
    print("\nResults table:\n")
    try:
        print(df_table.to_markdown(index=False))
    except ImportError:
        print(df_table.to_string(index=False))


def timestamped_preview_dir(preview_path: str | Path = CLASSIFICATION_PREVIEW_PATH) -> Path:
    output_path = Path(preview_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_HISTORY_DIR / f"{output_path.stem}_{stamp}"


def save_preview(
    report: dict[str, Any],
    preview_path: str | Path = CLASSIFICATION_PREVIEW_PATH,
) -> tuple[Path, Path]:
    output_path = Path(preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_dir = timestamped_preview_dir(output_path)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / output_path.name
    history_csv_path = history_dir / output_path.with_suffix(".csv").name
    table = report_table(report)
    classification = report["validation"].get("classification_algorithmic", "No data")
    lcoe = report["computed_metrics"].get("LCOE_EUR_per_MWh")
    capex = report["computed_metrics"].get("CAPEX_total_EUR")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wind Project Classification</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; }}
    h1 {{ margin-bottom: 8px; }}
    .cards {{ display: flex; gap: 16px; margin: 24px 0; }}
    .card {{ border: 1px solid #d8dee4; padding: 16px 20px; min-width: 180px; }}
    .label {{ color: #57606a; font-size: 13px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 9px 12px; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Wind Project Classification</h1>
  <div class="cards">
    <div class="card"><div class="label">Classification</div><div class="value">{classification}</div></div>
    <div class="card"><div class="label">LCOE</div><div class="value">{float(lcoe):.2f} EUR/MWh</div></div>
    <div class="card"><div class="label">Estimated CAPEX</div><div class="value">{float(capex) / 1_000_000:.1f}M EUR</div></div>
  </div>
  {table.to_html(index=False, escape=True)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    history_path.write_text(html, encoding="utf-8")
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    table.to_csv(history_csv_path, index=False)
    print(f"Preview saved to file '{output_path}'.")
    print(f"Historical preview saved to file '{history_path}'.")
    return output_path, history_path


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    df_valid = load_historic_projects(dataset_path)
    new_project_data = load_project_data(args.json)
    new_project_data = add_estimated_budget(new_project_data, df_valid)
    computed = classify_new_project(new_project_data)
    computed = apply_estimate_quality_to_computed(computed, new_project_data)
    computed["classification"] = classify_project_against_dataset(computed, df_valid)
    nearest = find_nearest_or_none(df_valid, new_project_data)
    ml_report = build_capex_ml_report(df_valid, new_project_data)
    report = round_report(build_report(new_project_data, computed, nearest, ml_report))
    save_report(report)
    save_preview(report)
    print_report_table(report)


if __name__ == "__main__":
    main()
