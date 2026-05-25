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
    candidates["unit_budget_EUR_per_MW"] = (
        pd.to_numeric(candidates["total_budget_EUR_2026"], errors="coerce")
        / pd.to_numeric(candidates["installed_capacity_MW"], errors="coerce")
    )
    candidates = candidates[
        candidates["unit_budget_EUR_per_MW"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()

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
        reference = candidates.nsmallest(nearest_count, "budget_estimate_distance")
    else:
        reference = candidates.head(nearest_count)

    unit_budget = float(reference["unit_budget_EUR_per_MW"].median())
    estimated_budget = unit_budget * capacity
    reference_projects = [
        str(name) for name in reference.get("wind_farm_name", pd.Series(dtype=str)).dropna().head(nearest_count)
    ]

    return {
        "estimated_total_budget_EUR_2026": round(estimated_budget, 0),
        "unit_budget_EUR_per_MW": round(unit_budget, 2),
        "reference_projects": reference_projects,
        "method": f"median EUR/MW from {len(reference)} nearest historic projects",
    }


def add_estimated_budget(
    project_data: dict[str, Any],
    df_valid: pd.DataFrame,
) -> dict[str, Any]:
    estimate = estimate_budget_from_dataset(df_valid, project_data)
    project_data = project_data.copy()
    project_data["total_budget_EUR_2026"] = estimate["estimated_total_budget_EUR_2026"]
    project_data["budget_estimate"] = estimate
    return project_data


def build_report(
    new_project_data: dict[str, Any],
    computed: dict[str, Any],
    nearest: pd.Series | None,
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
        "validation": {
            "status": computed.get("validation_status"),
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
        return ", ".join(str(item) for item in value)
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
    computed["classification"] = classify_project_against_dataset(computed, df_valid)
    nearest = find_nearest_or_none(df_valid, new_project_data)
    report = round_report(build_report(new_project_data, computed, nearest))
    save_report(report)
    save_preview(report)
    print_report_table(report)


if __name__ == "__main__":
    main()
