from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd

from .wind_calculations import (
    classify_new_project,
    classify_project_against_dataset,
    find_nearest_project,
    load_and_analyse,
)
from .wind_constants import (
    CLASSIFICATION_REPORT_PATH,
    COMPUTED_METRIC_KEYS,
    DEFAULT_COMMISSIONING_YEAR,
    DEFAULT_DATASET_CANDIDATES,
    DEFAULT_GRID_CONNECTION_MODEL,
    DEFAULT_PROJECT_LIFETIME_YEARS,
    FIELD_DESCRIPTIONS,
)
from .wind_utils import flatten_dict

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
            "output/windTurbineData_enriched_2026EUR.csv when present."
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to a JSON file containing new project data. If omitted, prompts are used.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_path: Optional[str]) -> Path:
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
    default: Optional[T],
    converter: Callable[[str], T],
    error_message: str,
    required: bool = False,
) -> Optional[T]:
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
    default: Optional[float] = None,
    required: bool = False,
) -> Optional[float]:
    return prompt_value(prompt, default, float, "Invalid number. Try again.", required)


def prompt_int(
    prompt: str,
    default: Optional[int] = None,
    required: bool = False,
) -> Optional[int]:
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
        "total_budget_EUR_2026": prompt_float("Total budget, EUR (2026)", required=True),
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
        "lat": prompt_float("Latitude"),
        "lon": prompt_float("Longitude"),
    }


def load_project_data(json_path: Optional[str]) -> dict[str, Any]:
    if not json_path:
        return prompt_new_project()
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_nearest_or_none(
    df_valid: pd.DataFrame,
    new_project_data: dict[str, Any],
) -> Optional[pd.Series]:
    if df_valid.empty:
        return None
    try:
        return find_nearest_project(df_valid, new_project_data)
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def build_report(
    new_project_data: dict[str, Any],
    computed: dict[str, Any],
    nearest: Optional[pd.Series],
) -> dict[str, Any]:
    return {
        "project_parameters": {
            key: value for key, value in new_project_data.items() if value is not None
        },
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


def save_report(
    report: dict[str, Any],
    json_filename: str | Path = CLASSIFICATION_REPORT_PATH,
) -> None:
    try:
        output_path = Path(json_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON report saved to file '{output_path}'.")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Could not write JSON to file: {exc}")


def print_report_table(report: dict[str, Any]) -> None:
    flat_items = sorted(flatten_dict(report), key=lambda item: item[0])
    df_table = pd.DataFrame(flat_items, columns=["Field", "Value"])
    df_table["Field"] = df_table["Field"].apply(lambda key: FIELD_DESCRIPTIONS.get(key, key))

    print("\nResults table:\n")
    try:
        print(df_table.to_markdown(index=False))
    except ImportError:
        print(df_table.to_string(index=False))


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    df_valid = load_historic_projects(dataset_path)
    new_project_data = load_project_data(args.json)
    computed = classify_new_project(new_project_data)
    computed["classification"] = classify_project_against_dataset(computed, df_valid)
    nearest = find_nearest_or_none(df_valid, new_project_data)
    report = build_report(new_project_data, computed, nearest)
    save_report(report)
    print_report_table(report)


if __name__ == "__main__":
    main()
