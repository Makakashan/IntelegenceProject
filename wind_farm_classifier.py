from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Slope and intercept of the linear regression used to estimate capacity factors
CF_SLOPE: float = 0.0960
CF_INTERCEPT: float = -0.4125

# Global discount rate used for calculating the capital recovery factor (CRF)
DISCOUNT_RATE: float = 0.05

# List of keywords that identify floating foundation systems
FLOATING_KEYWORDS = [
    "floating",
    "spar",
    "semi-submersible",
    "tlp",
    "damping",
    "sath",
]

def is_floating_foundation(foundation_type: Optional[str]) -> bool:
    if not isinstance(foundation_type, str):
        return False
    ftype = foundation_type.lower()
    return any(key in ftype for key in FLOATING_KEYWORDS)


def estimate_capacity_factor(row: pd.Series) -> float:
    cf = row.get("capacity_factor")
    try:
        cf_val = float(cf)
        if cf_val > 0:
            return cf_val
    except (TypeError, ValueError):
        pass

    wind = row.get("mean_hub_wind_speed")
    try:
        wind_val = float(wind)
    except (TypeError, ValueError):
        return 0.40  # fallback if wind speed missing

    estimate = CF_SLOPE * wind_val + CF_INTERCEPT
    return float(np.clip(estimate, 0.25, 0.65))


def calculate_capex(row: pd.Series) -> pd.Series:
    base_capex = row.get("total_budget_EUR_2026")
    capacity = row.get("installed_capacity_MW")
    try:
        base_capex_val = float(base_capex)
        capacity_val = float(capacity)
    except (TypeError, ValueError):
        return pd.Series({"CAPEX_total_EUR": np.nan, "CAPEX_unit_EUR_per_MW": np.nan})
    if base_capex_val <= 0 or capacity_val <= 0:
        return pd.Series({"CAPEX_total_EUR": np.nan, "CAPEX_unit_EUR_per_MW": np.nan})

    # Apply cable cost adder if TSO provides connection
    cable_multiplier = 1.15 if row.get("grid_connection_model") == "TSO_provided" else 1.0
    total_capex = base_capex_val * cable_multiplier
    return pd.Series(
        {
            "CAPEX_total_EUR": total_capex,
            "CAPEX_unit_EUR_per_MW": total_capex / capacity_val,
        }
    )


def calculate_opex(row: pd.Series) -> float:
    base = 85.0 if bool(row.get("is_floating")) else 65.0
    dist = row.get("distance_from_shore_km")
    try:
        dist_val = float(dist)
        if dist_val <= 50:
            corr = 5.0
        elif dist_val <= 100:
            corr = 15.0
        else:
            corr = 25.0
    except (TypeError, ValueError):
        corr = 10.0
    return base + corr


def calculate_crcf(n_years: float) -> float:
    try:
        n = float(n_years)
    except (TypeError, ValueError):
        return np.nan
    if n <= 0:
        return np.nan
    r = DISCOUNT_RATE
    return r / (1.0 - (1.0 + r) ** (-n))


def calculate_lcoe(row: pd.Series) -> pd.Series:
    capex_total = row.get("CAPEX_total_EUR")
    production = row.get("annual_production_MWh")
    opex = row.get("annual_OPEX_total_EUR")
    lifetime = row.get("project_lifetime_years")

    try:
        capex_val = float(capex_total)
        prod_val = float(production)
        opex_val = float(opex)
        life_val = float(lifetime)
    except (TypeError, ValueError):
        return pd.Series({"CRF": np.nan, "annual_CAPEX_EUR": np.nan, "LCOE_EUR_per_MWh": np.nan})

    if capex_val <= 0 or prod_val <= 0:
        return pd.Series({"CRF": np.nan, "annual_CAPEX_EUR": np.nan, "LCOE_EUR_per_MWh": np.nan})

    crf = calculate_crcf(life_val)
    if not np.isfinite(crf):
        return pd.Series({"CRF": np.nan, "annual_CAPEX_EUR": np.nan, "LCOE_EUR_per_MWh": np.nan})

    annual_capex = capex_val * crf
    lcoe = (annual_capex + opex_val) / prod_val
    return pd.Series(
        {
            "CRF": crf,
            "annual_CAPEX_EUR": annual_capex,
            "LCOE_EUR_per_MWh": lcoe,
        }
    )


def expected_lcoe_range(row: pd.Series) -> tuple[float, float]:
    is_float = bool(row.get("is_floating"))
    try:
        year = int(row.get("commissioning_year"))
    except (TypeError, ValueError):
        year = 2026

    if is_float:
        return 60.0, 200.0
    if year < 2015:
        return 50.0, 250.0
    return 30.0, 120.0


def validate_project(row: pd.Series) -> str:
    lcoe = row.get("LCOE_EUR_per_MWh")
    try:
        lcoe_val = float(lcoe)
    except (TypeError, ValueError):
        return "No data"

    low, high = expected_lcoe_range(row)

    if lcoe_val < low:
        return "Below expected"
    if lcoe_val > high:
        return "Above expected (outlier)"
    return "Valid"


def classify_project_quality(row: pd.Series) -> str:
    lcoe = row.get("LCOE_EUR_per_MWh")
    try:
        lcoe_val = float(lcoe)
    except (TypeError, ValueError):
        return "No data"

    low, high = expected_lcoe_range(row)
    score = (lcoe_val - low) / (high - low)

    if score <= 0.20:
        return "Good"
    if score <= 0.40:
        return "Above average"
    if score <= 0.60:
        return "Average"
    if score <= 0.80:
        return "Below average"
    return "Bad"


def load_and_analyse(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["capacity_factor"] = pd.to_numeric(df.get("capacity_factor"), errors="coerce")
    df["project_lifetime_years"] = df.get("project_lifetime_years").fillna(25.0)

    df["is_floating"] = df["foundation_type"].apply(is_floating_foundation)
    mask = df["total_budget_EUR_2026"].notna() & df["installed_capacity_MW"].notna()
    df_model = df.loc[mask].copy()

    # CAPEX
    capex_cols = ["CAPEX_total_EUR", "CAPEX_unit_EUR_per_MW"]
    df_model[capex_cols] = df_model.apply(calculate_capex, axis=1)

    # Energy yield
    df_model["capacity_factor_est"] = df_model.apply(estimate_capacity_factor, axis=1)
    # Productivity (brutto and netto)
    df_model["productivity_brutto_h"] = 8760.0 * df_model["capacity_factor_est"]
    df_model["productivity_netto_h"] = df_model["productivity_brutto_h"] * 0.85
    df_model["annual_production_MWh"] = df_model["productivity_netto_h"] * df_model["installed_capacity_MW"]

    # OPEX
    df_model["OPEX_unit_kEUR_per_MW_year"] = df_model.apply(calculate_opex, axis=1)
    df_model["annual_OPEX_total_EUR"] = df_model["OPEX_unit_kEUR_per_MW_year"] * 1000.0 * df_model["installed_capacity_MW"]

    # LCOE
    lcoe_cols = ["CRF", "annual_CAPEX_EUR", "LCOE_EUR_per_MWh"]
    df_model[lcoe_cols] = df_model.apply(calculate_lcoe, axis=1)

    # Only rows with valid LCOE are kept for validation
    df_valid = df_model[df_model["LCOE_EUR_per_MWh"].notna()].copy()
    df_valid["validation_status"] = df_valid.apply(validate_project, axis=1)
    df_valid["classification"] = df_valid.apply(classify_project_quality, axis=1)

    return df_valid


def classify_new_project(project_data: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "total_budget_EUR_2026",
        "installed_capacity_MW",
        "grid_connection_model",
        "foundation_type",
        "distance_from_shore_km",
        "mean_hub_wind_speed",
        "capacity_factor",
        "project_lifetime_years",
        "commissioning_year",
        "lat",
        "lon",
    ]
    row_data = {f: project_data.get(f) for f in fields}
    if row_data["grid_connection_model"] is None:
        row_data["grid_connection_model"] = "developer_provided"
    if row_data["project_lifetime_years"] is None:
        row_data["project_lifetime_years"] = 25.0

    row_data["is_floating"] = is_floating_foundation(row_data.get("foundation_type"))

    row = pd.Series(row_data)
    capex_series = calculate_capex(row)
    for key, value in capex_series.items():
        row[key] = value

    cf_est = estimate_capacity_factor(row)
    row["capacity_factor_est"] = cf_est
    row["productivity_brutto_h"] = 8760.0 * cf_est
    row["productivity_netto_h"] = row["productivity_brutto_h"] * 0.85
    try:
        capacity = float(row["installed_capacity_MW"])
        row["annual_production_MWh"] = row["productivity_netto_h"] * capacity
    except (TypeError, ValueError):
        row["annual_production_MWh"] = np.nan

    opex_unit = calculate_opex(row)
    row["OPEX_unit_kEUR_per_MW_year"] = opex_unit
    try:
        row["annual_OPEX_total_EUR"] = opex_unit * 1000.0 * float(row["installed_capacity_MW"])
    except (TypeError, ValueError):
        row["annual_OPEX_total_EUR"] = np.nan

    lcoe_series = calculate_lcoe(row)
    for key, value in lcoe_series.items():
        row[key] = value

    result = row.to_dict()
    result["validation_status"] = validate_project(row)
    result["classification"] = classify_project_quality(row)
    return result


def find_nearest_project(
    df: pd.DataFrame, new_data: Dict[str, Any], features: Optional[list] = None
) -> pd.Series:
    if df.empty:
        raise ValueError("Historic dataset is empty; cannot compute similarity.")
    if features is None:
        features = [
            "lat",
            "lon",
            "installed_capacity_MW",
            "distance_from_shore_km",
            "mean_hub_wind_speed",
        ]
    usable = []
    for f in features:
        if f in df.columns and new_data.get(f) is not None:
            usable.append(f)
    if not usable:
        row = df.iloc[0]
        row = row.copy()
        row["similarity_distance"] = np.nan
        return row
    X = df[usable].copy().astype(float)
    means = X.mean(axis=0)
    stds = X.std(axis=0).replace(0, 1.0)
    X_norm = (X - means) / stds
    new_vec = np.array([(float(new_data[f]) - means[f]) / stds[f] for f in usable], dtype=float)
    dists = np.linalg.norm(X_norm.values - new_vec, axis=1)
    nearest_idx = int(np.argmin(dists))
    row = df.iloc[nearest_idx].copy()
    row["similarity_distance"] = float(dists[nearest_idx])
    return row


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Analyse offshore wind projects and classify a new proposal via JSON or interactive input."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to the historic project dataset (CSV). If omitted, the script will look for "
            "'output/windTurbineData_enriched_2026EUR.csv' then 'windTurbineData_enriched_2026EUR.csv'"
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=("Path to a JSON file containing new project data to classify. If omitted, prompts will be used."),
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_path = args.dataset
    else:
        candidate = os.path.join("output", "windTurbineData_enriched_2026EUR.csv")
        if os.path.isfile(candidate):
            dataset_path = candidate
        elif os.path.isfile("windTurbineData_enriched_2026EUR.csv"):
            dataset_path = "windTurbineData_enriched_2026EUR.csv"
        else:
            dataset_path = args.dataset or "windTurbineData_enriched_2026EUR.csv"

    try:
        df_valid = load_and_analyse(dataset_path)
        print(f"Loaded {len(df_valid)} projects from '{dataset_path}'.")
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        df_valid = pd.DataFrame()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as fh:
            new_project_data = json.load(fh)
    else:
        print("Enter data for the new project (leave blank for the default value):")

        def prompt_float(prompt, default=None):
            val = input(f"{prompt}" + (f" [default: {default}]" if default is not None else "") + ": ")
            if not val.strip():
                return default
            try:
                return float(val)
            except ValueError:
                print("Invalid number, the value will be skipped.")
                return default

        def prompt_int(prompt, default=None):
            val = input(f"{prompt}" + (f" [default: {default}]" if default is not None else "") + ": ")
            if not val.strip():
                return default
            try:
                return int(val)
            except ValueError:
                print("Invalid integer, the value will be skipped.")
                return default

        def prompt_str(prompt, default=None):
            val = input(f"{prompt}" + (f" [default: {default}]" if default is not None else "") + ": ")
            return val.strip() or default

        print("Select the grid connection model:")
        print("  1 — Developer-funded connection")
        print("  2 — System operator-funded connection (TSO)")
        grid_choice = prompt_int("Enter a number (1/2)", 1)
        grid_map = {1: "developer_provided", 2: "TSO_provided"}
        grid_model = grid_map.get(grid_choice, "developer_provided")

        print("Select the foundation type:")
        print("  1 — Fixed (monopile/jacket)")
        print("  2 — Floating (semi-submersible, etc.)")
        foundation_choice = prompt_int("Enter a number (1/2)", 1)
        foundation_map = {1: "Monopile", 2: "Semi-submersible"}
        foundation_type = foundation_map.get(foundation_choice, "Monopile")

        new_project_data = {
            "total_budget_EUR_2026": prompt_float("Total budget, EUR (2026)"),
            "installed_capacity_MW": prompt_float("Installed capacity, MW"),
            "grid_connection_model": grid_model,
            "foundation_type": foundation_type,
            "distance_from_shore_km": prompt_float("Distance to shore, km"),
            "mean_hub_wind_speed": prompt_float("Mean wind speed at hub height, m/s"),
            "capacity_factor": None,
            "project_lifetime_years": 25.0,
            "commissioning_year": 2026,
            "lat": prompt_float("Latitude (°)"),
            "lon": prompt_float("Longitude (°)"),
        }

    computed = classify_new_project(new_project_data)

    if not df_valid.empty:
        try:
            nearest = find_nearest_project(df_valid, new_project_data)
            similarity_classification = nearest.get("classification")
        except Exception:
            nearest = None
            similarity_classification = None
    else:
        nearest = None
        similarity_classification = None

    report = {
        "project_parameters": {k: v for k, v in new_project_data.items() if v is not None},
        "computed_metrics": {
            "CAPEX_total_EUR": computed.get("CAPEX_total_EUR"),
            "CAPEX_unit_EUR_per_MW": computed.get("CAPEX_unit_EUR_per_MW"),
            "capacity_factor_est": computed.get("capacity_factor_est"),
            "annual_production_MWh": computed.get("annual_production_MWh"),
            "OPEX_unit_kEUR_per_MW_year": computed.get("OPEX_unit_kEUR_per_MW_year"),
            "annual_OPEX_total_EUR": computed.get("annual_OPEX_total_EUR"),
            "CRF": computed.get("CRF"),
            "annual_CAPEX_EUR": computed.get("annual_CAPEX_EUR"),
            "LCOE_EUR_per_MWh": computed.get("LCOE_EUR_per_MWh"),
        },
        "validation": {
            "status": computed.get("validation_status"),
            "classification_algorithmic": computed.get("classification"),
            "nearest_project_name": nearest.get("wind_farm_name") if nearest is not None else None,
            "nearest_project_distance": nearest.get("similarity_distance") if nearest is not None else None,
            "classification_nearest": similarity_classification,
        },
    }

    json_filename = "classification_result.json"
    try:
        with open(json_filename, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON report saved to file '{json_filename}'.")
    except Exception as e:
        print(f"Could not write JSON to file: {e}")

    def _flatten_dict(d, parent_key: str = "", sep: str = "."):
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))
        return items

    flat_items = _flatten_dict(report)
    flat_items.sort(key=lambda x: x[0])

    FIELD_DESCRIPTIONS = {
        "project_parameters.total_budget_EUR_2026": "Total budget (2026, €)",
        "project_parameters.installed_capacity_MW": "Installed capacity (MW)",
        "project_parameters.grid_connection_model": "Grid connection model",
        "project_parameters.foundation_type": "Foundation type",
        "project_parameters.distance_from_shore_km": "Distance to shore (km)",
        "project_parameters.mean_hub_wind_speed": "Mean wind speed (m/s)",
        "project_parameters.capacity_factor": "Specified capacity factor",
        "project_parameters.project_lifetime_years": "Project lifetime (years)",
        "project_parameters.commissioning_year": "Commissioning year",
        "project_parameters.lat": "Latitude",
        "project_parameters.lon": "Longitude",
        "computed_metrics.CAPEX_total_EUR": "Total capital expenditure (CAPEX), €",
        "computed_metrics.CAPEX_unit_EUR_per_MW": "CAPEX per 1 MW, €",
        "computed_metrics.capacity_factor_est": "Estimated capacity factor",
        "computed_metrics.annual_production_MWh": "Annual production (MWh)",
        "computed_metrics.OPEX_unit_kEUR_per_MW_year": "OPEX per 1 MW (kEUR/year)",
        "computed_metrics.annual_OPEX_total_EUR": "Annual operating expenses (OPEX), €",
        "computed_metrics.CRF": "Capital recovery factor (CRF)",
        "computed_metrics.annual_CAPEX_EUR": "Annual CAPEX, €",
        "computed_metrics.LCOE_EUR_per_MWh": "Levelized cost of energy (LCOE), €/MWh",
        "validation.status": "Threshold status",
        "validation.classification_algorithmic": "Algorithmic classification",
        "validation.nearest_project_name": "Nearest project (historical)",
        "validation.nearest_project_distance": "Distance to nearest (in normalized feature space)",
        "validation.classification_nearest": "Nearest-project classification",
    }

    try:
        import pandas as pd
        df_table = pd.DataFrame(flat_items, columns=["Field", "Value"])
        df_table["Field"] = df_table["Field"].apply(lambda x: FIELD_DESCRIPTIONS.get(x, x))
        print("\nResults table:\n")
        try:
            print(df_table.to_markdown(index=False))
        except Exception:
            print(df_table.to_string(index=False))
    except ImportError:
        print("\nResults table:")
        for key, value in flat_items:
            friendly = FIELD_DESCRIPTIONS.get(key, key)
            print(f"{friendly}\t{value}")
