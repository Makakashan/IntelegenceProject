from __future__ import annotations

import pandas as pd

from src.lcoe.calculations import classify_project_against_dataset
from wind_farm_classifier import estimate_budget_from_dataset


def test_outlier_classification_is_not_overridden_by_distribution() -> None:
    reference = pd.DataFrame(
        {
            "LCOE_EUR_per_MWh": [50.0, 60.0, 70.0],
            "validation_status": ["Valid", "Valid", "Valid"],
        }
    )

    label = classify_project_against_dataset(
        {"LCOE_EUR_per_MWh": 65.0, "validation_status": "Above expected (outlier)"},
        reference,
    )

    assert label == "Outlier"


def test_budget_estimator_uses_validated_references_only() -> None:
    df = pd.DataFrame(
        {
            "wind_farm_name": ["dirty low", "clean one", "clean two", "clean three", "clean four", "clean five"],
            "installed_capacity_MW": [500, 500, 520, 480, 510, 495],
            "distance_from_shore_km": [50, 50, 55, 45, 51, 49],
            "mean_hub_wind_speed": [9, 9, 9.2, 8.8, 9.1, 8.9],
            "commissioning_year": [2026, 2026, 2027, 2025, 2026, 2026],
            "foundation_type": ["Monopile"] * 6,
            "total_budget_EUR_2026": [10_000_000, 1_500_000_000, 1_600_000_000, 1_450_000_000, 1_550_000_000, 1_520_000_000],
            "budget_EUR_per_MW": [20_000, 3_000_000, 3_076_923, 3_020_833, 3_039_216, 3_070_707],
            "budget_quality_status": ["rejected", "valid", "valid", "valid", "valid", "valid"],
            "budget_confidence": [0.0, 0.95, 0.95, 0.95, 0.95, 0.95],
            "budget_verification_level": ["D", "B", "B", "B", "B", "B"],
        }
    )

    estimate = estimate_budget_from_dataset(
        df,
        {
            "installed_capacity_MW": 500,
            "distance_from_shore_km": 50,
            "mean_hub_wind_speed": 9,
            "commissioning_year": 2026,
            "foundation_type": "Monopile",
        },
    )

    assert estimate["unit_budget_EUR_per_MW"] > 2_900_000
    assert "dirty low" not in estimate["reference_projects"]
    assert estimate["confidence_label"] in {"Medium", "High"}
    assert estimate["estimated_budget_range_EUR_2026"]["low"] is not None
