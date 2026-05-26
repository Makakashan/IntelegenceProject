from __future__ import annotations

import pandas as pd

from src.budget.quality import (
    apply_budget_quality_columns,
    evaluate_budget_quality,
    is_blocked_source,
    parse_budget_text_to_eur,
    source_verification_level,
)


def test_declared_budget_is_preferred_over_dirty_lookup() -> None:
    df = pd.DataFrame(
        [
            {
                "wind_farm_name": "Dogger Bank A",
                "commissioning_year": 2025,
                "installed_capacity_MW": 1200.0,
                "foundation_type": "Monopile",
                "total_budget_EUR": "GBP 3 billion",
                "total_budget_EUR_2026": 58_970_000.0,
                "source": "https://www.facebook.com/example",
            }
        ]
    )

    result = apply_budget_quality_columns(df, overrides_path=None)

    assert result.loc[0, "budget_quality_status"] == "valid"
    assert result.loc[0, "budget_source_type"] == "declared_csv"
    assert result.loc[0, "total_budget_EUR_2026"] > 3_000_000_000
    assert result.loc[0, "budget_EUR_per_MW"] > 2_000_000


def test_absurd_lookup_budget_is_rejected() -> None:
    quality = evaluate_budget_quality(
        23_400_000_000,
        90,
        source="https://facebook.com/bad-budget",
        source_type="internet_lookup",
    )

    assert quality.status == "rejected"
    assert quality.confidence == 0


def test_budget_text_range_is_parsed_to_midpoint() -> None:
    value = parse_budget_text_to_eur("GBP 780-900 million", 2010)

    assert value is not None
    assert 1_000_000_000 < value < 1_500_000_000


def test_pln_budget_is_not_treated_as_eur() -> None:
    value = parse_budget_text_to_eur("PLN 10 billion", 2026)

    assert value == 2_300_000_000


def test_blocked_source_domains_include_social_media() -> None:
    assert is_blocked_source("https://www.facebook.com/example")
    assert is_blocked_source("https://linkedin.com/posts/example")
    assert not is_blocked_source("https://orsted.com/en/example")


def test_source_verification_ranking() -> None:
    assert source_verification_level("https://www.eib.org/en/projects/all/123", source_type="internet_lookup")[0] == "A"
    assert source_verification_level("https://windfarminfo.com/example", source_type="declared_csv")[0] == "B"
    assert source_verification_level("https://example.com/blog", source_type="internet_lookup")[0] == "C"
    assert source_verification_level("https://facebook.com/post", source_type="internet_lookup")[0] == "D"


def test_combined_budget_is_allocated_by_capacity() -> None:
    df = pd.DataFrame(
        [
            {
                "wind_farm_name": "Gode Wind 1",
                "commissioning_year": 2017,
                "installed_capacity_MW": 300.0,
                "foundation_type": "Monopile",
                "total_budget_EUR": "EUR 2 billion (combined 1+2)",
                "data_source": "windfarminfo.com",
            },
            {
                "wind_farm_name": "Gode Wind 2",
                "commissioning_year": 2017,
                "installed_capacity_MW": 100.0,
                "foundation_type": "Monopile",
                "total_budget_EUR": "EUR 2 billion (combined 1+2)",
                "data_source": "windfarminfo.com",
            },
        ]
    )

    result = apply_budget_quality_columns(df, overrides_path=None)

    first = result.loc[0, "total_budget_EUR_2026"]
    second = result.loc[1, "total_budget_EUR_2026"]
    assert first > second
    assert round(first / second, 1) == 3.0


def test_wind_budget_lookup_does_not_replace_valid_declared_budget() -> None:
    df = pd.DataFrame(
        [
            {
                "wind_farm_name": "Thanet",
                "commissioning_year": 2010,
                "installed_capacity_MW": 300.0,
                "foundation_type": "Monopile",
                "original_budget_raw": "GBP 780-900 million",
                "total_budget_raw": "EUR 700 million",
                "total_budget_EUR_2026": 969_500_000.0,
                "budget_source_type": "wind_budget.py",
                "budget_lookup_source": "https://www.eib.org/en/projects/all/unrelated",
                "data_source": "windfarminfo.com",
            }
        ]
    )

    result = apply_budget_quality_columns(df, overrides_path=None)

    assert result.loc[0, "budget_source_type"] == "declared_csv"
    assert result.loc[0, "total_budget_EUR_2026"] > 1_000_000_000
