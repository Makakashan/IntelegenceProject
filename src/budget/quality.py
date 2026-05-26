from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import DEFAULT_TARGET_YEAR
from .finance import FX_FALLBACK
from .parsing import UNIT_PATTERN, budget_currency

BLOCKED_SOURCE_DOMAINS = (
    "facebook.com",
    "fb.com",
    "linkedin.com",
    "instagram.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "youtube.com",
)

TRUSTED_SOURCE_HINTS = (
    "agentzero.energy",
    ".gov",
    "gov.",
    "orsted.com",
    "rwe.com",
    "vattenfall.",
    "sse.com",
    "scottishpowerrenewables.com",
    "eib.org",
    "ofgem.gov.uk",
    "planninginspectorate.gov.uk",
    "4coffshore.com",
    "offshorewind.biz",
    "windpowernl.com",
)

OFFICIAL_SOURCE_HINTS = (
    ".gov",
    "gov.",
    "orsted.com",
    "rwe.com",
    "vattenfall.",
    "sse.com",
    "scottishpowerrenewables.com",
    "eib.org",
    "ofgem.gov.uk",
    "planninginspectorate.gov.uk",
    "statkraft.com",
    "sserenewables.com",
)

INDUSTRY_SOURCE_HINTS = (
    "windfarminfo.com",
    "agentzero.energy",
    "offshorewind.biz",
    "4coffshore.com",
    "tgs4c.com",
    "nsenergybusiness.com",
    "power-technology.com",
    "windpowermonthly.com",
    "renewableuk.com",
    "windpowernl.com",
)

VERIFICATION_RANK = {"A": 3, "B": 2, "C": 1, "D": 0}
DEFAULT_OVERRIDES_PATH = Path("input") / "budget_overrides.csv"

QUALITY_COLUMNS = (
    "declared_budget_EUR_2026",
    "declared_budget_EUR_per_MW",
    "declared_budget_verification_level",
    "lookup_budget_EUR_per_MW",
    "lookup_budget_verification_level",
    "total_budget_EUR_2026_clean",
    "budget_EUR_per_MW",
    "budget_quality_status",
    "budget_quality_reason",
    "budget_confidence",
    "budget_source_type",
    "budget_source_url",
    "budget_verification_level",
    "budget_verification_reason",
)

LOCAL_CPI_FALLBACK = {
    2000: 80.4,
    2001: 82.3,
    2002: 84.1,
    2003: 86.0,
    2004: 88.0,
    2005: 90.0,
    2006: 91.9,
    2007: 94.0,
    2008: 97.1,
    2009: 98.3,
    2010: 100.0,
    2011: 102.7,
    2012: 105.2,
    2013: 106.7,
    2014: 107.1,
    2015: 107.2,
    2016: 107.3,
    2017: 109.1,
    2018: 111.0,
    2019: 112.1,
    2020: 112.3,
    2021: 114.8,
    2022: 122.8,
    2023: 130.2,
    2024: 133.5,
    2025: 136.0,
    2026: 138.5,
}


@dataclass(frozen=True)
class BudgetQuality:
    value: float | None
    unit_eur_per_mw: float | None
    status: str
    reason: str
    confidence: float
    verification_level: str = "D"
    verification_reason: str = "not verified"
    source_url: str = ""


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def source_domain(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(str(url))
    domain = parsed.netloc.lower() or parsed.path.lower().split("/", 1)[0]
    return domain.removeprefix("www.")


def is_blocked_source(url: str | None) -> bool:
    domain = source_domain(url)
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in BLOCKED_SOURCE_DOMAINS)


def is_trusted_source(url: str | None) -> bool:
    domain = source_domain(url)
    return any(hint in domain for hint in TRUSTED_SOURCE_HINTS)


def source_verification_level(
    source_url: str | None,
    *,
    source_type: str,
    requested_level: str | None = None,
) -> tuple[str, str]:
    if requested_level:
        level = str(requested_level).strip().upper()
        if level in VERIFICATION_RANK:
            return level, f"manual verification level {level}"

    if source_type == "manual_override":
        return "A", "manual curated override"

    if is_blocked_source(source_url):
        return "D", "blocked source domain"

    domain = source_domain(source_url)
    if any(hint in domain for hint in OFFICIAL_SOURCE_HINTS):
        return "A", "official/developer/government/EIB source"
    if any(hint in domain for hint in INDUSTRY_SOURCE_HINTS):
        return "B", "industry database or specialist source"
    if source_type == "declared_csv":
        return "B", "declared dataset value from windfarminfo-based source"
    if source_type == "internet_lookup" and domain:
        return "C", "web lookup source requires manual verification"
    return "D", "missing source"


def confidence_for_level(level: str, source_type: str) -> float:
    if level == "A":
        return 0.98
    if level == "B":
        return 0.88 if source_type == "internet_lookup" else 0.90
    if level == "C":
        return 0.60
    return 0.0


def nearest_fx_rate(currency: str, year: int) -> float:
    if currency == "EUR":
        return 1.0
    fallback = FX_FALLBACK.get(currency, {})
    if not fallback:
        return 1.0
    nearest_year = min(fallback.keys(), key=lambda candidate: abs(candidate - year))
    return float(fallback.get(year) or fallback[nearest_year])


def local_inflation_factor(from_year: int, to_year: int = DEFAULT_TARGET_YEAR) -> float:
    from_year = int(from_year)
    to_year = int(to_year)
    cpi_from = LOCAL_CPI_FALLBACK.get(from_year)
    if cpi_from is None:
        nearest_from = min(LOCAL_CPI_FALLBACK, key=lambda year: abs(year - from_year))
        cpi_from = LOCAL_CPI_FALLBACK[nearest_from]
    cpi_to = LOCAL_CPI_FALLBACK.get(to_year)
    if cpi_to is None:
        nearest_to = min(LOCAL_CPI_FALLBACK, key=lambda year: abs(year - to_year))
        cpi_to = LOCAL_CPI_FALLBACK[nearest_to]
    return float(cpi_to / cpi_from)


def parse_budget_text_to_eur(
    raw: Any,
    budget_year: int | float | None,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "not found", "none"}:
        return None

    number_match = re.search(r"(\d+(?:[,.]\d+)?)(?:\s*[-–]\s*(\d+(?:[,.]\d+)?))?", text)
    if not number_match:
        return None

    first = float(number_match.group(1).replace(",", ""))
    second = number_match.group(2)
    amount = (first + float(second.replace(",", ""))) / 2.0 if second else first

    unit_match = UNIT_PATTERN.search(text)
    if unit_match:
        unit = unit_match.group().lower()
        if unit in {"billion", "bn", "b"}:
            amount *= 1_000_000_000
        elif unit in {"million", "mn", "m"}:
            amount *= 1_000_000

    year = int(_to_float(budget_year)) if math.isfinite(_to_float(budget_year)) else target_year
    currency = budget_currency(text)
    return round(amount * nearest_fx_rate(currency, year) * local_inflation_factor(year, target_year), 0)


def is_combined_budget(raw: Any) -> bool:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return False
    return bool(re.search(r"\bcombined\b|\b1\s*\+\s*2\b|\bii\s*\+\s*iii\b|\biii\s+and\s+iv\b", str(raw), re.I))


def budget_unit_bounds(capacity_mw: float, is_floating: bool = False) -> tuple[float, float]:
    if is_floating:
        return 750_000.0, 35_000_000.0
    if capacity_mw < 100:
        return 500_000.0, 25_000_000.0
    return 500_000.0, 15_000_000.0


def evaluate_budget_quality(
    budget_eur: Any,
    capacity_mw: Any,
    *,
    is_floating: bool = False,
    source: str | None = None,
    source_type: str = "unknown",
    requested_verification_level: str | None = None,
) -> BudgetQuality:
    budget = _to_float(budget_eur)
    capacity = _to_float(capacity_mw)
    level, verification_reason = source_verification_level(
        source,
        source_type=source_type,
        requested_level=requested_verification_level,
    )
    if not math.isfinite(budget) or budget <= 0:
        return BudgetQuality(None, None, "missing", "budget missing", 0.0, "D", "budget missing", str(source or ""))
    if not math.isfinite(capacity) or capacity <= 0:
        return BudgetQuality(float(budget), None, "missing", "capacity missing", 0.0, "D", "capacity missing", str(source or ""))
    if source_type == "internet_lookup" and is_blocked_source(source):
        return BudgetQuality(float(budget), float(budget / capacity), "rejected", "blocked source domain", 0.0, "D", "blocked source domain", str(source or ""))

    unit = float(budget / capacity)
    low, high = budget_unit_bounds(capacity, is_floating=is_floating)
    if unit < low:
        return BudgetQuality(float(budget), unit, "rejected", f"below plausible EUR/MW floor ({low:.0f})", 0.0, "D", "failed plausibility check", str(source or ""))
    if unit > high:
        return BudgetQuality(float(budget), unit, "rejected", f"above plausible EUR/MW ceiling ({high:.0f})", 0.0, "D", "failed plausibility check", str(source or ""))

    confidence = confidence_for_level(level, source_type)
    return BudgetQuality(
        float(budget),
        unit,
        "valid",
        "within plausible EUR/MW range",
        confidence,
        level,
        verification_reason,
        str(source or ""),
    )


def _row_is_floating(row: pd.Series) -> bool:
    foundation = str(row.get("foundation_type", "")).lower()
    return any(keyword in foundation for keyword in ("floating", "spar", "semi-submersible", "tlp"))


def load_budget_overrides(path: str | Path | None = DEFAULT_OVERRIDES_PATH) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    override_path = Path(path)
    if not override_path.is_file():
        return pd.DataFrame()
    overrides = pd.read_csv(override_path, encoding="utf-8-sig")
    if "wind_farm_name" not in overrides.columns:
        return pd.DataFrame()
    return overrides.dropna(subset=["wind_farm_name"]).drop_duplicates("wind_farm_name", keep="last")


def _declared_budget_for_row(
    row: pd.Series,
    declared_raw: Any,
    parsed_budget: float | None,
    all_rows: pd.DataFrame,
) -> float | None:
    if parsed_budget is None:
        return None
    if not is_combined_budget(declared_raw):
        return parsed_budget

    same_raw = all_rows["__declared_raw_for_quality"].astype(str).eq(str(declared_raw))
    group = all_rows[same_raw].copy()
    if len(group) < 2:
        return parsed_budget

    capacities = pd.to_numeric(group["installed_capacity_MW"], errors="coerce")
    capacity = _to_float(row.get("installed_capacity_MW"))
    total_capacity = float(capacities.sum())
    if not math.isfinite(capacity) or capacity <= 0 or total_capacity <= 0:
        return parsed_budget
    return float(parsed_budget * capacity / total_capacity)


def _first_present(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if pd.notna(value):
            return value
    return None


def apply_budget_quality_columns(
    df: pd.DataFrame,
    *,
    target_year: int = DEFAULT_TARGET_YEAR,
    replace_total_budget: bool = True,
    overrides_path: str | Path | None = DEFAULT_OVERRIDES_PATH,
) -> pd.DataFrame:
    source_result = df.copy()
    if "total_budget_EUR_2026" in source_result.columns and "total_budget_EUR_2026_lookup" not in source_result.columns:
        source_result["total_budget_EUR_2026_lookup"] = source_result["total_budget_EUR_2026"]
    overrides = load_budget_overrides(overrides_path)
    overrides_by_name = overrides.set_index("wind_farm_name").to_dict("index") if not overrides.empty else {}

    source_result["__declared_raw_for_quality"] = source_result.apply(
        lambda row: row.get("total_budget_EUR", row.get("original_budget_raw", row.get("total_budget_raw"))),
        axis=1,
    )

    rows: list[dict[str, Any]] = []
    for _, row in source_result.iterrows():
        capacity = row.get("installed_capacity_MW")
        is_floating = _row_is_floating(row)
        declared_year = row.get("commissioning_year") if pd.notna(row.get("commissioning_year")) else row.get("budget_year")
        raw_budget = row.get("__declared_raw_for_quality")
        parsed_declared = parse_budget_text_to_eur(raw_budget, declared_year, target_year)
        declared_budget = _declared_budget_for_row(row, raw_budget, parsed_declared, source_result)
        if (
            declared_budget is None
            and pd.notna(raw_budget)
            and pd.notna(row.get("total_budget_EUR_2026_lookup"))
            and row.get("budget_source_type") in {"declared_csv", "manual_override"}
        ):
            declared_budget = _to_float(row.get("total_budget_EUR_2026_lookup"))
        declared_quality = evaluate_budget_quality(
            declared_budget,
            capacity,
            is_floating=is_floating,
            source=row.get("data_source", "windfarminfo.com"),
            source_type="declared_csv",
        )
        lookup_quality = evaluate_budget_quality(
            row.get("total_budget_EUR_2026_lookup"),
            capacity,
            is_floating=is_floating,
            source=_first_present(row, "source", "budget_lookup_source"),
            source_type="internet_lookup",
        )

        override_quality = BudgetQuality(None, None, "missing", "no override", 0.0)
        override = overrides_by_name.get(row.get("wind_farm_name"))
        if override:
            override_year = override.get("budget_year", declared_year)
            override_budget = parse_budget_text_to_eur(override.get("budget_raw"), override_year, target_year)
            override_quality = evaluate_budget_quality(
                override_budget,
                capacity,
                is_floating=is_floating,
                source=override.get("source_url"),
                source_type="manual_override",
                requested_verification_level=override.get("verification_level"),
            )

        if override_quality.status == "valid":
            selected = override_quality
            source_type = "manual_override"
        elif declared_quality.status == "valid":
            selected = declared_quality
            source_type = "declared_csv"
        elif lookup_quality.status == "valid":
            selected = lookup_quality
            source_type = "internet_lookup"
        else:
            selected = BudgetQuality(
                None,
                None,
                "rejected",
                f"declared: {declared_quality.reason}; lookup: {lookup_quality.reason}; override: {override_quality.reason}",
                0.0,
                "D",
                "no valid budget source",
                "",
            )
            source_type = "none"

        rows.append(
            {
                "declared_budget_EUR_2026": declared_quality.value,
                "declared_budget_EUR_per_MW": declared_quality.unit_eur_per_mw,
                "declared_budget_verification_level": declared_quality.verification_level,
                "lookup_budget_EUR_per_MW": lookup_quality.unit_eur_per_mw,
                "lookup_budget_verification_level": lookup_quality.verification_level,
                "total_budget_EUR_2026_clean": selected.value,
                "budget_EUR_per_MW": selected.unit_eur_per_mw,
                "budget_quality_status": selected.status,
                "budget_quality_reason": selected.reason,
                "budget_confidence": selected.confidence,
                "budget_source_type": source_type,
                "budget_source_url": selected.source_url,
                "budget_verification_level": selected.verification_level,
                "budget_verification_reason": selected.verification_reason,
            }
        )

    result = source_result.drop(columns=[column for column in QUALITY_COLUMNS if column in source_result.columns], errors="ignore")
    result = result.drop(columns=["__declared_raw_for_quality"], errors="ignore")
    quality_df = pd.DataFrame(rows, index=result.index)
    result = pd.concat([result, quality_df], axis=1)
    if replace_total_budget:
        result["total_budget_EUR_2026"] = result["total_budget_EUR_2026_clean"]
    return result
