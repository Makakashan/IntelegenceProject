from __future__ import annotations

import re

from .config import DEFAULT_TARGET_YEAR
from .finance import SYMBOL_TO_ISO, fx_rate_for_year, inflation_factor

BUDGET_PATTERNS = [
    r"(?:total\s+(?:investment|budget|cost|capex|project\s+cost|project\s+value)[^\n]{0,80}?)"
    r"((?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b))",
    r"((?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b)"
    r"[^\n]{0,50}?(?:investment|budget|cost|capex|project))",
    r"((?:£|€|\$|DKr|DKK|NOK|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b))",
]

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
SYMBOL_PATTERN = re.compile(r"(£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)")
UNIT_PATTERN = re.compile(r"billion|million|bn|mn|(?<=\d)[bm]", re.I)
CLEAN_RAW_PATTERN = re.compile(
    r"(?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*"
    r"[\d,\.]+(?:\s*[-–]\s*[\d,\.]+)?\s*"
    r"(?:billion|million|bn|mn|[bBmM])\b",
    re.IGNORECASE,
)


def extract_budget(texts: list[str]) -> str | None:
    combined = " ".join(texts)
    for pattern in BUDGET_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def clean_raw(raw: str) -> str:
    if not raw or raw == "Not found":
        return raw
    match = CLEAN_RAW_PATTERN.search(raw)
    return match.group().strip() if match else raw


def extract_year(
    texts: list[str],
    fallback: int | None = None,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> int:
    combined = " ".join(texts)
    contextual_years = re.findall(
        r"(20\d{2})[^\n]{0,60}?(?:investment|cost|budget|capex|FID|financial\s+close)",
        combined,
        re.IGNORECASE,
    )
    if contextual_years:
        return int(contextual_years[0])

    contextual_years = re.findall(
        r"(?:investment|cost|budget|FID|financial\s+close)[^\n]{0,60}?(20\d{2})",
        combined,
        re.IGNORECASE,
    )
    if contextual_years:
        return int(contextual_years[0])

    all_years = [int(year) for year in YEAR_PATTERN.findall(combined) if 2005 <= int(year) <= target_year]
    return max(all_years) if all_years else fallback or target_year - 2


def budget_currency(raw: str) -> str:
    match = SYMBOL_PATTERN.search(raw)
    symbol = match.group() if match else "€"
    return SYMBOL_TO_ISO.get(symbol, "EUR")


def parse_raw_to_target_eur(
    raw: str,
    budget_year: int,
    target_year: int = DEFAULT_TARGET_YEAR,
    frankfurter_url: str | None = None,
    worldbank_url: str | None = None,
) -> float | None:
    if not raw or raw == "Not found":
        return None

    currency = budget_currency(raw)
    number_match = re.search(r"[\d,\.]+", raw)
    if not number_match:
        return None

    try:
        amount = float(number_match.group().replace(",", ""))
    except ValueError:
        return None

    unit_match = UNIT_PATTERN.search(raw)
    if unit_match:
        unit = unit_match.group().lower()
        if unit in ("billion", "bn", "b"):
            amount *= 1_000_000_000
        elif unit in ("million", "mn", "m"):
            amount *= 1_000_000

    fx = fx_rate_for_year(currency, budget_year, frankfurter_url) if frankfurter_url else fx_rate_for_year(currency, budget_year)
    inflation = (
        inflation_factor(budget_year, target_year, worldbank_url)
        if worldbank_url
        else inflation_factor(budget_year, target_year)
    )
    return round(amount * fx * inflation, 0)


def parse_raw_to_eur2026(raw: str, budget_year: int) -> float | None:
    return parse_raw_to_target_eur(raw, budget_year, target_year=DEFAULT_TARGET_YEAR)
