from __future__ import annotations

import time
from datetime import date
from functools import lru_cache

import requests

from .config import DEFAULT_FRANKFURTER_URL, DEFAULT_TARGET_YEAR, DEFAULT_WORLDBANK_URL
from .logging import log, write_jsonl

SYMBOL_TO_ISO = {
    "£": "GBP",
    "GBP": "GBP",
    "$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "DKK": "DKK",
    "NOK": "NOK",
    "SEK": "SEK",
    "DKr": "DKK",
}

FX_FALLBACK: dict[str, dict[int, float]] = {
    "GBP": {
        2010: 1.16,
        2011: 1.15,
        2012: 1.23,
        2013: 1.18,
        2014: 1.26,
        2015: 1.38,
        2016: 1.22,
        2017: 1.14,
        2018: 1.13,
        2019: 1.18,
        2020: 1.12,
        2021: 1.16,
        2022: 1.17,
        2023: 1.15,
        2024: 1.18,
        2025: 1.19,
        2026: 1.17,
    },
    "USD": {
        2010: 0.75,
        2011: 0.72,
        2012: 0.78,
        2013: 0.75,
        2014: 0.75,
        2015: 0.90,
        2016: 0.90,
        2017: 0.89,
        2018: 0.85,
        2019: 0.89,
        2020: 0.88,
        2021: 0.84,
        2022: 0.95,
        2023: 0.92,
        2024: 0.92,
        2025: 0.91,
        2026: 0.92,
    },
    "DKK": {
        2010: 0.134,
        2011: 0.134,
        2012: 0.134,
        2013: 0.134,
        2014: 0.134,
        2015: 0.134,
        2016: 0.134,
        2017: 0.134,
        2018: 0.134,
        2019: 0.134,
        2020: 0.134,
        2021: 0.134,
        2022: 0.134,
        2023: 0.134,
        2024: 0.134,
        2025: 0.134,
        2026: 0.134,
    },
    "NOK": {
        2010: 0.124,
        2011: 0.128,
        2012: 0.136,
        2013: 0.119,
        2014: 0.106,
        2015: 0.104,
        2016: 0.108,
        2017: 0.108,
        2018: 0.106,
        2019: 0.100,
        2020: 0.093,
        2021: 0.099,
        2022: 0.097,
        2023: 0.086,
        2024: 0.088,
        2025: 0.087,
        2026: 0.086,
    },
    "EUR": {year: 1.0 for year in range(2000, 2027)},
}


@lru_cache(maxsize=256)
def fx_rate_for_year(
    currency: str,
    year: int,
    frankfurter_url: str = DEFAULT_FRANKFURTER_URL,
) -> float:
    if currency == "EUR":
        return 1.0

    lookup_year = min(year, date.today().year)
    date_str = f"{lookup_year}-07-01"
    url = f"{frankfurter_url}/{date_str}?base={currency}&symbols=EUR"
    started_at = time.perf_counter()

    try:
        resp = requests.get(url, timeout=8)
        elapsed = round(time.perf_counter() - started_at, 3)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        rate = (data.get("rates") or {}).get("EUR")
        if rate:
            write_jsonl(
                {
                    "api": "frankfurter",
                    "currency": currency,
                    "year": year,
                    "date": date_str,
                    "rate": rate,
                    "elapsed_s": elapsed,
                }
            )
            log.debug("FX %s/EUR %d -> %.4f (Frankfurter)", currency, year, rate)
            return float(rate)
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        log.warning("Frankfurter failed %s %d: %s - using fallback", currency, year, exc)
        write_jsonl(
            {"api": "frankfurter", "currency": currency, "year": year, "error": str(exc)}
        )

    fallback = FX_FALLBACK.get(currency, {})
    nearest_year = min(fallback.keys(), key=lambda fallback_year: abs(fallback_year - year), default=year)
    rate = fallback.get(year) or fallback.get(nearest_year)
    log.debug("FX %s/EUR %d -> %.4f (fallback)", currency, year, rate or 1.0)
    return float(rate or 1.0)


@lru_cache(maxsize=8)
def load_eurozone_cpi(worldbank_url: str = DEFAULT_WORLDBANK_URL) -> dict[int, float]:
    for country in ("XC", "EMU", "EUU", "DE", "FR"):
        url = worldbank_url.format(country=country)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
                continue

            cpi_map: dict[int, float] = {}
            for entry in payload[1]:
                if entry.get("value") is not None:
                    cpi_map[int(entry["date"])] = float(entry["value"])

            if len(cpi_map) >= 5:
                log.info(
                    "World Bank CPI loaded (%s): %d years (%d-%d)",
                    country,
                    len(cpi_map),
                    min(cpi_map),
                    max(cpi_map),
                )
                write_jsonl(
                    {
                        "api": "worldbank_cpi",
                        "country": country,
                        "years": len(cpi_map),
                        "min_yr": min(cpi_map),
                        "max_yr": max(cpi_map),
                    }
                )
                return cpi_map
        except (requests.RequestException, ValueError, TypeError) as exc:
            log.warning("World Bank CPI %s: %s", country, exc)

    log.warning("World Bank CPI unavailable - using built-in fallback")
    return {
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


def inflation_factor(
    from_year: int,
    to_year: int = DEFAULT_TARGET_YEAR,
    worldbank_url: str = DEFAULT_WORLDBANK_URL,
) -> float:
    cpi = load_eurozone_cpi(worldbank_url)
    if from_year == to_year:
        return 1.0

    max_year = max(cpi)
    cpi_from = cpi.get(from_year) or cpi.get(min(cpi.keys(), key=lambda year: abs(year - from_year)))
    cpi_to = cpi.get(to_year) or cpi.get(max_year)
    if not cpi_from or not cpi_to:
        return 1.0
    return cpi_to / cpi_from
