import re
import sys
import json
import time
import logging
import argparse
import threading
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from functools import lru_cache


SERPER_API_KEY  = "dd0415b5c6f150f16ac3b492368b25a8e47d3718"
SERPER_URL      = "https://google.serper.dev/search"
FRANKFURTER_URL = "https://api.frankfurter.app"
WORLDBANK_URL   = "https://api.worldbank.org/v2/country/{country}/indicator/FP.CPI.TOTL?format=json&per_page=200"

TARGET_YEAR     = 2026
TARGET_CCY      = "EUR"
MAX_WORKERS     = 5
REQUEST_DELAY   = 0.3
LOG_DIR         = Path("logs")

CPI_COUNTRY     = "XC"

SYMBOL_TO_ISO = {
    "£":   "GBP",
    "GBP": "GBP",
    "$":   "USD",
    "USD": "USD",
    "€":   "EUR",
    "EUR": "EUR",
    "DKK": "DKK",
    "NOK": "NOK",
    "SEK": "SEK",
    "DKr": "DKK",
}

FX_FALLBACK: dict[str, dict[int, float]] = {
    "GBP": {2010:1.16,2011:1.15,2012:1.23,2013:1.18,2014:1.26,2015:1.38,
            2016:1.22,2017:1.14,2018:1.13,2019:1.18,2020:1.12,2021:1.16,
            2022:1.17,2023:1.15,2024:1.18,2025:1.19,2026:1.17},
    "USD": {2010:0.75,2011:0.72,2012:0.78,2013:0.75,2014:0.75,2015:0.90,
            2016:0.90,2017:0.89,2018:0.85,2019:0.89,2020:0.88,2021:0.84,
            2022:0.95,2023:0.92,2024:0.92,2025:0.91,2026:0.92},
    "DKK": {2010:0.134,2011:0.134,2012:0.134,2013:0.134,2014:0.134,
            2015:0.134,2016:0.134,2017:0.134,2018:0.134,2019:0.134,
            2020:0.134,2021:0.134,2022:0.134,2023:0.134,2024:0.134,
            2025:0.134,2026:0.134},
    "NOK": {2010:0.124,2011:0.128,2012:0.136,2013:0.119,2014:0.106,
            2015:0.104,2016:0.108,2017:0.108,2018:0.106,2019:0.100,
            2020:0.093,2021:0.099,2022:0.097,2023:0.086,2024:0.088,
            2025:0.087,2026:0.086},
    "EUR": {y: 1.0 for y in range(2000, 2027)},
}

LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"requests_{date.today().isoformat()}.jsonl"

_log_lock = threading.Lock()

def _jsonl(entry: dict):
    """Thread-safe write to JSONL."""
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with _log_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

_fh = logging.FileHandler(log_file.with_suffix(".log"), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))

log = logging.getLogger("wind")
log.setLevel(logging.DEBUG)
log.addHandler(_fh)
log.addHandler(_ch)

@lru_cache(maxsize=256)
def fx_rate_for_year(currency: str, year: int) -> float:
    """
    Return FX rate currency → EUR for the given year.
    Attempt to use the ECB average for the year (date = July 1st).
    Data source: European Central Bank via frankfurter.dev
    """
    if currency == "EUR":
        return 1.0

    # Frankfurter has no data for future dates — cap to today
    lookup_year = min(year, date.today().year)
    date_str = f"{lookup_year}-07-01"
    # v2 API uses 'symbols', older app API uses same; response may be list or dict
    url = f"{FRANKFURTER_URL}/{date_str}?base={currency}&symbols=EUR"
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=8)
        elapsed = round(time.perf_counter() - t0, 3)
        resp.raise_for_status()
        data = resp.json()
        # v2 returns a list of objects: [{"date":..., "rates":{...}}]
        if isinstance(data, list):
            data = data[0] if data else {}
        rate = (data.get("rates") or {}).get("EUR")
        if rate:
            _jsonl({"api": "frankfurter", "currency": currency, "year": year,
                    "date": date_str, "rate": rate, "elapsed_s": elapsed})
            log.debug("FX  %s/%s  %d  →  %.4f  (Frankfurter)", currency, "EUR", year, rate)
            return float(rate)
    except Exception as e:
        log.warning("Frankfurter failed %s %d: %s — using fallback", currency, year, e)
        _jsonl({"api": "frankfurter", "currency": currency, "year": year, "error": str(e)})

    # Fallback
    fb = FX_FALLBACK.get(currency, {})
    rate = fb.get(year) or fb.get(min(fb.keys(), key=lambda y: abs(y - year), default=year))
    log.debug("FX  %s/%s  %d  →  %.4f  (fallback)", currency, "EUR", year, rate or 1.0)
    return float(rate or 1.0)


# WORLD BANK API — CPI DATA


@lru_cache(maxsize=1)
def _load_eurozone_cpi() -> dict[int, float]:
    """
    Load CPI (index, base ~2010=100) for the Eurozone from World Bank API.
    Returns a mapping {year: cpi_value}.
    """
    # World Bank Eurozone = "XC" or try "EMU"
    for country in ("XC", "EMU", "EUU", "EUU", "DE", "FR"):
        url = WORLDBANK_URL.format(country=country)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            # World Bank wraps data in [metadata, rows]
            if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
                continue
            cpi_map = {}
            for entry in payload[1]:
                if entry.get("value") is not None:
                    yr = int(entry["date"])
                    cpi_map[yr] = float(entry["value"])
            if len(cpi_map) >= 5:
                log.info("World Bank CPI loaded (%s): %d years (%d–%d)",
                         country, len(cpi_map), min(cpi_map), max(cpi_map))
                _jsonl({"api": "worldbank_cpi", "country": country,
                        "years": len(cpi_map), "min_yr": min(cpi_map), "max_yr": max(cpi_map)})
                return cpi_map
        except Exception as e:
            log.warning("World Bank CPI %s: %s", country, e)

    # Hard fallback — Eurozone HICP series converted to approximate 2010=100 base
    log.warning("World Bank CPI unavailable — using built-in fallback")
    return {
        2010:100.0, 2011:102.7, 2012:105.2, 2013:106.7, 2014:107.1,
        2015:107.2, 2016:107.3, 2017:109.1, 2018:111.0, 2019:112.1,
        2020:112.3, 2021:114.8, 2022:122.8, 2023:130.2, 2024:133.5,
        2025:136.0, 2026:138.5,
    }

def inflation_factor(from_year: int, to_year: int = TARGET_YEAR) -> float:
    """
    Inflation factor: how many EUR_{to_year} equals 1 EUR_{from_year}.
    factor = CPI(to_year) / CPI(from_year)
    """
    cpi = _load_eurozone_cpi()
    if from_year == to_year:
        return 1.0
    max_yr = max(cpi)
    cpi_from = cpi.get(from_year) or cpi.get(min(cpi.keys(), key=lambda y: abs(y - from_year)))
    cpi_to   = cpi.get(to_year)   or cpi.get(max_yr)
    if not cpi_from or not cpi_to:
        return 1.0
    return cpi_to / cpi_from


# BUDGET EXTRACTION (REGEX)


BUDGET_PATTERNS = [
    # "total investment of £9 billion" / "total cost: $3.6bn"
    r'(?:total\s+(?:investment|budget|cost|capex|project\s+cost|project\s+value)[^\n]{0,80}?)'
    r'((?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b))',

    # "£9 billion investment"
    r'((?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b)'
    r'[^\n]{0,50}?(?:investment|budget|cost|capex|project))',

    # bare: "£9 billion" / "DKK 70 billion"
    r'((?:£|€|\$|DKr|DKK|NOK|SEK)\s*[\d,\.]+(?:\s*[\–\-]\s*[\d,\.]+)?\s*(?:billion|million|bn|mn|B\b|M\b))',
]

YEAR_PATTERN    = re.compile(r'\b(20\d{2})\b')
SYMBOL_PATTERN  = re.compile(r'(£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)')
UNIT_PATTERN    = re.compile(r'billion|million|bn|mn|(?<=\d)[bm]', re.I)

def extract_budget(texts: list[str]) -> str | None:
    combined = " ".join(texts)
    for pat in BUDGET_PATTERNS:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

# Tight pattern: extract only SYMBOL + NUMBER + UNIT, drop surrounding context
_CLEAN_RAW = re.compile(
    r'(?:£|€|\$|GBP|EUR|USD|DKK|NOK|DKr|SEK)\s*'
    r'[\d,\.]+(?:\s*[-–]\s*[\d,\.]+)?\s*'
    r'(?:billion|million|bn|mn|[bBmM])\b',
    re.IGNORECASE,
)

def clean_raw(raw: str) -> str:
    """Strip junk context, keep only SYMBOL NUMBER UNIT (e.g. 'EUR 488 million')."""
    if not raw or raw == "Not found":
        return raw
    m = _CLEAN_RAW.search(raw)
    return m.group().strip() if m else raw

def extract_year(texts: list[str], fallback: int = TARGET_YEAR - 2) -> int:
    """
    Attempt to find the budget year in snippets.
    Prefer a year that appears near words like 'investment', 'cost', 'budget', 'FID'.
    """
    combined = " ".join(texts)
    # Contextual search: year near financial keywords
    ctx = re.findall(
        r'(20\d{2})[^\n]{0,60}?(?:investment|cost|budget|capex|FID|financial\s+close)',
        combined, re.IGNORECASE)
    if ctx:
        return int(ctx[0])
    ctx2 = re.findall(
        r'(?:investment|cost|budget|FID|financial\s+close)[^\n]{0,60}?(20\d{2})',
        combined, re.IGNORECASE)
    if ctx2:
        return int(ctx2[0])
    # All years found — pick the largest reasonable one
    all_years = [int(y) for y in YEAR_PATTERN.findall(combined)
                 if 2005 <= int(y) <= TARGET_YEAR]
    return max(all_years) if all_years else fallback

def parse_raw_to_eur2026(raw: str, budget_year: int) -> float | None:
    """
    Convert a string like '£9 billion' to EUR in TARGET_YEAR (e.g. 2026):
      1. number × unit → float in original currency
      2. × FX(currency, budget_year) → EUR in budget_year
      3. × inflation_factor(budget_year, TARGET_YEAR) → EUR TARGET_YEAR
    """
    if not raw or raw == "Not found":
        return None

    sym_m = SYMBOL_PATTERN.search(raw)
    symbol = sym_m.group() if sym_m else "€"
    currency = SYMBOL_TO_ISO.get(symbol, "EUR")

    num_m = re.search(r'[\d,\.]+', raw)
    if not num_m:
        return None
    try:
        amount = float(num_m.group().replace(",", ""))
    except ValueError:
        return None

    unit_m = UNIT_PATTERN.search(raw)
    if unit_m:
        u = unit_m.group().lower()
        if u in ("billion", "bn", "b"):
            amount *= 1_000_000_000
        elif u in ("million", "mn", "m"):
            amount *= 1_000_000

    # [1] → EUR in the budget year
    fx   = fx_rate_for_year(currency, budget_year)
    eur_historic = amount * fx

    # [2] → EUR in TARGET_YEAR (with inflation)
    inf  = inflation_factor(budget_year, TARGET_YEAR)
    eur_2026 = eur_historic * inf

    return round(eur_2026, 0)


#  SERPER — GOOGLE SEARCH API


def serper_search(query: str, api_key: str, num: int = 6) -> dict:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num, "gl": "gb", "hl": "en"}
    for attempt in range(4):
        t0 = time.perf_counter()
        resp = requests.post(SERPER_URL, headers=headers, json=payload, timeout=12)
        elapsed = round(time.perf_counter() - t0, 3)
        if resp.status_code == 429:
            wait = 2 ** attempt
            log.debug("Serper 429 — retry in %ds (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        _jsonl({"api": "serper", "query": query, "status": resp.status_code,
                "elapsed_s": elapsed, "hits": len(data.get("organic", []))})
        return data
    # All retries exhausted — raise the last 429
    resp.raise_for_status()
    return {}

def get_snippets(results: dict) -> tuple[list[str], list[str]]:
    snippets, sources = [], []
    for item in results.get("organic", []):
        s = item.get("snippet", "")
        if s:
            snippets.append(f"{item.get('title','')}. {s}")
            sources.append(item.get("link", ""))
    for key in ("answerBox", "knowledgeGraph"):
        box = results.get(key, {})
        for field in ("answer", "description", "snippet"):
            val = box.get(field, "")
            if val:
                snippets.insert(0, val)
    return snippets, sources


#  LOOKUP — SINGLE WIND FARM


def lookup(farm_name: str) -> dict:
    result = {
        "wind_farm_name":        farm_name,
        "total_budget_raw":      None,
        "budget_year":           None,
        "budget_currency":       None,
        "fx_rate_to_eur":        None,
        "inflation_factor":      None,
        "total_budget_EUR_2026": None,
        "source":                "",
        "lookup_date":           date.today().isoformat(),
        "error":                 None,
    }

    queries = [
        f'"{farm_name}" offshore wind farm total investment budget cost billion',
        f'{farm_name} wind farm total project cost financial close',
    ]

    all_snippets: list[str] = []
    all_sources:  list[str] = []

    for i, q in enumerate(queries):
        try:
            time.sleep(REQUEST_DELAY * i)
            data = serper_search(q, SERPER_API_KEY, num=6)
            snips, srcs = get_snippets(data)
            all_snippets.extend(snips)
            all_sources.extend(srcs)
            if extract_budget(all_snippets):
                break
        except requests.HTTPError as e:
            msg = f"HTTP {e.response.status_code}"
            log.warning("%-40s  ✗  %s", farm_name, msg)
            result["error"] = msg
            return result
        except Exception as e:
            log.warning("%-40s  ✗  %s", farm_name, str(e))
            result["error"] = str(e)
            return result

    raw = extract_budget(all_snippets)
    if not raw:
        log.info("%-40s  ?  not found", farm_name)
        result["total_budget_raw"] = "Not found"
        return result

    raw = clean_raw(raw)

    # Budget year
    byr = extract_year(all_snippets)

    # Currency
    sym_m = SYMBOL_PATTERN.search(raw)
    symbol   = sym_m.group() if sym_m else "€"
    currency = SYMBOL_TO_ISO.get(symbol, "EUR")

    # Conversion
    fx  = fx_rate_for_year(currency, byr)
    inf = inflation_factor(byr, TARGET_YEAR)
    eur = parse_raw_to_eur2026(raw, byr)

    result.update({
        "total_budget_raw":      raw,
        "budget_year":           byr,
        "budget_currency":       currency,
        "fx_rate_to_eur":        round(fx, 5),
        "inflation_factor":      round(inf, 4),
        "total_budget_EUR_2026": eur,
        "source":                all_sources[0] if all_sources else "",
    })

    log.info("%-40s  ✓  %s  (%s %d)  fx=%.3f  inf=%.3f  →  %s",
             farm_name, raw, currency, byr, fx, inf, eur)
    return result


#  MAIN


def run(csv_path: str, workers: int):
    # Pre-load CPI before starting threads
    log.info("Loading CPI from World Bank API...")
    _load_eurozone_cpi()

    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="warn")
    if "wind_farm_name" not in df.columns:
        sys.exit(f"Column 'wind_farm_name' not found. Available: {list(df.columns)}")

    farms = df["wind_farm_name"].dropna().unique().tolist()
    log.info("CSV: %s  |  unique farms: %d  |  workers: %d  |  target year: %d EUR",
             csv_path, len(farms), workers, TARGET_YEAR)
    log.info("JSONL log:  %s", log_file)
    print()

    budget_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(lookup, farm): farm for farm in farms}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            budget_map[res["wind_farm_name"]] = res
            pct = done * 100 // len(farms)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {done}/{len(farms)}", end="", flush=True)

    print()

    cols = ["wind_farm_name", "total_budget_raw", "budget_year",
            "budget_currency", "fx_rate_to_eur", "inflation_factor",
            "total_budget_EUR_2026", "source", "lookup_date"]

    results_df = pd.DataFrame(list(budget_map.values()))[cols]

    # Drop any stale enrichment columns that may exist from a previous run
    drop_cols = [c for c in cols[1:] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.merge(results_df, on="wind_farm_name", how="left")

    out_dir  = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_csv  = out_dir / (Path(csv_path).stem + f"_enriched_{TARGET_YEAR}EUR.csv")
    out_json = out_dir / (Path(csv_path).stem + f"_budgets_{TARGET_YEAR}EUR.json")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(list(budget_map.values()), f, ensure_ascii=False, indent=2, default=str)

    found    = sum(1 for r in budget_map.values() if r.get("total_budget_EUR_2026"))
    notfound = len(farms) - found

    print()
    print("═" * 64)
    print(f"  Done:    {found} found  |  {notfound} not found  |  total {len(farms)}")
    print(f"  Method:  FX (Frankfurter/ECB) + inflation (World Bank CPI)")
    print(f"  Target:  {TARGET_YEAR} EUR")
    print(f"  CSV   →  {out_csv}")
    print(f"  JSON  →  {out_json}")
    print(f"  Log   →  {log_file}")
    print("═" * 64)

    _jsonl({"event": "run_complete", "farms": len(farms),
            "found": found, "target_year": TARGET_YEAR})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Enrich wind farm CSV with total_budget_EUR_{TARGET_YEAR} "
                    f"(historical FX + CPI inflation)"
    )
    parser.add_argument("--input",   "-i", default="windTurbineData.csv")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS,
                        help=f"Number of parallel workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    run(args.input, args.workers)
