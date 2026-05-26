from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from . import logging as budget_logging
from .config import DEFAULT_TARGET_YEAR, RuntimeConfig, SERPER_API_KEY_ENV
from .finance import fx_rate_for_year, inflation_factor, load_eurozone_cpi
from .logging import log, write_jsonl
from .parsing import (
    budget_currency,
    clean_raw,
    extract_budget,
    extract_year,
    parse_raw_to_target_eur,
)
from .quality import apply_budget_quality_columns, evaluate_budget_quality
from .search import build_search_queries, get_snippets, serper_search

ENRICHMENT_COLUMNS = (
    "wind_farm_name",
    "total_budget_raw",
    "budget_year",
    "budget_currency",
    "fx_rate_to_eur",
    "inflation_factor",
    "total_budget_EUR_2026",
    "budget_EUR_per_MW",
    "budget_quality_status",
    "budget_quality_reason",
    "budget_confidence",
    "budget_source_type",
    "budget_verification_level",
    "budget_verification_reason",
    "source",
    "lookup_date",
)


def lookup(
    farm_name: str,
    capacity_mw: float | None,
    is_floating: bool,
    api_key: str,
    request_delay: float,
    search_results: int,
    target_year: int,
    serper_url: str,
    frankfurter_url: str,
    worldbank_url: str,
) -> dict:
    result = {
        "wind_farm_name": farm_name,
        "total_budget_raw": None,
        "budget_year": None,
        "budget_currency": None,
        "fx_rate_to_eur": None,
        "inflation_factor": None,
        "total_budget_EUR_2026": None,
        "budget_EUR_per_MW": None,
        "budget_quality_status": "missing",
        "budget_quality_reason": "not evaluated",
        "budget_confidence": 0.0,
        "budget_source_type": "internet_lookup",
        "budget_verification_level": "D",
        "budget_verification_reason": "not evaluated",
        "source": "",
        "lookup_date": date.today().isoformat(),
        "error": None,
    }

    selected_snippets: list[str] = []
    selected_source = ""
    raw = None

    for index, query in enumerate(build_search_queries(farm_name)):
        try:
            time.sleep(request_delay * index)
            data = serper_search(query, api_key, num=search_results, serper_url=serper_url)
            snippets, sources = get_snippets(data, farm_name=farm_name)
            for snippet, source in zip(snippets, sources, strict=False):
                candidate_raw = extract_budget([snippet])
                if candidate_raw:
                    raw = candidate_raw
                    selected_snippets = [snippet]
                    selected_source = source
                    break
            if raw:
                break
        except requests.HTTPError as exc:
            message = f"HTTP {exc.response.status_code}"
            log.warning("%-40s  x  %s", farm_name, message)
            result["error"] = message
            return result
        except (requests.RequestException, ValueError, TypeError) as exc:
            log.warning("%-40s  x  %s", farm_name, str(exc))
            result["error"] = str(exc)
            return result

    if not raw:
        log.info("%-40s  ?  not found", farm_name)
        result["total_budget_raw"] = "Not found"
        return result

    raw = clean_raw(raw)
    budget_year = extract_year(selected_snippets, target_year=target_year)
    currency = budget_currency(raw)
    fx = fx_rate_for_year(currency, budget_year, frankfurter_url)
    inflation = inflation_factor(budget_year, target_year, worldbank_url)
    target_eur = parse_raw_to_target_eur(
        raw,
        budget_year,
        target_year=target_year,
        frankfurter_url=frankfurter_url,
        worldbank_url=worldbank_url,
    )
    quality = evaluate_budget_quality(
        target_eur,
        capacity_mw,
        is_floating=is_floating,
        source=selected_source,
        source_type="internet_lookup",
    )

    result.update(
        {
            "total_budget_raw": raw,
            "budget_year": budget_year,
            "budget_currency": currency,
            "fx_rate_to_eur": round(fx, 5),
            "inflation_factor": round(inflation, 4),
            "total_budget_EUR_2026": target_eur,
            "budget_EUR_per_MW": quality.unit_eur_per_mw,
            "budget_quality_status": quality.status,
            "budget_quality_reason": quality.reason,
            "budget_confidence": quality.confidence,
            "budget_source_type": "internet_lookup",
            "budget_verification_level": quality.verification_level,
            "budget_verification_reason": quality.verification_reason,
            "source": selected_source,
        }
    )

    log.info(
        "%-40s  ok  %s  (%s %d)  fx=%.3f  inf=%.3f  ->  %s",
        farm_name,
        raw,
        currency,
        budget_year,
        fx,
        inflation,
        target_eur,
    )
    return result


def is_floating_foundation(value: object) -> bool:
    text = str(value or "").lower()
    return any(keyword in text for keyword in ("floating", "spar", "semi-submersible", "tlp"))


def progress_bar(done: int, total: int) -> str:
    pct = done * 100 // total if total else 100
    return "#" * (pct // 5) + "." * (20 - pct // 5)


def output_paths(input_path: Path, output_dir: Path, target_year: int = DEFAULT_TARGET_YEAR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    if stem.endswith("_for_clean"):
        stem = f"{stem.removesuffix('_for_clean')}_dataset"
    return (
        output_dir / f"{stem}.csv",
        output_dir / f"{stem}_budget_lookup_{target_year}EUR.json",
    )


def run(config: RuntimeConfig) -> None:
    if not config.serper_api_key:
        raise ValueError(
            f"Serper API key is required. Set {SERPER_API_KEY_ENV} in .env/env or pass --api-key."
        )

    log.info("Loading CPI from World Bank API...")
    load_eurozone_cpi(config.worldbank_url)

    df = pd.read_csv(config.input_path, encoding="utf-8-sig", on_bad_lines="warn")
    if "wind_farm_name" not in df.columns:
        raise ValueError(f"Column 'wind_farm_name' not found. Available: {list(df.columns)}")

    farms = df["wind_farm_name"].dropna().unique().tolist()
    if not farms:
        raise ValueError("Column 'wind_farm_name' does not contain any non-empty values.")
    farm_context = df.drop_duplicates("wind_farm_name").set_index("wind_farm_name", drop=False)

    log.info(
        "CSV: %s  |  unique farms: %d  |  workers: %d  |  target year: %d EUR",
        config.input_path,
        len(farms),
        config.workers,
        config.target_year,
    )
    log.info("JSONL log:  %s", budget_logging.log_file)
    print()

    budget_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(
                lookup,
                farm,
                farm_context.loc[farm].get("installed_capacity_MW"),
                is_floating_foundation(farm_context.loc[farm].get("foundation_type")),
                config.serper_api_key,
                config.request_delay,
                config.search_results,
                config.target_year,
                config.serper_url,
                config.frankfurter_url,
                config.worldbank_url,
            ): farm
            for farm in farms
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            budget_map[result["wind_farm_name"]] = result
            print(f"\r  [{progress_bar(done, len(farms))}] {done}/{len(farms)}", end="", flush=True)

    print()

    results_df = pd.DataFrame(list(budget_map.values()))[list(ENRICHMENT_COLUMNS)]

    drop_cols = [column for column in ENRICHMENT_COLUMNS[1:] if column in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.merge(results_df, on="wind_farm_name", how="left")
    df = apply_budget_quality_columns(df, target_year=config.target_year, replace_total_budget=True)
    out_csv, out_json = output_paths(config.input_path, config.output_dir, config.target_year)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(list(budget_map.values()), file, ensure_ascii=False, indent=2, default=str)

    found = int(pd.to_numeric(df["total_budget_EUR_2026"], errors="coerce").notna().sum())
    not_found = len(farms) - found

    print()
    print("=" * 64)
    print(f"  Done:    {found} found  |  {not_found} not found  |  total {len(farms)}")
    print("  Method:  FX (Frankfurter/ECB) + inflation (World Bank CPI)")
    print(f"  Target:  {config.target_year} EUR")
    print(f"  CSV   ->  {out_csv}")
    print(f"  JSON  ->  {out_json}")
    print(f"  Log   ->  {budget_logging.log_file}")
    print("=" * 64)

    write_jsonl(
        {
            "event": "run_complete",
            "farms": len(farms),
            "found": found,
            "target_year": config.target_year,
        }
    )
