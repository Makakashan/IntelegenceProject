from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_ENV_PATH,
    DEFAULT_FRANKFURTER_URL,
    DEFAULT_INPUT_PATH,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REQUEST_DELAY,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_SERPER_URL,
    DEFAULT_TARGET_YEAR,
    DEFAULT_WORLDBANK_URL,
    FRANKFURTER_URL_ENV,
    SERPER_API_KEY_ENV,
    SERPER_URL_ENV,
    WORLDBANK_URL_ENV,
    RuntimeConfig,
    env_url,
    load_env_file,
    normalize_api_key,
)
from .logging import configure_logging, log
from .runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Enrich wind farm CSV with total_budget_EUR_{DEFAULT_TARGET_YEAR} "
            "(historical FX + CPI inflation)"
        )
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    parser.add_argument("--search-results", type=int, default=DEFAULT_SEARCH_RESULTS)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    load_env_file(args.env_file)
    return RuntimeConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        env_path=args.env_file,
        target_year=args.target_year,
        workers=args.workers,
        request_delay=args.request_delay,
        search_results=args.search_results,
        serper_api_key=normalize_api_key(args.api_key or os.getenv(SERPER_API_KEY_ENV)),
        serper_url=env_url(SERPER_URL_ENV, DEFAULT_SERPER_URL),
        frankfurter_url=env_url(FRANKFURTER_URL_ENV, DEFAULT_FRANKFURTER_URL),
        worldbank_url=env_url(WORLDBANK_URL_ENV, DEFAULT_WORLDBANK_URL),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    configure_logging(config.log_dir)
    try:
        run(config)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        sys.exit(1)
