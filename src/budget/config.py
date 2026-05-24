from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SERPER_URL = "https://google.serper.dev/search"
DEFAULT_FRANKFURTER_URL = "https://api.frankfurter.app"
DEFAULT_WORLDBANK_URL = (
    "https://api.worldbank.org/v2/country/{country}/indicator/"
    "FP.CPI.TOTL?format=json&per_page=200"
)

SERPER_URL_ENV = "SERPER_URL"
FRANKFURTER_URL_ENV = "FRANKFURTER_URL"
WORLDBANK_URL_ENV = "WORLDBANK_URL"

DEFAULT_INPUT_PATH = Path("input") / "wind_for_clean.csv"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_TARGET_YEAR = 2026
DEFAULT_TARGET_CCY = "EUR"
DEFAULT_MAX_WORKERS = 5
DEFAULT_REQUEST_DELAY = 0.3
DEFAULT_SEARCH_RESULTS = 6
SERPER_API_KEY_ENV = "SERPER_API_KEY"
SERPER_API_KEY_PLACEHOLDER = "your_serper_api_key_here"


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    log_dir: Path = DEFAULT_LOG_DIR
    env_path: Path = DEFAULT_ENV_PATH
    target_year: int = DEFAULT_TARGET_YEAR
    target_ccy: str = DEFAULT_TARGET_CCY
    workers: int = DEFAULT_MAX_WORKERS
    request_delay: float = DEFAULT_REQUEST_DELAY
    search_results: int = DEFAULT_SEARCH_RESULTS
    serper_api_key: str = ""
    serper_url: str = DEFAULT_SERPER_URL
    frankfurter_url: str = DEFAULT_FRANKFURTER_URL
    worldbank_url: str = DEFAULT_WORLDBANK_URL


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> None:
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def normalize_api_key(api_key: str | None) -> str:
    value = (api_key or "").strip()
    if value == SERPER_API_KEY_PLACEHOLDER:
        return ""
    return value


def env_url(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default
