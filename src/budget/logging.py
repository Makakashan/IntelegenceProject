from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from .config import DEFAULT_LOG_DIR

log_file = DEFAULT_LOG_DIR / f"requests_{date.today().isoformat()}.jsonl"
_log_lock = threading.Lock()
log = logging.getLogger("wind")


def configure_logging(log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    global log_file

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"requests_{date.today().isoformat()}.jsonl"

    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    file_handler = logging.FileHandler(log_file.with_suffix(".log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
    )

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log_file


def write_jsonl(entry: dict) -> None:
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
