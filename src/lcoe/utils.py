from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nan_metrics(keys: tuple[str, ...]) -> pd.Series:
    return pd.Series({key: np.nan for key in keys})


def read_project_csv(csv_path: str | Path) -> pd.DataFrame:
    read_csv = pd.read_csv
    source_df = read_csv(str(csv_path))
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("CSV reader did not return a DataFrame.")
    return source_df


def flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = ".") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep))
        else:
            items.append((new_key, value))
    return items
