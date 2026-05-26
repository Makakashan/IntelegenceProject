from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lcoe.calculations import load_and_analyse
from src.lcoe.constants import DEFAULT_DATASET_CANDIDATES, OUTPUT_DIR
from src.lcoe.ml import build_capex_ml_report


def resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_DATASET_CANDIDATES[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare offshore wind CAPEX models.")
    parser.add_argument("--dataset", default=None, help="Path to the cleaned wind dataset.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "model_training_report.json"),
        help="Where to write the model comparison report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    df = load_and_analyse(dataset_path)
    report = build_capex_ml_report(df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Model training report saved to '{output_path}'.")


if __name__ == "__main__":
    main()
