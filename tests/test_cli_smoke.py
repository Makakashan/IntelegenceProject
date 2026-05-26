from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_classifier_cli_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "wind_farm_classifier.py",
            "--dataset",
            "input/wind_dataset.csv",
            "--json",
            "examples/new_project_500mw.json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "JSON report saved" in result.stdout
