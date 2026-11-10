"""ML-01 R01: el paquete instalado se ejecuta como módulo y como comando."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_python_dash_m_lists_the_stages() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "thehouse_ml", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    for name in ("download", "prepare", "train", "evaluate", "export"):
        assert name in result.stdout


def test_console_script_runs_a_stage(tmp_path: Path) -> None:
    executable = shutil.which("thehouse-ml")
    if executable is None:
        pytest.skip("Script de consola no instalado en este entorno")

    result = subprocess.run(
        [executable, "--root", str(tmp_path), "prepare"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "prepare: pendiente de ML-02"
