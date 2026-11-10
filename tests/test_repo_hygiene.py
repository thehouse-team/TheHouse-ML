"""ML-01 R03: datos y modelos nunca entran a git (constitution §6 y §7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("data/raw/", "data/processed/", "models/")
FORBIDDEN_SUFFIXES = (".pkl", ".pickle", ".joblib")


def forbidden_files(tracked: list[str]) -> list[str]:
    """Archivos versionados que la constitution prohíbe."""
    return [
        path
        for path in tracked
        if path.replace("\\", "/").startswith(FORBIDDEN_PREFIXES) or path.lower().endswith(FORBIDDEN_SUFFIXES)
    ]


def tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("Sin git o fuera de un repositorio")
    result = subprocess.run([git, "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


@pytest.mark.parametrize(
    "path",
    ["data/raw/listings.csv.gz", "data/processed/rooms.parquet", "models/model.onnx", "notas/modelo.pkl", "x.joblib"],
)
def test_detector_flags_forbidden_paths(path: str) -> None:
    assert forbidden_files([path]) == [path]


@pytest.mark.parametrize("path", ["data/manifests/censo2020.json", "reports/data_card.md", "src/thehouse_ml/cli.py"])
def test_detector_allows_versioned_paths(path: str) -> None:
    assert forbidden_files([path]) == []


def test_gitignore_excludes_data_and_models() -> None:
    lines = {line.strip() for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}

    assert {"data/raw/", "data/processed/", "models/"} <= lines


def test_no_data_or_model_files_are_tracked() -> None:
    assert forbidden_files(tracked_files()) == []
