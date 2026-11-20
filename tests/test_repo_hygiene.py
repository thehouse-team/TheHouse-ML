"""ML-01 R03: datos y modelos nunca entran a git (constitution §6 y §7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("data/raw/", "data/processed/", "models/")
FORBIDDEN_SUFFIXES = (".pkl", ".pickle", ".joblib", ".onnx")


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

    # `/models/` anclado a la raíz: solo la carpeta de artefactos (M-ML-030, H1).
    assert {"data/raw/", "data/processed/", "/models/"} <= lines


def _git_ignored(paths: list[str]) -> set[str]:
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("Sin git o fuera de un repositorio")
    result = subprocess.run(
        [git, "check-ignore", "--no-index", *paths], cwd=REPO, capture_output=True, text=True, check=False
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines()}


@pytest.mark.parametrize(
    "path",
    ["models/model.onnx", "models/baseline/model.onnx", "data/raw/censo.zip", "data/processed/pairs.parquet"],
)
def test_artifacts_and_data_are_ignored_by_git(path: str) -> None:
    assert _git_ignored([path]) == {path}


@pytest.mark.parametrize(
    "path",
    [
        "src/thehouse_ml/models/baseline_params.json",
        "src/thehouse_ml/models/baseline_config.py",
        "src/thehouse_ml/models/__init__.py",
    ],
)
def test_the_models_package_of_the_code_is_not_ignored(path: str) -> None:
    assert _git_ignored([path]) == set()


def test_onnx_files_are_flagged_wherever_they_are_tracked() -> None:
    # Un .onnx versionado fuera de /models/ también es un artefacto prohibido (constitution §13: va en el release).
    assert forbidden_files(["src/thehouse_ml/models/model.onnx", "reports/model.onnx"]) == [
        "src/thehouse_ml/models/model.onnx",
        "reports/model.onnx",
    ]


def test_no_data_or_model_files_are_tracked() -> None:
    assert forbidden_files(tracked_files()) == []
