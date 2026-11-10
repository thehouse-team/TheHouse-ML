"""ML-01: las rutas del proyecto son configurables (constitution §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from thehouse_ml.paths import ROOT_ENV_VAR, ProjectPaths


def test_explicit_root_wins_over_environment(tmp_path: Path) -> None:
    other = tmp_path / "otra"

    paths = ProjectPaths.resolve(tmp_path, environ={ROOT_ENV_VAR: str(other)})

    assert paths.root == tmp_path.resolve()


def test_environment_variable_sets_the_root(tmp_path: Path) -> None:
    paths = ProjectPaths.resolve(environ={ROOT_ENV_VAR: str(tmp_path)})

    assert paths.root == tmp_path.resolve()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_environment_variable_falls_back_to_current_directory(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    paths = ProjectPaths.resolve(environ={ROOT_ENV_VAR: value})

    assert paths.root == tmp_path.resolve()


def test_uses_process_environment_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))

    assert ProjectPaths.resolve().root == tmp_path.resolve()


def test_layout_matches_the_constitution(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    assert paths.data_raw == tmp_path / "data" / "raw"
    assert paths.data_processed == tmp_path / "data" / "processed"
    assert paths.data_manifests == tmp_path / "data" / "manifests"
    assert paths.models == tmp_path / "models"
    assert paths.reports == tmp_path / "reports"
