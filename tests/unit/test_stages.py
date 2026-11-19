"""ML-01: catálogo de etapas del pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from thehouse_ml import stages
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import STAGE_NAMES, STAGES, get_stage, run_stage


def test_stage_names_are_unique() -> None:
    assert len(set(STAGE_NAMES)) == len(STAGES)


def test_every_stage_points_to_an_existing_spec_folder() -> None:
    specs = Path(__file__).resolve().parents[2] / "specs"
    folders = {path.name.split("-")[0] + "-" + path.name.split("-")[1] for path in specs.glob("ml-*")}

    for stage in STAGES:
        assert stage.spec.lower() in folders, stage


def test_get_stage_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        get_stage("deploy")


def test_run_stage_returns_the_pending_message(tmp_path: Path) -> None:
    assert run_stage("train", ProjectPaths(root=tmp_path)) == "train: pendiente de ML-04"


def test_prepare_generates_the_data_card_after_the_other_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake(name: str) -> object:
        def run(_paths: ProjectPaths) -> str:
            calls.append(name)
            return f"{name}: ok"

        return run

    for stage_name in ("run_prepare", "run_regional", "run_universities", "run_denue", "run_data_card"):
        monkeypatch.setattr(stages, stage_name, fake(stage_name))

    output = run_stage("prepare", ProjectPaths(root=tmp_path))

    assert calls == ["run_prepare", "run_regional", "run_universities", "run_denue", "run_data_card"]
    assert "run_data_card: ok" in output
