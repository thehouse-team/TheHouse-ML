"""ML-01 R01: la CLI lista las cinco etapas y cada una termina sin error."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from thehouse_ml import __version__
from thehouse_ml.cli import build_parser, main
from thehouse_ml.stages import STAGE_NAMES, STAGES, Stage


def test_help_lists_the_five_stages_in_pipeline_order(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    positions = [out.index(f"    {name}") for name in STAGE_NAMES]
    assert positions == sorted(positions)
    assert STAGE_NAMES == ("download", "prepare", "train", "evaluate", "export")


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.name)
def test_each_empty_stage_reports_its_spec_and_exits_zero(
    stage: Stage, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path), stage.name])

    assert code == 0
    assert capsys.readouterr().out.strip() == f"{stage.name}: pendiente de {stage.spec}"


def test_empty_stages_do_not_create_files(tmp_path: Path) -> None:
    for name in STAGE_NAMES:
        main(["--root", str(tmp_path), name])

    assert list(tmp_path.iterdir()) == []


def test_unknown_stage_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["entrenar-todo"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_missing_stage_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2


def test_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"thehouse_ml {__version__}"


def test_cli_text_is_printable_on_a_windows_cp1252_console() -> None:
    # Hallazgo de ML-01 (RK01): la consola de Windows usa cp1252; un carácter
    # fuera de esa página (por ejemplo «→») rompe `--help` con UnicodeEncodeError.
    help_text = build_parser().format_help()
    messages = [f"{stage.name}: pendiente de {stage.spec}" for stage in STAGES]

    for text in [help_text, *messages]:
        text.encode("cp1252")


def test_module_entry_point_runs_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["thehouse_ml", "export"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("thehouse_ml", run_name="__main__", alter_sys=True)

    assert exit_info.value.code == 0
    assert "export: pendiente de ML-06" in capsys.readouterr().out
