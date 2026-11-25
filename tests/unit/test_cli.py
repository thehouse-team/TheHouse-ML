"""ML-01 R01: la CLI lista las cinco etapas y cada una termina sin error."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from thehouse_ml import __version__
from thehouse_ml.cli import build_parser, main
from thehouse_ml.models.baseline_report import EVALUATE_SKIP_MESSAGE
from thehouse_ml.models.train import TRAIN_SKIP_MESSAGE
from thehouse_ml.paths import ROOT_ENV_VAR
from thehouse_ml.stages import STAGE_NAMES

# ML-06 T01 implementa `export`: ya no quedan etapas vacías (ML-02, ML-04, ML-05 y ML-06).
EXPORT_SKIP_MESSAGE = "export: omitido; falta models/logreg.json (ejecute `train`)"


def test_help_lists_the_five_stages_in_pipeline_order(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    positions = [out.index(f"    {name}") for name in STAGE_NAMES]
    assert positions == sorted(positions)
    assert STAGE_NAMES == ("download", "prepare", "train", "evaluate", "export")


def test_export_without_a_model_is_skipped_and_creates_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path), "export"])

    assert code == 0
    assert capsys.readouterr().out.strip() == EXPORT_SKIP_MESSAGE
    assert list(tmp_path.iterdir()) == []


def test_evaluate_without_inputs_is_skipped_and_creates_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path), "evaluate"])

    assert code == 0
    assert capsys.readouterr().out.strip() == EVALUATE_SKIP_MESSAGE
    assert list(tmp_path.iterdir()) == []
    EVALUATE_SKIP_MESSAGE.encode("cp1252")


def test_train_without_inputs_is_skipped_and_creates_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path), "train"])

    assert code == 0
    assert capsys.readouterr().out.strip() == TRAIN_SKIP_MESSAGE
    assert list(tmp_path.iterdir()) == []
    TRAIN_SKIP_MESSAGE.encode("cp1252")


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
    messages = [EXPORT_SKIP_MESSAGE, "download_url pendiente"]

    for text in [help_text, *messages]:
        text.encode("cp1252")


def test_download_accepts_a_source_filter() -> None:
    args = build_parser().parse_args(["download", "--source", "inegi_envi_2020"])

    assert args.stage == "download"
    assert args.source == "inegi_envi_2020"


def test_download_source_filter_is_optional() -> None:
    args = build_parser().parse_args(["download"])

    assert args.source is None


def test_module_entry_point_runs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Raíz temporal: `export` ya no es una etapa vacía y no debe escribir en el repositorio.
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["thehouse_ml", "export"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("thehouse_ml", run_name="__main__", alter_sys=True)

    assert exit_info.value.code == 0
    assert EXPORT_SKIP_MESSAGE in capsys.readouterr().out
