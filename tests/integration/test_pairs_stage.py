"""ML-03 T06 (integración): etapa del conjunto de pares con estudiantes y cuartos de insumos versionados, sin red."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import versioned_synthetic_data
from thehouse_ml.cli import main
from thehouse_ml.features import default_schema
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.pairs_stage import (
    PAIRS_PARQUET,
    PAIRS_REPORT,
    PAIRS_SKIP_MESSAGE,
    PairsError,
    run_pairs,
)
from thehouse_ml.synthetic.rooms import SyntheticRoom
from thehouse_ml.synthetic.rooms_stage import ROOMS_PARQUET, rooms_frame
from thehouse_ml.synthetic.students_stage import STUDENTS_PARQUET, students_frame

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
# Un campus verificado y uno sin verificar: la etapa completa con 400 estudiantes y 600 cuartos.
CODES = {"UTTT", "UCAN"}


def _write_project(tmp_path: Path, *, unaffordable: bool = False) -> ProjectPaths:
    students, rooms, _campuses = versioned_synthetic_data()
    schema = default_schema()
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "universities.csv", paths.universities_csv)
    chosen_students = [item for item in students if item.profile.university_code in CODES]
    chosen_rooms = [item for item in rooms if item.room.university_code in CODES]
    if unaffordable:
        chosen_rooms = [
            SyntheticRoom(
                item.room_id, item.seed_verified, item.campus_distance_km, replace(item.room, price_mxn=50000)
            )
            for item in chosen_rooms
        ]
    students_frame(chosen_students, schema).to_parquet(paths.data_processed / STUDENTS_PARQUET, index=False)
    rooms_frame(chosen_rooms, schema).to_parquet(paths.data_processed / ROOMS_PARQUET, index=False)
    return paths


def test_run_pairs_writes_the_parquet_and_the_report(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)

    message = run_pairs(paths)

    assert message.startswith("pairs: ")
    assert "de 400 estudiantes" in message
    frame = pd.read_parquet(paths.data_processed / PAIRS_PARQUET)
    schema = default_schema()
    assert list(frame.columns) == [
        "student_id",
        "room_id",
        "university_code",
        "seed_verified",
        "split",
        *schema.feature_order,
        "synthetic",
    ]
    assert bool(frame["synthetic"].all())
    # Esquema 2.0.0 (ML-04 T02): rangos de las 3 features nuevas; distance_ratio dentro de su training_range.
    assert frame["distance_ratio"].between(0.0, 60.0).all()
    assert frame["price_below_min"].between(0.0, 1.0).all()
    assert frame["extra_services_ratio"].between(0.0, 1.0).all()
    assert set(frame["university_code"]) == CODES
    assert frame.groupby("student_id")["split"].nunique().eq(1).all()  # sin fuga
    for _code, group in frame.groupby("university_code"):
        assert set(group["split"]) == {"train", "validation", "test"}
    assert frame.groupby("student_id").size().max() <= 200
    assert set(frame.loc[~frame["seed_verified"], "university_code"]) == {"UCAN"}
    report = (paths.reports / PAIRS_REPORT).read_text(encoding="utf-8")
    assert report.startswith("# DATOS SINTÉTICOS")
    assert "**cumple**" in report
    assert "| UTTT | sí | 140 | 30 | 30 |" in report
    assert "| **Total sin `seed_sin_verificar`** | — | 140 | 30 | 30 |" in report
    assert str(tmp_path) not in report


def test_running_twice_does_not_change_any_file(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)

    def digests() -> dict[str, str]:
        files = [paths.reports / PAIRS_REPORT, paths.data_processed / PAIRS_PARQUET]
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    run_pairs(paths)
    first = digests()
    run_pairs(paths)
    assert digests() == first


def test_without_inputs_the_stage_is_skipped(tmp_path: Path) -> None:
    assert run_pairs(ProjectPaths(root=tmp_path)) == PAIRS_SKIP_MESSAGE
    paths = _write_project(tmp_path)
    (paths.data_processed / ROOMS_PARQUET).unlink()
    assert run_pairs(paths) == PAIRS_SKIP_MESSAGE
    assert not (paths.reports / PAIRS_REPORT).exists()


def test_threshold_not_met_is_an_error_and_the_report_says_so(tmp_path: Path) -> None:
    paths = _write_project(tmp_path, unaffordable=True)

    with pytest.raises(PairsError, match=r"UTTT 0.0 %"):
        run_pairs(paths)

    report = (paths.reports / PAIRS_REPORT).read_text(encoding="utf-8")
    assert "**NO CUMPLE**" in report
    assert "| UTTT | sí | 200 | 0 | 0.0 % | **no** |" in report
    assert len(pd.read_parquet(paths.data_processed / PAIRS_PARQUET)) == 0


def test_cli_reports_pairs_errors_with_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_project(tmp_path, unaffordable=True)

    def no_op(_paths: ProjectPaths) -> str:
        return "ok"

    from thehouse_ml import stages

    for name in (
        "run_prepare",
        "run_regional",
        "run_universities",
        "run_denue",
        "run_students",
        "run_rooms",
        "run_data_card",
    ):
        monkeypatch.setattr(stages, name, no_op)

    assert main(["--root", str(tmp_path), "prepare"]) == 1
    assert "umbral de M-ML-021" in capsys.readouterr().err
