"""ML-03 T04 (integración): etapa de estudiantes sintéticos con una calibración inventada y sin red."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.cli import main
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import MUNICIPIOS
from thehouse_ml.synthetic.students_stage import (
    STUDENTS_PARQUET,
    STUDENTS_REPORT,
    STUDENTS_SKIP_MESSAGE,
    StudentsError,
    quartiles,
    run_students,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
# Renta p50 inventada por entidad (pesos); el resto de columnas sigue el esquema de ML-02 T04.
INVENTED_P50 = {"13": 2000.0, "15": 3000.0}


def _write_project(tmp_path: Path, *, skip: str | None = None) -> ProjectPaths:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "universities.csv", tmp_path / "data" / "universities.csv")
    rows = [
        {
            "municipio_clave": municipality.clave,
            "municipio_nombre": municipality.nombre,
            "entidad": municipality.entidad,
            "indicador": indicator,
            "level": "estado" if indicator == "renta" else "municipio",
            "p50": INVENTED_P50[municipality.entidad] if indicator == "renta" else float("nan"),
        }
        for municipality in MUNICIPIOS
        for indicator in ("internet", "renta")
        if not (indicator == "renta" and municipality.clave == skip)
    ]
    pd.DataFrame(rows).to_parquet(paths.data_processed / "regional_calibration.parquet", index=False)
    return paths


def test_run_students_writes_the_parquet_and_the_report(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)

    message = run_students(paths)

    assert message.startswith("students: 2000 estudiantes sintéticos")
    frame = pd.read_parquet(paths.data_processed / STUDENTS_PARQUET)
    assert len(frame) == 2000
    assert bool(frame["synthetic"].all())
    assert frame.groupby("university_code").size().eq(200).all()
    assert set(frame.loc[~frame["seed_verified"], "university_code"]) == {"TESJI", "UMB-VDC", "UCAN"}
    report = (paths.reports / STUDENTS_REPORT).read_text(encoding="utf-8")
    assert report.startswith("# DATOS SINTÉTICOS")
    for heading in ("Calibración del presupuesto", "Transporte", "livingWith", "placeType", "Servicios", "Nulos"):
        assert heading in report
    assert "| UTTT | Tula de Allende | estado | $2,000 | $2,600 |" in report
    assert "| TESJI | Jilotepec | estado | $3,000 | $3,900 |" in report
    assert "**no**" in report  # campus sin verificar marcados en la tabla de presupuesto
    assert "| no |" not in report.split("## 3.")[0]  # toda la calibración dentro de la tolerancia
    assert str(tmp_path) not in report


def test_running_twice_does_not_change_any_file(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    run_students(paths)
    report = (paths.reports / STUDENTS_REPORT).read_bytes()
    parquet = (paths.data_processed / STUDENTS_PARQUET).read_bytes()

    run_students(paths)

    assert (paths.reports / STUDENTS_REPORT).read_bytes() == report
    assert (paths.data_processed / STUDENTS_PARQUET).read_bytes() == parquet


def test_without_regional_calibration_the_stage_is_skipped(tmp_path: Path) -> None:
    assert run_students(ProjectPaths(root=tmp_path)) == STUDENTS_SKIP_MESSAGE
    assert not (tmp_path / "reports").exists()


def test_missing_rent_for_a_campus_municipality_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path, skip="15112")

    with pytest.raises(StudentsError, match="UMB-VDC"):
        run_students(paths)


def test_missing_campus_in_universities_csv_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    lines = paths.universities_csv.read_text(encoding="utf-8").splitlines()
    paths.universities_csv.write_text(
        "\n".join(line for line in lines if not line.startswith("UCAN,")) + "\n", encoding="utf-8"
    )

    with pytest.raises(StudentsError, match="UCAN"):
        run_students(paths)


def test_cli_reports_students_errors_with_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise StudentsError("sin renta")

    monkeypatch.setattr("thehouse_ml.cli.run_stage", fail)

    assert main(["--root", str(tmp_path), "prepare"]) == 1
    assert "sin renta" in capsys.readouterr().err


def test_quartiles_of_a_single_value() -> None:
    assert quartiles([2500.0]) == (2500.0, 2500.0, 2500.0)
