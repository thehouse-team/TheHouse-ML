"""ML-04 T04 (integración): etiquetas sintéticas sobre pares de insumos versionados, sin red."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest

from conftest import copy_project
from thehouse_ml.cli import main
from thehouse_ml.features import default_schema
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic import labels_stage
from thehouse_ml.synthetic.labels import load_params, relevance_by_position
from thehouse_ml.synthetic.labels_stage import (
    HIDDEN_UTILITY_PARQUET,
    HIDDEN_WEIGHTS_PARQUET,
    LABEL_COLUMNS,
    LABELS_PARQUET,
    LABELS_REPORT,
    LABELS_SKIP_MESSAGE,
    LABELS_TITLE,
    LabelsError,
    run_labels,
)
from thehouse_ml.synthetic.pairs_stage import PAIRS_PARQUET, run_pairs

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}  # Un campus verificado y uno sin verificar.
PERSONAL_OR_SENSITIVE = ("gender", "genero", "living", "name", "nombre", "email", "phone", "age", "edad")


@pytest.fixture(scope="module")
def project(two_campus_pairs_template: Path, tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Proyecto con 400 estudiantes y 600 cuartos (2 campus), pares de T06 y etiquetas de T04."""
    paths: ProjectPaths = copy_project(two_campus_pairs_template, tmp_path_factory.mktemp("labels") / "project")
    return paths


@pytest.fixture(scope="module")
def first_labels(project: ProjectPaths) -> str:
    """Una sola corrida de `labels` por módulo (T07, T05-H3); las pruebas que solo leen sus salidas la reutilizan."""
    return run_labels(project)


def _digests(paths: ProjectPaths) -> dict[str, str]:
    files = [
        paths.data_processed / PAIRS_PARQUET,
        paths.data_processed / LABELS_PARQUET,
        paths.data_processed / HIDDEN_UTILITY_PARQUET,
        paths.data_processed / HIDDEN_WEIGHTS_PARQUET,
        paths.reports / LABELS_REPORT,
    ]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def test_run_labels_writes_one_synthetic_label_per_candidate_pair(project: ProjectPaths, first_labels: str) -> None:
    message = first_labels

    assert message.startswith("labels: ")
    assert "de 400 estudiantes" in message
    pairs = pd.read_parquet(project.data_processed / PAIRS_PARQUET)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    assert list(labels.columns) == list(LABEL_COLUMNS)
    assert bool(labels["synthetic"].all())
    assert set(labels["labels_params_version"]) == {"1.0.0"}
    assert set(labels["relevance"]) <= {0, 1, 2, 3}
    # Cada par tiene exactamente una etiqueta y no hay etiquetas fuera de los candidatos.
    assert len(labels) == len(pairs)
    assert not labels.duplicated(["student_id", "room_id"]).any()
    merged = pairs.merge(labels, on=["student_id", "room_id"], how="outer", indicator=True, suffixes=("", "_label"))
    assert set(merged["_merge"]) == {"both"}
    assert (merged["split"] == merged["split_label"]).all()


def test_relevance_proportions_per_student_follow_the_cuts(project: ProjectPaths, first_labels: str) -> None:
    del first_labels
    params = load_params(default_schema())
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    utilities = pd.read_parquet(project.data_processed / HIDDEN_UTILITY_PARQUET)
    joined = labels.merge(utilities, on=["student_id", "room_id"])
    for _student_id, group in joined.groupby("student_id"):
        ranked = group.sort_values(["utility", "room_id"], ascending=[False, True], kind="mergesort")
        expected = relevance_by_position(len(group), params)
        assert ranked["relevance"].tolist() == expected
        assert int((group["relevance"] == 3).sum()) >= 1


def test_hidden_weights_are_per_student_and_sum_to_100(project: ProjectPaths, first_labels: str) -> None:
    del first_labels
    weights = pd.read_parquet(project.data_processed / HIDDEN_WEIGHTS_PARQUET)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    columns = [column for column in weights.columns if column.startswith("weight_")]
    assert len(columns) == 6
    assert weights[columns].sum(axis=1).sub(100.0).abs().max() < 1e-9
    assert weights["student_id"].tolist() == sorted(labels["student_id"].unique().tolist())
    assert weights.set_index("student_id")["candidates"].to_dict() == labels.groupby("student_id").size().to_dict()


def test_report_is_marked_synthetic_and_the_correlation_is_in_range(project: ProjectPaths, first_labels: str) -> None:
    del first_labels
    report = (project.reports / LABELS_REPORT).read_text(encoding="utf-8")
    assert report.startswith(LABELS_TITLE + "\n")
    assert LABELS_TITLE == "# DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES"
    assert "no son preferencias reales" in report
    assert "ML-07" in report
    assert "ML-04-RK02" in report
    assert "**cumple**" in report
    assert "| `UCAN` (sin verificar) |" in report
    assert "| `train` |" in report
    assert str(project.root) not in report
    pooled = labels_stage.spearman(
        labels_stage.baseline_totals(
            pd.read_parquet(project.data_processed / PAIRS_PARQUET), default_schema(), labels_stage.default_config()
        ),
        pd.read_parquet(project.data_processed / HIDDEN_UTILITY_PARQUET)["utility"].tolist(),
    )
    low, high = load_params(default_schema()).spearman_range
    assert low <= pooled <= high
    assert not math.isclose(pooled, 1.0)
    assert f"**{pooled:.3f}**" in report


def test_outputs_have_no_personal_or_gender_columns(project: ProjectPaths, first_labels: str) -> None:
    del first_labels
    for name in (LABELS_PARQUET, HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET):
        for column in pd.read_parquet(project.data_processed / name).columns:
            assert not set(column.lower().split("_")) & set(PERSONAL_OR_SENSITIVE), (name, column)


def test_running_twice_does_not_change_any_file(project: ProjectPaths, first_labels: str) -> None:
    del first_labels  # 1.ª corrida de `pairs` (plantilla) y de `labels` (fixture); aquí va la 2.ª
    first = _digests(project)
    run_pairs(project)
    run_labels(project)
    assert _digests(project) == first


def test_without_pairs_the_stage_is_skipped(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    assert run_labels(paths) == LABELS_SKIP_MESSAGE
    assert not (paths.reports / LABELS_REPORT).exists()


def test_correlation_out_of_range_is_an_error_and_the_report_says_so(
    project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    strict = load_params(default_schema()).replace("check.spearman_range", [0.95, 1.0])
    monkeypatch.setattr(labels_stage, "load_params", lambda _schema: strict)

    with pytest.raises(LabelsError, match=r"fuera de \[0.95, 1\]"):
        run_labels(project)

    report = (project.reports / LABELS_REPORT).read_text(encoding="utf-8")
    assert "**NO CUMPLE**" in report


def test_missing_feature_columns_are_an_error(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    pd.DataFrame({"student_id": ["S0001"], "room_id": ["R00001"], "split": ["train"]}).to_parquet(
        paths.data_processed / PAIRS_PARQUET, index=False
    )
    with pytest.raises(LabelsError, match="faltan las columnas"):
        run_labels(paths)


def test_cli_reports_label_errors_with_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    pd.DataFrame({"student_id": ["S0001"], "room_id": ["R00001"], "split": ["train"]}).to_parquet(
        paths.data_processed / PAIRS_PARQUET, index=False
    )

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
        "run_pairs",
        "run_data_card",
    ):
        monkeypatch.setattr(stages, name, no_op)

    assert main(["--root", str(tmp_path), "prepare"]) == 1
    assert "faltan las columnas" in capsys.readouterr().err
