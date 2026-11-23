"""ML-04 T06 (integración): etapa ``evaluate`` del modelo base sobre pares y etiquetas de insumos versionados."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import versioned_synthetic_data
from thehouse_ml.cli import main
from thehouse_ml.evaluation import METRIC_NAMES, student_metrics
from thehouse_ml.features import default_schema
from thehouse_ml.models import default_config, rank, score_features
from thehouse_ml.models.baseline_report import (
    BASELINE_METRICS,
    BASELINE_REPORT,
    EVALUATE_SKIP_MESSAGE,
    EvaluationError,
    load_eval_params,
    run_evaluate,
)
from thehouse_ml.models.baseline_score import ScoredRoom
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.labels_stage import (
    HIDDEN_UTILITY_PARQUET,
    HIDDEN_WEIGHTS_PARQUET,
    LABELS_PARQUET,
    run_labels,
)
from thehouse_ml.synthetic.pairs_stage import PAIRS_PARQUET, run_pairs
from thehouse_ml.synthetic.rooms_stage import ROOMS_PARQUET, rooms_frame
from thehouse_ml.synthetic.students_stage import STUDENTS_PARQUET, students_frame

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}  # Un campus verificado y uno sin verificar.
PERSONAL_OR_SENSITIVE = ("gender", "email", "phone", "teléfono", "correo")  # el texto sí dice «no usa género»


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Proyecto con 400 estudiantes y 600 cuartos (2 campus), pares (ML-03 T06) y etiquetas (T04)."""
    students, rooms, _campuses = versioned_synthetic_data()
    schema = default_schema()
    paths = ProjectPaths(root=tmp_path_factory.mktemp("evaluate"))
    paths.data_processed.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "universities.csv", paths.universities_csv)
    chosen_students = [item for item in students if item.profile.university_code in CODES]
    chosen_rooms = [item for item in rooms if item.room.university_code in CODES]
    students_frame(chosen_students, schema).to_parquet(paths.data_processed / STUDENTS_PARQUET, index=False)
    rooms_frame(chosen_rooms, schema).to_parquet(paths.data_processed / ROOMS_PARQUET, index=False)
    run_pairs(paths)
    run_labels(paths)
    run_evaluate(paths)
    return paths


def _metrics(paths: ProjectPaths) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((paths.reports / BASELINE_METRICS).read_text(encoding="utf-8"))
    return data


def _digests(paths: ProjectPaths) -> dict[str, str]:
    files = [paths.reports / BASELINE_REPORT, paths.reports / BASELINE_METRICS]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def test_report_and_metrics_are_marked_synthetic_with_versions(project: ProjectPaths) -> None:
    report = (project.reports / BASELINE_REPORT).read_text(encoding="utf-8")
    title = report.splitlines()[0]
    assert "DATOS SINTÉTICOS" in title
    assert "baseline-1.0.0" in title
    assert "esquema 2.0.0" in title
    assert "no son preferencias reales" in report
    assert "ML-07" in report
    data = _metrics(project)
    assert data["synthetic"] is True
    assert "SINTÉTICAS" in data["warning"]
    assert data["model_version"] == "baseline-1.0.0"
    assert data["feature_schema_version"] == "2.0.0"
    assert data["labels_params_version"] == "1.0.0"
    assert data["primary_metric"] == "ndcg@10"
    assert data["bootstrap"]["resamples"] == 1000
    assert data["bootstrap"]["confidence"] == 0.95
    assert data["metrics"]["relevance_threshold"] == 2


def test_only_test_students_with_two_or_more_candidates_are_evaluated(project: ProjectPaths) -> None:
    pairs = pd.read_parquet(project.data_processed / PAIRS_PARQUET)
    test = pairs[pairs["split"] == "test"]
    counts = test.groupby("student_id").size()
    data = _metrics(project)
    evaluation = data["evaluation"]

    assert evaluation["split"] == "test"
    assert evaluation["students_in_split"] == len(counts)
    assert evaluation["excluded_fewer_than_min_candidates"] == int((counts < 2).sum())
    assert evaluation["students_evaluated"] == (
        len(counts) - evaluation["excluded_fewer_than_min_candidates"] - evaluation["excluded_no_relevant_room"]
    )
    evaluated = {row["student_id"] for row in data["per_student_baseline"]}
    assert evaluated <= set(counts.index)
    assert all(counts[student_id] >= 2 for student_id in evaluated)


def test_baseline_metrics_match_an_independent_recomputation(project: ProjectPaths) -> None:
    # Recalcula NDCG@10 etc. del baseline uniendo pares y etiquetas aquí, con
    # score_features + rank (T03) y las métricas puras, y compara con el JSON.
    pairs = pd.read_parquet(project.data_processed / PAIRS_PARQUET)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    merged = pairs.merge(labels[["student_id", "room_id", "relevance"]], on=["student_id", "room_id"])
    merged = merged[merged["split"] == "test"]
    schema, config = default_schema(), default_config()
    expected: dict[str, dict[str, float]] = {}
    for student_id, group in merged.groupby("student_id"):
        scored = [
            ScoredRoom(str(row.room_id), score_features([float(getattr(row, f)) for f in schema.feature_order]))
            for row in group.itertuples(index=False)
        ]
        relevance = dict(zip(group["room_id"].astype(str), group["relevance"].astype(int), strict=True))
        metrics = student_metrics([relevance[item.room_id] for item in rank(scored)], threshold=2)
        if metrics is not None and len(group) >= 2:
            expected[str(student_id)] = metrics.as_dict()
    del config
    data = _metrics(project)
    observed = {row["student_id"]: {m: row[m] for m in METRIC_NAMES} for row in data["per_student_baseline"]}
    assert observed.keys() == expected.keys()
    for student_id, values in expected.items():
        for metric, value in values.items():
            assert observed[student_id][metric] == pytest.approx(value, abs=1e-12)
    total = next(segment for segment in data["segments"] if segment["id"] == "total")
    for metric in METRIC_NAMES:
        mean = math.fsum(values[metric] for values in expected.values()) / len(expected)
        assert total["baseline"][metric]["mean"] == pytest.approx(mean, abs=1e-12)


def test_every_segment_has_intervals_and_the_baseline_beats_random(project: ProjectPaths) -> None:
    data = _metrics(project)
    ids = [segment["id"] for segment in data["segments"]]
    assert ids[0] == "total"
    assert {"campus:UTTT", "campus:UCAN", "without_seed_sin_verificar"} <= set(ids)
    assert [i for i in ids if i.startswith("budget:")] == [
        "budget:budget_le_2000",
        "budget:budget_2001_3500",
        "budget:budget_gt_3500",
    ]
    by_id = {segment["id"]: segment for segment in data["segments"]}
    total = by_id["total"]
    budget_students = sum(by_id[i]["students_in_split"] for i in ids if i.startswith("budget:"))
    assert budget_students == total["students_in_split"]
    assert by_id["without_seed_sin_verificar"]["students_in_split"] == by_id["campus:UTTT"]["students_in_split"]
    for segment in data["segments"]:
        if not segment["students_evaluated"]:
            continue
        for ranker in ("baseline", "random", "lift_baseline_minus_random"):
            for metric in METRIC_NAMES:
                interval = segment[ranker][metric]
                assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"], (segment["id"], ranker, metric)
        assert segment["ideal"]["ndcg@10"]["mean"] == pytest.approx(1.0, abs=1e-12)
        assert segment["ideal"]["map"]["mean"] == pytest.approx(1.0, abs=1e-12)
    for metric in METRIC_NAMES:
        assert total["baseline"][metric]["mean"] > total["random"][metric]["mean"]
        assert total["lift_baseline_minus_random"][metric]["ci_low"] > 0


def test_report_has_distribution_example_and_tables(project: ProjectPaths) -> None:
    report = (project.reports / BASELINE_REPORT).read_text(encoding="utf-8")
    for heading in (
        "## 2. Distribución de `matchScore`",
        "## 3. Ejemplo explicado: un estudiante con su top 5",
        "## 4. Métricas de ranking por segmento",
        "## 5. Comparación contra el azar y el techo",
    ):
        assert heading in report
    assert "```text" in report  # histograma
    assert '"matchScore"' in report  # salida §6.4 del ejemplo
    assert "Sin `seed_sin_verificar`" in report
    assert "`budget.max` ≤ $2,000" in report
    params = load_eval_params()
    assert f"semilla {params.bootstrap_seed}" in report
    assert f"semilla {params.random_seed}" in report


def test_outputs_have_no_personal_or_sensitive_fields(project: ProjectPaths) -> None:
    text = (project.reports / BASELINE_REPORT).read_text(encoding="utf-8").lower()
    text += (project.reports / BASELINE_METRICS).read_text(encoding="utf-8").lower()
    for word in PERSONAL_OR_SENSITIVE:
        assert word not in text, word
    data = _metrics(project)
    assert set(data["per_student_baseline"][0]) == {
        "student_id",
        "university_code",
        "seed_verified",
        "budget_max",
        "candidates",
        *METRIC_NAMES,
    }


def test_evaluate_never_reads_the_hidden_label_files(project: ProjectPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _digests(project)
    read: list[str] = []
    original = pd.read_parquet

    def spy(path: object, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read.append(Path(str(path)).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    hidden = [project.data_processed / HIDDEN_UTILITY_PARQUET, project.data_processed / HIDDEN_WEIGHTS_PARQUET]
    stash = {path: path.read_bytes() for path in hidden}
    for path in hidden:
        path.unlink()
    try:
        run_evaluate(project)
    finally:
        for path, content in stash.items():
            path.write_bytes(content)

    assert sorted(read) == sorted([PAIRS_PARQUET, LABELS_PARQUET, STUDENTS_PARQUET, ROOMS_PARQUET])
    assert HIDDEN_UTILITY_PARQUET not in read
    assert HIDDEN_WEIGHTS_PARQUET not in read
    # Sin los archivos ocultos el resultado es idéntico.
    assert _digests(project) == before
    source = (REPO / "src" / "thehouse_ml" / "models" / "baseline_report.py").read_text(encoding="utf-8")
    assert "HIDDEN_" not in source


def test_running_twice_does_not_change_any_file(project: ProjectPaths) -> None:
    first = _digests(project)
    run_evaluate(project)
    assert _digests(project) == first


def test_without_inputs_the_stage_is_skipped_and_writes_nothing(tmp_path: Path) -> None:
    assert run_evaluate(ProjectPaths(root=tmp_path)) == EVALUATE_SKIP_MESSAGE
    assert list(tmp_path.iterdir()) == []


def test_labels_that_do_not_match_the_pairs_are_an_error(project: ProjectPaths, tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    for name in (PAIRS_PARQUET, STUDENTS_PARQUET, ROOMS_PARQUET):
        shutil.copyfile(project.data_processed / name, paths.data_processed / name)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    labels.iloc[1:].to_parquet(paths.data_processed / LABELS_PARQUET, index=False)

    with pytest.raises(EvaluationError, match="sin etiqueta"):
        run_evaluate(paths)


def test_cli_runs_evaluate_and_reports_errors_with_exit_code_one(
    project: ProjectPaths, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(project.root), "evaluate"]) == 0
    assert "DATOS SINTÉTICOS" in capsys.readouterr().out

    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    for name in (PAIRS_PARQUET, STUDENTS_PARQUET, ROOMS_PARQUET):
        shutil.copyfile(project.data_processed / name, paths.data_processed / name)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    labels.assign(synthetic=False).to_parquet(paths.data_processed / LABELS_PARQUET, index=False)
    assert main(["--root", str(tmp_path), "evaluate"]) == 1
    assert "synthetic" in capsys.readouterr().err
