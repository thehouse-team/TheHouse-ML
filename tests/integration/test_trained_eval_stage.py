"""ML-05 T03 (integración): medición única en test de los modelos de `train` y decisión de promoción.

Proyecto temporal con 2 campus, `prepare` parcial (pares y etiquetas), `train` con una rejilla
pequeña y `evaluate` del baseline; después, la etapa de T03. Sin red. La corrida con los datos de
ML-03/ML-04 queda en la evidencia.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pytest

from conftest import copy_project, small_train_params
from thehouse_ml.evaluation.ranking import METRIC_NAMES, student_metrics
from thehouse_ml.features import default_schema
from thehouse_ml.models.baseline_report import BASELINE_METRICS, SCORE_PERCENTILES
from thehouse_ml.models.model_card import MODEL_CARD_SKIP_MESSAGE
from thehouse_ml.models.train import LIGHTGBM_MODEL, LOGREG_MODEL, LogRegModel
from thehouse_ml.models.train_config import PROFILE_ENV_VAR, TrainParams, load_train_params
from thehouse_ml.models.trained_breakdown_report import HISTOGRAM_BINS, TRAINED_BREAKDOWN
from thehouse_ml.models.trained_eval import (
    DECISION_BASELINE,
    DECISION_PROMOTED,
    TRAINED_METRICS,
    TRAINED_REPORT,
    TRAINED_SKIP_MESSAGE,
    TrainedEvalError,
    run_trained_evaluate,
)
from thehouse_ml.models.training_data import TRAINING_INPUTS, load_training_data
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage
from thehouse_ml.synthetic.labels_stage import HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}
ML05_EVAL_MODULES = tuple(
    REPO / "src" / "thehouse_ml" / "models" / name
    for name in ("trained_eval.py", "trained_report.py", "trained_breakdown.py", "trained_breakdown_report.py")
)
OUTPUTS = (f"reports/{TRAINED_METRICS}", f"reports/{TRAINED_REPORT}", f"reports/{TRAINED_BREAKDOWN}")


def small_params() -> TrainParams:
    params: TrainParams = small_train_params()  # la rejilla de la plantilla de sesión (conftest)
    return params


@pytest.fixture(scope="module")
def project(two_campus_trained_template: Path, tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Proyecto con pares, etiquetas (con archivos ocultos), modelos de `train` y métricas del baseline."""
    paths: ProjectPaths = copy_project(two_campus_trained_template, tmp_path_factory.mktemp("trained_eval") / "project")
    return paths


def _snapshot(paths: ProjectPaths, names: tuple[str, ...] = OUTPUTS) -> dict[str, str]:
    return {name: hashlib.sha256((paths.root / name).read_bytes()).hexdigest() for name in names}


@pytest.fixture(scope="module")
def first_run(project: ProjectPaths) -> dict[str, str]:
    message = run_trained_evaluate(project, small_params(), "full")
    assert message.startswith("evaluate (ML-05, perfil full): ")
    assert "DATOS SINTÉTICOS" in message
    message.encode("cp1252")  # la CLI lo imprime en una consola de Windows (T03-H1)
    TRAINED_SKIP_MESSAGE.encode("cp1252")
    return _snapshot(project)


def _metrics(paths: ProjectPaths, directory: str = "reports") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((paths.root / directory / TRAINED_METRICS).read_text(encoding="utf-8"))
    return data


def test_only_the_test_students_of_per_student_baseline_are_evaluated(
    project: ProjectPaths, first_run: dict[str, str]
) -> None:
    del first_run
    data = load_training_data(project)
    metrics = _metrics(project)
    reference = json.loads((project.reports / BASELINE_METRICS).read_text(encoding="utf-8"))
    test_students = set(data.students.loc[data.students["split"] == "test", "student_id"])

    per_student = [row["student_id"] for row in metrics["per_student_test"]]
    assert set(per_student) == test_students
    assert per_student == [row["student_id"] for row in reference["per_student_baseline"]]
    assert metrics["evaluation"]["students_evaluated"] == len(test_students)
    assert metrics["evaluation"]["pairs"] == int((data.split == "test").sum())
    # Mismo bootstrap que el total del baseline: su intervalo se reproduce exactamente.
    total = next(item for item in reference["segments"] if item["id"] == "total")
    assert metrics["test"]["baseline"] == total["baseline"]


def test_only_test_pairs_are_scored(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del first_run
    test_pairs = int((load_training_data(project).split == "test").sum())
    scored: list[tuple[str, int]] = []
    logreg_original = LogRegModel.decision_function
    predict_original = lightgbm.Booster.predict

    def spy_logreg(self: LogRegModel, features: np.ndarray) -> np.ndarray:
        scored.append(("logreg", len(features)))
        return logreg_original(self, features)

    def spy_predict(self: lightgbm.Booster, data: Any, *args: Any, **kwargs: Any) -> Any:
        scored.append(("lightgbm", len(data)))
        return predict_original(self, data, *args, **kwargs)

    monkeypatch.setattr(LogRegModel, "decision_function", spy_logreg)
    monkeypatch.setattr(lightgbm.Booster, "predict", spy_predict)

    run_trained_evaluate(project, small_params(), "full")

    # Los modelos oficiales (T03) puntúan solo test, una vez cada uno.
    assert scored[:2] == [("logreg", test_pairs), ("lightgbm", test_pairs)]
    # T04: los modelos de cada pliegue puntúan solo a sus retenidos de train + validación, una vez por pasada
    # (modelo completo + 6 grupos de la ablación); ningún par de test.
    cv_pairs = int((load_training_data(project).split != "test").sum())
    for name in ("logreg", "lightgbm"):
        assert sum(rows for model, rows in scored[2:] if model == name) == cv_pairs * 7


def test_metrics_match_an_independent_recomputation_with_ranking_py(
    project: ProjectPaths, first_run: dict[str, str]
) -> None:
    del first_run
    data = load_training_data(project)
    mask = (data.split == "test").to_numpy()
    features = data.features[mask].to_numpy()
    logreg = LogRegModel.from_dict(json.loads((project.models / LOGREG_MODEL).read_text(encoding="utf-8")))
    booster = lightgbm.Booster(model_file=str(project.models / LIGHTGBM_MODEL))
    frame = pd.DataFrame(
        {
            "student": data.group[mask].to_numpy(),
            "room": data.room_id[mask].to_numpy(),
            "relevance": data.relevance[mask].to_numpy(),
            "logreg": logreg.decision_function(features),
            "lightgbm": booster.predict(features, num_threads=1),
        }
    )
    metrics = _metrics(project)
    observed = {row["student_id"]: row for row in metrics["per_student_test"]}
    for name in ("logreg", "lightgbm"):
        for student, rows in frame.groupby("student"):
            ordered = sorted(rows.itertuples(index=False), key=lambda row: (-getattr(row, name), row.room))
            expected = student_metrics([int(row.relevance) for row in ordered], 2)
            assert expected is not None
            assert observed[str(student)][name] == expected.as_dict()
        means = {m: float(np.mean([observed[s][name][m] for s in observed])) for m in METRIC_NAMES}
        for metric in METRIC_NAMES:
            assert metrics["test"][name][metric]["mean"] == pytest.approx(means[metric], abs=1e-12)
            low = metrics["paired_difference_vs_baseline"][name][metric]["ci_low"]
            high = metrics["paired_difference_vs_baseline"][name][metric]["ci_high"]
            assert low <= metrics["paired_difference_vs_baseline"][name][metric]["mean"] <= high


def test_outputs_are_synthetic_and_record_the_single_measurement_and_decision(
    project: ProjectPaths, first_run: dict[str, str]
) -> None:
    del first_run
    metrics = _metrics(project)
    report = (project.reports / TRAINED_REPORT).read_text(encoding="utf-8")

    assert metrics["synthetic"] is True
    assert metrics["profile"] == "full"
    assert "una sola vez" in metrics["single_test_measurement"]
    assert "ML-07" in metrics["warning"]
    assert metrics["promotion"]["required_folds"] == small_params().n_splits
    assert metrics["promotion"]["decision"]["outcome"] in (DECISION_PROMOTED, DECISION_BASELINE)
    assert metrics["promotion"]["candidates"]["lightgbm"]["outcome"] == "reference_not_promoted_in_v1"
    assert len(metrics["cross_validation"]["baseline"]["ndcg@10_by_fold"]) == small_params().n_splits
    for name in ("logreg", "lightgbm"):
        digest = hashlib.sha256((project.models / metrics["models"][name]["file"]).read_bytes()).hexdigest()
        assert metrics["models"][name]["sha256"] == digest
    for text in ("DATOS SINTÉTICOS", "una sola vez", "ML-07", "Criterio de promoción", "**Decisión:**", "T04"):
        assert text in report, text
    assert set(metrics["per_student_test"][0]) == {"student_id", "logreg", "lightgbm"}  # sin datos personales


def test_distribution_and_examples_are_in_the_json_and_match_the_markdown(
    project: ProjectPaths, first_run: dict[str, str]
) -> None:
    """ML-05 T07 (T06-H1): `breakdown.distribution` y `breakdown.examples` de `trained_metrics.json`."""
    del first_run
    metrics = _metrics(project)
    markdown = (project.reports / TRAINED_BREAKDOWN).read_text(encoding="utf-8")
    distribution = metrics["breakdown"]["distribution"]
    score = distribution["matchScore"]

    assert distribution["pairs"] == metrics["evaluation"]["pairs"] == metrics["breakdown"]["sums"]["pairs"]
    assert [(item["low"], item["high"]) for item in score["histogram"]] == list(HISTOGRAM_BINS)
    assert sum(item["count"] for item in score["histogram"]) == distribution["pairs"]
    assert list(score["percentiles"]) == [f"p{round(q * 100)}" for q in SCORE_PERCENTILES]
    assert score["min"] <= min(score["percentiles"].values()) <= max(score["percentiles"].values()) <= score["max"]
    row = " | ".join([str(score["min"]), *(f"{value:g}" for value in score["percentiles"].values()), str(score["max"])])
    assert f"| {row} | {score['mean']:.1f} |" in markdown  # el Markdown se renderiza desde el JSON
    for item in score["histogram"]:
        assert f"{item['count']:>7,} (" in markdown
    factors = metrics["breakdown"]["calibration"]["factors"]
    assert [item["factor"] for item in distribution["factor_points"]] == [item["factor"] for item in factors]
    for item in distribution["factor_points"]:
        assert 0 <= item["p25"] <= item["p50"] <= item["p75"] <= item["raw_max_points"]
        assert item["mean_share_of_max"] == pytest.approx(item["mean"] / item["raw_max_points"])

    examples = metrics["breakdown"]["examples"]
    assert examples, "la regla fija debe encontrar al menos un ejemplo en el proyecto temporal"
    test_students = {row["student_id"] for row in metrics["per_student_test"]}
    for example in examples:
        assert example["student_id"] in test_students
        assert f"`{example['student_id']}` — {example['university_code']}" in markdown
        top = example["top"]
        assert [item["rank"] for item in top] == list(range(1, len(top) + 1))
        assert [item["matchScore"] for item in top] == sorted((item["matchScore"] for item in top), reverse=True)
        first = example["first_place"]
        assert first["room_id"] == top[0]["room_id"]
        assert first["output"]["matchScore"] == top[0]["matchScore"]
        assert [item["points"] for item in first["output"]["breakdown"]] == list(top[0]["points"].values())
        assert abs(first["raw_total"] - first["output"]["matchScore"]) <= 0.5
        assert set(first["inputs"]) == set(default_schema().feature_order)
    assert json.dumps(examples[0]["first_place"]["output"], ensure_ascii=False, indent=2) in markdown


def test_running_twice_does_not_change_any_file(project: ProjectPaths, first_run: dict[str, str]) -> None:
    run_trained_evaluate(project, small_params(), "full")

    assert _snapshot(project) == first_run


def test_evaluate_never_reads_the_hidden_label_files(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
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
        run_trained_evaluate(project, small_params(), "full")
    finally:
        for path, content in stash.items():
            path.write_bytes(content)

    assert sorted(read) == sorted(TRAINING_INPUTS)
    assert _snapshot(project) == first_run
    for module in ML05_EVAL_MODULES:
        source = module.read_text(encoding="utf-8")
        for forbidden in ("HIDDEN_", "labels_hidden", HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET):
            assert forbidden not in source, (module.name, forbidden)


@pytest.fixture
def copy(project: ProjectPaths, first_run: dict[str, str], tmp_path: Path) -> ProjectPaths:
    del first_run
    target = tmp_path / "copy"
    shutil.copytree(project.root, target)
    return ProjectPaths(root=target)


@pytest.mark.parametrize("name", [LOGREG_MODEL, LIGHTGBM_MODEL])
def test_a_model_that_does_not_match_its_sha256_is_an_error(copy: ProjectPaths, name: str) -> None:
    before = _snapshot(copy)
    with (copy.models / name).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(TrainedEvalError, match="SHA-256"):
        run_trained_evaluate(copy, small_params(), "full")
    assert _snapshot(copy) == before  # no se escribió nada


def test_inputs_different_from_the_training_ones_are_an_error(copy: ProjectPaths) -> None:
    pairs = copy.data_processed / "pairs.parquet"
    pd.read_parquet(pairs).iloc[:-1].to_parquet(pairs, index=False)

    with pytest.raises(TrainedEvalError, match="otros datos"):
        run_trained_evaluate(copy, small_params(), "full")


def test_without_models_the_stage_is_skipped(copy: ProjectPaths) -> None:
    (copy.models / LIGHTGBM_MODEL).unlink()
    before = _snapshot(copy)

    assert run_trained_evaluate(copy, small_params(), "full") == TRAINED_SKIP_MESSAGE
    assert _snapshot(copy) == before


def test_evaluate_stage_runs_the_baseline_and_then_the_trained_models(
    copy: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    (copy.reports / TRAINED_METRICS).unlink()

    message = run_stage("evaluate", copy)  # train_params.json oficial: la condición 1 exige 5 pliegues

    baseline, trained, ablation, breakdown, card = message.splitlines()
    assert card == MODEL_CARD_SKIP_MESSAGE  # ML-05 T06: el proyecto temporal no tiene uv.lock
    assert baseline.startswith("evaluate: baseline-1.0.0")
    assert trained.startswith("evaluate (ML-05, perfil full): ")
    assert ablation.startswith("evaluate (ML-05 T04, perfil full): ")
    assert breakdown.startswith("evaluate (ML-05 T05, perfil full): desglose 0-100 de logreg-1.0.0")
    message.encode("cp1252")  # la CLI lo imprime en una consola de Windows
    metrics = _metrics(copy)
    assert metrics["promotion"]["required_folds"] == 5
    assert metrics["promotion"]["decision"]["outcome"] == DECISION_BASELINE  # 3 pliegues no cumplen la condición 1


def test_ci_profile_does_not_modify_the_versioned_reports(copy: ProjectPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROFILE_ENV_VAR, "ci")
    versioned = (*OUTPUTS, f"reports/{BASELINE_METRICS}")
    run_stage("train", copy)  # el CI corre train y evaluate con la variable de entorno
    before = _snapshot(copy, versioned)

    message = run_stage("evaluate", copy)

    assert "evaluate (ML-05, perfil ci): " in message
    assert _snapshot(copy, versioned) == before  # reports/trained_* intactos; baseline determinista
    ci_dir = load_train_params().ci_profile.output_dir
    metrics = _metrics(copy, ci_dir)
    assert metrics["profile"] == "ci"
    assert metrics["official"] is False
    assert metrics["cross_validation"]["n_splits"] == 2
    assert metrics["promotion"]["candidates"]["logreg"]["conditions"]["1_cv_all_folds_above_baseline"]["passed"] is (
        False
    )
    assert "no oficial" in (copy.root / ci_dir / TRAINED_REPORT).read_text(encoding="utf-8")
