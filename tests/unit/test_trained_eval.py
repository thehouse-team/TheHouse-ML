"""ML-05 T03 (unitarias): verificación de modelos, métricas, comparación pareada y criterio de promoción."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval, StudentMetrics, bootstrap_means, student_metrics
from thehouse_ml.models.train import LIGHTGBM_MODEL, LOGREG_MODEL
from thehouse_ml.models.trained_eval import (
    DECISION_BASELINE,
    DECISION_CANDIDATE,
    OUTCOME_CANDIDATE,
    OUTCOME_NOT_PROMOTED,
    OUTCOME_REFERENCE,
    FoldSeries,
    TrainedEvalError,
    check_baseline_reference,
    cv_condition,
    fold_series,
    holdout_condition,
    paired_intervals,
    promotion,
    ranked_metrics,
    select_split,
    verify_models,
)
from thehouse_ml.models.training_data import TrainingData

# ---------------------------------------------------------------------------
# SHA-256 de los modelos
# ---------------------------------------------------------------------------


def _models_dir(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    (tmp_path / LOGREG_MODEL).write_bytes(b'{"format": "x"}\n')
    (tmp_path / LIGHTGBM_MODEL).write_bytes(b"tree\n")
    manifest = {
        "models": {
            name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in (LOGREG_MODEL, LIGHTGBM_MODEL)
        }
    }
    return tmp_path, manifest


def test_verify_models_accepts_the_files_of_the_manifest(tmp_path: Path) -> None:
    models_dir, manifest = _models_dir(tmp_path)

    digests = verify_models(models_dir, manifest)

    assert digests == {name: manifest["models"][name]["sha256"] for name in (LOGREG_MODEL, LIGHTGBM_MODEL)}


@pytest.mark.parametrize("name", [LOGREG_MODEL, LIGHTGBM_MODEL])
def test_verify_models_fails_if_a_model_changed(tmp_path: Path, name: str) -> None:
    models_dir, manifest = _models_dir(tmp_path)
    with (models_dir / name).open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(TrainedEvalError, match=f"{name}: SHA-256"):
        verify_models(models_dir, manifest)


def test_verify_models_fails_if_a_model_or_its_digest_is_missing(tmp_path: Path) -> None:
    models_dir, manifest = _models_dir(tmp_path)
    (models_dir / LIGHTGBM_MODEL).unlink()
    with pytest.raises(TrainedEvalError, match="falta el modelo"):
        verify_models(models_dir, manifest)

    del manifest["models"][LOGREG_MODEL]
    with pytest.raises(TrainedEvalError, match="falta el SHA-256"):
        verify_models(models_dir, manifest)


# ---------------------------------------------------------------------------
# Solo test y métricas iguales a ranking.py
# ---------------------------------------------------------------------------


def _training_data() -> TrainingData:
    rows = [
        # student, room, split, relevance, x
        ("S1", "R1", "train", 3, 0.1),
        ("S1", "R2", "train", 0, 0.2),
        ("S2", "R3", "test", 2, 0.3),
        ("S2", "R4", "test", 0, 0.4),
        ("S3", "R5", "validation", 1, 0.5),
        ("S3", "R6", "validation", 3, 0.6),
        ("S4", "R7", "test", 3, 0.7),
        ("S4", "R8", "test", 1, 0.8),
    ]
    frame = pd.DataFrame(rows, columns=["student_id", "room_id", "split", "relevance", "x"])
    return TrainingData(
        feature_order=("x",),
        schema_version="2.0.0",
        labels_params_version="1.0.0",
        features=frame[["x"]].astype("float64"),
        relevance=frame["relevance"].astype("int64"),
        group=frame["student_id"].astype("string"),
        room_id=frame["room_id"].astype("string"),
        split=frame["split"].astype("string"),
        students=pd.DataFrame(),
        excluded=(),
    )


def test_select_split_keeps_only_the_test_pairs() -> None:
    test = select_split(_training_data(), "test")

    assert test.students == ["S2", "S4"]
    assert test.room_id.tolist() == ["R3", "R4", "R7", "R8"]
    assert test.features.ravel().tolist() == [0.3, 0.4, 0.7, 0.8]
    assert test.relevance.tolist() == [2, 0, 3, 1]
    with pytest.raises(TrainedEvalError, match="no hay pares"):
        select_split(_training_data(), "holdout")


def test_ranked_metrics_order_by_score_then_room_id_and_use_ranking_py() -> None:
    # S1: R2 0.9 primero; R1 y R3 empatan en 0.5 y se desempatan por room_id; R4 al final.
    scores = np.array([0.5, 0.9, 0.5, 0.1, 2.0, 1.0])
    relevance = np.array([3, 0, 2, 1, 0, 3])
    group = np.array(["S1", "S1", "S1", "S1", "S0", "S0"])
    room_id = np.array(["R3", "R2", "R1", "R4", "R9", "R8"])

    result = ranked_metrics(scores, relevance, group, room_id, threshold=2)

    assert result == {
        "S0": student_metrics([0, 3], 2),
        "S1": student_metrics([0, 2, 3, 1], 2),  # R2, R1, R3, R4
    }


def test_ranked_metrics_rejects_non_finite_scores() -> None:
    with pytest.raises(TrainedEvalError, match="no finitos"):
        ranked_metrics(np.array([np.nan, 1.0]), np.array([1, 2]), np.array(["S", "S"]), np.array(["a", "b"]), 2)


# ---------------------------------------------------------------------------
# Comparación pareada (ejemplo a mano)
# ---------------------------------------------------------------------------


class FixedChoices(random.Random):
    """Generador que devuelve índices fijos en cada remuestreo (para el ejemplo a mano)."""

    def __init__(self, draws: Sequence[Sequence[int]]) -> None:
        super().__init__(0)
        self._draws = [list(draw) for draw in draws]

    def choices(self, population: Any, weights: Any = None, *, cum_weights: Any = None, k: int = 1) -> list[Any]:
        del population, weights, cum_weights
        draw = self._draws.pop(0)
        assert len(draw) == k
        return draw


def _metrics(ndcg10: float) -> StudentMetrics:
    return StudentMetrics(ndcg10, ndcg10 / 2, 0.2, ndcg10 / 4)


def test_paired_difference_matches_a_hand_computed_example() -> None:
    students = ["S1", "S2", "S3"]
    baseline = {"S1": _metrics(0.5), "S2": _metrics(0.6), "S3": _metrics(0.7)}
    model = {"S1": _metrics(0.6), "S2": _metrics(0.6), "S3": _metrics(0.9)}
    # Diferencias por estudiante: +0.1, 0.0, +0.2 → media 0.1.
    # Remuestreo 1: índices [0, 0, 1] → (0.1 + 0.1 + 0.0) / 3 = 0.0667; remuestreo 2: [2, 2, 2] → 0.2.
    # Percentil 2.5 % = 0.0667 + 0.025 * (0.2 - 0.0667) = 0.07; 97.5 % = 0.0667 + 0.975 * 0.1333 = 0.19667.
    result = paired_intervals(
        students,
        baseline,
        {"logreg": model},
        resamples=2,
        confidence=0.95,
        rng=FixedChoices([[0, 0, 1], [2, 2, 2]]),
    )

    diff = result.differences["logreg"]["ndcg@10"]
    assert diff.mean == pytest.approx(0.1)
    assert diff.low == pytest.approx(0.07)
    assert diff.high == pytest.approx(0.2 / 3 + 0.975 * (0.2 - 0.2 / 3))
    assert result.baseline["ndcg@10"].mean == pytest.approx(0.6)
    assert result.models["logreg"]["ndcg@10"].mean == pytest.approx(0.7)
    assert result.students == ("S1", "S2", "S3")


def _seeded() -> random.Random:
    return random.Random("bootstrap:1:total")  # noqa: S311 - bootstrap reproducible


def test_paired_intervals_use_the_same_resamples_as_the_baseline_bootstrap() -> None:
    generator = np.random.default_rng(7)
    students = [f"S{index:02d}" for index in range(40)]
    baseline = {student: _metrics(float(value)) for student, value in zip(students, generator.random(40), strict=True)}
    models = {
        "logreg": {
            student: _metrics(float(value)) for student, value in zip(students, generator.random(40), strict=True)
        }
    }

    result = paired_intervals(students, baseline, models, resamples=200, confidence=0.95, rng=_seeded())

    # El mismo generador sobre las series solas da los mismos intervalos (pareo por índices de estudiante).
    alone = bootstrap_means(
        {
            "baseline": [baseline[s].ndcg_10 for s in students],
            "diff": [models["logreg"][s].ndcg_10 - baseline[s].ndcg_10 for s in students],
        },
        resamples=200,
        confidence=0.95,
        rng=_seeded(),
    )
    assert result.baseline["ndcg@10"] == alone["baseline"]
    assert result.differences["logreg"]["ndcg@10"] == alone["diff"]
    assert set(result.differences["logreg"]) == set(METRIC_NAMES)


# ---------------------------------------------------------------------------
# per_student_baseline
# ---------------------------------------------------------------------------


def _reference(values: dict[str, float]) -> list[dict[str, Any]]:
    return [{"student_id": student, **_metrics(value).as_dict()} for student, value in values.items()]


def test_baseline_reference_must_have_the_same_students_and_metrics() -> None:
    recomputed = {"S1": _metrics(0.5), "S2": _metrics(0.6)}

    assert check_baseline_reference(_reference({"S2": 0.6, "S1": 0.5}), recomputed) == ["S2", "S1"]
    with pytest.raises(TrainedEvalError, match="no son los de per_student_baseline"):
        check_baseline_reference(_reference({"S1": 0.5, "S3": 0.6}), recomputed)
    with pytest.raises(TrainedEvalError, match="no reproduce per_student_baseline"):
        check_baseline_reference(_reference({"S1": 0.5, "S2": 0.61}), recomputed)


# ---------------------------------------------------------------------------
# Condiciones y decisión (spec §6.5)
# ---------------------------------------------------------------------------


def _series(name: str, values: Sequence[float]) -> FoldSeries:
    return FoldSeries(name, name, tuple(values), sum(values) / len(values), 0.0)


BASE = _series("baseline", [0.5, 0.5, 0.5, 0.5, 0.5])
ABOVE = [0.6, 0.6, 0.6, 0.6, 0.6]
ONE_TIE = [0.6, 0.6, 0.5, 0.6, 0.6]


def _interval(low: float) -> Interval:
    return Interval(mean=low + 0.01, low=low, high=low + 0.02)


def _differences(lr_low: float, gbm_low: float) -> dict[str, dict[str, Interval]]:
    return {
        "logreg": {metric: _interval(lr_low) for metric in METRIC_NAMES},
        "lightgbm": {metric: _interval(gbm_low) for metric in METRIC_NAMES},
    }


def test_cv_condition_needs_every_required_fold_strictly_above() -> None:
    assert cv_condition(_series("m", ABOVE), BASE, 5).passed is True
    assert cv_condition(_series("m", ONE_TIE), BASE, 5).passed is False  # empate en un pliegue no basta
    short = cv_condition(_series("m", ABOVE[:2]), _series("baseline", [0.5, 0.5]), 5)
    assert short.passed is False  # perfil ci: 2 pliegues no cumplen la condición de 5
    assert short.detail["folds_above_baseline"] == 2


@pytest.mark.parametrize(("low", "expected"), [(0.001, True), (0.0, False), (-0.01, False)])
def test_holdout_condition_needs_the_lower_bound_above_zero(low: float, expected: bool) -> None:
    assert holdout_condition(_interval(low)).passed is expected


def test_logreg_meeting_cv_and_test_is_a_candidate_subject_to_t05() -> None:
    folds = {"baseline": BASE, "logreg": _series("logreg", ABOVE), "lightgbm": _series("lightgbm", ABOVE)}

    candidates, decision = promotion(folds, _differences(0.02, 0.03), 5)

    assert decision.outcome == DECISION_CANDIDATE
    assert decision.promoted_model == "logreg"
    assert "sujeta a T05" in decision.text
    assert candidates["logreg"].outcome == OUTCOME_CANDIDATE
    assert candidates["logreg"].conditions[2].passed is None  # condición 3: se verifica en T05
    assert candidates["logreg"].as_dict()["conditions"]["3_additive_breakdown"]["status"] == "pending_t05"
    assert candidates["lightgbm"].meets_cv_and_test is True
    assert candidates["lightgbm"].outcome == OUTCOME_REFERENCE  # referencia aunque cumpla 1 y 2 (decisión 5)
    assert candidates["lightgbm"].conditions[2].passed is False


def test_only_lightgbm_meeting_the_conditions_keeps_the_baseline() -> None:
    folds = {"baseline": BASE, "logreg": _series("logreg", ONE_TIE), "lightgbm": _series("lightgbm", ABOVE)}

    candidates, decision = promotion(folds, _differences(0.02, 0.03), 5)

    assert decision.outcome == DECISION_BASELINE
    assert decision.promoted_model is None
    assert "se queda el baseline" in decision.text
    assert "no se promueve en v1" in decision.text
    assert candidates["logreg"].outcome == OUTCOME_NOT_PROMOTED
    assert candidates["lightgbm"].outcome == OUTCOME_REFERENCE


@pytest.mark.parametrize(
    ("logreg_folds", "lr_low"),
    [(ABOVE, -0.01), (ONE_TIE, 0.02), (ONE_TIE, -0.01)],
    ids=["fails-test", "fails-cv", "fails-both"],
)
def test_without_a_candidate_meeting_cv_and_test_the_baseline_stays(logreg_folds: list[float], lr_low: float) -> None:
    folds = {"baseline": BASE, "logreg": _series("logreg", logreg_folds), "lightgbm": _series("lightgbm", ONE_TIE)}

    candidates, decision = promotion(folds, _differences(lr_low, -0.01), 5)

    assert decision.outcome == DECISION_BASELINE
    assert decision.promoted_model is None
    assert "LightGBM" not in decision.text
    assert candidates["logreg"].outcome == OUTCOME_NOT_PROMOTED


# ---------------------------------------------------------------------------
# Tabla por pliegue de train_cv_metrics.json
# ---------------------------------------------------------------------------


def _cv_and_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    files = {LOGREG_MODEL: {"sha256": "a"}, LIGHTGBM_MODEL: {"sha256": "b"}}
    entry = {"ndcg@10_by_fold": [0.8, 0.9], "ndcg@10_mean": 0.85, "ndcg@10_std": 0.07}
    cv = {
        "baseline": {"model_version": "baseline-1.0.0", **entry},
        "selected": {"logreg": {"config_id": "lr", **entry}, "lightgbm": {"config_id": "gbm", **entry}},
        "final_models": {"files": files},
    }
    manifest = {"selected": {"logreg": "lr", "lightgbm": "gbm"}, "models": files}
    return cv, manifest


def test_fold_series_reads_the_selected_models_of_the_same_run() -> None:
    cv, manifest = _cv_and_manifest()

    series = fold_series(cv, manifest)

    assert series["logreg"].values == (0.8, 0.9)
    assert series["baseline"].config_id == "baseline-1.0.0"

    manifest["selected"]["logreg"] = "other"
    with pytest.raises(TrainedEvalError, match="no es la de"):
        fold_series(cv, manifest)
    cv, manifest = _cv_and_manifest()
    manifest["models"] = {LOGREG_MODEL: {"sha256": "z"}, LIGHTGBM_MODEL: {"sha256": "b"}}
    with pytest.raises(TrainedEvalError, match="misma corrida"):
        fold_series(cv, manifest)
