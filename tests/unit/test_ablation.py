"""ML-05 T04 (unitarias): grupos de features, pliegues, segmentos e hipótesis del segmento > $3,500."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval, StudentMetrics
from thehouse_ml.features import default_schema
from thehouse_ml.models.ablation import (
    CI_ABLATION_MODELS,
    VERDICT_CONFIRMED,
    VERDICT_INCONCLUSIVE,
    VERDICT_REFUTED,
    AblationResult,
    Segment,
    ablated_models_for,
    build_segments,
    check_segments,
    cv_from_manifest,
    cv_scores,
    hypothesis,
    segment_results,
    selected_configs,
    summarize_folds,
)
from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_report import load_eval_params
from thehouse_ml.models.factor_groups import (
    FACTOR_GROUPS,
    OUTSIDE_GROUPS,
    FactorGroup,
    FactorGroupError,
    check_groups,
    kept_columns,
)
from thehouse_ml.models.trained_eval import TrainedEvalError
from thehouse_ml.models.training_data import TrainingData

FEATURE_ORDER = tuple(default_schema().feature_order)

# ---------------------------------------------------------------------------
# Grupos de features
# ---------------------------------------------------------------------------


def test_groups_cover_feature_order_once_and_match_the_baseline_factors() -> None:
    check_groups(FEATURE_ORDER)

    assigned = [name for group in FACTOR_GROUPS for name in group.features]
    assert sorted(assigned + list(OUTSIDE_GROUPS)) == sorted(FEATURE_ORDER)
    assert len(assigned) == len(set(assigned)) == len(FEATURE_ORDER) - 1
    assert list(OUTSIDE_GROUPS) == ["pets_compatible"]
    assert [group.id for group in FACTOR_GROUPS] == [factor.id for factor in default_config().factors]


@pytest.mark.parametrize("group", FACTOR_GROUPS, ids=[group.id for group in FACTOR_GROUPS])
def test_each_group_removes_exactly_its_features(group: FactorGroup) -> None:
    kept = [FEATURE_ORDER[i] for i in kept_columns(FEATURE_ORDER, group.features)]

    assert kept == [name for name in FEATURE_ORDER if name not in group.features]  # mismo orden
    assert set(FEATURE_ORDER) - set(kept) == set(group.features)
    assert "pets_compatible" in kept


def test_invalid_groups_are_rejected() -> None:
    with pytest.raises(FactorGroupError, match="repetidas"):
        check_groups(FEATURE_ORDER, [*FACTOR_GROUPS, FactorGroup("x", "X", ("distance_km",))])
    with pytest.raises(FactorGroupError, match="sin grupo"):
        check_groups(FEATURE_ORDER, FACTOR_GROUPS[1:])
    with pytest.raises(FactorGroupError, match="ajenas"):
        check_groups(FEATURE_ORDER, [*FACTOR_GROUPS, FactorGroup("x", "X", ("utility",))])
    with pytest.raises(FactorGroupError, match="fuera de feature_order"):
        kept_columns(FEATURE_ORDER, ["utility"])


# ---------------------------------------------------------------------------
# Pliegues
# ---------------------------------------------------------------------------


def _data() -> TrainingData:
    rows = [
        ("S1", "R1", "train", 3),
        ("S1", "R2", "train", 0),
        ("S2", "R3", "validation", 2),
        ("S2", "R4", "validation", 0),
        ("S3", "R5", "test", 3),
        ("S3", "R6", "test", 1),
        ("S4", "R7", "train", 1),
        ("S4", "R8", "train", 3),
    ]
    frame = pd.DataFrame(rows, columns=["student_id", "room_id", "split", "relevance"])
    frame["x"] = np.arange(len(frame), dtype=np.float64)
    students = frame.groupby("student_id", as_index=False).agg(split=("split", "first"))
    return TrainingData(
        feature_order=("x",),
        schema_version="2.0.0",
        labels_params_version="1.0.0",
        features=frame[["x"]],
        relevance=frame["relevance"].astype("int64"),
        group=frame["student_id"].astype("string"),
        room_id=frame["room_id"].astype("string"),
        split=frame["split"].astype("string"),
        students=students,
        excluded=(),
    )


def test_cv_from_manifest_uses_the_manifest_folds_and_never_test() -> None:
    cv = cv_from_manifest(_data(), {"S1": 0, "S2": 1, "S4": 1}, ["train", "validation"])

    assert sorted(set(cv.group.tolist())) == ["S1", "S2", "S4"]
    assert cv.fold.tolist() == [0, 0, 1, 1, 1, 1]
    assert cv.room_id.tolist() == ["R1", "R2", "R3", "R4", "R7", "R8"]
    with pytest.raises(TrainedEvalError, match="no cubre"):
        cv_from_manifest(_data(), {"S1": 0, "S2": 1}, ["train", "validation"])
    with pytest.raises(TrainedEvalError, match="test nunca"):
        cv_from_manifest(_data(), {"S1": 0, "S2": 1, "S3": 0, "S4": 1}, ["train", "validation", "test"])


def test_cv_scores_fit_on_other_folds_and_score_each_held_out_pair_once() -> None:
    cv = cv_from_manifest(_data(), {"S1": 0, "S2": 1, "S4": 1}, ["train", "validation"])
    calls: list[tuple[list[str], list[str], tuple[int, ...]]] = []

    def fit_score(fit: np.ndarray, held: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
        calls.append((sorted(set(cv.group[fit].tolist())), sorted(set(cv.group[held].tolist())), columns))
        return cv.features[held][:, 0] * 10

    scores = cv_scores(cv, 2, fit_score, (0,))

    assert calls == [(["S2", "S4"], ["S1"], (0,)), (["S1"], ["S2", "S4"], (0,))]
    assert scores.tolist() == (cv.features[:, 0] * 10).tolist()


def test_selected_configs_come_from_train_cv_metrics() -> None:
    cv_metrics: dict[str, Any] = {
        "selected": {
            "logreg": {"config_id": "logreg:standard:C=0.01", "config": {"transform": "standard", "C": 0.01}},
            "lightgbm": {
                "config_id": "lightgbm:leaves=15:lr=0.05:trees=300",
                "config": {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300},
            },
        }
    }
    logreg, lightgbm = selected_configs(cv_metrics)
    assert (logreg.transform, logreg.c) == ("standard", 0.01)
    assert (lightgbm.num_leaves, lightgbm.n_estimators) == (15, 300)

    cv_metrics["selected"]["logreg"]["config_id"] = "logreg:standard:C=1"
    with pytest.raises(TrainedEvalError, match="distinta de config_id"):
        selected_configs(cv_metrics)


def test_summarize_folds_averages_each_metric_by_fold_and_counts_undefined() -> None:
    per_student: dict[str, StudentMetrics | None] = {
        "S1": StudentMetrics(0.2, 0.4, 0.6, 0.8),
        "S2": StudentMetrics(0.4, 0.6, 0.8, 1.0),
        "S3": StudentMetrics(1.0, 1.0, 1.0, 1.0),
        "S4": None,
    }

    summary = summarize_folds("m", per_student, {"S1": 0, "S2": 0, "S3": 1, "S4": 1}, 2)

    assert summary.values("ndcg@10") == [pytest.approx(0.3), 1.0]
    assert summary.values("map") == [pytest.approx(0.9), 1.0]
    assert summary.students == (2, 1)
    assert summary.undefined == (0, 1)
    assert summary.mean("ndcg@10") == pytest.approx(0.65)
    assert summary.std("ndcg@10") == pytest.approx(math.sqrt(2 * 0.35**2))


def test_ci_profile_only_ablates_the_logistic_regression() -> None:
    assert ablated_models_for("full") == ("logreg", "lightgbm")
    assert ablated_models_for("ci") == CI_ABLATION_MODELS == ("logreg",)


# ---------------------------------------------------------------------------
# Hipótesis > $3,500 (casos a mano)
# ---------------------------------------------------------------------------

BUDGET = FACTOR_GROUPS[0]


def _ablation(deltas: list[float]) -> AblationResult:
    full = (0.8,) * len(deltas)
    return AblationResult("logreg", BUDGET, (), tuple(0.8 + delta for delta in deltas), full)


def _deltas(low_band: float, mid_band: float, high: Interval) -> dict[str, Interval]:
    return {
        "budget:budget_le_2000": Interval(low_band, low_band - 0.05, low_band + 0.05),
        "budget:budget_2001_3500": Interval(mid_band, mid_band - 0.05, mid_band + 0.05),
        "budget:budget_gt_3500": high,
    }


TARGET = "budget:budget_gt_3500"


def test_hypothesis_confirmed_when_the_three_conditions_hold() -> None:
    result = hypothesis(_deltas(0.30, 0.33, Interval(0.40, 0.35, 0.45)), TARGET, _ablation([-0.1, -0.2, -0.1]))

    assert result.verdict == VERDICT_CONFIRMED
    assert (result.largest_improvement, result.interval_excludes_zero, result.budget_ablation_reduces) == (
        True,
        True,
        True,
    )
    assert len(result.reasons) == 3


@pytest.mark.parametrize(
    ("high", "ablation", "why"),
    [
        (Interval(0.32, 0.27, 0.37), [-0.1, -0.1], "not-largest"),  # 0.32 < 0.33 de $2,001-$3,500
        (Interval(0.40, 0.35, 0.45), [0.01, -0.005], "ablation-does-not-reduce"),  # media +0.0025
        (Interval(0.40, 0.35, 0.45), [0.0, 0.0], "ablation-unchanged"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_hypothesis_refuted_when_a_point_condition_fails(high: Interval, ablation: list[float], why: str) -> None:
    result = hypothesis(_deltas(0.30, 0.33, high), TARGET, _ablation(ablation))

    assert result.verdict == VERDICT_REFUTED, why


def test_hypothesis_refuted_when_the_interval_is_below_zero() -> None:
    result = hypothesis(_deltas(-0.30, -0.20, Interval(-0.10, -0.15, -0.05)), TARGET, _ablation([-0.1, -0.1]))

    assert result.largest_improvement is True
    assert result.interval_excludes_zero is False
    assert result.verdict == VERDICT_REFUTED


def test_hypothesis_inconclusive_when_the_interval_includes_zero() -> None:
    result = hypothesis(_deltas(0.01, 0.02, Interval(0.05, -0.01, 0.11)), TARGET, _ablation([-0.1, -0.1]))

    assert result.interval_excludes_zero is None
    assert result.verdict == VERDICT_INCONCLUSIVE
    assert result.as_dict()["conditions"]["2_ci_excludes_zero_above"] is None


def test_hypothesis_needs_the_budget_ranges() -> None:
    with pytest.raises(TrainedEvalError, match="faltan los rangos"):
        hypothesis({TARGET: Interval(0.1, 0.0, 0.2)}, TARGET, _ablation([-0.1]))


# ---------------------------------------------------------------------------
# Segmentos
# ---------------------------------------------------------------------------


def _reference() -> list[dict[str, Any]]:
    generator = np.random.default_rng(3)
    campuses = list(default_schema().catalog("university_code"))[:3]
    budgets = [1800, 3000, 4500]
    rows: list[dict[str, Any]] = []
    for index in range(60):
        values = generator.random(4)
        rows.append(
            {
                "student_id": f"S{index:03d}",
                "university_code": campuses[index % 3],
                "seed_verified": index % 3 != 2,
                "budget_max": budgets[(index // 3) % 3],
                **dict(zip(METRIC_NAMES, (float(value) for value in values), strict=True)),
            }
        )
    return rows


def _metrics(rows: list[dict[str, Any]], shift: float) -> dict[str, StudentMetrics]:
    return {
        row["student_id"]: StudentMetrics(*(min(1.0, row[metric] + shift) for metric in METRIC_NAMES)) for row in rows
    }


def test_segments_follow_the_baseline_ids_and_aggregate_to_the_total() -> None:
    rows = _reference()
    params = load_eval_params()
    segments = build_segments(rows, default_schema(), params)
    ids = [segment.id for segment in segments]
    assert ids[0] == "total"
    assert [i for i in ids if i.startswith("budget:")] == [f"budget:{band.id}" for band in params.budget_ranges]
    assert "without_seed_sin_verificar" in ids

    baseline = _metrics(rows, 0.0)
    results = segment_results(
        segments, baseline, {"logreg": _metrics(rows, 0.1), "lightgbm": _metrics(rows, 0.2)}, params
    )
    reference = {
        "segments": [
            {
                "id": item.segment.id,
                "students_evaluated": len(item.segment.students),
                "baseline": {metric: item.paired.baseline[metric].as_dict() for metric in METRIC_NAMES},
            }
            for item in results
            if item is not None
        ]
    }

    check_segments(results, reference)  # baseline igual a la referencia; campus y presupuesto se agregan al total

    broken = [dict(entry) for entry in reference["segments"]]
    broken[1] = {**broken[1], "students_evaluated": broken[1]["students_evaluated"] + 1}
    with pytest.raises(TrainedEvalError, match="número de estudiantes"):
        check_segments(results, {"segments": broken})


def test_segment_check_detects_partitions_that_do_not_cover_the_total() -> None:
    rows = _reference()
    params = load_eval_params()
    segments = [s for s in build_segments(rows, default_schema(), params) if s.id != "budget:budget_le_2000"]
    results = segment_results(segments, _metrics(rows, 0.0), {"logreg": _metrics(rows, 0.1)}, params)
    reference = {
        "segments": [
            {
                "id": item.segment.id,
                "students_evaluated": len(item.segment.students),
                "baseline": {metric: item.paired.baseline[metric].as_dict() for metric in METRIC_NAMES},
            }
            for item in results
            if item is not None
        ]
    }

    with pytest.raises(TrainedEvalError, match="no cubren"):
        check_segments(results, reference)


def test_empty_segments_have_no_result() -> None:
    rows = _reference()
    results = segment_results([Segment("x", "budget", "x", ())], _metrics(rows, 0.0), {}, load_eval_params())
    assert results == [None]
