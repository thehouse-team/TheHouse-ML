"""ML-05 T02 (ML-05-R02): pliegues sin fuga, búsqueda solo con pliegues, modelos sin pickle y reproducibles.

Datos construidos en la prueba (sin red ni archivos de data/): 3 campus, estudiantes de
train, validación y test, y una relevancia que baja con el precio y la distancia.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from thehouse_ml.evaluation.ranking import ndcg_at_k
from thehouse_ml.features import default_schema
from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_score import ScoredRoom, rank, score_features
from thehouse_ml.models.train import (
    FoldScore,
    GridResult,
    LogRegModel,
    TrainError,
    assign_folds,
    build_cv_data,
    clip_bounds,
    cross_validate,
    fit_lightgbm,
    fit_logreg,
    fold_counts,
    group_sizes,
    lightgbm_scores,
    logreg_pipeline,
    monotone_violations,
    ndcg_by_student,
    select_best,
    train_models,
)
from thehouse_ml.models.train_config import LightGBMConfig, LogRegConfig, TrainParams, load_train_params
from thehouse_ml.models.training_data import TrainingData

SCHEMA = default_schema()
ORDER = tuple(SCHEMA.feature_order)
CAMPUSES = ("AAA", "BBB", "CCC")
DECREASING = ("price_to_budget", "distance_km", "effective_distance_km", "distance_ratio")


def small_params() -> TrainParams:
    """Rejilla pequeña para pruebas rápidas (la completa queda en la evidencia)."""
    return load_train_params().with_values(
        n_splits=3,
        logreg_c=(0.1, 1.0),
        num_leaves=(4,),
        learning_rate=(0.1,),
        n_estimators=(5, 10),
        min_data_in_leaf=5,
    )


def toy_data(students_per_split: dict[str, int] | None = None, rooms: int = 12, seed: int = 7) -> TrainingData:
    counts = students_per_split or {"train": 36, "validation": 12, "test": 12}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    number = 0
    for split, total in counts.items():
        for _ in range(total):
            student = f"S{number:04d}"
            campus = CAMPUSES[number % len(CAMPUSES)]
            number += 1
            utilities = []
            for room in range(rooms):
                distance = float(rng.uniform(0.2, 12.0))
                price = float(rng.uniform(0.2, 1.0))
                values = {
                    "distance_km": distance,
                    "effective_distance_km": distance * float(rng.uniform(1.0, 1.4)),
                    "price_to_budget": price,
                    "services_coverage": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                    "place_type_match": float(rng.integers(0, 2)),
                    "property_type_match": float(rng.integers(0, 2)),
                    "pets_compatible": 1.0,
                    "distance_ratio": distance / 5.0,
                    "price_below_min": 0.0,
                    "extra_services_ratio": float(rng.uniform(0.0, 1.0)),
                }
                utility = 1.0 - price - distance / 12.0 + 0.3 * values["services_coverage"] + rng.normal(0, 0.1)
                utilities.append(utility)
                rows.append({"student_id": student, "room_id": f"R{room:03d}", "split": split, **values})
            cut = np.quantile(utilities, [0.6, 0.8, 0.9])
            for offset, utility in enumerate(utilities):
                rows[-rooms + offset]["relevance"] = int(np.searchsorted(cut, utility, side="right"))
            meta.append(
                {
                    "student_id": student,
                    "split": split,
                    "university_code": campus,
                    "seed_verified": True,
                    "budget_max": 3000,
                    "candidates": rooms,
                }
            )
    frame = pd.DataFrame(rows)
    return TrainingData(
        feature_order=ORDER,
        schema_version=SCHEMA.version,
        labels_params_version="1.0.0",
        features=frame[list(ORDER)].astype("float64"),
        relevance=frame["relevance"].astype("int64"),
        group=frame["student_id"].astype("string"),
        room_id=frame["room_id"].astype("string"),
        split=frame["split"].astype("string"),
        students=pd.DataFrame(meta),
        excluded=(),
    )


def _with_test_rows_changed(data: TrainingData) -> TrainingData:
    """Mismos datos con las features y la relevancia de **test** alteradas."""
    test = (data.split == "test").to_numpy()
    features = data.features.copy()
    features.loc[test, "price_to_budget"] = 0.05
    relevance = data.relevance.copy()
    relevance[test] = 3 - relevance[test]
    return replace(data, features=features, relevance=relevance)


# ---------------------------------------------------------------------------
# Pliegues
# ---------------------------------------------------------------------------


def test_folds_have_no_shared_students_and_never_include_test() -> None:
    data = toy_data()
    params = small_params()

    folds = assign_folds(data.students, params)

    test_students = set(data.students.loc[data.students["split"] == "test", "student_id"])
    cv_students = set(data.students.loc[data.students["split"] != "test", "student_id"])
    assert set(folds.index) == cv_students  # cada estudiante de train + validación en exactamente un pliegue
    assert folds.index.is_unique
    assert not set(folds.index) & test_students
    assert set(folds.tolist()) == set(range(params.n_splits))


def test_folds_are_stratified_by_campus_and_reproducible() -> None:
    data = toy_data()
    params = small_params()

    first = assign_folds(data.students, params)
    second = assign_folds(data.students, params)
    other = assign_folds(data.students, params.with_values(seed=params.seed + 1))

    pd.testing.assert_series_equal(first, second)
    assert not first.equals(other)
    campus = data.students.set_index("student_id")["university_code"]
    table = pd.crosstab(first, campus.loc[first.index])
    assert (table.max(axis=0) - table.min(axis=0)).max() <= 1  # 16 por campus en 3 pliegues: 5 o 6


def test_cv_data_drops_every_test_pair_and_keeps_pairs_contiguous() -> None:
    data = toy_data()
    params = small_params()

    cv = build_cv_data(data, params)

    test_students = set(data.students.loc[data.students["split"] == "test", "student_id"])
    assert not set(cv.group.tolist()) & test_students
    assert cv.n_pairs == int((data.split != "test").sum())
    assert sum(group_sizes(cv.group)) == cv.n_pairs
    counts = fold_counts(cv)
    assert counts["students"].sum() == len(cv.students) == 48
    assert counts["pairs"].sum() == cv.n_pairs
    for fold in range(params.n_splits):  # el pliegue de cada par es el de su estudiante
        held = set(cv.group[cv.fold == fold].tolist())
        assert held == set(cv.students.loc[cv.students["fold"] == fold, "student_id"])


def test_group_sizes_requires_contiguous_students() -> None:
    assert group_sizes(np.array(["a", "a", "b", "c", "c", "c"])) == [2, 1, 3]
    assert group_sizes(np.array([], dtype=str)) == []
    with pytest.raises(TrainError, match="contiguos"):
        group_sizes(np.array(["a", "b", "a"]))


def test_assign_folds_requires_cv_students() -> None:
    data = toy_data({"test": 6})

    with pytest.raises(TrainError, match="no hay estudiantes"):
        assign_folds(data.students, small_params())


# ---------------------------------------------------------------------------
# Búsqueda solo con los pliegues
# ---------------------------------------------------------------------------


def test_cross_validation_fits_on_other_folds_and_scores_only_held_out_students() -> None:
    data = toy_data()
    params = small_params()
    cv = build_cv_data(data, params)
    calls: list[tuple[set[str], set[str]]] = []

    def fit_and_score(fit: np.ndarray, held: np.ndarray) -> np.ndarray:
        calls.append((set(cv.group[fit].tolist()), set(cv.group[held].tolist())))
        return -cv.features[held][:, ORDER.index("price_to_budget")]

    scores = cross_validate(cv, params.n_splits, fit_and_score)

    assert len(scores) == params.n_splits
    seen: set[str] = set()
    for fold, (fit_students, held_students) in enumerate(calls):
        assert not fit_students & held_students  # sin fuga dentro del pliegue
        assert fit_students | held_students == set(cv.group.tolist())  # solo train + validación
        assert held_students == set(cv.students.loc[cv.students["fold"] == fold, "student_id"])
        assert scores[fold].students == len(held_students)
        seen |= held_students
    assert seen == set(cv.students["student_id"])


def test_cross_validation_rejects_a_wrong_number_of_scores() -> None:
    cv = build_cv_data(toy_data(), small_params())

    with pytest.raises(TrainError, match="puntajes"):
        cross_validate(cv, 3, lambda fit, held: np.zeros(3))


def test_training_ignores_the_test_partition_entirely() -> None:
    """Si cambian las features y la relevancia de test, pliegues, rejillas, baseline y modelos no cambian."""
    data = toy_data()
    params = small_params()

    original = train_models(data, params, 2, SCHEMA, default_config())
    changed = train_models(_with_test_rows_changed(data), params, 2, SCHEMA, default_config())

    assert original.logreg_results == changed.logreg_results
    assert original.lightgbm_results == changed.lightgbm_results
    assert original.baseline_folds == changed.baseline_folds
    assert original.logreg_model == changed.logreg_model
    assert original.lightgbm_booster.model_to_string() == changed.lightgbm_booster.model_to_string()


def test_ndcg_by_student_breaks_ties_by_room_id_like_the_baseline_rank() -> None:
    data = toy_data({"train": 8})
    cv = build_cv_data(data, small_params().with_values(n_splits=2))
    config = default_config()
    results = [score_features([float(value) for value in row], config, SCHEMA) for row in cv.features]
    raw = np.array([item.raw_total for item in results])

    generic = ndcg_by_student(raw, cv.relevance, cv.group, cv.room_id)

    for student in sorted(set(cv.group.tolist())):
        rows = np.flatnonzero(cv.group == student)
        ordered = rank(ScoredRoom(str(cv.room_id[i]), results[i]) for i in rows)
        relevance = {str(cv.room_id[i]): int(cv.relevance[i]) for i in rows}
        assert generic[student] == ndcg_at_k([relevance[item.room_id] for item in ordered], 10)
    ties = ndcg_by_student(np.zeros(3), np.array([0, 3, 0]), np.array(["s"] * 3), np.array(["R2", "R1", "R0"]))
    assert ties["s"] == pytest.approx(1 / np.log2(3))  # empate total: orden R0, R1, R2; el 3 queda en la posición 2


def test_select_best_uses_mean_then_std_then_grid_order() -> None:
    def result(name: str, values: list[float]) -> GridResult:
        return GridResult(name, {}, tuple(FoldScore(i, 10, 0, value) for i, value in enumerate(values)))

    assert select_best([result("a", [0.5, 0.5]), result("b", [0.6, 0.6])]).config_id == "b"
    assert select_best([result("a", [0.4, 0.6]), result("b", [0.5, 0.5])]).config_id == "b"  # menor desviación
    assert select_best([result("a", [0.5, 0.5]), result("b", [0.5, 0.5])]).config_id == "a"  # primera
    with pytest.raises(TrainError, match="vacía"):
        select_best([])


# ---------------------------------------------------------------------------
# Regresión logística: pipeline exportable y JSON sin pickle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["standard", "quantile_bins"])
def test_logreg_rebuilt_from_json_gives_the_same_predictions(transform: str) -> None:
    cv = build_cv_data(toy_data(), small_params())
    params = small_params()
    bounds = clip_bounds(SCHEMA)
    target = (cv.relevance >= 2).astype(np.int64)

    model, pipeline, _warnings = fit_logreg(cv.features, target, LogRegConfig(transform, 1.0), params, bounds, ORDER)
    rebuilt = LogRegModel.from_dict(json.loads(json.dumps(model.to_dict())))

    expected = pipeline.decision_function(np.clip(cv.features, *bounds))
    np.testing.assert_array_equal(rebuilt.decision_function(cv.features), expected)
    assert rebuilt == model


def test_logreg_clips_each_feature_to_its_training_range() -> None:
    cv = build_cv_data(toy_data(), small_params())
    bounds = clip_bounds(SCHEMA)
    model, _pipeline, _ = fit_logreg(
        cv.features, (cv.relevance >= 2).astype(np.int64), LogRegConfig("standard", 1.0), small_params(), bounds, ORDER
    )
    lower, upper = bounds
    assert upper[ORDER.index("distance_km")] == 30.0
    assert upper[ORDER.index("distance_ratio")] == 60.0
    assert upper[ORDER.index("price_to_budget")] == 1.0
    assert upper[ORDER.index("services_coverage")] == 1.0  # sin training_range: su range

    far = cv.features[:5].copy()
    far[:, ORDER.index("distance_km")] = 500.0
    limit = cv.features[:5].copy()
    limit[:, ORDER.index("distance_km")] = 30.0
    np.testing.assert_array_equal(model.decision_function(far), model.decision_function(limit))
    assert (lower == 0.0).all()


def test_logreg_pipeline_uses_only_transformers_with_skl2onnx_converters() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

    allowed = (StandardScaler, KBinsDiscretizer, LogisticRegression)
    for config in load_train_params().logreg_grid():
        pipeline = logreg_pipeline(config, load_train_params())
        assert all(isinstance(step, allowed) for _name, step in pipeline.steps)
        assert isinstance(pipeline.steps[-1][1], LogisticRegression)


def test_logreg_json_rejects_unknown_formats() -> None:
    cv = build_cv_data(toy_data(), small_params())
    model, _pipeline, _ = fit_logreg(
        cv.features,
        (cv.relevance >= 2).astype(np.int64),
        LogRegConfig("standard", 1.0),
        small_params(),
        clip_bounds(SCHEMA),
        ORDER,
    )
    document = model.to_dict()

    with pytest.raises(TrainError, match="formato"):
        LogRegModel.from_dict({**document, "format": "pickle"})
    with pytest.raises(TrainError, match=r"transform.type"):
        LogRegModel.from_dict({**document, "transform": {**document["transform"], "type": "quantile_bins"}})


# ---------------------------------------------------------------------------
# LightGBM: restricciones monotónicas efectivas
# ---------------------------------------------------------------------------


def _monotone_case(constrained: tuple[str, ...]) -> dict[str, int]:
    """Datos en los que la relevancia **sube** con el precio (tienta a violar la restricción)."""
    data = toy_data({"train": 60})
    cv = build_cv_data(data, small_params().with_values(n_splits=2))
    relevance = np.digitize(cv.features[:, ORDER.index("price_to_budget")], [0.5, 0.7, 0.9]).astype(np.int64)
    params = small_params().with_values(monotone_decreasing=constrained, min_data_in_leaf=3)
    booster = fit_lightgbm(cv.features, relevance, cv.group, LightGBMConfig(8, 0.2, 30), params, ORDER)
    sample = cv.features[::7]
    return monotone_violations(
        lambda matrix: lightgbm_scores(booster, matrix), sample, ORDER, DECREASING, clip_bounds(SCHEMA)
    )


def test_lightgbm_monotone_constraints_are_effective() -> None:
    with_constraints = _monotone_case(DECREASING)
    without_constraints = _monotone_case(())

    assert with_constraints == dict.fromkeys(DECREASING, 0)  # al subir cada feature restringida, el puntaje no sube
    assert without_constraints["price_to_budget"] > 0  # la verificación sí detecta una violación


def test_final_models_are_reproducible_and_monotone() -> None:
    data = toy_data()
    params = small_params()

    first = train_models(data, params, 2, SCHEMA, default_config())
    second = train_models(data, params, 2, SCHEMA, default_config())

    assert json.dumps(first.logreg_model.to_dict()) == json.dumps(second.logreg_model.to_dict())
    assert first.lightgbm_booster.model_to_string() == second.lightgbm_booster.model_to_string()
    assert first.logreg_results == second.logreg_results
    assert first.lightgbm_results == second.lightgbm_results
    assert first.monotone_check == dict.fromkeys(params.monotone_decreasing, 0)
    assert len(first.logreg_results) == 4  # 2 transformaciones por 2 valores de C
    assert len(first.lightgbm_results) == 2
    assert first.logreg_selected in first.logreg_results
    assert all(len(item.folds) == params.n_splits for item in first.logreg_results)
    assert first.cv.n_pairs == 48 * 12  # modelo final: todo train + validación, nada de test


def test_train_models_rejects_a_feature_order_different_from_the_schema() -> None:
    data = toy_data()
    shuffled = replace(data, feature_order=tuple(reversed(ORDER)))

    with pytest.raises(TrainError, match="feature_order"):
        train_models(shuffled, small_params(), 2, SCHEMA, default_config())
