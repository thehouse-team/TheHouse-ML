"""ML-05 T05 (unitarias): calibración 0-100 y desglose por factor de la regresión logística (ML-05-R05).

Pruebas de propiedades con vectores aleatorios (semilla fija) dentro y fuera del `training_range`,
formato de salida, pares excluidos y perfil incompleto (casos dorados de ML-04), orden del logit,
monotonía por factor y lista de transformadores con convertidor en skl2onnx.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thehouse_ml.features import (
    Campus,
    IncompleteProfileError,
    compute_pair_features,
    default_schema,
    feature_vector,
    load_campuses,
    room_from_record,
    student_from_record,
)
from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_score import ExcludedPair, ScoreInputError, factor_points, match_score
from thehouse_ml.models.factor_groups import FACTOR_GROUPS, OUTSIDE_GROUPS
from thehouse_ml.models.train import LogRegModel, clip_bounds, logreg_pipeline
from thehouse_ml.models.train_config import LogRegConfig, load_train_params
from thehouse_ml.models.trained_breakdown import (
    ACCEPTED_DEVIATIONS,
    LOGREG_MODEL_VERSION,
    BreakdownError,
    BreakdownEvaluation,
    BreakdownModel,
    LogRegScoreResult,
    RankedRoom,
    budget_monotonicity,
    budget_ratio_threshold,
    build_breakdown_model,
    check_monotonicity,
    check_order,
    check_sums,
    coefficient_table,
    distance_monotonicity,
    evaluate_breakdown,
    observed_budget_ratio,
    rank_logreg,
    score_features_logreg,
    score_pair_logreg,
    score_pairs,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA = default_schema()
ORDER = SCHEMA.feature_order
CONFIG = default_config()
LOWER, UPPER = clip_bounds(SCHEMA)
RULES = CONFIG.rounding
GOLDEN: dict[str, Any] = json.loads((REPO / "tests" / "fixtures" / "baseline_golden_cases.json").read_text("utf-8"))
SEED = 20261124

# Coeficientes estandarizados de un modelo de juguete con los signos «correctos» (orden de feature_order).
GOOD_COEF = {
    "distance_km": -3.0,
    "effective_distance_km": -1.5,
    "price_to_budget": -1.0,
    "services_coverage": 0.6,
    "place_type_match": 1.0,
    "property_type_match": 0.4,
    "pets_compatible": 0.0,
    "distance_ratio": -0.1,
    "price_below_min": 0.05,
    "extra_services_ratio": 0.4,
}
MEAN = {name: (low + high) / 4 for name, low, high in zip(ORDER, LOWER, UPPER, strict=True)}
SCALE = {name: max((high - low) / 5, 0.1) for name, low, high in zip(ORDER, LOWER, UPPER, strict=True)}
MEAN["pets_compatible"], SCALE["pets_compatible"] = 1.0, 1.0  # constante en los candidatos (StandardScaler)


def toy_logreg(coef: dict[str, float] | None = None, intercept: float = -2.0) -> LogRegModel:
    values = {**GOOD_COEF, **(coef or {})}
    return LogRegModel(
        feature_order=ORDER,
        config=LogRegConfig("standard", 0.01),
        clip_lower=tuple(float(value) for value in LOWER),
        clip_upper=tuple(float(value) for value in UPPER),
        coef=tuple(values[name] for name in ORDER),
        intercept=intercept,
        mean=tuple(MEAN[name] for name in ORDER),
        scale=tuple(SCALE[name] for name in ORDER),
    )


@pytest.fixture(scope="module")
def model() -> BreakdownModel:
    return build_breakdown_model(toy_logreg(), SCHEMA, CONFIG)


def random_vectors(rng: np.random.Generator, n: int, spread: float = 1.0) -> np.ndarray:
    """Vectores válidos para el esquema: uniformes en el recorte (x `spread` en las sin límite superior)."""
    rows = rng.uniform(LOWER, UPPER * spread, size=(n, len(ORDER)))
    for name in ("place_type_match", "property_type_match", "pets_compatible"):
        rows[:, ORDER.index(name)] = rng.integers(0, 2, size=n)
    for name in ("services_coverage", "price_below_min", "extra_services_ratio"):  # range [0, 1] cerrado
        rows[:, ORDER.index(name)] = rng.uniform(0, 1, size=n)
    return rows


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------


def test_calibration_has_the_six_baseline_factors_with_max_points_summing_100(model: BreakdownModel) -> None:
    assert [factor.id for factor in model.factors] == [factor.id for factor in CONFIG.factors]
    assert [factor.features for factor in model.factors] == [group.features for group in FACTOR_GROUPS]
    assert math.isclose(sum(f.raw_max_points for f in model.factors), 100.0, rel_tol=0, abs_tol=1e-9)
    for factor in model.factors:
        assert factor.max_points == factor_points(factor.raw_max_points, CONFIG)
        assert math.isclose(factor.max_logit - factor.min_logit, factor.span, rel_tol=1e-12)
        assert math.isclose(factor.raw_max_points, 100 * factor.span / model.total_span, rel_tol=1e-12)
    document = model.as_dict()
    assert document["model_version"] == LOGREG_MODEL_VERSION == "logreg-1.0.0"
    assert document["feature_schema_version"] == SCHEMA.version


def test_max_points_are_proportional_to_the_contribution_range_of_each_factor() -> None:
    doubled = build_breakdown_model(toy_logreg({"place_type_match": 2.0}), SCHEMA, CONFIG)
    base = build_breakdown_model(toy_logreg(), SCHEMA, CONFIG)
    ratio = doubled.factors[3].raw_max_points / doubled.factors[2].raw_max_points
    assert math.isclose(ratio, 2 * base.factors[3].raw_max_points / base.factors[2].raw_max_points, rel_tol=1e-12)


def test_pets_compatible_goes_to_the_fixed_term_not_to_a_factor(model: BreakdownModel) -> None:
    assert [term.name for term in model.fixed_terms] == list(OUTSIDE_GROUPS) == ["pets_compatible"]
    assert all("pets_compatible" not in factor.features for factor in model.factors)
    assert model.as_dict()["fixed_terms"][0]["constant_in_candidates"] is True
    # Con coeficiente ≠ 0, el término fijo cambia con pets_compatible pero los puntos no; en los candidatos vale 1.
    pets = build_breakdown_model(toy_logreg({"pets_compatible": 0.7}), SCHEMA, CONFIG)
    rows = random_vectors(np.random.default_rng(SEED), 50)
    rows[:, ORDER.index("pets_compatible")] = 1.0
    fixed = pets.fixed_term(rows)
    assert np.all(fixed == fixed[0])
    other = rows.copy()
    other[:, ORDER.index("pets_compatible")] = 0.0
    assert np.array_equal(pets.raw_points(rows), pets.raw_points(other))


def test_only_the_standard_transform_and_the_schema_order_are_accepted() -> None:
    bins = LogRegModel(
        feature_order=ORDER,
        config=LogRegConfig("quantile_bins", 0.01),
        clip_lower=tuple(LOWER),
        clip_upper=tuple(UPPER),
        coef=(0.0,) * 50,
        intercept=0.0,
        inner_edges=((0.1, 0.2, 0.3, 0.4),) * len(ORDER),
    )
    with pytest.raises(BreakdownError, match="standard"):
        build_breakdown_model(bins, SCHEMA, CONFIG)
    shuffled = toy_logreg()
    shuffled = LogRegModel(**{**shuffled.__dict__, "feature_order": tuple(reversed(ORDER))})
    with pytest.raises(BreakdownError, match="feature_order"):
        build_breakdown_model(shuffled, SCHEMA, CONFIG)
    flat = toy_logreg(dict.fromkeys(GOOD_COEF, 0.0))
    with pytest.raises(BreakdownError, match="constante"):
        build_breakdown_model(flat, SCHEMA, CONFIG)


# ---------------------------------------------------------------------------
# Propiedades del puntaje (dentro y fuera del training_range)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spread", [1.0, 1.6], ids=["inside-training-range", "beyond-training-range"])
def test_breakdown_properties_hold_for_random_vectors(model: BreakdownModel, spread: float) -> None:
    rows = random_vectors(np.random.default_rng(SEED), 400, spread)
    raw_max = [factor.raw_max_points for factor in model.factors]
    for row in rows:
        result = score_features_logreg([float(value) for value in row], model, SCHEMA)
        assert 0 <= result.match_score <= 100
        assert result.match_score == match_score(result.raw_total, CONFIG)  # floor(round6(Σ) + 0.5)
        assert abs(result.raw_total - result.match_score) <= RULES.total_tolerance
        assert abs(result.reported_sum - result.raw_total) <= RULES.breakdown_tolerance + 1e-12
        running = 0.0
        for item, maximum in zip(result.breakdown, raw_max, strict=True):
            assert 0.0 <= item.raw_points <= maximum
            assert 0.0 <= item.points <= item.max_points
            assert item.points == factor_points(item.raw_points, CONFIG)  # round6 y 1 decimal mitad hacia arriba
            running += item.raw_points
        assert running == result.raw_total  # suma de izquierda a derecha en binary64
        logit = float(model.logreg.decision_function(row.reshape(1, -1))[0])
        assert math.isclose(result.logit, logit, rel_tol=0.0, abs_tol=1e-9)  # afín exacta salvo binary64


def test_inputs_beyond_the_training_range_score_as_the_clipped_input(model: BreakdownModel) -> None:
    rng = np.random.default_rng(SEED + 1)
    for row in random_vectors(rng, 100, spread=2.0):
        clipped = np.clip(row, LOWER, UPPER)
        far = score_features_logreg([float(v) for v in row], model, SCHEMA)
        near = score_features_logreg([float(v) for v in clipped], model, SCHEMA)
        assert far.raw_total == near.raw_total
        assert far.match_score == near.match_score
        assert [item.points for item in far.breakdown] == [item.points for item in near.breakdown]
        # `value` es el valor de entrada (sin recortar).
        assert far.breakdown[1].features[0] == ("distance_km", float(row[ORDER.index("distance_km")]))


def test_extremes_give_0_and_100(model: BreakdownModel) -> None:
    best = [0.0] * len(ORDER)
    worst = [0.0] * len(ORDER)
    for factor in model.factors:
        for term in factor.terms:
            best[term.index], worst[term.index] = (
                (term.upper, term.lower) if term.slope > 0 else (term.lower, term.upper)
            )
    assert score_features_logreg(best, model, SCHEMA).match_score == 100
    top = score_features_logreg(best, model, SCHEMA)
    assert [item.points for item in top.breakdown] == [factor.max_points for factor in model.factors]
    bottom = score_features_logreg(worst, model, SCHEMA)
    assert bottom.match_score == 0
    assert all(item.raw_points == 0.0 for item in bottom.breakdown)


def test_ranking_by_raw_total_is_the_logit_ranking_inside_the_range(model: BreakdownModel) -> None:
    rng = np.random.default_rng(SEED + 2)
    for _student in range(30):
        rows = random_vectors(rng, 40)
        ids = [f"R{index:03d}" for index in range(len(rows))]
        scored = [
            RankedRoom(room, score_features_logreg([float(v) for v in row], model, SCHEMA))
            for room, row in zip(ids, rows, strict=True)
        ]
        logit = model.logreg.decision_function(rows)
        by_logit = sorted(range(len(rows)), key=lambda i: (-float(logit[i]), ids[i]))
        assert [item.room_id for item in rank_logreg(scored)] == [ids[i] for i in by_logit]


def test_vector_validation_is_the_baseline_one(model: BreakdownModel) -> None:
    with pytest.raises(ScoreInputError, match="se esperaban 10"):
        score_features_logreg([0.5] * 9, model, SCHEMA)
    bad = [0.5] * len(ORDER)
    bad[ORDER.index("services_coverage")] = 1.5  # fuera de range [0, 1]
    with pytest.raises(ScoreInputError, match="services_coverage"):
        score_features_logreg(bad, model, SCHEMA)


# ---------------------------------------------------------------------------
# Formato de salida
# ---------------------------------------------------------------------------


def test_output_has_the_baseline_format_with_features_lists_and_variable_max_points(model: BreakdownModel) -> None:
    row = [float(value) for value in random_vectors(np.random.default_rng(SEED + 3), 1)[0]]
    result = score_features_logreg(row, model, SCHEMA)
    produced = result.to_dict()

    assert list(produced) == ["modelVersion", "featureSchemaVersion", "matchScore", "breakdown"]
    assert produced["modelVersion"] == "logreg-1.0.0"
    assert produced["featureSchemaVersion"] == SCHEMA.version
    assert isinstance(produced["matchScore"], int)
    assert [entry["factor"] for entry in produced["breakdown"]] == [factor.id for factor in CONFIG.factors]
    for entry, factor in zip(produced["breakdown"], model.factors, strict=True):
        assert entry["maxPoints"] == factor.max_points
        if len(factor.features) == 1:
            assert set(entry) == {"factor", "points", "maxPoints", "feature", "value"}
            assert entry["feature"] == factor.features[0]
        else:
            assert set(entry) == {"factor", "points", "maxPoints", "features"}
            assert [item["feature"] for item in entry["features"]] == list(factor.features)
            assert [item["value"] for item in entry["features"]] == [row[ORDER.index(n)] for n in factor.features]
    raw = result.to_dict(include_raw=True)
    assert {"rawTotal", "logit", "probability"} <= set(raw)
    assert raw["probability"] == pytest.approx(1 / (1 + math.exp(-result.logit)))
    assert all({"rawPoints", "rawMaxPoints"} <= set(entry) for entry in raw["breakdown"])
    json.dumps(raw)  # serializable


# ---------------------------------------------------------------------------
# Pares excluidos (HC01-HC03) y perfil incompleto, como en ML-04
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(REPO / "data" / "universities.csv")


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[case["id"] for case in GOLDEN["cases"]])
def test_excluded_pairs_and_incomplete_profiles_behave_as_in_ml04(
    case: dict[str, Any], campuses: dict[str, Campus], model: BreakdownModel
) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    expected = case["expected"]["result"]
    if expected.get("incomplete"):
        with pytest.raises(IncompleteProfileError) as error:
            score_pair_logreg(student, room, campuses, model, SCHEMA)
        assert list(error.value.missing) == expected["missingFields"]
        return
    result = score_pair_logreg(student, room, campuses, model, SCHEMA)
    if expected.get("excluded"):
        assert isinstance(result, ExcludedPair)
        assert result.to_dict() == {**expected, "modelVersion": "logreg-1.0.0"}  # sin matchScore ni breakdown
        return
    assert isinstance(result, LogRegScoreResult)
    vector = feature_vector(compute_pair_features(student, room, campuses, SCHEMA), SCHEMA)
    assert result == score_features_logreg(list(vector), model, SCHEMA)


# ---------------------------------------------------------------------------
# Verificación en lote (condición 3)
# ---------------------------------------------------------------------------


def _batch(model: BreakdownModel, n_students: int = 6, per_student: int = 12) -> tuple[np.ndarray, ...]:
    rows = random_vectors(np.random.default_rng(SEED + 4), n_students * per_student)
    group = np.repeat([f"S{index:04d}" for index in range(n_students)], per_student)
    room = np.asarray([f"R{index:05d}" for index in range(len(rows))])
    return rows, model.logreg.decision_function(rows), group, room


def test_batch_scores_match_the_single_pair_scores(model: BreakdownModel) -> None:
    rows, _logit, _group, _room = _batch(model)
    scored = score_pairs(model, rows)
    for index, row in enumerate(rows):
        single = score_features_logreg([float(v) for v in row], model, SCHEMA)
        assert scored.raw_total[index] == single.raw_total
        assert int(scored.match_score[index]) == single.match_score
        assert list(scored.points[index]) == [item.points for item in single.breakdown]


def test_evaluate_breakdown_passes_for_a_model_with_consistent_signs(model: BreakdownModel) -> None:
    rows, logit, group, room = _batch(model)
    evaluation = evaluate_breakdown(model, rows, logit, group, room, rows, True, SCHEMA)

    assert evaluation.sums.passed and evaluation.sums.pairs == len(rows)
    assert evaluation.order.passed and evaluation.order.students == 6
    assert evaluation.monotonicity.passed
    assert evaluation.passed
    detail = evaluation.condition_detail()
    assert detail["status"] == "verified_t05"
    assert detail["failed_checks"] == []
    assert evaluation.metrics_unchanged is True
    assert not evaluate_breakdown(model, rows, logit, group, room, rows, False, SCHEMA).passed


def test_check_sums_and_check_order_detect_a_logit_that_is_not_the_calibrated_one(model: BreakdownModel) -> None:
    rows, logit, group, room = _batch(model)
    scored = score_pairs(model, rows)
    wrong = logit.copy()
    wrong[0] += 1.0
    assert check_sums(model, rows, wrong, scored).passed is False
    reversed_logit = -logit
    order = check_order(scored.raw_total, reversed_logit, group, room)
    assert order.passed is False
    assert order.identical == 0
    assert order.as_dict()["differing_students"] == sorted(set(group.tolist()))


# ---------------------------------------------------------------------------
# Monotonía por factor
# ---------------------------------------------------------------------------


def test_coefficient_table_flags_signs_against_the_natural_variable() -> None:
    model = build_breakdown_model(toy_logreg({"price_below_min": -0.3}), SCHEMA, CONFIG)
    rows = {row.feature: row for row in coefficient_table(model)}
    assert rows["price_below_min"].consistent is False  # negativo: por debajo del mínimo, más renta suma
    assert rows["price_to_budget"].consistent is True
    assert rows["distance_km"].consistent is True
    assert rows["pets_compatible"].consistent is None
    assert rows["pets_compatible"].factor is None
    assert rows["services_coverage"].slope == pytest.approx(GOOD_COEF["services_coverage"] / SCALE["services_coverage"])


def test_budget_factor_is_monotone_when_price_below_min_has_the_expected_sign(model: BreakdownModel) -> None:
    check = budget_monotonicity(model, SCHEMA)
    assert check.passed
    assert check.curves > 50 and check.steps > 5000
    assert budget_ratio_threshold(model) is None
    assert distance_monotonicity(model, SCHEMA).passed


def test_collinear_budget_features_are_reported_with_the_exact_ratio_threshold() -> None:
    # slope(p2b) = -1 / SCALE, slope(pbm) = -0.3 / SCALE: el factor sube con la renta si max/min > cociente.
    model = build_breakdown_model(toy_logreg({"price_below_min": -0.3}), SCHEMA, CONFIG)
    threshold = budget_ratio_threshold(model)
    expected = (GOOD_COEF["price_to_budget"] / SCALE["price_to_budget"]) / (-0.3 / SCALE["price_below_min"])
    assert threshold == pytest.approx(expected)
    check = budget_monotonicity(model, SCHEMA)
    assert not check.passed
    assert check.worst_case is not None
    ratio = check.worst_case["budget.max"] / check.worst_case["budget.min"]
    assert ratio > threshold  # solo sube en la zona predicha
    assert check.worst_case["rent_mxn_to"] <= check.worst_case["budget.min"]  # con renta por debajo del mínimo


def test_distance_factor_with_a_wrong_sign_is_detected() -> None:
    model = build_breakdown_model(toy_logreg({"distance_ratio": 10.0}), SCHEMA, CONFIG)
    check = distance_monotonicity(model, SCHEMA)
    assert not check.passed
    assert check.violations > 0


def test_observed_budget_ratio_reads_the_implicit_max_over_min_of_the_pairs() -> None:
    rows = np.zeros((3, len(ORDER)))
    p2b, pbm = ORDER.index("price_to_budget"), ORDER.index("price_below_min")
    rows[0, [p2b, pbm]] = [0.25, 0.5]  # renta = 0.5 · min = 0.25 · max → max/min = 2
    rows[1, [p2b, pbm]] = [0.1, 0.7]  # 0.3 / 0.1 = 3
    rows[2, [p2b, pbm]] = [0.9, 0.0]  # sin mínimo: no cuenta
    ratio, pairs = observed_budget_ratio(rows, ORDER)
    assert pairs == 2
    assert ratio == pytest.approx(3.0)
    assert observed_budget_ratio(rows[2:], ORDER) == (None, 0)


def test_check_monotonicity_reports_observed_pairs_outside_the_violation_zone() -> None:
    model = build_breakdown_model(toy_logreg({"price_below_min": -0.3}), SCHEMA, CONFIG)
    rows = np.zeros((1, len(ORDER)))
    rows[0, [ORDER.index("price_to_budget"), ORDER.index("price_below_min")]] = [0.5, 0.2]  # max/min = 1.6
    result = check_monotonicity(model, rows, SCHEMA)
    assert not result.passed
    assert result.observed_violations is False
    assert result.as_dict()["sign_inconsistent_features"] == ["price_below_min"]


# ---------------------------------------------------------------------------
# Pipeline exportable (spec R05: lista verificada por prueba)
# ---------------------------------------------------------------------------

# Transformadores del pipeline y su convertidor registrado en skl2onnx (ML-06 los exporta); el recorte va antes
# del pipeline (`np.clip`) y en ONNX es un nodo `Clip` (M-ML-038 decisión 5).
SKL2ONNX_CONVERTERS = {
    "StandardScaler": "SklearnScaler",
    "KBinsDiscretizer": "SklearnKBinsDiscretizer",
    "LogisticRegression": "SklearnLinearClassifier",
}


def test_pipeline_uses_only_estimators_listed_with_a_skl2onnx_converter() -> None:
    params = load_train_params()
    for config in params.logreg_grid():
        pipeline = logreg_pipeline(config, params)
        names = [type(step).__name__ for _name, step in pipeline.steps]
        assert set(names) <= set(SKL2ONNX_CONVERTERS), names
        assert names[-1] == "LogisticRegression"
    # El modelo que se calibra usa la transformación «standard» (StandardScaler) y guarda el recorte en el JSON.
    document = toy_logreg().to_dict()
    assert document["transform"]["type"] == "standard"
    assert document["clip"] == {"lower": list(LOWER), "upper": list(UPPER)}


# ---------------------------------------------------------------------------
# Desviación aceptada (M-ML-041, T05-H1): la condición 3 es el criterio de la spec §6.5
# ---------------------------------------------------------------------------


def _collinear_evaluation(coef: dict[str, float]) -> BreakdownEvaluation:
    model = build_breakdown_model(toy_logreg(coef), SCHEMA, CONFIG)
    rows, logit, group, room = _batch(model)
    return evaluate_breakdown(model, rows, logit, group, room, rows, True, SCHEMA)


def test_budget_non_monotonicity_is_an_accepted_deviation_that_does_not_block_condition_3() -> None:
    evaluation = _collinear_evaluation({"price_below_min": -0.3})

    assert evaluation.sums_total
    assert not evaluation.monotonicity.budget.passed
    assert evaluation.failed_checks == []
    assert evaluation.deviations_applied == ["monotonicity_budget"]
    assert evaluation.passed
    detail = evaluation.condition_detail()
    assert detail["failed_checks"] == []
    assert "M-ML-041" in detail["accepted_deviations"]["monotonicity_budget"]
    assert list(ACCEPTED_DEVIATIONS) == ["monotonicity_budget"]  # solo la desviación aprobada


def test_without_the_accepted_deviation_the_budget_non_monotonicity_blocks_condition_3() -> None:
    evaluation = replace(_collinear_evaluation({"price_below_min": -0.3}), accepted_deviations={})

    assert not evaluation.passed
    assert evaluation.failed_checks == ["monotonicity_budget"]
    assert evaluation.deviations_applied == []


def test_a_non_accepted_monotonicity_failure_still_blocks_condition_3() -> None:
    evaluation = _collinear_evaluation({"distance_ratio": 10.0})

    assert evaluation.sums_total
    assert not evaluation.passed
    assert evaluation.failed_checks == ["monotonicity_distance"]


def test_a_breakdown_that_does_not_sum_the_total_is_never_rescued_by_a_deviation(model: BreakdownModel) -> None:
    rows, logit, group, room = _batch(model)
    evaluation = evaluate_breakdown(model, rows, logit, group, room, rows, False, SCHEMA)  # métricas cambian

    assert not evaluation.sums_total
    assert not evaluation.passed
    assert evaluation.failed_checks == ["metrics_unchanged"]
