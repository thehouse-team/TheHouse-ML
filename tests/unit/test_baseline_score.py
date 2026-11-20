"""ML-04 T03 (R03): puntaje de 0 a 100 con desglose por factor del modelo base.

Sin red. Las pruebas de propiedades usan rejillas y `random.Random` con semilla
fija (sin dependencias nuevas). Los valores esperados de los ejemplos y de los
bordes del redondeo se calcularon a mano a partir de la spec ML-04 §6.2 a §6.4.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import random
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import (
    Campus,
    FeatureSchema,
    IncompleteProfileError,
    PairInputError,
    Room,
    StudentProfile,
    compute_pair_features,
    default_schema,
    feature_vector,
    load_campuses,
    room_from_record,
    student_from_record,
)
from thehouse_ml.models import (
    CONFIG_RESOURCE,
    BaselineConfig,
    ExcludedPair,
    ScoredRoom,
    ScoreInputError,
    ScoreResult,
    default_config,
    factor_points,
    match_score,
    parse_config,
    rank,
    round_half_up,
    score_features,
    score_pair,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"
PAIR_CASES: dict[str, Any] = json.loads((FIXTURES / "pair_feature_cases.json").read_text(encoding="utf-8"))
HARD_CASES: dict[str, Any] = json.loads((FIXTURES / "hard_constraint_cases.json").read_text(encoding="utf-8"))
SEED = 20261120  # semilla de las pruebas de propiedades (no es la de ningún generador de datos)
SAMPLES = 2000

# Features en el orden de feature_order 2.0.0.
ORDER = (
    "distance_km",
    "effective_distance_km",
    "price_to_budget",
    "services_coverage",
    "place_type_match",
    "property_type_match",
    "pets_compatible",
    "distance_ratio",
    "price_below_min",
    "extra_services_ratio",
)
NON_SCORING = ("distance_km", "effective_distance_km", "price_to_budget", "pets_compatible")
PERFECT = {
    "distance_km": 0.0,
    "effective_distance_km": 0.0,
    "price_to_budget": 1.0,
    "services_coverage": 1.0,
    "place_type_match": 1,
    "property_type_match": 1,
    "pets_compatible": 1,
    "distance_ratio": 0.0,
    "price_below_min": 0.0,
    "extra_services_ratio": 1.0,
}
WORST = {
    **PERFECT,
    "distance_km": 12.0,
    "effective_distance_km": 12.0,
    "services_coverage": 0.0,
    "place_type_match": 0,
    "property_type_match": 0,
    "distance_ratio": 1.5,
    "price_below_min": 1.0,
    "extra_services_ratio": 0.0,
}
# Ejemplo de la spec ML-04 §6.4.
SPEC_EXAMPLE = {
    "distance_km": 6.6,
    "effective_distance_km": 6.6,
    "price_to_budget": 0.6,
    "services_coverage": 2 / 3,
    "place_type_match": 1,
    "property_type_match": 1,
    "pets_compatible": 1,
    "distance_ratio": 0.66,
    "price_below_min": 0.15,
    "extra_services_ratio": 3 / 7,
}


def _vector(values: dict[str, float], **changes: float) -> list[float]:
    merged = {**values, **changes}
    return [float(merged[name]) for name in ORDER]


def _random_features(rng: random.Random) -> dict[str, float]:
    distance_km = rng.uniform(0, 30)
    return {
        "distance_km": distance_km,
        "effective_distance_km": distance_km * rng.choice((1.0, 0.5, 0.4, 0.25)),
        "price_to_budget": rng.uniform(0, 1),
        "services_coverage": rng.choice((0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, rng.random())),
        "place_type_match": rng.randint(0, 1),
        "property_type_match": rng.randint(0, 1),
        "pets_compatible": rng.randint(0, 1),
        "distance_ratio": rng.choice((rng.uniform(0, 1.2), rng.uniform(0, 60), 0.5, 1.0)),
        "price_below_min": rng.choice((0.0, rng.random())),
        "extra_services_ratio": rng.randint(0, 11) / 11 if rng.random() < 0.5 else rng.random(),
    }


def _samples(salt: str) -> Iterator[dict[str, float]]:
    rng = random.Random(f"score:{SEED}:{salt}")  # noqa: S311 - pruebas reproducibles
    for _ in range(SAMPLES):
        yield _random_features(rng)


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


@pytest.fixture(scope="module")
def config() -> BaselineConfig:
    return default_config()


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(UNIVERSITIES_CSV)


@pytest.fixture(scope="module")
def raw_config() -> dict[str, Any]:
    text = resources.files("thehouse_ml.models").joinpath(CONFIG_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def _pair_case(case_id: str) -> dict[str, Any]:
    return next(case for case in PAIR_CASES["cases"] if case["id"] == case_id)


def _student(**changes: Any) -> StudentProfile:
    return dataclasses.replace(student_from_record(_pair_case("budget_exact")["input"]["student"]), **changes)


def _room(**changes: Any) -> Room:
    return dataclasses.replace(room_from_record(_pair_case("budget_exact")["input"]["room"]), **changes)


# --- Contrato con el esquema y la configuración ----------------------------


def test_the_test_order_is_the_schema_order(schema: FeatureSchema, config: BaselineConfig) -> None:
    assert schema.feature_order == ORDER
    assert {item.feature for item in config.non_scoring_features} == set(NON_SCORING)


def test_the_breakdown_follows_the_config_order_and_weights(config: BaselineConfig) -> None:
    result = score_features(_vector(SPEC_EXAMPLE))
    assert [item.factor for item in result.breakdown] == [factor.id for factor in config.factors]
    assert [item.feature for item in result.breakdown] == [factor.feature for factor in config.factors]
    assert [item.max_points for item in result.breakdown] == [factor.weight for factor in config.factors]
    assert sum(item.max_points for item in result.breakdown) == config.total_points
    assert result.model_version == "baseline-1.0.0"
    assert result.feature_schema_version == "2.0.0"


def test_weights_and_curves_are_read_from_the_config(raw_config: dict[str, Any], schema: FeatureSchema) -> None:
    data = copy.deepcopy(raw_config)
    weights = {"budget": 25, "distance": 30, "essential_services": 20, "place_type": 10, "property_type": 10}
    for factor in data["factors"]:
        factor["weight"] = weights.get(factor["id"], factor["weight"])
        if factor["id"] == "distance":
            factor["curve"]["points"] = [[0, 1.0], [1, 0.0]]
    data["metadata"]["model_version"] = "baseline-9.9.9"
    changed = parse_config(data, schema)
    result = score_features(_vector(SPEC_EXAMPLE), changed, schema)
    # A mano: 25·0.85 + 30·0.34 + 20·(2/3) + 10 + 10 + 5·(3/7) = 21.25 + 10.2 + 13.333333 + 10 + 10 + 2.142857.
    assert result.raw_total == pytest.approx(66.926190, abs=1e-6)
    assert result.match_score == 67
    assert result.model_version == "baseline-9.9.9"
    assert [item.max_points for item in result.breakdown] == [25, 30, 20, 10, 10, 5]


def test_an_incompatible_schema_is_rejected(schema: FeatureSchema, config: BaselineConfig) -> None:
    old = dataclasses.replace(schema, metadata=dataclasses.replace(schema.metadata, schema_version="1.1.0"))
    with pytest.raises(ScoreInputError, match=r"requiere el esquema 2\.0\.0"):
        score_features(_vector(PERFECT), config, old)
    with pytest.raises(ScoreInputError, match=r"requiere el esquema 2\.0\.0"):
        score_pair(_student(), _room(), {}, config, old)


def test_a_config_feature_missing_from_the_schema_order_is_rejected(
    schema: FeatureSchema, config: BaselineConfig
) -> None:
    reduced = dataclasses.replace(schema, feature_order=schema.feature_order[:-1])
    with pytest.raises(ScoreInputError, match="extra_services_ratio"):
        score_features(_vector(PERFECT)[:-1], config, reduced)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (_vector(PERFECT)[:-1], "se esperaban 10 valores"),
        ([*_vector(PERFECT), 0.0], "se esperaban 10 valores"),
        ("0123456789", "se esperaba una secuencia"),
        ({"distance_km": 0.0}, "se esperaba una secuencia"),
        (_vector(PERFECT, distance_ratio=math.nan), "distance_ratio: se esperaba un número finito"),
        (_vector(PERFECT, distance_ratio=math.inf), "distance_ratio: se esperaba un número finito"),
        ([*_vector(PERFECT)[:4], True, *_vector(PERFECT)[5:]], "place_type_match: se esperaba un número finito"),
        ([*_vector(PERFECT)[:4], "1", *_vector(PERFECT)[5:]], "place_type_match: se esperaba un número finito"),
        (_vector(PERFECT, price_below_min=1.2), r"price_below_min: 1\.2 fuera del rango \[0, 1\]"),
        (_vector(PERFECT, distance_ratio=-0.1), r"distance_ratio: -0\.1 fuera del rango \[0, ∞\)"),
        (_vector(PERFECT, place_type_match=0.5), r"place_type_match: 0\.5 no es entero"),
    ],
)
def test_invalid_vectors_are_rejected(features: Any, message: str) -> None:
    with pytest.raises(ScoreInputError, match=message):
        score_features(features)


def test_every_problem_of_a_vector_is_reported_at_once() -> None:
    with pytest.raises(ScoreInputError) as error:
        score_features(_vector(PERFECT, price_below_min=2.0, extra_services_ratio=-1.0, pets_compatible=0.5))
    text = str(error.value)
    assert "price_below_min" in text
    assert "extra_services_ratio" in text
    assert "pets_compatible" in text


def test_integer_features_accept_python_ints() -> None:
    vector: list[float] = _vector(PERFECT)
    vector[4] = 1
    assert score_features(vector).match_score == 100


# --- Ejemplo de la spec §6.4 y formato JSON ---------------------------------


def test_the_example_of_spec_6_4_scores_78() -> None:
    result = score_features(_vector(SPEC_EXAMPLE))
    # A mano: 30·0.85 + 25·2·(1 - 0.66) + 20·(2/3) + 10 + 10 + 5·(3/7) = 77.976190…
    assert result.raw_total == pytest.approx(25.5 + 17.0 + 40 / 3 + 10 + 10 + 15 / 7, abs=1e-12)
    assert round(result.raw_total, 6) == 77.976190
    assert result.match_score == 78
    assert [item.points for item in result.breakdown] == [25.5, 17.0, 13.3, 10.0, 10.0, 2.1]
    assert result.reported_sum == pytest.approx(77.9, abs=1e-9)


def test_the_json_output_is_the_format_of_spec_6_4() -> None:
    output = score_features(_vector(SPEC_EXAMPLE)).to_dict()
    decoded = json.loads(json.dumps(output))
    assert decoded == output
    assert list(output) == ["modelVersion", "featureSchemaVersion", "matchScore", "breakdown"]
    assert isinstance(output["matchScore"], int)
    assert output["matchScore"] == 78
    expected = [
        {"factor": "budget", "points": 25.5, "maxPoints": 30, "feature": "price_below_min", "value": 0.15},
        {"factor": "distance", "points": 17.0, "maxPoints": 25, "feature": "distance_ratio", "value": 0.66},
        {
            "factor": "essential_services",
            "points": 13.3,
            "maxPoints": 20,
            "feature": "services_coverage",
            "value": 0.666667,
        },
        {"factor": "place_type", "points": 10.0, "maxPoints": 10, "feature": "place_type_match", "value": 1},
        {"factor": "property_type", "points": 10.0, "maxPoints": 10, "feature": "property_type_match", "value": 1},
        {"factor": "extras", "points": 2.1, "maxPoints": 5, "feature": "extra_services_ratio", "value": 0.428571},
    ]
    assert len(output["breakdown"]) == len(expected)
    for got, want in zip(output["breakdown"], expected, strict=True):
        assert list(got) == ["factor", "points", "maxPoints", "feature", "value"]
        assert {key: got[key] for key in ("factor", "points", "maxPoints", "feature")} == {
            key: want[key] for key in ("factor", "points", "maxPoints", "feature")
        }
        assert got["value"] == pytest.approx(want["value"], abs=5e-7)
        assert isinstance(got["maxPoints"], int)
    assert isinstance(output["breakdown"][3]["value"], int)


def test_include_raw_adds_the_unrounded_values() -> None:
    result = score_features(_vector(SPEC_EXAMPLE))
    output = result.to_dict(include_raw=True)
    assert list(output) == ["modelVersion", "featureSchemaVersion", "matchScore", "rawTotal", "breakdown"]
    assert output["rawTotal"] == result.raw_total
    assert [item["rawPoints"] for item in output["breakdown"]] == [item.raw_points for item in result.breakdown]
    assert json.loads(json.dumps(output)) == output


# --- Propiedades ------------------------------------------------------------


def test_perfect_features_score_100() -> None:
    result = score_features(_vector(PERFECT))
    assert result.raw_total == 100.0
    assert result.match_score == 100
    assert [item.points for item in result.breakdown] == [30.0, 25.0, 20.0, 10.0, 10.0, 5.0]


def test_the_worst_features_score_0() -> None:
    result = score_features(_vector(WORST))
    assert result.raw_total == 0.0
    assert result.match_score == 0
    assert all(item.points == 0.0 for item in result.breakdown)


def test_a_vector_of_zeros_only_keeps_budget_and_distance() -> None:
    # price_below_min = 0 y distance_ratio = 0 son el mejor valor de su factor: 30 + 25 (spec §6.2).
    result = score_features([0.0] * len(ORDER))
    assert result.raw_total == 55.0
    assert result.match_score == 55
    assert [item.points for item in result.breakdown] == [30.0, 25.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("salt", ["a", "b"])
def test_price_monotonicity(salt: str) -> None:
    rng = random.Random(f"score:{SEED}:price:{salt}")  # noqa: S311 - pruebas reproducibles
    grid = [step / 100 for step in range(101)]
    for _ in range(200):
        base = _random_features(rng)
        scores = [score_features(_vector(base, price_below_min=value)) for value in grid]
        for before, after in itertools.pairwise(scores):
            assert after.raw_total <= before.raw_total
            assert after.match_score <= before.match_score
        assert scores[0].raw_total - scores[-1].raw_total == pytest.approx(30.0)


@pytest.mark.parametrize("salt", ["a", "b"])
def test_distance_monotonicity(salt: str) -> None:
    rng = random.Random(f"score:{SEED}:distance:{salt}")  # noqa: S311 - pruebas reproducibles
    grid = [step / 100 for step in range(301)] + [10.0, 59.797, 60.0, 1000.0]
    for _ in range(200):
        base = _random_features(rng)
        scores = [score_features(_vector(base, distance_ratio=value)) for value in grid]
        for before, after in itertools.pairwise(scores):
            assert after.raw_total <= before.raw_total
            assert after.match_score <= before.match_score
        flat = {round(item.raw_total, 9) for item, value in zip(scores, grid, strict=True) if value <= 0.5}
        assert len(flat) == 1  # 1.0 hasta 0.5 (spec §6.2)
        beyond = {round(item.raw_total, 9) for item, value in zip(scores, grid, strict=True) if value >= 1.0}
        assert len(beyond) == 1  # 0 después de 1


def test_distance_curve_at_its_breakpoints() -> None:
    points = {
        value: score_features(_vector(PERFECT, distance_ratio=value)).breakdown[1].raw_points
        for value in (0.0, 0.5, 0.75, 1.0, 1.000754, 8.006)
    }
    assert points == {0.0: 25.0, 0.5: 25.0, 0.75: 12.5, 1.0: 0.0, 1.000754: 0.0, 8.006: 0.0}


def test_total_is_between_0_and_100() -> None:
    for features in _samples("range"):
        result = score_features(_vector(features))
        assert 0.0 <= result.raw_total <= 100.0
        assert 0 <= result.match_score <= 100
        assert isinstance(result.match_score, int)
        for item in result.breakdown:
            assert 0.0 <= item.raw_points <= item.max_points
            assert 0.0 <= item.points <= item.max_points


def test_the_total_is_between_0_and_100_on_the_corners_of_the_grid() -> None:
    corners = {
        "services_coverage": (0.0, 1.0),
        "place_type_match": (0, 1),
        "property_type_match": (0, 1),
        "distance_ratio": (0.0, 0.5, 1.0, 60.0),
        "price_below_min": (0.0, 1.0),
        "extra_services_ratio": (0.0, 1.0),
    }
    seen = set()
    stack: list[dict[str, float]] = [dict(PERFECT)]
    for name, values in corners.items():
        stack = [{**item, name: value} for item in stack for value in values]
    for features in stack:
        result = score_features(_vector(features))
        assert 0 <= result.match_score <= 100
        seen.add(result.match_score)
    assert {0, 100} <= seen


def test_the_breakdown_is_coherent_with_the_total(config: BaselineConfig) -> None:
    tolerances = config.rounding
    assert (tolerances.total_tolerance, tolerances.breakdown_tolerance) == (0.5, 0.3)
    factors = len(config.factors)
    for features in _samples("coherence"):
        result = score_features(_vector(features))
        raw = math.fsum(item.raw_points for item in result.breakdown)
        assert result.raw_total == pytest.approx(raw, abs=1e-9)
        assert abs(result.match_score - result.raw_total) <= tolerances.total_tolerance + 1e-6
        assert abs(result.reported_sum - result.raw_total) <= tolerances.breakdown_tolerance + 1e-9
        assert abs(result.reported_sum - result.raw_total) <= 0.05 * factors + 1e-9
        for item in result.breakdown:
            assert abs(item.points - item.raw_points) <= 0.05 + 1e-9
            assert round(item.points * 10) == pytest.approx(item.points * 10, abs=1e-9)
        assert abs(result.reported_sum - result.match_score) <= 0.8 + 1e-9


def test_the_result_is_deterministic() -> None:
    for features in list(_samples("determinism"))[:200]:
        assert score_features(_vector(features)) == score_features(_vector(features))


@pytest.mark.parametrize("feature", NON_SCORING)
def test_non_scoring_features_do_not_change_the_result(feature: str, schema: FeatureSchema) -> None:
    spec = next(item for item in schema.pair_features if item.name == feature)
    values = (0, 1) if spec.type == "integer" else (0.0, 0.3, 1.0, 7.5, 59.0)
    for base in list(_samples(f"non-scoring:{feature}"))[:100]:
        results = {score_features(_vector(base, **{feature: value})) for value in values}
        assert len({(item.raw_total, item.breakdown) for item in results}) == 1


def test_pets_compatible_does_not_change_the_example() -> None:
    with_pets = score_features(_vector(SPEC_EXAMPLE, pets_compatible=1))
    without = score_features(_vector(SPEC_EXAMPLE, pets_compatible=0))
    assert with_pets.to_dict(include_raw=True) == without.to_dict(include_raw=True)


# --- Redondeo (spec §6.3) ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw_total", "expected"),
    [
        # A mano: round6 (mitad hacia arriba en el 6.º decimal) y después entero, mitad hacia arriba.
        (77.5, 78),  # 77.500000 → 78
        (77.4999995, 78),  # round6 → 77.500000 → 78
        (77.5000004, 78),  # round6 → 77.500000 → 78
        (77.4999994, 77),  # round6 → 77.499999 → 77
        (77.49999949, 77),  # round6 → 77.499999 → 77
        (77.4999999999999, 78),  # error de punto flotante absorbido → 77.500000 → 78
        (77.976190, 78),
        (78.4999995, 79),
        (0.0, 0),
        (0.4999994, 0),
        (0.4999995, 1),
        (0.5, 1),
        (2.5, 3),  # el round de Python (mitad al par) daría 2
        (99.4999994, 99),
        (99.5, 100),
        (100.0, 100),
        (100.00000000000001, 100),
    ],
)
def test_match_score_rounding_edges(raw_total: float, expected: int) -> None:
    assert match_score(raw_total) == expected


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        # Puntos del desglose: 1 decimal, mitad hacia arriba (a mano sobre la lectura decimal).
        (0.05, 1, 0.1),
        (0.15, 1, 0.2),
        (0.25, 1, 0.3),
        (1.05, 1, 1.1),
        (2.25, 1, 2.3),
        (13.35, 1, 13.4),
        (13.333333333333334, 1, 13.3),
        (2.142857142857143, 1, 2.1),
        (0.0499999, 1, 0.0),
        (29.95, 1, 30.0),
        (77.4999995, 6, 77.5),
        (77.4999994, 6, 77.499999),
        (0.0, 0, 0.0),
    ],
)
def test_round_half_up(value: float, decimals: int, expected: float) -> None:
    assert round_half_up(value, decimals) == expected


@pytest.mark.parametrize("value", [-0.5, math.nan, math.inf])
def test_round_half_up_rejects_negative_or_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="round_half_up"):
        round_half_up(value, 1)


def test_a_total_of_exactly_77_5_from_features_scores_78() -> None:
    # Presupuesto 30·(1 - 0.75) = 7.5 y lo demás perfecto: 7.5 + 25 + 20 + 10 + 10 + 5 = 77.5.
    result = score_features(_vector(PERFECT, price_below_min=0.75))
    assert result.raw_total == 77.5
    assert result.match_score == 78
    assert result.breakdown[0].points == 7.5


def test_factor_points_ending_in_5_round_up_from_features() -> None:
    # Extras 5·0.45 = 2.25 → 2.3; presupuesto 30·(1 - 0.125) = 26.25 → 26.3 (mitad hacia arriba).
    result = score_features(_vector(PERFECT, extra_services_ratio=0.45, price_below_min=0.125))
    assert result.breakdown[5].raw_points == 2.25
    assert result.breakdown[5].points == 2.3
    assert result.breakdown[0].raw_points == 26.25
    assert result.breakdown[0].points == 26.3


@pytest.mark.parametrize(
    ("changes", "factor_index", "binary64_points", "expected"),
    [
        # T03-H1 (M-ML-031): a mano, cada caso termina en .x5 exacto; en binary64 queda justo debajo.
        # round6 lo devuelve a .x5 y la mitad hacia arriba sube (sin round6 bajaría).
        # Presupuesto: 30 * (1 - 0.555) = 30 * 0.445 = 13.35 → 13.4.
        ({"price_below_min": 0.555}, 0, 13.349999999999998, 13.4),
        # Distancia: 25 * 2 * (1 - 0.513) = 50 * 0.487 = 24.35 → 24.4.
        ({"distance_ratio": 0.513}, 1, 24.349999999999998, 24.4),
        # Extras: 5 * 0.47 = 2.35 → 2.4.
        ({"extra_services_ratio": 0.47}, 5, 2.3499999999999996, 2.4),
    ],
)
def test_factor_points_are_pre_rounded_to_6_decimals(
    changes: dict[str, float], factor_index: int, binary64_points: float, expected: float
) -> None:
    result = score_features(_vector(PERFECT, **changes))
    item = result.breakdown[factor_index]
    assert item.raw_points == binary64_points
    assert round_half_up(item.raw_points, 1) == pytest.approx(expected - 0.1)  # sin round6 bajaría
    assert item.points == expected
    assert factor_points(item.raw_points) == expected


@pytest.mark.parametrize(
    ("raw_points", "expected"),
    [
        # floor(round6(p) * 10 + 0.5) / 10, a mano.
        (13.349999999999998, 13.4),
        (13.3499995, 13.4),  # round6 → 13.35
        (13.3499994, 13.3),  # round6 → 13.349999
        (0.04999995, 0.1),  # round6 → 0.05
        (0.0499994, 0.0),  # round6 → 0.049999
        (29.95, 30.0),
    ],
)
def test_factor_points_rounding_edges(raw_points: float, expected: float) -> None:
    assert factor_points(raw_points) == expected


# --- Pares: restricciones duras y una sola implementación ------------------


def test_a_pair_that_passes_is_scored_with_its_features(campuses: dict[str, Campus], schema: FeatureSchema) -> None:
    student, room = _student(), _room()
    result = score_pair(student, room, campuses)
    assert isinstance(result, ScoreResult)
    features = compute_pair_features(student, room, campuses, schema)
    assert result == score_features(feature_vector(features, schema))
    # budget_exact: renta = budget.max, distancia 0, servicios pedidos cubiertos, extras 1/9.
    assert result.raw_total == pytest.approx(30 + 25 + 20 + 10 + 10 + 5 / 9)
    assert result.match_score == 96


@pytest.mark.parametrize(
    ("student_changes", "room_changes", "violated"),
    [
        ({}, {"price_mxn": 4001}, ("HC01_budget",)),
        ({"living_with": "women"}, {"living_with": "men"}, ("HC02_living_with",)),
        ({"pets": True}, {"pets_allowed": False}, ("HC03_pets",)),
        (
            {"living_with": "women", "pets": True},
            {"price_mxn": 5000, "living_with": "men", "pets_allowed": False},
            ("HC01_budget", "HC02_living_with", "HC03_pets"),
        ),
    ],
)
def test_an_excluded_pair_has_no_score(
    student_changes: dict[str, Any],
    room_changes: dict[str, Any],
    violated: tuple[str, ...],
    campuses: dict[str, Campus],
) -> None:
    result = score_pair(_student(**student_changes), _room(**room_changes), campuses)
    assert isinstance(result, ExcludedPair)
    assert result.violated == violated
    output = result.to_dict()
    assert "matchScore" not in output
    assert "breakdown" not in output
    assert output == {
        "modelVersion": "baseline-1.0.0",
        "featureSchemaVersion": "2.0.0",
        "excluded": True,
        "violatedConstraints": list(violated),
    }
    assert json.loads(json.dumps(output)) == output


def test_an_excluded_pair_does_not_need_campus_coordinates() -> None:
    # Si el par se excluye no se calculan features (ni la distancia).
    result = score_pair(_student(), _room(price_mxn=4001), {})
    assert isinstance(result, ExcludedPair)
    with pytest.raises(PairInputError, match="sin coordenadas de campus"):
        score_pair(_student(), _room(), {})


@pytest.mark.parametrize("case", HARD_CASES["cases"], ids=lambda case: case["id"])
def test_hard_constraint_cases(case: dict[str, Any], campuses: dict[str, Campus]) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    if "expected_error" in case:
        with pytest.raises(PairInputError):
            score_pair(student, room, campuses)
        return
    result = score_pair(student, room, campuses)
    if case["expected"]["allowed"]:
        assert isinstance(result, ScoreResult)
        assert 0 <= result.match_score <= 100
    else:
        assert isinstance(result, ExcludedPair)
        assert list(result.violated) == case["expected"]["violated"]


@pytest.mark.parametrize("case", PAIR_CASES["cases"], ids=lambda case: case["id"])
def test_pair_feature_cases(case: dict[str, Any], campuses: dict[str, Campus], schema: FeatureSchema) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    if "expected_error" in case:
        with pytest.raises(IncompleteProfileError):
            score_pair(student, room, campuses)
        return
    result = score_pair(student, room, campuses)
    if isinstance(result, ExcludedPair):
        assert result.violated
        return
    expected = score_features([float(case["expected"][name]) for name in schema.feature_order])
    assert result.match_score == expected.match_score
    assert result.raw_total == pytest.approx(expected.raw_total, abs=1e-6)


# --- Ranking ----------------------------------------------------------------


def _scored(room_id: str, **changes: float) -> ScoredRoom:
    return ScoredRoom(room_id=room_id, score=score_features(_vector(PERFECT, **changes)))


def test_rank_orders_by_unrounded_total_descending() -> None:
    low = _scored("R0001", price_below_min=0.02)  # 99.4 → 99
    mid = _scored("R0002", price_below_min=0.01)  # 99.7 → 100
    high = _scored("R0003", price_below_min=0.005)  # 99.85 → 100
    ranked = rank([low, mid, high])
    assert [item.room_id for item in ranked] == ["R0003", "R0002", "R0001"]
    assert ranked[0].score.match_score == ranked[1].score.match_score == 100


def test_rank_breaks_ties_by_room_id_ascending() -> None:
    ids = ["R0300", "R0010", "R0200", "R0002", "R0100"]
    tied = [_scored(room_id, price_below_min=0.5) for room_id in ids]
    best = _scored("R9999")
    ranked = rank([*tied, best])
    assert [item.room_id for item in ranked] == ["R9999", "R0002", "R0010", "R0100", "R0200", "R0300"]
    shuffled = list(reversed([*tied, best]))
    assert rank(shuffled) == ranked


def test_rank_is_stable_against_input_order() -> None:
    rng = random.Random(f"score:{SEED}:rank")  # noqa: S311 - pruebas reproducibles
    candidates = [
        ScoredRoom(room_id=f"R{index:04d}", score=score_features(_vector(features)))
        for index, features in enumerate(list(_samples("rank"))[:300])
    ]
    candidates += [dataclasses.replace(item, room_id=item.room_id + "b") for item in candidates[:50]]
    expected = rank(candidates)
    for _ in range(5):
        rng.shuffle(candidates)
        assert rank(candidates) == expected
    totals = [item.score.raw_total for item in expected]
    assert totals == sorted(totals, reverse=True)


def test_rank_of_nothing_is_empty() -> None:
    assert rank([]) == []


def test_round_half_up_rejects_negative_decimals() -> None:
    with pytest.raises(ValueError, match="decimals"):
        round_half_up(1.0, -1)
