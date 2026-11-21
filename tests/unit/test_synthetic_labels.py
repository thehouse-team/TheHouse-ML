"""ML-04 T04 (ML-04-R04): simulador de etiquetas sintéticas y `labels_params.json`, sin red ni archivos de datos."""

from __future__ import annotations

import copy
import json
import math
import random
from importlib import resources
from typing import Any

import pytest

from thehouse_ml.features import default_schema
from thehouse_ml.models import default_config
from thehouse_ml.synthetic.labels import (
    FACTORS,
    LABELS_PARAMS_RESOURCE,
    LabelInput,
    LabelInputError,
    LabelParams,
    LabelParamsError,
    UtilityCurve,
    grade_cutoffs,
    hidden_weights,
    load_params,
    parse_params,
    relevance_by_position,
    simulate_labels,
    utility,
)
from thehouse_ml.synthetic.labels_stage import spearman

SENSITIVE_OR_PERSONAL = ("gender", "genero", "género", "living", "name", "nombre", "email", "phone", "age", "edad")


def _raw() -> dict[str, Any]:
    text = resources.files("thehouse_ml.synthetic").joinpath(LABELS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@pytest.fixture(scope="module")
def params() -> LabelParams:
    return load_params(default_schema())


def _pair(student_id: str, room_id: str, **overrides: float) -> LabelInput:
    features = {
        "price_to_budget": 0.5,
        "effective_distance_km": 2.0,
        "services_coverage": 0.5,
        "place_type_match": 1.0,
        "property_type_match": 1.0,
        "extra_services_ratio": 0.5,
    }
    features.update(overrides)
    return LabelInput(student_id, room_id, features)


def _synthetic_pairs(students: int = 12, seed: int = 7) -> list[LabelInput]:
    """Pares con features aleatorias; el estudiante k tiene k candidatos (1, 2, 3, …)."""
    rng = random.Random(seed)  # noqa: S311 - pruebas reproducibles
    pairs: list[LabelInput] = []
    for index in range(1, students + 1):
        for room in range(index * 3 if index > 3 else index):
            pairs.append(
                _pair(
                    f"S{index:04d}",
                    f"R{room:05d}",
                    price_to_budget=rng.random(),
                    effective_distance_km=rng.uniform(0, 30),
                    services_coverage=rng.random(),
                    place_type_match=float(rng.randint(0, 1)),
                    property_type_match=float(rng.randint(0, 1)),
                    extra_services_ratio=rng.random(),
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# labels_params.json
# ---------------------------------------------------------------------------


def test_packaged_params_load_and_declare_synthetic(params: LabelParams) -> None:
    assert params.version == "1.0.0"
    assert params.metadata["synthetic"] is True
    assert all(parameter.basis.startswith("SUPUESTO documentado") for parameter in params.parameters.values())
    assert params.multiplier_sigma == 0.3
    assert params.noise_sigma == 5.0
    assert params.spearman_range == (0.3, 0.9)


def test_seed_is_fixed_and_distinct_from_students_rooms_and_pairs(params: LabelParams) -> None:
    seeds = {}
    for name in ("students_params.json", "rooms_params.json", "pairs_params.json"):
        text = resources.files("thehouse_ml.synthetic").joinpath(name).read_text(encoding="utf-8")
        seeds[name] = json.loads(text)["parameters"]["seed"]["value"]
    assert params.seed == 20261121
    assert params.seed not in seeds.values()


def test_hidden_base_weights_differ_from_the_baseline_and_sum_to_100(params: LabelParams) -> None:
    baseline = default_config().weights
    assert list(params.base_weights) == list(FACTORS)
    assert set(baseline) == set(FACTORS)
    assert math.isclose(sum(params.base_weights.values()), 100.0)
    assert params.base_weights != baseline
    differing = [factor for factor in FACTORS if params.base_weights[factor] != baseline[factor]]
    assert len(differing) >= 3
    assert params.base_weights == {
        "budget": 20.0,
        "distance": 35.0,
        "essential_services": 20.0,
        "place_type": 10.0,
        "property_type": 5.0,
        "extras": 10.0,
    }


def test_utility_shape_differs_from_the_baseline_in_budget_and_distance(params: LabelParams) -> None:
    config = default_config()
    curves = params.curves
    assert curves["budget"] == UtilityCurve("price_to_budget", "one_minus")
    assert curves["distance"] == UtilityCurve("effective_distance_km", "exp_decay", 3.0)
    non_scoring = {item.feature for item in config.non_scoring_features}
    # Las dos features que la utilidad usa en presupuesto y distancia no puntúan en el baseline.
    assert {"price_to_budget", "effective_distance_km"} <= non_scoring
    differing = [factor for factor in FACTORS if curves[factor].feature != config.factor(factor).feature]
    assert differing == ["budget", "distance"]


def test_curves_are_bounded_in_zero_one() -> None:
    assert UtilityCurve("price_to_budget", "one_minus").evaluate(0.25) == 0.75
    assert UtilityCurve("price_to_budget", "one_minus").evaluate(1.5) == 0.0
    assert UtilityCurve("services_coverage", "identity").evaluate(0.4) == 0.4
    decay = UtilityCurve("effective_distance_km", "exp_decay", 3.0)
    assert decay.evaluate(0.0) == 1.0
    assert math.isclose(decay.evaluate(3.0), math.exp(-1))
    assert decay.evaluate(1.0) > decay.evaluate(2.0) > decay.evaluate(10.0) > 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["parameters"]["utility.base_weights"]["value"].update(budget=25), "deben sumar 100"),
        (lambda d: d["parameters"]["utility.base_weights"]["value"].pop("extras"), "utility.base_weights"),
        (lambda d: d["parameters"]["utility.curves"]["value"]["budget"].update(shape="cubic"), "shape"),
        (
            lambda d: d["parameters"]["utility.curves"]["value"]["budget"].update(feature="nope"),
            "no está en el esquema",
        ),
        (lambda d: d["parameters"]["utility.curves"]["value"]["distance"].pop("scale_km"), "scale_km"),
        (lambda d: d["parameters"]["utility.curves"]["value"]["budget"].update(scale_km=2), "solo aplica"),
        (lambda d: d["parameters"]["relevance.bands"]["value"].update({"1": 0.9}), "más de 1"),
        (lambda d: d["parameters"]["relevance.bands"]["value"].update({"1": 0}), "(0, 1)"),
        (lambda d: d["parameters"]["check.spearman_range"].update(value=[0.9, 0.3]), "spearman_range"),
        (lambda d: d["parameters"]["noise.sigma"].update(value=-1), "noise.sigma"),
        (lambda d: d["parameters"]["seed"].update(value=-3), "seed"),
        (lambda d: d["parameters"]["seed"].update(basis="sin base"), "basis"),
        (lambda d: d["parameters"].pop("noise.sigma"), "faltan"),
        (lambda d: d["parameters"].update(extra={"value": 1, "basis": "SUPUESTO documentado"}), "desconocidos"),
        (lambda d: d["metadata"].update(synthetic=False), "metadata.synthetic"),
    ],
)
def test_invalid_params_are_rejected(mutate: Any, message: str) -> None:
    data = copy.deepcopy(_raw())
    mutate(data)
    with pytest.raises(LabelParamsError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_params(data, default_schema())


def test_non_object_params_are_rejected() -> None:
    with pytest.raises(LabelParamsError, match="raíz"):
        parse_params([], default_schema())
    with pytest.raises(LabelParamsError, match="parameters"):
        parse_params({"metadata": {"synthetic": True, "params_version": "1.0.0"}}, default_schema())


# ---------------------------------------------------------------------------
# Relevancia por posición
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, [3]),
        (2, [3, 0]),
        (3, [3, 0, 0]),
        (4, [3, 1, 0, 0]),
        (5, [3, 1, 0, 0, 0]),
        (6, [3, 2, 0, 0, 0, 0]),
        (10, [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_relevance_for_few_candidates(params: LabelParams, n: int, expected: list[int]) -> None:
    assert relevance_by_position(n, params) == expected


@pytest.mark.parametrize("n", range(1, 251))
def test_relevance_counts_follow_the_cumulative_ceiling_cuts(params: LabelParams, n: int) -> None:
    grades = relevance_by_position(n, params)
    top3, top2, top1 = math.ceil(n / 10), math.ceil(2 * n / 10), math.ceil(3 * n / 10)
    assert grades == sorted(grades, reverse=True)
    assert grades.count(3) == top3 >= 1
    assert grades.count(2) == top2 - top3
    assert grades.count(1) == top1 - top2
    assert grades.count(0) == n - top1
    assert grade_cutoffs(n, params) == {3: top3, 2: top2, 1: top1}


def test_two_hundred_candidates_split_20_20_20_140(params: LabelParams) -> None:
    grades = relevance_by_position(200, params)
    assert [grades.count(grade) for grade in (3, 2, 1, 0)] == [20, 20, 20, 140]


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------


def test_hidden_weights_are_renormalized_per_student(params: LabelParams) -> None:
    result = simulate_labels(_synthetic_pairs(), params)
    weight_sets = [dict(item.weights) for item in result.students]
    for weights in weight_sets:
        assert list(weights) == list(FACTORS)
        assert math.isclose(sum(weights.values()), 100.0, abs_tol=1e-9)
        assert all(value > 0 for value in weights.values())
    assert len({tuple(weights.values()) for weights in weight_sets}) == len(weight_sets)  # cada estudiante distinto


def test_hidden_weights_use_the_lognormal_multipliers(params: LabelParams) -> None:
    rng = random.Random("labels:20261121:S0001")  # noqa: S311 - pruebas reproducibles
    z = [rng.gauss(0.0, 1.0) for _ in FACTORS]
    raw = {
        factor: params.base_weights[factor] * math.exp(0.3 * value) for factor, value in zip(FACTORS, z, strict=True)
    }
    expected = {factor: 100 * value / sum(raw.values()) for factor, value in raw.items()}
    assert hidden_weights(random.Random("labels:20261121:S0001"), params) == expected  # noqa: S311


def test_without_noise_or_variation_utility_is_the_weighted_sum(params: LabelParams) -> None:
    exact = params.replace("noise.sigma", 0.0).replace("weights.multiplier_sigma", 0.0)
    pair = _pair("S0001", "R00001", price_to_budget=0.4, effective_distance_km=3.0)
    result = simulate_labels([pair], exact)
    expected = 20 * 0.6 + 35 * math.exp(-1) + 20 * 0.5 + 10 * 1 + 5 * 1 + 10 * 0.5
    assert math.isclose(result.labels[0].utility, expected)
    assert math.isclose(utility(pair, exact.base_weights, exact.curves), expected)


def test_same_seed_gives_the_same_labels_and_another_seed_changes_them(params: LabelParams) -> None:
    pairs = _synthetic_pairs()
    first = simulate_labels(pairs, params)
    assert simulate_labels(pairs, params) == first
    shuffled = list(pairs)
    random.Random(3).shuffle(shuffled)  # noqa: S311 - pruebas reproducibles
    assert simulate_labels(shuffled, params) == first  # no depende del orden de entrada
    other = simulate_labels(pairs, params.with_seed(1))
    assert [item.relevance for item in other.labels] != [item.relevance for item in first.labels]


def test_exactly_one_label_per_input_pair_in_order(params: LabelParams) -> None:
    pairs = _synthetic_pairs()
    result = simulate_labels(pairs, params)
    keys = [(item.student_id, item.room_id) for item in result.labels]
    assert keys == sorted((pair.student_id, pair.room_id) for pair in pairs)
    assert len(set(keys)) == len(pairs)
    assert {item.relevance for item in result.labels} <= {0, 1, 2, 3}


def test_relevance_per_student_follows_utility_and_the_cuts(params: LabelParams) -> None:
    result = simulate_labels(_synthetic_pairs(), params)
    by_student: dict[str, list[tuple[float, str, int]]] = {}
    for item in result.labels:
        by_student.setdefault(item.student_id, []).append((item.utility, item.room_id, item.relevance))
    for rows in by_student.values():
        ranked = sorted(rows, key=lambda row: (-row[0], row[1]))
        assert [row[2] for row in ranked] == relevance_by_position(len(rows), params)


def test_ties_in_utility_are_broken_by_room_id(params: LabelParams) -> None:
    flat = params.replace("noise.sigma", 0.0)
    pairs = [_pair("S0001", f"R{index:05d}") for index in (4, 2, 3, 1, 5)]
    result = simulate_labels(pairs, flat)
    assert [(item.room_id, item.relevance) for item in result.labels] == [
        ("R00001", 3),
        ("R00002", 1),
        ("R00003", 0),
        ("R00004", 0),
        ("R00005", 0),
    ]


def test_duplicate_pairs_and_bad_features_are_rejected(params: LabelParams) -> None:
    with pytest.raises(LabelInputError, match="duplicado"):
        simulate_labels([_pair("S1", "R1"), _pair("S1", "R1")], params)
    with pytest.raises(LabelInputError, match="falta la feature"):
        simulate_labels([LabelInput("S1", "R1", {"price_to_budget": 0.5})], params)
    with pytest.raises(LabelInputError, match="finito"):
        simulate_labels([_pair("S1", "R1", effective_distance_km=math.nan)], params)


def test_simulator_has_no_personal_or_gender_fields(params: LabelParams) -> None:
    result = simulate_labels(_synthetic_pairs(), params)
    fields = set(LabelInput.__dataclass_fields__) | set(type(result.labels[0]).__dataclass_fields__)
    used_features = {curve.feature for curve in params.curves.values()}
    for name in fields | used_features:
        assert not set(name.lower().split("_")) & set(SENSITIVE_OR_PERSONAL), name


def test_simulator_does_not_import_the_baseline() -> None:
    source = resources.files("thehouse_ml.synthetic").joinpath("labels.py").read_text(encoding="utf-8")
    assert "thehouse_ml.models" not in source.split('"""', 2)[2]


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def test_spearman_known_values() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Con empates usa rangos promedio: rangos x = 1, 2.5, 2.5, 4 e y = 1, 2, 3, 4.
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1], [1]))
    with pytest.raises(ValueError, match="longitud"):
        spearman([1, 2], [1])
