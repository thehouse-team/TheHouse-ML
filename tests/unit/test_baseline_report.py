"""ML-04 T06 (ML-04-R06): partes puras de la evaluación del modelo base (parámetros, exclusiones, órdenes)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from thehouse_ml.features import default_schema
from thehouse_ml.models import default_config
from thehouse_ml.models.baseline_report import (
    EVAL_PARAMS_RESOURCE,
    BudgetRange,
    EvaluationError,
    build_segments,
    evaluate_students,
    explain_factor,
    ideal_order,
    join_labels,
    load_eval_params,
    parse_eval_params,
    pick_example,
    random_order,
    score_frame,
    summarize_segment,
)

PARAMS_PATH = Path(__file__).resolve().parents[2] / "src" / "thehouse_ml" / "models" / EVAL_PARAMS_RESOURCE


def _raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    return data


def test_packaged_params_hold_the_supuestos_of_m_ml_034() -> None:
    params = load_eval_params()

    assert params.version == "1.0.0"
    assert params.decision.startswith("M-ML-034")
    assert params.split == "test"
    assert params.min_candidates == 2
    assert params.relevance_threshold == 2
    assert (params.resamples, params.confidence) == (1000, 0.95)
    assert params.bootstrap_seed != params.random_seed
    # Semillas distintas de estudiantes, cuartos, pares y etiquetas.
    assert {params.bootstrap_seed, params.random_seed}.isdisjoint({20261119, 20261905, 20261120, 20261121})
    assert [(b.min, b.max) for b in params.budget_ranges] == [(None, 2000), (2001, 3500), (3501, None)]
    assert load_eval_params(PARAMS_PATH) == params


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (500, "budget_le_2000"),
        (2000, "budget_le_2000"),
        (2001, "budget_2001_3500"),
        (3500, "budget_2001_3500"),
        (3501, "budget_gt_3500"),
        (9000, "budget_gt_3500"),
    ],
)
def test_budget_ranges_are_inclusive_and_cover_every_value(budget: int, expected: str) -> None:
    ranges = load_eval_params().budget_ranges
    assert [band.id for band in ranges if band.contains(budget)] == [expected]


def test_budget_range_without_limits_contains_everything() -> None:
    assert BudgetRange("all", "todo", None, None).contains(0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["parameters"].pop("split"), "split"),
        (lambda d: d["parameters"].update(extra={"value": 1, "basis": "SUPUESTO documentado"}), "desconocidos"),
        (lambda d: d["parameters"]["split"].update(basis="sin base"), "basis"),
        (lambda d: d["parameters"]["split"].update(value="holdout"), "split"),
        (lambda d: d["parameters"]["min_candidates"].update(value=1), "min_candidates"),
        (lambda d: d["parameters"]["relevance_threshold"].update(value=True), "relevance_threshold"),
        (lambda d: d["parameters"]["bootstrap.confidence"].update(value=1.0), "confidence"),
        (lambda d: d["parameters"]["budget_ranges"].update(value=[]), "budget_ranges"),
        (lambda d: d["parameters"]["budget_ranges"]["value"][1].update(min=2500), "contiguos"),
        (lambda d: d["parameters"]["budget_ranges"]["value"][0].update(min=0), "primer rango"),
        (lambda d: d["parameters"]["budget_ranges"]["value"][0].update(max="2000"), "tipos"),
        (lambda d: d["parameters"]["budget_ranges"]["value"][0].pop("label"), "id, label"),
    ],
)
def test_invalid_params_are_rejected(mutate: Any, message: str) -> None:
    data = copy.deepcopy(_raw())
    mutate(data)
    with pytest.raises(EvaluationError, match=message):
        parse_eval_params(data)


def test_params_without_structure_are_rejected() -> None:
    with pytest.raises(EvaluationError):
        parse_eval_params([])
    with pytest.raises(EvaluationError):
        parse_eval_params({"metadata": {}})


def test_random_order_is_seeded_per_student_and_independent_of_input_order() -> None:
    rooms = [f"R{i:05d}" for i in range(20)]
    first = random_order(rooms, "S0001", 7)
    assert first == random_order(list(reversed(rooms)), "S0001", 7)
    assert sorted(first) == rooms
    assert first != random_order(rooms, "S0002", 7)
    assert first != random_order(rooms, "S0001", 8)


def test_ideal_order_sorts_by_relevance_then_room_id() -> None:
    assert ideal_order({"R3": 1, "R1": 3, "R2": 1, "R0": 0}) == ["R1", "R2", "R3", "R0"]


# --- Exclusiones con un conjunto mínimo ---------------------------------------


def _frame() -> pd.DataFrame:
    """3 estudiantes de test: S1 con 3 candidatos, S2 con 1 y S3 sin ningún relevante (≥ 2)."""
    schema = default_schema()
    # Vector válido base: todo en el mejor valor salvo lo que se cambia por cuarto.
    base = dict.fromkeys(schema.feature_order, 0.0)
    base.update(services_coverage=1.0, place_type_match=1, property_type_match=1, pets_compatible=1)
    rows = [
        ("S1", "R1", 0.9, 3),  # peor puntaje, más relevante
        ("S1", "R2", 0.0, 0),
        ("S1", "R3", 0.5, 2),
        ("S2", "R1", 0.0, 3),
        ("S3", "R1", 0.0, 1),
        ("S3", "R2", 0.2, 0),
    ]
    records = []
    for student_id, room_id, price_below_min, relevance in rows:
        record: dict[str, Any] = {**base, "price_below_min": price_below_min}
        record.update(
            student_id=student_id,
            room_id=room_id,
            split="test",
            relevance=relevance,
            university_code="UTTT" if student_id != "S3" else "UCAN",
            seed_verified=student_id != "S3",
        )
        records.append(record)
    return pd.DataFrame(records)


def test_students_with_one_candidate_or_no_relevant_room_are_excluded_and_counted() -> None:
    frame = _frame()
    params = load_eval_params()
    results = score_frame(frame, default_schema(), default_config())
    students = evaluate_students(frame, results, {"S1": 1500, "S2": 3000, "S3": 5000}, params)

    by_id = {item.student_id: item for item in students}
    assert by_id["S1"].included
    assert by_id["S1"].baseline_order == ("R2", "R3", "R1")  # sin price_below_min primero
    assert not by_id["S2"].included and by_id["S2"].candidates == 1
    assert not by_id["S3"].included and by_id["S3"].relevant == 0
    segments = build_segments(students, default_schema(), params)
    total = segments[0]
    assert (len(total.included), total.excluded_few_candidates, total.excluded_no_relevant) == (1, 1, 1)
    # S1: relevancias en orden [0, 2, 3] → AP = (1/2 + 2/3) / 2
    metrics = by_id["S1"].metrics
    assert metrics is not None
    assert metrics["baseline"].average_precision == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics["ideal"].ndcg_10 == pytest.approx(1.0)
    summary = summarize_segment(total, params)
    assert summary is not None
    assert summary.intervals["baseline"]["map"].mean == pytest.approx((1 / 2 + 2 / 3) / 2)
    # Un segmento sin estudiantes evaluados no tiene resumen.
    ucan = next(segment for segment in segments if segment.id == "campus:UCAN")
    assert summarize_segment(ucan, params) is None
    assert pick_example(students, params) is None  # ninguno con 10 candidatos


def test_missing_student_budget_is_an_error() -> None:
    frame = _frame()
    results = score_frame(frame, default_schema(), default_config())
    with pytest.raises(EvaluationError, match="S1"):
        evaluate_students(frame, results, {}, load_eval_params())


def test_join_labels_requires_one_label_per_pair_with_the_same_split() -> None:
    pairs = pd.DataFrame({"student_id": ["S1", "S1"], "room_id": ["R1", "R2"], "split": ["test", "test"]})
    labels = pairs.assign(relevance=[3, 0], synthetic=True)
    joined = join_labels(pairs, labels)
    assert joined["relevance"].tolist() == [3, 0]

    with pytest.raises(EvaluationError, match="duplicados"):
        join_labels(pd.concat([pairs, pairs]), labels)
    with pytest.raises(EvaluationError, match="partición"):
        join_labels(pairs, labels.assign(split="train"))
    with pytest.raises(EvaluationError, match="synthetic"):
        join_labels(pairs, labels.assign(synthetic=False))
    with pytest.raises(EvaluationError, match="sin etiqueta"):
        join_labels(pairs, labels.iloc[:1])


# --- Explicaciones -------------------------------------------------------------

STUDENT = {
    "budget.min": 1000,
    "budget.max": 2000,
    "essentialServices": ["wifi", "water"],
    "placeType": "room",
    "propertyTypes": ["house"],
}
ROOM = {
    "price_mxn": 800,
    "services": ["wifi", "gas", "laundry"],
    "placeType": "room",
    "propertyType": "apartment",
    "_distance_km": 1.25,
}


@pytest.mark.parametrize(
    ("factor", "value", "student", "expected"),
    [
        ("budget", 0.2, STUDENT, "renta $800, 20 % por debajo de su mínimo $1,000"),
        ("budget", 0.0, {**STUDENT, "budget.min": None}, "renta $800 dentro de su rango (sin mínimo a $2,000)"),
        ("distance", 0.3, STUDENT, "1.2 km del campus, `distance_ratio` 0.30 (≤ 0.5 de su máximo: puntos completos)"),
        ("distance", 0.75, STUDENT, "entre 0.5 y 1 de su máximo: decae"),
        ("distance", 1.5, STUDENT, "más lejos que su máximo: 0 puntos"),
        ("essential_services", 0.5, STUDENT, "cubre 1 de 2 servicios pedidos"),
        ("essential_services", 1.0, {**STUDENT, "essentialServices": None}, "no pidió servicios esenciales"),
        ("place_type", 1.0, STUDENT, "`room` coincide con lo pedido (`room`)"),
        ("place_type", 0.0, {**STUDENT, "placeType": "shared"}, "no coincide con"),
        ("property_type", 0.0, STUDENT, "`apartment` no está en lo pedido (`house`)"),
        ("property_type", 1.0, {**STUDENT, "propertyTypes": None}, "sin preferencia de inmueble"),
        ("extras", 2 / 9, STUDENT, "2 servicios extra no pedidos (`extra_services_ratio` 0.22)"),
        ("extras", 0.1, {**STUDENT, "essentialServices": ["wifi", "gas"]}, "1 servicio extra no pedido"),
    ],
)
def test_explain_factor_describes_each_factor(
    factor: str, value: float, student: dict[str, Any], expected: str
) -> None:
    assert expected in explain_factor(factor, value, student, ROOM)


def test_explain_factor_treats_nan_lists_as_empty() -> None:
    assert explain_factor("essential_services", 1.0, {**STUDENT, "essentialServices": float("nan")}, ROOM) == (
        "no pidió servicios esenciales"
    )
