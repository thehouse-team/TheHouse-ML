"""ML-04 T06 (ML-04-R06): métricas de ranking y bootstrap con ejemplos calculados a mano.

Los esperados se escriben con la fórmula a mano (sumas de `ganancia / log2`),
no con las funciones del módulo.
"""

from __future__ import annotations

import math
import random

import pytest

from thehouse_ml.evaluation import (
    METRIC_NAMES,
    RankingInputError,
    average_precision,
    bootstrap_means,
    dcg_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    student_metrics,
)

TOL = 1e-12


def _rng(key: str) -> random.Random:
    return random.Random(key)  # noqa: S311 - semilla fija en pruebas


# --- DCG / NDCG -------------------------------------------------------------


def test_dcg_uses_graded_gain_and_log2_discount() -> None:
    # [3, 2, 0, 1]: 7/log2(2) + 3/log2(3) + 0/log2(4) + 1/log2(5)
    expected = 7 / 1 + 3 / math.log2(3) + 0 + 1 / math.log2(5)
    assert dcg_at_k([3, 2, 0, 1], 10) == pytest.approx(expected, abs=TOL)


def test_ndcg_divides_by_the_ideal_order_of_the_same_candidates() -> None:
    # Ideal [3, 2, 1, 0]: 7 + 3/log2(3) + 1/log2(4) = 7 + 1.8928 + 0.5
    dcg = 7 + 3 / math.log2(3) + 1 / math.log2(5)
    ideal = 7 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k([3, 2, 0, 1], 10) == pytest.approx(dcg / ideal, abs=TOL)
    assert ndcg_at_k([3, 2, 0, 1], 10) == pytest.approx(0.992620, abs=1e-6)


def test_ndcg_of_the_best_item_in_second_place_is_one_over_log2_3() -> None:
    # [0, 3]: DCG = 7/log2(3); ideal [3, 0] = 7 → NDCG = 1/log2(3) ≈ 0.630930
    assert ndcg_at_k([0, 3], 10) == pytest.approx(1 / math.log2(3), abs=TOL)
    assert ndcg_at_k([0, 3], 10) == pytest.approx(0.630930, abs=1e-6)


def test_ndcg_at_k_cuts_both_the_ranking_and_the_ideal_at_k() -> None:
    ranking = [0, 0, 0, 0, 0, 3]  # el único relevante queda en la posición 6
    assert ndcg_at_k(ranking, 5) == 0.0
    # @10: 7/log2(7) frente al ideal 7/log2(2) = 7
    assert ndcg_at_k(ranking, 10) == pytest.approx(1 / math.log2(7), abs=TOL)


def test_perfect_order_gives_ndcg_one_and_order_matters() -> None:
    assert ndcg_at_k([3, 2, 1, 0, 0], 5) == pytest.approx(1.0, abs=TOL)
    assert ndcg_at_k([0, 0, 1, 2, 3], 5) < ndcg_at_k([3, 0, 0, 1, 2], 5)  # type: ignore[operator]


def test_ndcg_is_undefined_without_any_gain() -> None:
    assert ndcg_at_k([0, 0, 0], 10) is None
    assert ndcg_at_k([], 10) is None


# --- Precision@k ------------------------------------------------------------


def test_precision_at_5_counts_relevance_at_least_the_threshold() -> None:
    # Top 5 de [3, 0, 2, 1, 0, 2]: relevantes (≥ 2) 3 y 2 → 2/5
    assert precision_at_k([3, 0, 2, 1, 0, 2], 5, threshold=2) == pytest.approx(0.4, abs=TOL)
    assert precision_at_k([3, 0, 2, 1, 0, 2], 5, threshold=1) == pytest.approx(0.6, abs=TOL)


def test_precision_divides_by_k_even_with_fewer_candidates() -> None:
    # 2 candidatos, 1 relevante: 1/5, no 1/2 (denominador fijo, como trec_eval)
    assert precision_at_k([2, 0], 5, threshold=2) == pytest.approx(0.2, abs=TOL)


# --- AP / MAP ---------------------------------------------------------------


def test_average_precision_by_hand() -> None:
    # Relevantes (≥ 2) en las posiciones 1, 3 y 6: (1/1 + 2/3 + 3/6) / 3 = 13/18
    assert average_precision([3, 0, 2, 1, 0, 2], threshold=2) == pytest.approx(13 / 18, abs=TOL)


def test_average_precision_uses_the_whole_list() -> None:
    # El único relevante en la posición 8: AP = 1/8
    assert average_precision([0] * 7 + [2], threshold=2) == pytest.approx(1 / 8, abs=TOL)


def test_student_without_any_relevant_room_has_undefined_ap_and_is_excluded() -> None:
    # Tratamiento documentado (M-ML-034): AP no definido → `None`; quien evalúa
    # excluye al estudiante y lo cuenta. NDCG sí está definido (hay ganancia de 1).
    ranking = [1, 0, 1]
    assert average_precision(ranking, threshold=2) is None
    assert ndcg_at_k(ranking, 10) == pytest.approx((1 + 1 / 2) / (1 + 1 / math.log2(3)), abs=TOL)
    assert precision_at_k(ranking, 5, threshold=2) == 0.0
    assert student_metrics(ranking, threshold=2) is None
    assert student_metrics([0, 0], threshold=2) is None


def test_student_metrics_bundle_in_the_documented_order() -> None:
    metrics = student_metrics([0, 3, 2, 0, 0, 0], threshold=2)

    assert metrics is not None
    ideal = 7 + 3 / math.log2(3)
    dcg = 7 / math.log2(3) + 3 / math.log2(4)
    assert metrics.ndcg_10 == pytest.approx(dcg / ideal, abs=TOL)
    assert metrics.ndcg_5 == pytest.approx(dcg / ideal, abs=TOL)
    assert metrics.precision_5 == pytest.approx(2 / 5, abs=TOL)
    assert metrics.average_precision == pytest.approx((1 / 2 + 2 / 3) / 2, abs=TOL)
    assert tuple(metrics.as_dict()) == METRIC_NAMES == ("ndcg@10", "ndcg@5", "precision@5", "map")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: dcg_at_k([1, -1], 5), "relevancia"),
        (lambda: dcg_at_k([1, 1.5], 5), "relevancia"),  # type: ignore[list-item]
        (lambda: dcg_at_k([True], 5), "relevancia"),
        (lambda: dcg_at_k([1], 0), "k inválido"),
        (lambda: precision_at_k([1], 5, threshold=0), "umbral"),
        (lambda: average_precision([1], threshold=0), "umbral"),
    ],
)
def test_invalid_inputs_are_rejected(call: object, message: str) -> None:
    with pytest.raises(RankingInputError, match=message):
        call()  # type: ignore[operator]


# --- Percentil y bootstrap ---------------------------------------------------


def test_percentile_interpolates_linearly() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.25) == pytest.approx(1.75, abs=TOL)  # posición 0.75
    assert percentile(values, 0.5) == pytest.approx(2.5, abs=TOL)
    # 1,000 medias: p2.5 en la posición 24.975 y p97.5 en 974.025
    thousand = [float(i) for i in range(1000)]
    assert percentile(thousand, 0.025) == pytest.approx(24.975, abs=1e-9)
    assert percentile(thousand, 0.975) == pytest.approx(974.025, abs=1e-9)
    with pytest.raises(RankingInputError):
        percentile([], 0.5)
    with pytest.raises(RankingInputError):
        percentile(values, 1.5)


def test_bootstrap_of_a_constant_series_is_a_point_interval() -> None:
    result = bootstrap_means({"x": [0.5] * 7}, resamples=50, confidence=0.95, rng=_rng("t"))

    assert result["x"].mean == 0.5
    assert result["x"].low == result["x"].high == pytest.approx(0.5, abs=TOL)


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    values = [float(i % 5) for i in range(40)]
    first = bootstrap_means({"x": values}, resamples=200, confidence=0.95, rng=_rng("seed"))
    second = bootstrap_means({"x": values}, resamples=200, confidence=0.95, rng=_rng("seed"))
    other = bootstrap_means({"x": values}, resamples=200, confidence=0.95, rng=_rng("other"))

    assert first == second
    assert first != other
    assert first["x"].mean == pytest.approx(sum(values) / len(values), abs=TOL)
    assert first["x"].low < first["x"].mean < first["x"].high


def test_bootstrap_is_paired_across_series() -> None:
    # Con los mismos índices en cada remuestreo, la media remuestreada de a - b
    # es exactamente la diferencia de medias, así que una diferencia constante
    # da un intervalo puntual aunque a y b varíen.
    a = [0.1, 0.9, 0.4, 0.7, 0.2]
    b = [value - 0.25 for value in a]
    diff = [x - y for x, y in zip(a, b, strict=True)]
    result = bootstrap_means({"a": a, "b": b, "diff": diff}, resamples=300, confidence=0.9, rng=_rng("p"))

    assert result["diff"].low == pytest.approx(0.25, abs=1e-12)
    assert result["diff"].high == pytest.approx(0.25, abs=1e-12)
    assert result["a"].high > result["a"].low


def test_bootstrap_rejects_bad_parameters() -> None:
    rng = _rng("x")
    with pytest.raises(RankingInputError):
        bootstrap_means({"x": [1.0]}, resamples=0, confidence=0.95, rng=rng)
    with pytest.raises(RankingInputError):
        bootstrap_means({"x": [1.0]}, resamples=10, confidence=1.0, rng=rng)
    with pytest.raises(RankingInputError):
        bootstrap_means({"x": [1.0], "y": [1.0, 2.0]}, resamples=10, confidence=0.95, rng=rng)
    with pytest.raises(RankingInputError):
        bootstrap_means({"x": []}, resamples=10, confidence=0.95, rng=rng)
