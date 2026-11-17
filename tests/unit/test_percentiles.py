"""ML-02 T03: percentil ponderado según la definición documentada (M-ML-014)."""

from __future__ import annotations

import pytest

from thehouse_ml.percentiles import weighted_percentiles, weighted_quantile


def test_weighted_quantile_matches_hand_calculation() -> None:
    # Valores 10, 20, 30 con factores 1, 1, 2; total = 4.
    # p25: objetivo 1 -> acumulado 1 en 10. p50: objetivo 2 -> acumulado 2 en 20.
    # p90: objetivo 3.6 -> acumulado 1, 2, 4 -> el menor valor que lo alcanza es 30.
    values = [10.0, 20.0, 30.0]
    weights = [1.0, 1.0, 2.0]

    assert weighted_quantile(values, weights, 0.25) == 10.0
    assert weighted_quantile(values, weights, 0.50) == 20.0
    assert weighted_quantile(values, weights, 0.90) == 30.0


def test_tied_values_are_grouped_before_accumulating() -> None:
    # 10 aparece dos veces con pesos 1 y 2: su peso acumulado es 3 de un total de 4.
    assert weighted_quantile([10.0, 10.0, 20.0], [1.0, 2.0, 1.0], 0.50) == 10.0
    assert weighted_quantile([10.0, 10.0, 20.0], [1.0, 2.0, 1.0], 0.90) == 20.0


def test_weighted_percentiles_are_ordered() -> None:
    result = weighted_percentiles([500.0, 100.0, 300.0, 200.0], [3.0, 1.0, 2.0, 4.0])

    assert result["p10"] <= result["p25"] <= result["p50"] <= result["p75"] <= result["p90"]


def test_q_zero_returns_smallest_value() -> None:
    assert weighted_quantile([30.0, 10.0, 20.0], [5.0, 1.0, 2.0], 0.0) == 10.0


@pytest.mark.parametrize(
    ("values", "weights", "q"),
    [
        ([], [], 0.5),
        ([1.0], [], 0.5),
        ([1.0, 2.0], [1.0, -1.0], 0.5),
        ([1.0, 2.0], [0.0, 0.0], 0.5),
        ([1.0, 2.0], [1.0, 1.0], 1.5),
    ],
)
def test_invalid_inputs_fail(values: list[float], weights: list[float], q: float) -> None:
    with pytest.raises(ValueError):
        weighted_quantile(values, weights, q)
