"""ML-06 T02 (ML-06-R02): paridad Python ↔ onnxruntime con un modelo sintético (spec ML-06 §6.3).

No necesita `models/`: las pruebas con el modelo oficial `logreg-1.0.0` están en
`tests/integration/test_onnx_parity_official.py`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx
import pytest

from thehouse_ml.export.onnx_graph import build_onnx_model, graph_constants, model_metadata, serialize_model
from thehouse_ml.export.parity import (
    GRID_OFFSETS,
    RAW_TOLERANCE,
    backend_factor_points,
    backend_match_score,
    backend_round,
    check_parity,
    clip_matches_np_clip,
    out_of_range_grid,
    python_reference,
    run_onnx_clipped,
)
from thehouse_ml.features import default_schema
from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_score import factor_points, match_score
from thehouse_ml.models.train import LOGREG_FORMAT, LogRegModel
from thehouse_ml.models.trained_breakdown import BreakdownModel, build_breakdown_model

LOWER = [0.0] * 10
UPPER = [30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 1.0, 1.0]
INTEGER_COLUMNS = (4, 5, 6)  # place_type_match, property_type_match, pets_compatible


def logreg_document(coef: list[float] | None = None) -> dict[str, Any]:
    """`logreg.json` sintético con la forma de `thehouse-ml/logreg-json/1` (signos mixtos, pets en 0)."""
    return {
        "synthetic": True,
        "format": LOGREG_FORMAT,
        "feature_order": list(default_schema().feature_order),
        "config": {"transform": "standard", "C": 0.01},
        "clip": {"lower": LOWER, "upper": UPPER},
        "transform": {
            "type": "standard",
            "mean": [5.2, 2.5, 0.55, 0.73, 0.3, 0.7, 1.0, 1.33, 0.14, 0.48],
            "scale": [6.3, 3.7, 0.23, 0.18, 0.46, 0.46, 1.0, 3.42, 0.22, 0.21],
        },
        "coef": coef or [-3.35, -1.87, -0.996, 0.566, 1.006, 0.36, 0.0, -0.1, -0.0123, 0.439],
        "intercept": -4.13,
    }


def onnx_bytes(model: BreakdownModel) -> bytes:
    constants = graph_constants(model)
    return serialize_model(build_onnx_model(constants, model_metadata(model, constants, "0" * 64, "abc", "ci")))


@pytest.fixture(scope="module")
def breakdown() -> BreakdownModel:
    return build_breakdown_model(LogRegModel.from_dict(logreg_document()))


@pytest.fixture(scope="module")
def model_bytes(breakdown: BreakdownModel) -> bytes:
    return onnx_bytes(breakdown)


def in_range_vectors(seed: int = 5, rows: int = 400) -> np.ndarray:
    """Vectores válidos para el esquema 2.0.0 (dentro del `range`; enteros donde aplica)."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.3, size=(rows, 10)) * np.asarray(UPPER)
    for column in (2, 3, 4, 5, 6, 8, 9):
        values[:, column] = np.clip(values[:, column], 0.0, 1.0)
    for column in INTEGER_COLUMNS:
        values[:, column] = np.round(values[:, column])
    return values


# ---------------------------------------------------------------------------
# Redondeo del backend (spec §6.1 paso 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1.25, 1, 1.3),
        (86.5, 0, 87.0),
        (0.05, 1, 0.1),
        (13.35, 1, 13.4),
        (2.0000004, 6, 2.0),
        (-1.25, 1, -1.3),  # AwayFromZero: la magnitud se redondea hacia arriba
        (-2.4518376962080584e-16, 6, 0.0),
    ],
)
def test_backend_round_is_dotnet_away_from_zero(value: float, decimals: int, expected: float) -> None:
    assert backend_round(value, decimals) == expected


def test_backend_round_normalizes_negative_zero() -> None:
    assert math.copysign(1.0, backend_round(-1e-16, 6)) == 1.0
    assert math.copysign(1.0, backend_round(-0.0, 1)) == 1.0


def test_backend_round_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finito"):
        backend_round(math.nan, 1)


def test_backend_rounding_agrees_with_the_package_for_non_negative_values() -> None:
    config = default_config()
    rng = np.random.default_rng(13)
    values = [*rng.uniform(0.0, 100.0, size=2000), 68.49999999999999, 13.349999999999998, 0.0, 99.5, 100.0]
    for value in values:
        assert backend_factor_points(float(value), config) == factor_points(float(value), config), value
        assert backend_match_score(float(value), config) == match_score(float(value), config), value


def test_backend_rounding_applies_round6_before_the_final_rounding() -> None:
    config = default_config()

    # Bordes de M-ML-033: sin round6, 68.49999999999999 → 68 y 13.349999999999998 → 13.3.
    assert backend_match_score(68.49999999999999, config) == 69
    assert backend_factor_points(13.349999999999998, config) == 13.4
    assert backend_match_score(68.4999994, config) == 68
    assert backend_match_score(68.5000004, config) == 69


def test_backend_rounding_of_tiny_negative_outputs_is_zero() -> None:
    config = default_config()

    assert backend_factor_points(-2.4518376962080584e-16, config) == 0.0
    assert backend_match_score(-1e-15, config) == 0


# ---------------------------------------------------------------------------
# Rejilla fuera del training_range
# ---------------------------------------------------------------------------


def test_grid_is_deterministic_and_has_the_documented_size() -> None:
    grid = out_of_range_grid(LOWER, UPPER)
    one_at_a_time = 3 * 10 * (2 + 2 * len(GRID_OFFSETS))
    all_at_once = 4 * len(GRID_OFFSETS)

    assert grid.shape == (one_at_a_time + all_at_once, 10)
    assert np.array_equal(grid, out_of_range_grid(LOWER, UPPER))


def test_grid_goes_below_and_above_every_limit_one_at_a_time_and_all_at_once() -> None:
    grid = out_of_range_grid(LOWER, UPPER)
    low, high = np.asarray(LOWER), np.asarray(UPPER)
    below, above = grid < low, grid > high

    for column in range(10):
        alone_below = below[:, column] & ~np.delete(below | above, column, axis=1).any(axis=1)
        alone_above = above[:, column] & ~np.delete(below | above, column, axis=1).any(axis=1)
        assert alone_below.any() and alone_above.any(), column
        assert (grid[:, column] == low[column]).any() and (grid[:, column] == high[column]).any(), column
    assert below.all(axis=1).any() and above.all(axis=1).any()
    assert (below.any(axis=1) & above.any(axis=1)).any()
    # Justo fuera del límite: a 1e-9 del ancho.
    assert np.isclose(grid[:, 0], -30e-9, rtol=0.0, atol=1e-15).any()


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([0.0, 0.0], [1.0]), ([1.0, 0.0], [1.0, 1.0]), ([[0.0]], [[1.0]])],
)
def test_grid_rejects_invalid_limits(lower: Any, upper: Any) -> None:
    with pytest.raises(ValueError, match="límites"):
        out_of_range_grid(lower, upper)


# ---------------------------------------------------------------------------
# Referencia de Python y comparación
# ---------------------------------------------------------------------------


def test_python_reference_uses_score_features_logreg_inside_the_range(breakdown: BreakdownModel) -> None:
    vectors = in_range_vectors(rows=50)
    reference = python_reference(breakdown, vectors)

    assert set(reference.source) == {"score_features_logreg"}
    assert reference.points.shape == (50, 6) and reference.match_score.shape == (50,)


def test_python_reference_falls_back_to_score_matrix_outside_the_range(breakdown: BreakdownModel) -> None:
    grid = out_of_range_grid(LOWER, UPPER)
    reference = python_reference(breakdown, grid)

    assert "score_matrix" in reference.source and "score_features_logreg" in reference.source
    assert (reference.raw_points >= 0).all()


def test_onnx_matches_python_inside_the_range(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    result = check_parity("sintético", model_bytes, breakdown, in_range_vectors())

    assert result.passed, result.as_dict()
    assert result.rows == result.rows_score_features == 400
    assert result.max_abs_diff_factor_points < 1e-12 and result.max_abs_diff_raw_total < 1e-12
    assert result.mismatches == ()
    assert result.summary().endswith("-> PASS")


def test_onnx_matches_python_on_the_out_of_range_grid(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    grid = out_of_range_grid(LOWER, UPPER)
    result = check_parity("rejilla", model_bytes, breakdown, grid)

    assert result.passed, result.as_dict()
    assert result.rows == len(grid) and result.rows_score_matrix > 0
    assert result.as_dict()["raw_tolerance"] == RAW_TOLERANCE


def test_parity_detects_a_different_model(breakdown: BreakdownModel) -> None:
    coef = list(logreg_document()["coef"])
    coef[0] *= 1.5  # distance_km: otro modelo
    other = onnx_bytes(build_breakdown_model(LogRegModel.from_dict(logreg_document(coef))))

    result = check_parity("otro modelo", other, breakdown, in_range_vectors())

    assert not result.passed
    assert result.max_abs_diff_factor_points > RAW_TOLERANCE
    assert result.rounded_points_mismatches > 0 and result.match_score_mismatches > 0
    assert 0 < len(result.mismatches) <= 10
    first = result.mismatches[0].as_dict()
    assert set(first) == {"row", "features", "onnx_points", "python_points", "onnx_match_score", "python_match_score"}
    assert result.summary().endswith("-> FALLA")


def test_an_empty_set_does_not_pass(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    result = check_parity("vacío", model_bytes, breakdown, np.zeros((0, 10)))

    assert result.rows == 0 and not result.passed  # un control sin vectores no es PASS


# ---------------------------------------------------------------------------
# Recorte contra np.clip
# ---------------------------------------------------------------------------


def test_onnx_clip_is_np_clip_bit_for_bit(model_bytes: bytes) -> None:
    grid = out_of_range_grid(LOWER, UPPER)

    assert clip_matches_np_clip(model_bytes, grid, LOWER, UPPER)
    assert np.array_equal(run_onnx_clipped(model_bytes, grid), np.clip(grid, LOWER, UPPER))
    assert not clip_matches_np_clip(model_bytes, grid, LOWER, [value / 2 for value in UPPER])


def test_run_onnx_clipped_requires_the_clipped_tensor() -> None:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["features"], ["out"])],
        "identity",
        [onnx.helper.make_tensor_value_info("features", onnx.TensorProto.DOUBLE, ["N", 10])],
        [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.DOUBLE, ["N", 10])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)

    with pytest.raises(ValueError, match="clipped"):
        run_onnx_clipped(model.SerializeToString(), np.zeros((1, 10)))
