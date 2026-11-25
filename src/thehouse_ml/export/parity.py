"""Paridad Python ↔ onnxruntime de `logreg-1.0.0` (ML-06 T02, ML-06-R02, spec §6.3).

Compara las salidas del ONNX (`factor_points` [N, 6] y `raw_total` [N], sin redondear) con la
referencia de Python, `score_features_logreg` de `trained_breakdown.py`:

- **Sin redondear:** |dif| ≤ `RAW_TOLERANCE` (1e-4) en cada factor y en `raw_total`; se reporta
  además el máximo |dif| observado.
- **Redondeados:** 0 diferencias en los puntos por factor y en `matchScore`. A las salidas del ONNX
  se les aplica la regla del backend (spec §6.1 paso 6): round6 y 1 decimal por factor y
  `matchScore = floor(round6(raw_total) + 0.5)`, con `MidpointRounding.AwayFromZero` como en .NET
  (`backend_round`). A la referencia de Python, la del paquete (`factor_points`, `match_score`).

Los vectores fuera del `range` del esquema (p. ej. negativos, solo posibles en la rejilla de la
spec §6.3) no los acepta `score_features_logreg`, que valida como el baseline; para ellos la
referencia es `score_matrix` (el mismo cálculo que usa `score_features_logreg`) con el mismo
redondeo del paquete. El reporte dice cuántas filas se compararon con cada una.

El recorte se comprueba además contra `np.clip` con los límites leídos de `logreg.json`,
exponiendo como salida el tensor intermedio `clipped` del grafo (`run_onnx_clipped`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx

from thehouse_ml.export.onnx_graph import FACTOR_OUTPUT, INPUT_NAME, TOTAL_OUTPUT, run_onnx
from thehouse_ml.features import default_schema
from thehouse_ml.models.baseline_config import BaselineConfig
from thehouse_ml.models.baseline_score import ScoreInputError, factor_points, match_score, round_half_up
from thehouse_ml.models.trained_breakdown import BreakdownModel, score_features_logreg, score_matrix

RAW_TOLERANCE = 1e-4  # |dif| máximo en factor_points y raw_total (spec §6.3)
CLIPPED_TENSOR = "clipped"  # salida de `Min(Max(features, lower), upper)` en el grafo de T01
MAX_REPORTED_MISMATCHES = 10
REFERENCE_SCORE = "score_features_logreg"
REFERENCE_MATRIX = "score_matrix"

# Desplazamientos de la rejilla fuera del `training_range` (spec §6.3), relativos al ancho
# `upper - lower` de cada feature: justo fuera del límite, un poco y muy lejos.
GRID_OFFSETS = (1e-9, 1e-6, 0.1, 1.0, 10.0)


# ---------------------------------------------------------------------------
# Redondeo del backend (spec §6.1 paso 6)
# ---------------------------------------------------------------------------


def backend_round(value: float, decimals: int) -> float:
    """.NET `Math.Round(value, decimals, MidpointRounding.AwayFromZero)` para cualquier signo.

    Para `value` ≥ 0 es `round_half_up` del paquete. Un valor negativo se redondea por su magnitud
    y conserva el signo (lo que hace .NET); el resultado `-0.0` se normaliza a `0.0`. El ONNX puede
    devolver negativos del orden de 1e-16 en un factor que está en su mínimo (T02-H1).
    """
    if not math.isfinite(value):
        raise ValueError(f"backend_round: se esperaba un número finito, llegó {value!r}")
    rounded = math.copysign(round_half_up(abs(value), decimals), value)
    return rounded + 0.0  # -0.0 + 0.0 == 0.0


def backend_factor_points(raw_points: float, config: BaselineConfig) -> float:
    """Puntos de un factor = round(round6(puntos), 1), mitad lejos de cero (M-ML-031)."""
    rules = config.rounding
    return backend_round(backend_round(raw_points, rules.factor_pre_round_decimals), rules.factor_decimals)


def backend_match_score(raw_total: float, config: BaselineConfig) -> int:
    """`matchScore` = round(round6(raw_total), 0), mitad lejos de cero, recortado a [0, total_points]."""
    total = backend_round(backend_round(raw_total, config.rounding.total_pre_round_decimals), 0)
    return min(max(int(total), 0), config.total_points)


# ---------------------------------------------------------------------------
# Referencia de Python
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PythonReference:
    """Puntos y total sin redondear, puntos redondeados y `matchScore` de la referencia de Python."""

    raw_points: np.ndarray  # [N, 6]
    raw_total: np.ndarray  # [N]
    points: np.ndarray  # [N, 6] redondeados
    match_score: np.ndarray  # [N] enteros
    source: tuple[str, ...]  # REFERENCE_SCORE o REFERENCE_MATRIX por fila


def python_reference(model: BreakdownModel, features: np.ndarray) -> PythonReference:
    """`score_features_logreg` fila por fila; `score_matrix` si el vector está fuera del `range`."""
    schema = default_schema()
    rows = np.atleast_2d(np.asarray(features, dtype=np.float64))
    matrix_points, matrix_total, _ = score_matrix(model, rows)
    n_factors = len(model.factors)
    raw_points = np.empty((len(rows), n_factors), dtype=np.float64)
    raw_total = np.empty(len(rows), dtype=np.float64)
    points = np.empty((len(rows), n_factors), dtype=np.float64)
    scores = np.empty(len(rows), dtype=np.int64)
    source: list[str] = []
    for index, row in enumerate(rows):
        try:
            result = score_features_logreg(row.tolist(), model, schema)
        except ScoreInputError:
            raw_points[index] = matrix_points[index]
            raw_total[index] = matrix_total[index]
            points[index] = [factor_points(float(value), model.rounding) for value in matrix_points[index]]
            scores[index] = match_score(float(matrix_total[index]), model.rounding)
            source.append(REFERENCE_MATRIX)
            continue
        raw_points[index] = [item.raw_points for item in result.breakdown]
        raw_total[index] = result.raw_total
        points[index] = [item.points for item in result.breakdown]
        scores[index] = result.match_score
        source.append(REFERENCE_SCORE)
    return PythonReference(raw_points, raw_total, points, scores, tuple(source))


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """Fila con una diferencia en puntos redondeados o en `matchScore` (para el reporte)."""

    row: int
    features: tuple[float, ...]
    onnx_points: tuple[float, ...]
    python_points: tuple[float, ...]
    onnx_match_score: int
    python_match_score: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "features": list(self.features),
            "onnx_points": list(self.onnx_points),
            "python_points": list(self.python_points),
            "onnx_match_score": self.onnx_match_score,
            "python_match_score": self.python_match_score,
        }


@dataclass(frozen=True)
class ParityResult:
    """Paridad de un conjunto de vectores (spec §6.3)."""

    name: str
    rows: int
    rows_score_features: int  # comparadas contra score_features_logreg
    rows_score_matrix: int  # fuera del `range`: comparadas contra score_matrix
    max_abs_diff_factor_points: float
    max_abs_diff_raw_total: float
    rounded_points_mismatches: int  # filas con algún factor redondeado distinto
    match_score_mismatches: int
    negative_onnx_points: int  # celdas de factor_points < 0 (T02-H1)
    min_onnx_points: float
    mismatches: tuple[Mismatch, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return (
            self.rows > 0
            and self.max_abs_diff_factor_points <= RAW_TOLERANCE
            and self.max_abs_diff_raw_total <= RAW_TOLERANCE
            and self.rounded_points_mismatches == 0
            and self.match_score_mismatches == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "rows_score_features_logreg": self.rows_score_features,
            "rows_score_matrix": self.rows_score_matrix,
            "max_abs_diff_factor_points": self.max_abs_diff_factor_points,
            "max_abs_diff_raw_total": self.max_abs_diff_raw_total,
            "raw_tolerance": RAW_TOLERANCE,
            "rounded_points_mismatches": self.rounded_points_mismatches,
            "match_score_mismatches": self.match_score_mismatches,
            "negative_onnx_points": self.negative_onnx_points,
            "min_onnx_points": self.min_onnx_points,
            "passed": self.passed,
            "mismatches": [item.as_dict() for item in self.mismatches],
        }

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FALLA"
        return (
            f"{self.name}: {self.rows:,} vectores; max |dif| factor_points {self.max_abs_diff_factor_points:.3e}, "
            f"raw_total {self.max_abs_diff_raw_total:.3e} (tolerancia {RAW_TOLERANCE:g}); "
            f"dif en puntos redondeados {self.rounded_points_mismatches}, en matchScore "
            f"{self.match_score_mismatches} -> {verdict}"
        )


def check_parity(name: str, model_bytes: bytes, model: BreakdownModel, features: np.ndarray) -> ParityResult:
    """Compara el ONNX con la referencia de Python sobre `features` [N, 10] (spec §6.3)."""
    rows = np.ascontiguousarray(np.atleast_2d(np.asarray(features, dtype=np.float64)))
    onnx_points, onnx_total = run_onnx(model_bytes, rows)
    reference = python_reference(model, rows)
    if onnx_points.shape != reference.raw_points.shape or onnx_total.shape != reference.raw_total.shape:
        raise ValueError(
            f"formas del ONNX {onnx_points.shape}/{onnx_total.shape} distintas de las de Python "
            f"{reference.raw_points.shape}/{reference.raw_total.shape}"
        )
    rounded = np.asarray(
        [[backend_factor_points(float(value), model.rounding) for value in row] for row in onnx_points],
        dtype=np.float64,
    ).reshape(onnx_points.shape)
    scores = np.asarray([backend_match_score(float(value), model.rounding) for value in onnx_total], dtype=np.int64)
    points_differ = (rounded != reference.points).any(axis=1)
    score_differ = scores != reference.match_score
    mismatches = tuple(
        Mismatch(
            row=int(index),
            features=tuple(float(value) for value in rows[index]),
            onnx_points=tuple(float(value) for value in rounded[index]),
            python_points=tuple(float(value) for value in reference.points[index]),
            onnx_match_score=int(scores[index]),
            python_match_score=int(reference.match_score[index]),
        )
        for index in np.flatnonzero(points_differ | score_differ)[:MAX_REPORTED_MISMATCHES]
    )
    return ParityResult(
        name=name,
        rows=len(rows),
        rows_score_features=reference.source.count(REFERENCE_SCORE),
        rows_score_matrix=reference.source.count(REFERENCE_MATRIX),
        max_abs_diff_factor_points=float(np.abs(onnx_points - reference.raw_points).max(initial=0.0)),
        max_abs_diff_raw_total=float(np.abs(onnx_total - reference.raw_total).max(initial=0.0)),
        rounded_points_mismatches=int(points_differ.sum()),
        match_score_mismatches=int(score_differ.sum()),
        negative_onnx_points=int((onnx_points < 0).sum()),
        min_onnx_points=float(onnx_points.min(initial=math.inf)),
        mismatches=mismatches,
    )


# ---------------------------------------------------------------------------
# Rejilla fuera del training_range (spec §6.3)
# ---------------------------------------------------------------------------


def out_of_range_grid(lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """Rejilla determinista de vectores fuera de `[lower, upper]` (los límites del recorte).

    - **Una feature a la vez:** sobre tres bases (esquina inferior, punto medio y esquina superior
      de la caja), cada feature por debajo de `lower` y por encima de `upper` con cada desplazamiento
      de `GRID_OFFSETS` (relativo al ancho), además de exactamente en cada límite.
    - **Todas a la vez:** todas por debajo, todas por encima y alternadas (pares abajo / impares
      arriba y al revés), con cada desplazamiento.
    """
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 1 or not (high > low).all():
        raise ValueError("out_of_range_grid: límites inválidos")
    width = high - low
    bases = (low, (low + high) / 2.0, high)
    rows: list[np.ndarray] = []
    for base in bases:
        for column in range(len(low)):
            for value in (low[column], high[column]):
                row = base.copy()
                row[column] = value
                rows.append(row)
            for offset in GRID_OFFSETS:
                for value in (low[column] - offset * width[column], high[column] + offset * width[column]):
                    row = base.copy()
                    row[column] = value
                    rows.append(row)
    even = np.arange(len(low)) % 2 == 0
    for offset in GRID_OFFSETS:
        below = low - offset * width
        above = high + offset * width
        rows.extend((below, above, np.where(even, below, above), np.where(even, above, below)))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Recorte contra np.clip
# ---------------------------------------------------------------------------


def run_onnx_clipped(model_bytes: bytes, features: np.ndarray) -> np.ndarray:
    """Tensor intermedio `clipped` [N, 10] del grafo, expuesto como salida en una copia del modelo."""
    import onnxruntime  # Diferido, como en `run_onnx` (override de mypy, M-ML-045).

    model = onnx.load_from_string(model_bytes)
    if CLIPPED_TENSOR not in {output for node in model.graph.node for output in node.output}:
        raise ValueError(f"el grafo no tiene el tensor {CLIPPED_TENSOR!r}")
    n_features = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(CLIPPED_TENSOR, onnx.TensorProto.DOUBLE, ["N", n_features])
    )
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    batch = np.ascontiguousarray(np.atleast_2d(np.asarray(features, dtype=np.float64)))
    (clipped,) = session.run([CLIPPED_TENSOR], {INPUT_NAME: batch})
    return np.asarray(clipped, dtype=np.float64)


def clip_matches_np_clip(
    model_bytes: bytes, features: np.ndarray, lower: Sequence[float], upper: Sequence[float]
) -> bool:
    """`Min(Max(features, lower), upper)` del ONNX es idéntico bit a bit a `np.clip` por columna."""
    expected = np.clip(np.atleast_2d(np.asarray(features, dtype=np.float64)), lower, upper)
    return bool(np.array_equal(run_onnx_clipped(model_bytes, features), expected))


__all__ = [
    "CLIPPED_TENSOR",
    "FACTOR_OUTPUT",
    "GRID_OFFSETS",
    "RAW_TOLERANCE",
    "TOTAL_OUTPUT",
    "Mismatch",
    "ParityResult",
    "PythonReference",
    "backend_factor_points",
    "backend_match_score",
    "backend_round",
    "check_parity",
    "clip_matches_np_clip",
    "out_of_range_grid",
    "python_reference",
    "run_onnx_clipped",
]
