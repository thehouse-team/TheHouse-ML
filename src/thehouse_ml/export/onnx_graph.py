"""Grafo ONNX de `logreg-1.0.0` construido con la biblioteca `onnx` (ML-06 T01, spec §6.2).

Opción (a) de la spec ML-06 §6.2 (M-ML-044 decisión 2): el grafo se escribe operación por
operación desde los parámetros de `models/logreg.json` y de `build_breakdown_model`, sin pasar
por scikit-learn ni `skl2onnx` y sin pickle. Todo en `tensor(double)` (riesgo ML-06-RK01):

1. **Recorte por columna:** `Max(features, clip_lower)` y `Min(·, clip_upper)` con constantes
   [10] (equivale a `np.clip`; el operador `Clip` solo acepta límites escalares).
2. **Escalado:** `Sub(mean)` y `Div(scale)` con constantes [10].
3. **Contribuciones lineales:** `Mul(coef)` con constante [10].
4. **Suma por factor:** `MatMul` con la matriz de asignación [10, 6] de ceros y unos
   (`pets_compatible` en ceros: su término va al término fijo, que no se exporta).
5. **Calibración afín:** `Sub(min_k)` y `Mul(100 / S)` con constantes [6] → `factor_points` [N, 6].
6. **Total:** `Gather` de cada columna y `Add` encadenado de izquierda a derecha,
   `((((p1 + p2) + p3) + p4) + p5) + p6`, el mismo orden que `BreakdownModel.raw_total`
   (M-ML-031 decisión 3) → `raw_total` [N].

El ONNX no valida, no excluye, no redondea ni ordena (spec §6.1): eso lo hace el backend.
La serialización es determinista (sin marcas de tiempo; productor y versión fijos), así que el
mismo `logreg.json` y los mismos metadatos dan un `model.onnx` idéntico byte a byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from thehouse_ml import __version__
from thehouse_ml.export import ExportError
from thehouse_ml.models.baseline_config import RoundingRules
from thehouse_ml.models.train import LOGREG_MODEL, TRAIN_MANIFEST, sha256_file
from thehouse_ml.models.trained_breakdown import BreakdownModel

# SHA-256 de `models/logreg.json` de `logreg-1.0.0` (spec ML-06 §1, M-ML-043 herencia 1, M-ML-044 decisión 1).
EXPECTED_LOGREG_SHA256 = "341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa"

# Opset 17 (spec §6.2; todas las operaciones existen desde el 13) con IR 8, la versión de IR que
# corresponde al opset 17: el archivo lo carga cualquier onnxruntime >= 1.13, incluido el de .NET.
OPSET = 17
IR_VERSION = 8
PRODUCER_NAME = "thehouse-ml"
GRAPH_NAME = "logreg_factor_points"
INPUT_NAME = "features"
FACTOR_OUTPUT = "factor_points"
TOTAL_OUTPUT = "raw_total"
BATCH_DIM = "N"
FULL_SCALE = 100.0
DOUBLE = TensorProto.DOUBLE


# ---------------------------------------------------------------------------
# Verificación del modelo de entrada
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExportError(f"{path.name}: se esperaba un objeto JSON")
    return data


def verify_logreg(models_dir: Path, expected_sha256: str | None, profile: str) -> str:
    """SHA-256 de `logreg.json` contra `train_manifest.json` y, si se da, contra el valor fijado.

    `expected_sha256` es `EXPECTED_LOGREG_SHA256` en el perfil `full` (spec §3.4 y §7.5: si no
    coincide, la exportación falla y no se reentrena) y `None` en el perfil `ci`, cuyo modelo cambia
    con cada corrida de `train`. Devuelve el SHA-256 verificado.
    """
    model_path = models_dir / LOGREG_MODEL
    manifest_path = models_dir / TRAIN_MANIFEST
    for path in (model_path, manifest_path):
        if not path.is_file():
            raise ExportError(f"falta {path.name} en {models_dir} (ejecute `train`)")
    manifest = _read_json(manifest_path)
    if manifest.get("profile") != profile:
        raise ExportError(f"{TRAIN_MANIFEST}: perfil {manifest.get('profile')!r}, se esperaba {profile!r}")
    entry = manifest.get("models", {}).get(LOGREG_MODEL) if isinstance(manifest.get("models"), dict) else None
    recorded = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(recorded, str):
        raise ExportError(f"{TRAIN_MANIFEST}: falta el SHA-256 de {LOGREG_MODEL}")
    actual = sha256_file(model_path)
    if actual != recorded:
        raise ExportError(
            f"{LOGREG_MODEL}: SHA-256 {actual} distinto del de {TRAIN_MANIFEST} ({recorded}); no se exporta"
        )
    if expected_sha256 is not None and actual != expected_sha256:
        raise ExportError(
            f"{LOGREG_MODEL}: SHA-256 {actual} distinto del de logreg-1.0.0 ({expected_sha256}, spec ML-06 §1); "
            "no se exporta ni se reentrena"
        )
    document = _read_json(model_path)
    if document.get("synthetic") is not True:
        raise ExportError(f"{LOGREG_MODEL}: synthetic debe ser true")
    return actual


# ---------------------------------------------------------------------------
# Constantes del grafo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConstants:
    """Constantes binary64 del grafo, copiadas de `logreg.json` y de la calibración (sin float32)."""

    feature_order: tuple[str, ...]
    factor_order: tuple[str, ...]
    clip_lower: np.ndarray  # [10]
    clip_upper: np.ndarray  # [10]
    mean: np.ndarray  # [10]
    scale: np.ndarray  # [10]
    coef: np.ndarray  # [10]
    factor_assignment: np.ndarray  # [10, 6] de ceros y unos
    factor_min_logit: np.ndarray  # [6] min_k
    points_per_logit: np.ndarray  # [6] 100 / S
    max_points_raw: tuple[float, ...]  # [6] sin redondear


def _vector(values: Sequence[float]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=np.float64)


def graph_constants(model: BreakdownModel) -> GraphConstants:
    """Constantes del grafo desde el modelo calibrado de ML-05 (`build_breakdown_model`)."""
    logreg = model.logreg
    if logreg.mean is None or logreg.scale is None:
        raise ExportError("el grafo de ML-06 §6.2 exige la transformación standard (mean y scale)")
    n_features = len(logreg.feature_order)
    assignment = np.zeros((n_features, len(model.factors)), dtype=np.float64)
    for column, factor in enumerate(model.factors):
        for term in factor.terms:
            assignment[term.index, column] = 1.0
    if (assignment.sum(axis=1) > 1).any():
        raise ExportError("una feature pertenece a más de un factor")
    return GraphConstants(
        feature_order=tuple(logreg.feature_order),
        factor_order=tuple(factor.id for factor in model.factors),
        clip_lower=_vector(logreg.clip_lower),
        clip_upper=_vector(logreg.clip_upper),
        mean=_vector(logreg.mean),
        scale=_vector(logreg.scale),
        coef=_vector(logreg.coef),
        factor_assignment=assignment,
        factor_min_logit=_vector([factor.min_logit for factor in model.factors]),
        points_per_logit=np.full(len(model.factors), FULL_SCALE / model.total_span, dtype=np.float64),
        max_points_raw=tuple(float(factor.raw_max_points) for factor in model.factors),
    )


# ---------------------------------------------------------------------------
# Metadatos y grafo
# ---------------------------------------------------------------------------


def rounding_rule(rules: RoundingRules) -> dict[str, Any]:
    """Regla de redondeo que aplica el backend a las salidas (M-ML-029/031; spec §6.1 paso 6)."""
    return {
        "factor": rules.factor_formula,
        "factor_decimals": rules.factor_decimals,
        "total": rules.total_formula,
        "pre_round_decimals": rules.factor_pre_round_decimals,
        "mode": rules.factor_mode,
        "dotnet": rules.dotnet,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def model_metadata(
    model: BreakdownModel, constants: GraphConstants, logreg_sha256: str, commit: str, profile: str
) -> dict[str, str]:
    """`metadata_props` (spec §6.2), legibles en .NET con `ModelMetadata.CustomMetadataMap`.

    Los valores son texto; las listas y la regla de redondeo van en JSON compacto. `maxPointsRaw`
    son los 6 `maxPoints` sin redondear (el backend los redondea con la misma regla que los puntos).
    """
    return {
        "modelVersion": model.model_version,
        "featureSchemaVersion": model.feature_schema_version,
        "featureOrder": _json(list(constants.feature_order)),
        "factorOrder": _json(list(constants.factor_order)),
        "maxPointsRaw": _json(list(constants.max_points_raw)),
        "roundingRule": _json(rounding_rule(model.rounding.rounding)),
        "logregSha256": logreg_sha256,
        "gitCommit": commit,
        "profile": profile,
        "synthetic": "true",
    }


def _initializer(name: str, values: np.ndarray) -> TensorProto:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return numpy_helper.from_array(array, name=name)


def _index(name: str, column: int) -> TensorProto:
    return numpy_helper.from_array(np.asarray(column, dtype=np.int64), name=name)


def build_onnx_model(constants: GraphConstants, metadata: Mapping[str, str]) -> onnx.ModelProto:
    """Grafo de la spec §6.2 con sus `metadata_props`; pasa `onnx.checker` con inferencia de formas."""
    n_features = len(constants.feature_order)
    n_factors = len(constants.factor_order)
    initializers = [
        _initializer("clip_lower", constants.clip_lower),
        _initializer("clip_upper", constants.clip_upper),
        _initializer("scaler_mean", constants.mean),
        _initializer("scaler_scale", constants.scale),
        _initializer("logreg_coef", constants.coef),
        _initializer("factor_assignment", constants.factor_assignment),
        _initializer("factor_min_logit", constants.factor_min_logit),
        _initializer("points_per_logit", constants.points_per_logit),
    ]
    nodes = [
        helper.make_node("Max", [INPUT_NAME, "clip_lower"], ["clipped_low"], name="clip_max"),
        helper.make_node("Min", ["clipped_low", "clip_upper"], ["clipped"], name="clip_min"),
        helper.make_node("Sub", ["clipped", "scaler_mean"], ["centered"], name="scale_sub_mean"),
        helper.make_node("Div", ["centered", "scaler_scale"], ["standardized"], name="scale_div_scale"),
        helper.make_node("Mul", ["standardized", "logreg_coef"], ["contributions"], name="contributions_mul_coef"),
        helper.make_node("MatMul", ["contributions", "factor_assignment"], ["factor_logit"], name="factor_sum"),
        helper.make_node("Sub", ["factor_logit", "factor_min_logit"], ["factor_gain"], name="calibration_sub_min"),
        helper.make_node("Mul", ["factor_gain", "points_per_logit"], [FACTOR_OUTPUT], name="calibration_mul_scale"),
    ]
    columns: list[str] = []
    for column in range(n_factors):
        index_name = f"factor_column_{column}"
        initializers.append(_index(index_name, column))
        columns.append(f"points_{constants.factor_order[column]}")
        nodes.append(
            helper.make_node("Gather", [FACTOR_OUTPUT, index_name], [columns[-1]], name=f"take_{column}", axis=1)
        )
    running = columns[0]
    for column in range(1, n_factors):
        target = TOTAL_OUTPUT if column == n_factors - 1 else f"partial_total_{column}"
        nodes.append(helper.make_node("Add", [running, columns[column]], [target], name=f"total_add_{column}"))
        running = target
    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        inputs=[helper.make_tensor_value_info(INPUT_NAME, DOUBLE, [BATCH_DIM, n_features])],
        outputs=[
            helper.make_tensor_value_info(FACTOR_OUTPUT, DOUBLE, [BATCH_DIM, n_factors]),
            helper.make_tensor_value_info(TOTAL_OUTPUT, DOUBLE, [BATCH_DIM]),
        ],
        initializer=initializers,
        doc_string=(
            "Puntos sin redondear por factor y total de logreg (ML-06 §6.2): recorte Max/Min, escalado, "
            "contribuciones, suma por factor, calibración afín y Add encadenado. Sin redondeo ni reglas previas."
        ),
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name=PRODUCER_NAME,
        producer_version=__version__,
        ir_version=IR_VERSION,
        doc_string="TheHouse Smart Match: modelo de compatibilidad (DATOS SINTÉTICOS; ver model_card.md).",
    )
    helper.set_model_props(model, dict(metadata))
    try:
        onnx.checker.check_model(model, full_check=True)
    except onnx.checker.ValidationError as exc:  # pragma: no cover - el grafo fijo siempre es válido
        raise ExportError(f"onnx.checker rechazó el grafo: {exc}") from exc
    return model


def serialize_model(model: onnx.ModelProto) -> bytes:
    """Bytes del modelo con serialización determinista (mismo modelo → mismos bytes)."""
    data: bytes = model.SerializeToString(deterministic=True)
    return data


# ---------------------------------------------------------------------------
# Ejecución en onnxruntime (CPU)
# ---------------------------------------------------------------------------


def run_onnx(model_bytes: bytes, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`factor_points` [N, 6] y `raw_total` [N] con onnxruntime en CPU (`CPUExecutionProvider`)."""
    import onnxruntime  # Diferido: solo al ejecutar (override de mypy en pyproject.toml, M-ML-045).

    session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    batch = np.ascontiguousarray(np.atleast_2d(np.asarray(features, dtype=np.float64)))
    points, total = session.run([FACTOR_OUTPUT, TOTAL_OUTPUT], {INPUT_NAME: batch})
    return np.asarray(points, dtype=np.float64), np.asarray(total, dtype=np.float64)


__all__ = [
    "EXPECTED_LOGREG_SHA256",
    "FACTOR_OUTPUT",
    "INPUT_NAME",
    "IR_VERSION",
    "OPSET",
    "TOTAL_OUTPUT",
    "ExportError",
    "GraphConstants",
    "build_onnx_model",
    "graph_constants",
    "model_metadata",
    "rounding_rule",
    "run_onnx",
    "serialize_model",
    "verify_logreg",
]
