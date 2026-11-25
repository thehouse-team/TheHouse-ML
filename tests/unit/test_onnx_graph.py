"""ML-06 T01 (ML-06-R01): grafo ONNX de la regresión logística calibrada (spec ML-06 §6.2).

Usa un modelo sintético con la misma forma que `logreg.json` (no necesita `models/`): las
pruebas con el modelo oficial están en `tests/integration/test_export_stage.py`.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, numpy_helper

from thehouse_ml.export import ExportError
from thehouse_ml.export.onnx_graph import (
    FACTOR_OUTPUT,
    INPUT_NAME,
    IR_VERSION,
    OPSET,
    TOTAL_OUTPUT,
    build_onnx_model,
    graph_constants,
    model_metadata,
    run_onnx,
    serialize_model,
    verify_logreg,
)
from thehouse_ml.features import default_schema
from thehouse_ml.models.train import LOGREG_FORMAT, LogRegModel, sha256_file
from thehouse_ml.models.trained_breakdown import BreakdownModel, build_breakdown_model, score_matrix

REPO = Path(__file__).resolve().parents[2]
UPPER = [30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 1.0, 1.0]


def logreg_document() -> dict[str, Any]:
    """`logreg.json` sintético con la forma de `thehouse-ml/logreg-json/1` (signos mixtos, pets en 0)."""
    return {
        "synthetic": True,
        "format": LOGREG_FORMAT,
        "feature_order": list(default_schema().feature_order),
        "config": {"transform": "standard", "C": 0.01},
        "clip": {"lower": [0.0] * 10, "upper": UPPER},
        "transform": {
            "type": "standard",
            "mean": [5.2, 2.5, 0.55, 0.73, 0.3, 0.7, 1.0, 1.33, 0.14, 0.48],
            "scale": [6.3, 3.7, 0.23, 0.18, 0.46, 0.46, 1.0, 3.42, 0.22, 0.21],
        },
        "coef": [-3.35, -1.87, -0.996, 0.566, 1.006, 0.36, 0.0, -0.1, -0.0123, 0.439],
        "intercept": -4.13,
    }


@pytest.fixture(scope="module")
def breakdown() -> BreakdownModel:
    return build_breakdown_model(LogRegModel.from_dict(logreg_document()))


def build(model: BreakdownModel, commit: str = "abc123", profile: str = "ci") -> onnx.ModelProto:
    constants = graph_constants(model)
    return build_onnx_model(constants, model_metadata(model, constants, "0" * 64, commit, profile))


@pytest.fixture(scope="module")
def onnx_model(breakdown: BreakdownModel) -> onnx.ModelProto:
    return build(breakdown)


@pytest.fixture(scope="module")
def model_bytes(onnx_model: onnx.ModelProto) -> bytes:
    return serialize_model(onnx_model)


def vectors(seed: int = 7, rows: int = 500) -> np.ndarray:
    """Vectores dentro y fuera del `training_range` (por debajo de 0 y por encima del límite)."""
    rng = np.random.default_rng(seed)
    upper = np.asarray(UPPER)
    return rng.uniform(-0.5 * upper, 1.5 * upper, size=(rows, len(UPPER)))


# ---------------------------------------------------------------------------
# Dependencias (ML-06 decisión 5)
# ---------------------------------------------------------------------------


def test_onnx_and_onnxruntime_are_declared_and_installed_from_the_lock() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {spec.split(">")[0].split("=")[0].split("<")[0].strip() for spec in project["dependencies"]}
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    locked = {item["name"]: item["version"] for item in lock["package"] if "version" in item}

    assert {"onnx", "onnxruntime"} <= declared
    assert "skl2onnx" not in declared and "skl2onnx" not in locked  # opción (a) de la spec §6.2
    for name in ("onnx", "onnxruntime", "protobuf", "ml-dtypes", "flatbuffers"):
        assert importlib.metadata.version(name) == locked[name], name


# ---------------------------------------------------------------------------
# Estructura del grafo
# ---------------------------------------------------------------------------


def test_graph_passes_checker_with_opset_17_and_ir_8(onnx_model: onnx.ModelProto) -> None:
    onnx.checker.check_model(onnx_model, full_check=True)

    assert [(item.domain, item.version) for item in onnx_model.opset_import] == [("", OPSET)]
    assert OPSET == 17
    assert onnx_model.ir_version == IR_VERSION == 8
    assert onnx_model.producer_name == "thehouse-ml"


def test_inputs_outputs_and_float_constants_are_double(onnx_model: onnx.ModelProto) -> None:
    graph = onnx_model.graph
    shapes = {
        value.name: (
            value.type.tensor_type.elem_type,
            [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim],
        )
        for value in (*graph.input, *graph.output)
    }

    assert shapes == {
        INPUT_NAME: (TensorProto.DOUBLE, ["N", 10]),
        FACTOR_OUTPUT: (TensorProto.DOUBLE, ["N", 6]),
        TOTAL_OUTPUT: (TensorProto.DOUBLE, ["N"]),
    }
    for tensor in graph.initializer:
        expected = TensorProto.INT64 if tensor.name.startswith("factor_column_") else TensorProto.DOUBLE
        assert tensor.data_type == expected, tensor.name
    inferred = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True).graph.value_info
    assert inferred and all(value.type.tensor_type.elem_type == TensorProto.DOUBLE for value in inferred)


def test_node_sequence_follows_the_spec(onnx_model: onnx.ModelProto) -> None:
    ops = [node.op_type for node in onnx_model.graph.node]

    assert ops == ["Max", "Min", "Sub", "Div", "Mul", "MatMul", "Sub", "Mul", *["Gather"] * 6, *["Add"] * 5]
    assert "Clip" not in ops


def test_raw_total_is_a_left_to_right_add_chain(onnx_model: onnx.ModelProto) -> None:
    factors = json.loads(dict((p.key, p.value) for p in onnx_model.metadata_props)["factorOrder"])
    adds = [(list(node.input), list(node.output)) for node in onnx_model.graph.node if node.op_type == "Add"]
    columns = [f"points_{factor}" for factor in factors]

    expected_left = [columns[0], "partial_total_1", "partial_total_2", "partial_total_3", "partial_total_4"]
    expected = [[left, right] for left, right in zip(expected_left, columns[1:], strict=True)]
    assert [inputs for inputs, _ in adds] == expected
    assert adds[-1][1] == [TOTAL_OUTPUT]


def test_constants_are_copied_from_the_model_in_binary64(
    onnx_model: onnx.ModelProto, breakdown: BreakdownModel
) -> None:
    values = {tensor.name: numpy_helper.to_array(tensor) for tensor in onnx_model.graph.initializer}
    logreg = breakdown.logreg

    assert values["clip_lower"].tolist() == list(logreg.clip_lower)
    assert values["clip_upper"].tolist() == list(logreg.clip_upper)
    assert values["scaler_mean"].tolist() == list(logreg.mean or ())
    assert values["scaler_scale"].tolist() == list(logreg.scale or ())
    assert values["logreg_coef"].tolist() == list(logreg.coef)
    assert values["factor_min_logit"].tolist() == [factor.min_logit for factor in breakdown.factors]
    assert values["points_per_logit"].tolist() == [100.0 / breakdown.total_span] * 6
    assert [int(values[f"factor_column_{k}"]) for k in range(6)] == list(range(6))


def test_assignment_matrix_maps_each_feature_to_its_factor(
    onnx_model: onnx.ModelProto, breakdown: BreakdownModel
) -> None:
    matrix = next(numpy_helper.to_array(t) for t in onnx_model.graph.initializer if t.name == "factor_assignment")
    order = breakdown.logreg.feature_order

    assert matrix.shape == (10, 6)
    assert set(np.unique(matrix).tolist()) == {0.0, 1.0}
    assert matrix[order.index("pets_compatible")].tolist() == [0.0] * 6
    for column, factor in enumerate(breakdown.factors):
        assert sorted(order[i] for i in np.flatnonzero(matrix[:, column])) == sorted(factor.features)
    assert int(matrix.sum()) == 9


def test_metadata_props(onnx_model: onnx.ModelProto, breakdown: BreakdownModel) -> None:
    props = {p.key: p.value for p in onnx_model.metadata_props}

    assert list(props) == [
        "modelVersion",
        "featureSchemaVersion",
        "featureOrder",
        "factorOrder",
        "maxPointsRaw",
        "roundingRule",
        "logregSha256",
        "gitCommit",
        "profile",
        "synthetic",
    ]
    assert props["modelVersion"] == "logreg-1.0.0"
    assert props["featureSchemaVersion"] == default_schema().version == "2.0.0"
    assert json.loads(props["featureOrder"]) == list(default_schema().feature_order)
    assert json.loads(props["factorOrder"]) == [
        "budget",
        "distance",
        "essential_services",
        "place_type",
        "property_type",
        "extras",
    ]
    assert json.loads(props["maxPointsRaw"]) == [factor.raw_max_points for factor in breakdown.factors]
    rule = json.loads(props["roundingRule"])
    assert rule["factor_decimals"] == 1 and rule["pre_round_decimals"] == 6 and rule["mode"] == "half_up"
    assert "AwayFromZero" in rule["dotnet"]
    assert (props["gitCommit"], props["profile"], props["synthetic"]) == ("abc123", "ci", "true")


def test_graph_requires_the_standard_transform(breakdown: BreakdownModel) -> None:
    from dataclasses import replace

    with pytest.raises(ExportError, match="standard"):
        graph_constants(replace(breakdown, logreg=replace(breakdown.logreg, mean=None)))


# ---------------------------------------------------------------------------
# Ejecución en onnxruntime y determinismo
# ---------------------------------------------------------------------------


def test_onnxruntime_reproduces_the_python_reference(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    features = vectors()
    points, total = run_onnx(model_bytes, features)
    expected_points, expected_total, _ = score_matrix(breakdown, features)

    assert points.shape == (len(features), 6) and total.shape == (len(features),)
    assert points.dtype == total.dtype == np.float64
    # Sanidad de T01; la paridad formal (1e-4 y 0 en redondeados) es T02.
    assert np.abs(points - expected_points).max() < 1e-9
    assert np.abs(total - expected_total).max() < 1e-9


def test_clip_matches_np_clip_per_column(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    features = vectors(seed=11)
    clipped = np.clip(features, breakdown.logreg.clip_lower, breakdown.logreg.clip_upper)

    points, total = run_onnx(model_bytes, features)
    points_clipped, total_clipped = run_onnx(model_bytes, clipped)

    assert np.array_equal(points, points_clipped) and np.array_equal(total, total_clipped)


def test_raw_total_equals_the_sequential_sum_bit_for_bit(model_bytes: bytes) -> None:
    points, total = run_onnx(model_bytes, vectors(seed=3))
    expected = np.zeros(len(points))
    for column in range(points.shape[1]):
        expected = expected + points[:, column]

    assert np.array_equal(total, expected)


def test_serialization_is_deterministic(breakdown: BreakdownModel, model_bytes: bytes) -> None:
    assert serialize_model(build(breakdown)) == model_bytes
    assert serialize_model(build(breakdown, commit="other")) != model_bytes


# ---------------------------------------------------------------------------
# Verificación de logreg.json
# ---------------------------------------------------------------------------


def write_models(directory: Path, document: dict[str, Any], profile: str = "ci", digest: str | None = None) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "logreg.json"
    model_path.write_text(json.dumps(document), encoding="utf-8")
    actual = sha256_file(model_path)
    manifest = {"profile": profile, "models": {"logreg.json": {"sha256": digest or actual}}}
    (directory / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return actual


def test_verify_logreg_accepts_the_manifest_and_expected_digest(tmp_path: Path) -> None:
    digest = write_models(tmp_path, logreg_document())

    assert verify_logreg(tmp_path, None, "ci") == digest
    assert verify_logreg(tmp_path, digest, "ci") == digest


def test_verify_logreg_rejects_a_digest_other_than_the_fixed_one(tmp_path: Path) -> None:
    write_models(tmp_path, logreg_document())

    with pytest.raises(ExportError, match="spec ML-06 §1"):
        verify_logreg(tmp_path, "f" * 64, "ci")


def test_verify_logreg_rejects_a_model_that_does_not_match_the_manifest(tmp_path: Path) -> None:
    write_models(tmp_path, logreg_document(), digest="e" * 64)

    with pytest.raises(ExportError, match=r"train_manifest\.json"):
        verify_logreg(tmp_path, None, "ci")


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"profile": "full", "models": {}}, "perfil"),
        ({"profile": "ci"}, "falta el SHA-256"),
        ({"profile": "ci", "models": {"logreg.json": {}}}, "falta el SHA-256"),
    ],
)
def test_verify_logreg_rejects_incomplete_manifests(tmp_path: Path, manifest: dict[str, Any], message: str) -> None:
    write_models(tmp_path, logreg_document())
    (tmp_path / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportError, match=message):
        verify_logreg(tmp_path, None, "ci")


def test_verify_logreg_requires_both_files(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match=r"falta logreg\.json"):
        verify_logreg(tmp_path, None, "ci")


def test_verify_logreg_requires_synthetic_true(tmp_path: Path) -> None:
    write_models(tmp_path, {**logreg_document(), "synthetic": False})

    with pytest.raises(ExportError, match="synthetic"):
        verify_logreg(tmp_path, None, "ci")


def test_verify_logreg_rejects_non_object_json(tmp_path: Path) -> None:
    write_models(tmp_path, logreg_document())
    (tmp_path / "train_manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ExportError, match="objeto JSON"):
        verify_logreg(tmp_path, None, "ci")
