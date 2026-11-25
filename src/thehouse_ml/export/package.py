"""Etapa `export` (ML-06): verifica `logreg.json`, construye `model.onnx` y lo ejecuta en onnxruntime.

ML-06 T01 (ML-06-R01) implementa la parte del modelo de la etapa: verificación del SHA-256 de
`logreg.json` (contra `train_manifest.json` y, en el perfil `full`, contra el valor fijado en la
spec §1), grafo de la spec §6.2 con sus metadatos, `onnx.checker`, escritura determinista de
`model.onnx` y ejecución en onnxruntime CPU sobre los pares de test (si `data/processed/` existe).
El resto del paquete de entrega (casos dorados, model card, métricas y `SHA256SUMS`) es de T05.

Perfiles (`THEHOUSE_ML_TRAIN_PROFILE`, como `train` y `evaluate`): `full` lee `models/` y escribe
en `dist/model-v1.0.0/`; `ci` lee `profiles.ci.output_dir` y escribe en `dist/ci/` (no publicable).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thehouse_ml.export import ExportError
from thehouse_ml.export.onnx_graph import (
    EXPECTED_LOGREG_SHA256,
    OPSET,
    build_onnx_model,
    graph_constants,
    model_metadata,
    run_onnx,
    serialize_model,
    verify_logreg,
)
from thehouse_ml.features import default_schema
from thehouse_ml.models.train import LOGREG_MODEL, LogRegModel, output_dirs
from thehouse_ml.models.train_config import DEFAULT_PROFILE, load_train_params, resolve_profile
from thehouse_ml.models.trained_breakdown import BreakdownModel, build_breakdown_model, score_matrix
from thehouse_ml.models.training_data import TRAINING_INPUTS, load_training_data
from thehouse_ml.paths import ProjectPaths

MODEL_ONNX = "model.onnx"
DIST_DIR = "dist"
PACKAGE_DIRS = {DEFAULT_PROFILE: "model-v1.0.0", "ci": "ci"}
TEST_SPLIT = "test"
UNKNOWN_COMMIT = "unknown"


def export_skip_message(models_dir: Path, root: Path) -> str:
    location = models_dir.relative_to(root).as_posix() if models_dir.is_relative_to(root) else str(models_dir)
    return f"export: omitido; falta {location}/{LOGREG_MODEL} (ejecute `train`)"


def git_commit(root: Path) -> str:
    """Commit de `root` (`HEAD`), con `-dirty` si hay cambios en archivos versionados; `unknown` sin git.

    Los archivos sin versionar (p. ej. `dist/`) no cuentan, para que dos exportaciones seguidas del
    mismo commit den el mismo `model.onnx`.
    """
    git = shutil.which("git")
    if git is None:
        return UNKNOWN_COMMIT
    # Argumentos fijos, sin shell ni entradas del usuario.
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603
    if head.returncode != 0 or not head.stdout.strip():
        return UNKNOWN_COMMIT
    status = subprocess.run(  # noqa: S603
        [git, "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=False
    )
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return head.stdout.strip() + ("-dirty" if dirty else "")


@dataclass(frozen=True)
class OnnxRunSummary:
    """Ejecución de `model.onnx` en onnxruntime CPU sobre los pares de test (T01; la paridad formal es T02)."""

    pairs: int
    finite: bool
    max_abs_diff_factor_points: float
    max_abs_diff_raw_total: float


@dataclass(frozen=True)
class ExportResult:
    profile: str
    model_path: Path
    model_sha256: str
    logreg_sha256: str
    commit: str
    test_pairs: OnnxRunSummary | None


def run_on_test_pairs(model_bytes: bytes, model: BreakdownModel, paths: ProjectPaths) -> OnnxRunSummary | None:
    """Ejecuta el ONNX sobre los pares de test; `None` si faltan los datos de `data/processed/`."""
    if not all((paths.data_processed / name).is_file() for name in TRAINING_INPUTS):
        return None
    data = load_training_data(paths, default_schema())
    mask = (data.split == TEST_SPLIT).to_numpy()
    features = data.features[mask].to_numpy(dtype=np.float64, copy=True)
    points, total = run_onnx(model_bytes, features)
    expected_points, expected_total, _ = score_matrix(model, features)
    if points.shape != expected_points.shape or total.shape != expected_total.shape:
        raise ExportError(f"formas del ONNX {points.shape}/{total.shape} distintas de las de Python")
    return OnnxRunSummary(
        pairs=len(features),
        finite=bool(np.isfinite(points).all() and np.isfinite(total).all()),
        max_abs_diff_factor_points=float(np.abs(points - expected_points).max(initial=0.0)),
        max_abs_diff_raw_total=float(np.abs(total - expected_total).max(initial=0.0)),
    )


def export_model(
    paths: ProjectPaths, profile: str | None = None, commit: str | None = None, expected_sha256: str | None = None
) -> ExportResult | None:
    """Verifica `logreg.json`, construye y escribe `model.onnx` y lo ejecuta; `None` si no hay modelo.

    `expected_sha256` solo se usa en pruebas; en el perfil `full` el valor fijado es
    `EXPECTED_LOGREG_SHA256` y en `ci` no hay valor fijado (solo el del manifiesto).
    """
    profile = profile or resolve_profile()
    models_dir, _ = output_dirs(paths, load_train_params(), profile)
    if not (models_dir / LOGREG_MODEL).is_file():
        return None
    if expected_sha256 is None and profile == DEFAULT_PROFILE:
        expected_sha256 = EXPECTED_LOGREG_SHA256
    logreg_sha256 = verify_logreg(models_dir, expected_sha256, profile)
    logreg = LogRegModel.from_dict(json.loads((models_dir / LOGREG_MODEL).read_text(encoding="utf-8")))
    breakdown = build_breakdown_model(logreg)
    constants = graph_constants(breakdown)
    commit = commit or git_commit(paths.root)
    metadata = model_metadata(breakdown, constants, logreg_sha256, commit, profile)
    model_bytes = serialize_model(build_onnx_model(constants, metadata))
    out_dir = paths.root / DIST_DIR / PACKAGE_DIRS[profile]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_ONNX
    model_path.write_bytes(model_bytes)
    return ExportResult(
        profile=profile,
        model_path=model_path,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        logreg_sha256=logreg_sha256,
        commit=commit,
        test_pairs=run_on_test_pairs(model_bytes, breakdown, paths),
    )


def run_export(paths: ProjectPaths, profile: str | None = None, commit: str | None = None) -> str:
    """Etapa `export` (T01): `model.onnx` verificado; mensaje para la consola."""
    profile = profile or resolve_profile()
    result = export_model(paths, profile=profile, commit=commit)
    if result is None:
        models_dir, _ = output_dirs(paths, load_train_params(), profile)
        return export_skip_message(models_dir, paths.root)
    location = result.model_path.relative_to(paths.root).as_posix()
    lines = [
        f"export: {location} (perfil {result.profile}, opset {OPSET}, onnx.checker sin errores)",
        f"  logreg.json SHA-256 {result.logreg_sha256} (verificado)",
        f"  model.onnx  SHA-256 {result.model_sha256}",
        f"  commit {result.commit}",
    ]
    run = result.test_pairs
    if run is None:
        lines.append("  pares de test: omitido; faltan los datos de data/processed/ (ejecute `prepare`)")
    else:
        lines.append(
            f"  onnxruntime CPU sobre {run.pairs:,} pares de test: salidas finitas {run.finite}; "
            f"max |dif| vs Python: factor_points {run.max_abs_diff_factor_points:.3e}, "
            f"raw_total {run.max_abs_diff_raw_total:.3e} "
            "(informativo; la paridad formal es T02)"
        )
    return "\n".join(lines)


__all__ = [
    "MODEL_ONNX",
    "PACKAGE_DIRS",
    "ExportResult",
    "OnnxRunSummary",
    "export_model",
    "export_skip_message",
    "git_commit",
    "run_export",
    "run_on_test_pairs",
]
