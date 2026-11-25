"""ML-06 T01 (ML-06-R01): etapa `export` con un modelo sintético y con el modelo oficial.

Las pruebas con el modelo oficial (`models/logreg.json` de `logreg-1.0.0`, fuera de git) se
omiten con motivo explícito si no está (p. ej. en CI, ML-06-RK05): no se reportan como PASS.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import pytest

from thehouse_ml.cli import main
from thehouse_ml.export import ExportError
from thehouse_ml.export.onnx_graph import EXPECTED_LOGREG_SHA256, OPSET
from thehouse_ml.export.package import UNKNOWN_COMMIT, export_model, git_commit, run_export
from thehouse_ml.models.train import LOGREG_FORMAT, sha256_file
from thehouse_ml.models.train_config import PROFILE_ENV_VAR
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_MODELS = REPO / "models"


def logreg_document() -> dict[str, Any]:
    return {
        "synthetic": True,
        "format": LOGREG_FORMAT,
        "feature_order": [
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
        ],
        "config": {"transform": "standard", "C": 0.01},
        "clip": {"lower": [0.0] * 10, "upper": [30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 1.0, 1.0]},
        "transform": {
            "type": "standard",
            "mean": [5.0, 2.5, 0.5, 0.7, 0.3, 0.7, 1.0, 1.3, 0.1, 0.5],
            "scale": [6.0, 3.7, 0.2, 0.2, 0.5, 0.5, 1.0, 3.4, 0.2, 0.2],
        },
        "coef": [-3.0, -2.0, -1.0, 0.5, 1.0, 0.4, 0.0, -0.1, -0.01, 0.4],
        "intercept": -4.0,
    }


def write_models(directory: Path, profile: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "logreg.json").write_text(json.dumps(logreg_document()), encoding="utf-8")
    digest = sha256_file(directory / "logreg.json")
    manifest = {"profile": profile, "models": {"logreg.json": {"sha256": digest}}}
    (directory / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return digest


def test_export_skips_without_a_model(tmp_path: Path) -> None:
    assert run_stage("export", ProjectPaths(root=tmp_path)) == (
        "export: omitido; falta models/logreg.json (ejecute `train`)"
    )
    assert not (tmp_path / "dist").exists()


def test_ci_profile_writes_a_reproducible_model(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    digest = write_models(tmp_path / "models" / "ci", "ci")

    message = run_export(paths, profile="ci", commit="abc123")
    first = (tmp_path / "dist" / "ci" / "model.onnx").read_bytes()
    run_export(paths, profile="ci", commit="abc123")
    second = (tmp_path / "dist" / "ci" / "model.onnx").read_bytes()

    assert first == second
    assert f"model.onnx  SHA-256 {hashlib.sha256(first).hexdigest()}" in message
    assert f"logreg.json SHA-256 {digest} (verificado)" in message
    assert "dist/ci/model.onnx (perfil ci, opset 17" in message
    assert "pares de test: omitido" in message
    model = onnx.load_from_string(first)
    onnx.checker.check_model(model, full_check=True)
    props = {p.key: p.value for p in model.metadata_props}
    assert (props["logregSha256"], props["gitCommit"], props["profile"]) == (digest, "abc123", "ci")


def test_full_profile_rejects_a_model_other_than_logreg_1_0_0(tmp_path: Path) -> None:
    write_models(tmp_path / "models", "full")

    with pytest.raises(ExportError, match="spec ML-06 §1"):
        export_model(ProjectPaths(root=tmp_path), profile="full")
    assert not (tmp_path / "dist").exists()


def test_cli_reports_export_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_models(tmp_path / "models", "full")
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)

    assert main(["--root", str(tmp_path), "export"]) == 1
    assert "no se exporta ni se reentrena" in capsys.readouterr().err


def test_git_commit_without_git_or_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert git_commit(tmp_path) == UNKNOWN_COMMIT
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert git_commit(REPO) == UNKNOWN_COMMIT


def test_git_commit_in_this_repository() -> None:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("Sin git o fuera de un repositorio")
    commit = git_commit(REPO)

    assert len(commit.removesuffix("-dirty")) == 40


# ---------------------------------------------------------------------------
# Modelo oficial logreg-1.0.0 (local; se omite en CI con motivo)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def official_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = OFFICIAL_MODELS / "logreg.json"
    if not model.is_file() or not (OFFICIAL_MODELS / "train_manifest.json").is_file():
        pytest.skip("models/logreg.json oficial ausente (fuera de git; en CI se usa el perfil ci, ML-06-RK05)")
    if sha256_file(model) != EXPECTED_LOGREG_SHA256:
        pytest.skip("models/logreg.json local no es logreg-1.0.0 (SHA-256 distinto del de la spec ML-06 §1)")
    root = tmp_path_factory.mktemp("official")
    (root / "models").mkdir()
    for name in ("logreg.json", "train_manifest.json"):
        shutil.copyfile(OFFICIAL_MODELS / name, root / "models" / name)
    return root


def test_official_model_exports_and_is_reproducible(official_root: Path) -> None:
    paths = ProjectPaths(root=official_root)
    first = export_model(paths, profile="full", commit="test-commit")
    assert first is not None
    first_bytes = first.model_path.read_bytes()
    second = export_model(paths, profile="full", commit="test-commit")
    assert second is not None

    assert first.logreg_sha256 == EXPECTED_LOGREG_SHA256
    assert first.model_path == official_root / "dist" / "model-v1.0.0" / "model.onnx"
    assert second.model_sha256 == first.model_sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert second.model_path.read_bytes() == first_bytes
    model = onnx.load_from_string(first_bytes)
    onnx.checker.check_model(model, full_check=True)
    assert model.opset_import[0].version == OPSET
    props = {p.key: p.value for p in model.metadata_props}
    assert props["modelVersion"] == "logreg-1.0.0" and props["profile"] == "full"
    assert [round(v, 1) for v in json.loads(props["maxPointsRaw"])] == [9.6, 72.6, 6.7, 4.9, 1.7, 4.5]
