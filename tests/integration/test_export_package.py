"""ML-06 T05 (ML-06-R05): paquete de entrega de la etapa `export` (spec ML-06 §6.6).

Con un modelo sintético (corre en CI): los 7 archivos, `SHA256SUMS` verificable, copias del contenido
versionado (CRLF → LF), sección «Artefacto ONNX» en la model card, paquete reproducible y las
condiciones de «publicable» (perfil, paridad, casos dorados, archivos de origen y árbol de git
limpio; M-ML-045, M-ML-046 y M-ML-048). Con el modelo oficial (local; se omite con motivo sin
él, ML-06-RK05): los casos dorados reales de `logreg-1.0.0` contra `model.onnx`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thehouse_ml.export.benchmark import BENCHMARK_ROOMS, BenchmarkResult
from thehouse_ml.export.onnx_graph import EXPECTED_LOGREG_SHA256, graph_constants, run_onnx
from thehouse_ml.export.package import (
    CHECKSUMMED_FILES,
    NOT_PUBLISHABLE_MARKER,
    PACKAGE_FILES,
    SHA256SUMS,
    ExportResult,
    GoldenResult,
    build_package,
    canonical_text,
    check_golden_cases,
    export_model,
    latency_bound,
    publishability,
    run_export,
    sha256sums_text,
    verify_sha256sums,
)
from thehouse_ml.export.parity import backend_factor_points, backend_match_score
from thehouse_ml.models.train import sha256_file
from thehouse_ml.paths import ProjectPaths

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_MODELS = REPO / "models"
LOWER = [0.0] * 10
UPPER = [30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 1.0, 1.0]
CLEAN = "0123456789abcdef0123456789abcdef01234567"
CARD_SOURCE = "# Model card — sintética\r\n\r\nLínea con CRLF.\r\n"
METRICS_SOURCE = '{\r\n  "synthetic": true\r\n}\r\n'


def logreg_document() -> dict[str, Any]:
    return {
        "synthetic": True,
        "format": "thehouse-ml/logreg-json/1",
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
        "clip": {"lower": LOWER, "upper": UPPER},
        "transform": {
            "type": "standard",
            "mean": [5.0, 2.5, 0.5, 0.7, 0.3, 0.7, 1.0, 1.3, 0.1, 0.5],
            "scale": [6.0, 3.7, 0.2, 0.2, 0.5, 0.5, 1.0, 3.4, 0.2, 0.2],
        },
        "coef": [-3.0, -2.0, -1.0, 0.5, 1.0, 0.4, 0.0, -0.1, -0.01, 0.4],
        "intercept": -4.0,
    }


def synthetic_features(rows: int, seed: int = 20261125) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(np.asarray(LOWER), np.asarray(UPPER), size=(rows, 10))
    matrix[:, 4:7] = np.round(matrix[:, 4:7])
    return matrix


def write_models(directory: Path, profile: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "logreg.json").write_text(json.dumps(logreg_document()), encoding="utf-8")
    digest = sha256_file(directory / "logreg.json")
    manifest = {"profile": profile, "models": {"logreg.json": {"sha256": digest}}}
    (directory / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return digest


def write_golden(path: Path, export: ExportResult, vectors: np.ndarray, *, corrupt: bool = False) -> None:
    """Casos dorados sintéticos con el esperado del propio ONNX y el redondeo del backend."""
    assert export.breakdown is not None
    constants = graph_constants(export.breakdown)
    points, totals = run_onnx(export.model_path.read_bytes(), vectors)
    order, factors = list(constants.feature_order), list(constants.factor_order)
    cases: list[dict[str, Any]] = []
    for row, vector in enumerate(vectors):
        breakdown = [
            {"factor": name, "points": backend_factor_points(float(points[row][k]), export.breakdown.rounding)}
            for k, name in enumerate(factors)
        ]
        score = backend_match_score(float(totals[row]), export.breakdown.rounding)
        cases.append(
            {
                "id": f"case_{row}",
                "input": {"features": dict(zip(order, vector.tolist(), strict=True))},
                "expected": {
                    "onnx": {"factor_points": points[row].tolist(), "raw_total": float(totals[row])},
                    "result": {"matchScore": score + (1 if corrupt and row == 0 else 0), "breakdown": breakdown},
                },
            }
        )
    cases.append(
        {"id": "incomplete", "input": {"student": {}, "room": {}}, "expected": {"result": {"incomplete": True}}}
    )
    meta = {
        "modelVersion": export.breakdown.model_version,
        "derived_from": {"sha256": export.logreg_sha256},
        "featureOrder": order,
        "factorOrder": factors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"metadata": meta, "cases": cases}, indent=2).replace("\n", "\r\n").encode("utf-8"))


def write_sources(root: Path, reports_dir: Path) -> None:
    """Archivos de origen del paquete con CRLF, como en un árbol de trabajo de Windows."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_card.md").write_bytes(CARD_SOURCE.encode("utf-8"))
    (reports_dir / "trained_metrics.json").write_bytes(METRICS_SOURCE.encode("utf-8"))
    fixtures = root / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "baseline_golden_cases.json").write_bytes(b'{\r\n  "cases": []\r\n}\r\n')


@dataclasses.dataclass
class Built:
    root: Path
    paths: ProjectPaths
    export: ExportResult
    features: np.ndarray
    digest: str


def build_full(tmp_path: Path, commit: str = CLEAN, *, corrupt: bool = False, features: bool = True) -> Built:
    """Exportación `full` con un modelo sintético (SHA-256 esperado inyectado) y sus archivos de origen."""
    paths = ProjectPaths(root=tmp_path)
    digest = write_models(tmp_path / "models", "full")
    write_sources(tmp_path, tmp_path / "reports")
    matrix = synthetic_features(300)
    export = export_model(paths, profile="full", commit=commit, expected_sha256=digest, test_features=matrix)
    assert export is not None
    write_golden(tmp_path / "tests" / "fixtures" / "logreg_golden_cases.json", export, matrix[:12], corrupt=corrupt)
    return Built(tmp_path, paths, export, matrix if features else np.empty((0, 10)), digest)


def package(built: Built) -> Any:
    features = built.features if len(built.features) else None
    return build_package(built.paths, built.export, features, expected_logreg_sha256=built.digest)


# ---------------------------------------------------------------------------
# Contenido del paquete
# ---------------------------------------------------------------------------


def test_full_package_has_the_seven_files_and_is_publishable_from_a_clean_commit(tmp_path: Path) -> None:
    built = build_full(tmp_path)
    result = package(built)
    directory = tmp_path / "dist" / "model-v1.0.0"

    assert result.publishable and result.reasons == ()
    assert sorted(p.name for p in directory.iterdir()) == sorted(PACKAGE_FILES)
    assert not (directory / NOT_PUBLISHABLE_MARKER).exists()
    assert verify_sha256sums(directory) == dict.fromkeys(CHECKSUMMED_FILES, True)
    lines = (directory / SHA256SUMS).read_bytes().decode("ascii").splitlines(keepends=True)
    assert [line.split("  ")[1].rstrip("\n") for line in lines] == list(CHECKSUMMED_FILES)
    assert all(line.endswith("\n") and not line.endswith("\r\n") for line in lines)
    assert result.golden.passed and result.golden.scored == 12 and result.golden.cases == 13
    assert result.benchmark is not None and result.benchmark.rooms == BENCHMARK_ROOMS


def test_copies_are_the_versioned_content_with_lf_line_endings(tmp_path: Path) -> None:
    built = build_full(tmp_path)
    package(built)
    directory = tmp_path / "dist" / "model-v1.0.0"
    schema = REPO / "src" / "thehouse_ml" / "features" / "feature_schema.json"
    fixtures = tmp_path / "tests" / "fixtures"

    assert (directory / "feature_schema.json").read_bytes() == canonical_text(schema.read_bytes())
    assert (directory / "golden_cases.json").read_bytes() == canonical_text(
        (fixtures / "logreg_golden_cases.json").read_bytes()
    )
    assert (directory / "baseline_golden_cases.json").read_bytes() == b'{\n  "cases": []\n}\n'
    assert (directory / "metrics.json").read_bytes() == METRICS_SOURCE.replace("\r\n", "\n").encode("utf-8")
    for name in ("feature_schema.json", "golden_cases.json", "metrics.json", "model_card.md"):
        assert b"\r\n" not in (directory / name).read_bytes(), name


def test_model_card_is_the_report_plus_the_onnx_section(tmp_path: Path) -> None:
    built = build_full(tmp_path)
    package(built)
    card = (tmp_path / "dist" / "model-v1.0.0" / "model_card.md").read_text(encoding="utf-8")

    assert card.startswith(CARD_SOURCE.replace("\r\n", "\n").rstrip("\n") + "\n\n## 11. Artefacto ONNX\n")
    assert "| Publicable | **SÍ** |" in card
    assert f"| Commit de git (`gitCommit`) | `{CLEAN}` |" in card
    assert f"| `model.onnx` SHA-256 | `{built.export.model_sha256}` |" in card
    assert "| Opset / IR | 17 / 8 |" in card
    for name in ("`features` (entrada)", "`factor_points` (salida)", "`raw_total` (salida)", "`maxPointsRaw`"):
        assert name in card
    assert "| pares de test | 300 |" in card and "| rejilla fuera del training_range | 380 |" in card
    assert "| casos dorados | 12 |" in card
    # Rendimiento (M-ML-048, T04-H5): veredicto y cota estable, sin cifras que cambien entre corridas.
    assert "| Criterio (constitución §13.5) | mediana < 20 ms: **PASS** |" in card
    assert "| Mediana | < 1 ms |" in card or "| Mediana | < 5 ms |" in card or "| Mediana | < 20 ms |" in card
    assert "### Desviación conocida ML-05 T05-H1" in card and "DATOS SINTÉTICOS" in card


def test_package_is_reproducible(tmp_path: Path) -> None:
    built = build_full(tmp_path)
    first = package(built)
    first_bytes = {name: (first.directory / name).read_bytes() for name in PACKAGE_FILES}
    again = export_model(
        built.paths, profile="full", commit=CLEAN, expected_sha256=built.digest, test_features=built.features
    )
    assert again is not None
    second = build_package(built.paths, again, built.features, expected_logreg_sha256=built.digest)

    assert second.digests == first.digests
    assert {name: (second.directory / name).read_bytes() for name in PACKAGE_FILES} == first_bytes


# ---------------------------------------------------------------------------
# Condiciones de «publicable» (spec §6.6, M-ML-045, M-ML-046)
# ---------------------------------------------------------------------------


def test_dirty_tree_is_not_publishable_and_the_marker_says_why(tmp_path: Path) -> None:
    built = build_full(tmp_path, commit=CLEAN + "-dirty")
    result = package(built)
    directory = tmp_path / "dist" / "model-v1.0.0"

    assert not result.publishable
    assert result.reasons == ("arbol de git sucio (gitCommit con -dirty)",)
    marker = (directory / NOT_PUBLISHABLE_MARKER).read_text(encoding="utf-8")
    assert "arbol de git sucio" in marker
    assert "| Publicable | **NO** — arbol de git sucio" in (directory / "model_card.md").read_text(encoding="utf-8")
    # El aviso queda fuera de SHA256SUMS: el release lleva solo los 7 archivos.
    assert NOT_PUBLISHABLE_MARKER not in (directory / SHA256SUMS).read_text(encoding="ascii")
    assert verify_sha256sums(directory) == dict.fromkeys(CHECKSUMMED_FILES, True)


def test_a_clean_rebuild_removes_the_stale_marker(tmp_path: Path) -> None:
    built = build_full(tmp_path, commit=CLEAN + "-dirty")
    package(built)
    clean = export_model(
        built.paths, profile="full", commit=CLEAN, expected_sha256=built.digest, test_features=built.features
    )
    assert clean is not None

    assert build_package(built.paths, clean, built.features, expected_logreg_sha256=built.digest).publishable
    assert not (tmp_path / "dist" / "model-v1.0.0" / NOT_PUBLISHABLE_MARKER).exists()


def test_failed_parity_is_not_publishable(tmp_path: Path) -> None:
    built = build_full(tmp_path)
    assert built.export.test_pairs is not None
    broken = dataclasses.replace(built.export.test_pairs, match_score_mismatches=1)
    export = dataclasses.replace(built.export, test_pairs=broken, grid=dataclasses.replace(built.export.grid, rows=0))
    result = build_package(built.paths, export, built.features, expected_logreg_sha256=built.digest)

    assert not result.publishable
    assert "la paridad en los pares de test falla" in result.reasons
    assert "la paridad en la rejilla fuera del training_range falla" in result.reasons
    assert (tmp_path / "dist" / "model-v1.0.0" / NOT_PUBLISHABLE_MARKER).is_file()


def test_golden_case_mismatch_is_not_publishable(tmp_path: Path) -> None:
    built = build_full(tmp_path, corrupt=True)
    result = package(built)

    assert not result.publishable
    assert result.golden.match_score_mismatches == 1 and result.golden.mismatched_ids == ("case_0",)
    assert "los casos dorados o su paridad fallan" in result.reasons


def test_without_test_pairs_parity_and_benchmark_are_not_executed(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    digest = write_models(tmp_path / "models", "full")
    write_sources(tmp_path, tmp_path / "reports")
    export = export_model(paths, profile="full", commit=CLEAN, expected_sha256=digest)
    assert export is not None and export.test_pairs is None
    write_golden(tmp_path / "tests" / "fixtures" / "logreg_golden_cases.json", export, synthetic_features(5))
    result = build_package(paths, export, None, expected_logreg_sha256=digest)
    card = (tmp_path / "dist" / "model-v1.0.0" / "model_card.md").read_text(encoding="utf-8")

    assert not result.publishable and result.benchmark is None
    assert result.reasons == ("paridad en los pares de test no ejecutada (faltan los datos de data/processed/)",)
    assert "| pares de test | no ejecutado (faltan los datos) |" in card
    assert "**NO EJECUTADO**: faltan los pares de test" in card


def test_missing_sources_and_foreign_golden_cases_are_not_publishable(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    digest = write_models(tmp_path / "models", "full")
    export = export_model(paths, profile="full", commit=CLEAN, expected_sha256=digest)
    assert export is not None
    result = build_package(paths, export, synthetic_features(300), expected_logreg_sha256=digest)

    assert not result.publishable
    assert result.missing == (
        "tests/fixtures/logreg_golden_cases.json",
        "tests/fixtures/baseline_golden_cases.json",
        "reports/model_card.md",
        "reports/trained_metrics.json",
    )
    assert "casos dorados no ejecutados (falta logreg_golden_cases.json)" in result.reasons
    assert sorted(result.digests) == ["feature_schema.json", "model.onnx"]

    golden = tmp_path / "tests" / "fixtures" / "logreg_golden_cases.json"
    write_golden(golden, export, synthetic_features(3))
    document = json.loads(golden.read_text(encoding="utf-8"))
    document["metadata"]["derived_from"]["sha256"] = "f" * 64
    golden.write_text(json.dumps(document), encoding="utf-8")
    assert export.breakdown is not None
    other = check_golden_cases(golden, export.model_path.read_bytes(), export.breakdown, export.logreg_sha256)
    assert not other.passed and other.skipped_reason is not None and "otro modelo" in other.skipped_reason


def test_ci_profile_package_is_never_publishable(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    write_models(tmp_path / "models" / "ci", "ci")
    write_sources(tmp_path, tmp_path / "models" / "ci")

    message = run_export(paths, profile="ci", commit=CLEAN)
    directory = tmp_path / "dist" / "ci"

    assert "PUBLICABLE: NO" in message and "perfil ci (solo el perfil full es publicable)" in message
    assert (directory / NOT_PUBLISHABLE_MARKER).is_file()
    assert verify_sha256sums(directory)["model.onnx"]
    message.encode("cp1252")  # consola de Windows (M-ML-045)


def test_publishability_lists_every_reason() -> None:
    from thehouse_ml.export.parity import ParityResult

    def parity(passed: bool) -> ParityResult:
        return ParityResult("x", 1 if passed else 0, 1, 0, 0.0, 0.0, 0, 0, 0, 0.0)

    export = ExportResult("ci", Path("m.onnx"), "a" * 64, "b" * 64, "unknown", None, parity(False))
    golden = GoldenResult(cases=1, scored=1, match_score_mismatches=1, parity=parity(True))

    assert publishability(export, golden, ["reports/model_card.md"], "c" * 64) == (
        "perfil ci (solo el perfil full es publicable)",
        "logreg.json no tiene el SHA-256 de la spec ML-06 seccion 1",
        "paridad en los pares de test no ejecutada (faltan los datos de data/processed/)",
        "la paridad en la rejilla fuera del training_range falla",
        "los casos dorados o su paridad fallan",
        "faltan archivos de origen: reports/model_card.md",
        "commit de git desconocido",
    )


def test_verify_sha256sums_detects_a_modified_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"a\n")
    (tmp_path / SHA256SUMS).write_bytes(sha256sums_text({"model.onnx": hashlib.sha256(b"a\n").hexdigest()}))
    (tmp_path / "model.onnx").write_bytes(b"a\n")
    assert verify_sha256sums(tmp_path) == {"model.onnx": True}
    (tmp_path / "model.onnx").write_bytes(b"b\n")
    assert verify_sha256sums(tmp_path) == {"model.onnx": False}


@pytest.mark.parametrize(
    ("median_ms", "expected"), [(0.05, "< 1 ms"), (1.0, "< 5 ms"), (7.0, "< 20 ms"), (20.0, ">= 20 ms")]
)
def test_latency_bound_is_coarse(median_ms: float, expected: str) -> None:
    result = BenchmarkResult(
        rooms=200, warmup_runs=10, timings_ns=(round(median_ms * 1e6),), intra_op_threads=0, inter_op_threads=0
    )
    assert latency_bound(result) == expected


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
    root = tmp_path_factory.mktemp("official_package")
    (root / "models").mkdir()
    for name in ("logreg.json", "train_manifest.json"):
        shutil.copyfile(OFFICIAL_MODELS / name, root / "models" / name)
    shutil.copytree(REPO / "reports", root / "reports")
    shutil.copytree(REPO / "tests" / "fixtures", root / "tests" / "fixtures")
    return root


def test_official_golden_cases_pass_against_the_exported_model(official_root: Path) -> None:
    paths = ProjectPaths(root=official_root)
    export = export_model(paths, profile="full", commit=CLEAN)
    assert export is not None
    result = build_package(paths, export, synthetic_features(300))

    assert result.golden.passed, result.golden.summary()
    assert (result.golden.cases, result.golden.scored) == (109, 98)
    assert result.golden.max_abs_diff_raw_total < 1e-10
    assert result.missing == ()
    # Sin los pares de test oficiales en la raíz temporal: no publicable por ese único motivo.
    assert result.reasons == ("paridad en los pares de test no ejecutada (faltan los datos de data/processed/)",)
    directory = official_root / "dist" / "model-v1.0.0"
    assert (directory / "golden_cases.json").read_bytes() == canonical_text(
        (REPO / "tests" / "fixtures" / "logreg_golden_cases.json").read_bytes()
    )
    assert (directory / "metrics.json").read_bytes() == canonical_text(
        (REPO / "reports" / "trained_metrics.json").read_bytes()
    )
    assert verify_sha256sums(directory) == dict.fromkeys(CHECKSUMMED_FILES, True)
