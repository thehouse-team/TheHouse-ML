"""Etapa `export` (ML-06): verifica `logreg.json`, construye `model.onnx` y arma el paquete de entrega.

ML-06 T01 (ML-06-R01) implementa la parte del modelo de la etapa: verificación del SHA-256 de
`logreg.json` (contra `train_manifest.json` y, en el perfil `full`, contra el valor fijado en la
spec §1), grafo de la spec §6.2 con sus metadatos, `onnx.checker`, escritura determinista de
`model.onnx` y ejecución en onnxruntime CPU sobre los pares de test (si `data/processed/` existe).
ML-06 T02 (ML-06-R02) agrega la paridad formal Python ↔ onnxruntime (`parity.py`) en los pares de
test y en la rejilla fuera del `training_range`.

ML-06 T05 (ML-06-R05, spec §6.6) completa el paquete en `dist/<paquete>/`:

| Archivo | Origen |
| --- | --- |
| `model.onnx` | Grafo de la spec §6.2 (`export_model`) |
| `feature_schema.json` | `src/thehouse_ml/features/feature_schema.json` 2.0.0 |
| `golden_cases.json` | `tests/fixtures/logreg_golden_cases.json` |
| `baseline_golden_cases.json` | `tests/fixtures/baseline_golden_cases.json` |
| `model_card.md` | `reports/model_card.md` + sección «Artefacto ONNX» generada aquí |
| `metrics.json` | `reports/trained_metrics.json` |
| `SHA256SUMS` | SHA-256 de los 6 anteriores, formato `sha256sum` |

Las copias son del **contenido versionado en git**: los finales de línea CRLF del árbol de trabajo
de Windows se normalizan a LF (`.gitattributes`: `* text=auto eol=lf`), para que el paquete tenga
los mismos SHA-256 en cualquier clon. Nada del paquete depende de la hora ni de corridas
aleatorias: la latencia exacta varía entre corridas, así que la model card lleva el veredicto y una
cota (`mediana < 1 ms`), y las cifras exactas van a la consola y a la evidencia.

**Publicable** (spec §6.6, M-ML-046 y M-ML-045) solo si: perfil `full`; `logreg.json` con el SHA-256
de la spec §1; paridad en los pares de test, en la rejilla y en los casos dorados en verde (un
control no ejecutado no cuenta como verde); los casos dorados coinciden con `model.onnx`; los 6
archivos de origen existen; y el árbol de git está limpio (`gitCommit` sin `-dirty`). Si no, el
paquete se escribe igual, la model card dice **NO publicable** con los motivos y se agrega
`NO_PUBLICABLE.txt` (fuera de `SHA256SUMS` y de los 7 archivos del release).

Perfiles (`THEHOUSE_ML_TRAIN_PROFILE`, como `train` y `evaluate`): `full` lee `models/` y escribe
en `dist/model-v1.0.0/`; `ci` lee `profiles.ci.output_dir` y escribe en `dist/ci/` (no publicable).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from thehouse_ml import __version__
from thehouse_ml.export.benchmark import BENCHMARK_ROOMS, BenchmarkResult, benchmark_batch, run_benchmark
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
from thehouse_ml.export.parity import (
    RAW_TOLERANCE,
    ParityResult,
    backend_factor_points,
    backend_match_score,
    check_parity,
    out_of_range_grid,
)
from thehouse_ml.features import default_schema
from thehouse_ml.models.model_card import MODEL_CARD, TRAINED_METRICS_JSON
from thehouse_ml.models.train import LOGREG_MODEL, LogRegModel, output_dirs
from thehouse_ml.models.train_config import DEFAULT_PROFILE, load_train_params, resolve_profile
from thehouse_ml.models.trained_breakdown import BreakdownModel, build_breakdown_model
from thehouse_ml.models.training_data import TRAINING_INPUTS, load_training_data
from thehouse_ml.paths import ProjectPaths

MODEL_ONNX = "model.onnx"
DIST_DIR = "dist"
PACKAGE_DIRS = {DEFAULT_PROFILE: "model-v1.0.0", "ci": "ci"}
TEST_SPLIT = "test"
UNKNOWN_COMMIT = "unknown"
DIRTY_SUFFIX = "-dirty"

# Archivos del paquete (spec §6.6), en el orden de la tabla.
FEATURE_SCHEMA = "feature_schema.json"
GOLDEN_CASES = "golden_cases.json"
BASELINE_GOLDEN_CASES = "baseline_golden_cases.json"
METRICS = "metrics.json"
SHA256SUMS = "SHA256SUMS"
CHECKSUMMED_FILES = (MODEL_ONNX, FEATURE_SCHEMA, GOLDEN_CASES, BASELINE_GOLDEN_CASES, MODEL_CARD, METRICS)
PACKAGE_FILES = (*CHECKSUMMED_FILES, SHA256SUMS)
NOT_PUBLISHABLE_MARKER = "NO_PUBLICABLE.txt"

FIXTURES_DIR = Path("tests") / "fixtures"
LOGREG_GOLDEN_FIXTURE = "logreg_golden_cases.json"
BASELINE_GOLDEN_FIXTURE = "baseline_golden_cases.json"
SCHEMA_SOURCE = Path(__file__).resolve().parents[1] / "features" / FEATURE_SCHEMA

ONNX_SECTION_TITLE = "## 11. Artefacto ONNX"
# Cotas de la mediana de latencia en la model card (ms): la menor que la mediana no alcanza.
LATENCY_BOUNDS_MS = (1.0, 5.0)


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
    return head.stdout.strip() + (DIRTY_SUFFIX if dirty else "")


@dataclass(frozen=True)
class ExportResult:
    profile: str
    model_path: Path
    model_sha256: str
    logreg_sha256: str
    commit: str
    test_pairs: ParityResult | None
    grid: ParityResult
    breakdown: BreakdownModel | None = field(default=None, compare=False, repr=False)


def load_test_pair_features(paths: ProjectPaths) -> np.ndarray | None:
    """Vectores [N, 10] de los pares de test de ML-05; `None` si faltan los datos de `data/processed/`."""
    if not all((paths.data_processed / name).is_file() for name in TRAINING_INPUTS):
        return None
    data = load_training_data(paths, default_schema())
    mask = (data.split == TEST_SPLIT).to_numpy()
    features: np.ndarray = data.features[mask].to_numpy(dtype=np.float64, copy=True)
    return features


def run_on_test_pairs(
    model_bytes: bytes, model: BreakdownModel, paths: ProjectPaths, features: np.ndarray | None = None
) -> ParityResult | None:
    """Paridad Python ↔ ONNX en los pares de test (spec §6.3); `None` si faltan los datos."""
    if features is None:
        features = load_test_pair_features(paths)
    if features is None:
        return None
    return check_parity("pares de test", model_bytes, model, features)


def run_on_grid(model_bytes: bytes, model: BreakdownModel) -> ParityResult:
    """Paridad en la rejilla fuera del `training_range` (límites del recorte de `logreg.json`)."""
    grid = out_of_range_grid(model.logreg.clip_lower, model.logreg.clip_upper)
    return check_parity("rejilla fuera del training_range", model_bytes, model, grid)


def export_model(
    paths: ProjectPaths,
    profile: str | None = None,
    commit: str | None = None,
    expected_sha256: str | None = None,
    test_features: np.ndarray | None = None,
) -> ExportResult | None:
    """Verifica `logreg.json`, construye y escribe `model.onnx` y lo ejecuta; `None` si no hay modelo.

    `expected_sha256` solo se usa en pruebas; en el perfil `full` el valor fijado es
    `EXPECTED_LOGREG_SHA256` y en `ci` no hay valor fijado (solo el del manifiesto).
    `test_features` evita volver a leer los pares de test si quien llama ya los cargó.
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
    out_dir = package_dir(paths, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_ONNX
    model_path.write_bytes(model_bytes)
    return ExportResult(
        profile=profile,
        model_path=model_path,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        logreg_sha256=logreg_sha256,
        commit=commit,
        test_pairs=run_on_test_pairs(model_bytes, breakdown, paths, test_features),
        grid=run_on_grid(model_bytes, breakdown),
        breakdown=breakdown,
    )


def package_dir(paths: ProjectPaths, profile: str) -> Path:
    return paths.root / DIST_DIR / PACKAGE_DIRS[profile]


# ---------------------------------------------------------------------------
# Casos dorados contra el ONNX (spec §6.4 y §6.6: «casos dorados en verde»)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenResult:
    """Casos dorados de `logreg-1.0.0` contra `model.onnx`: esperado del recuento independiente y paridad."""

    cases: int
    scored: int
    skipped_reason: str | None = None
    max_abs_diff_factor_points: float = 0.0
    max_abs_diff_raw_total: float = 0.0
    rounded_points_mismatches: int = 0
    match_score_mismatches: int = 0
    mismatched_ids: tuple[str, ...] = ()
    parity: ParityResult | None = None

    @property
    def passed(self) -> bool:
        return (
            self.skipped_reason is None
            and self.scored > 0
            and self.max_abs_diff_factor_points <= RAW_TOLERANCE
            and self.max_abs_diff_raw_total <= RAW_TOLERANCE
            and self.rounded_points_mismatches == 0
            and self.match_score_mismatches == 0
            and self.parity is not None
            and self.parity.passed
        )

    def summary(self) -> str:
        if self.skipped_reason is not None:
            return f"casos dorados: no ejecutado; {self.skipped_reason}"
        verdict = "PASS" if self.passed else "FALLA"
        return (
            f"casos dorados: {self.cases} casos, {self.scored} puntuados contra el esperado; max |dif| "
            f"factor_points {self.max_abs_diff_factor_points:.3e}, raw_total {self.max_abs_diff_raw_total:.3e} "
            f"(tolerancia {RAW_TOLERANCE:g}); dif en puntos redondeados {self.rounded_points_mismatches}, "
            f"en matchScore {self.match_score_mismatches} -> {verdict}"
        )


def _golden_vector(case: dict[str, Any], order: Sequence[str]) -> list[float]:
    source = case["input"]["features"] if "features" in case["input"] else case["expected"]["features"]
    return [float(source[name]) for name in order]


def check_golden_cases(
    golden_path: Path, model_bytes: bytes, model: BreakdownModel, logreg_sha256: str
) -> GoldenResult:
    """Corre los casos dorados puntuados en `model.onnx` y los compara con su esperado (spec §6.4).

    Sin redondear: |dif| ≤ 1e-4 contra `expected.onnx`; redondeados con la regla del backend
    (`backend_factor_points`, `backend_match_score`): 0 diferencias contra `expected.result`.
    Además, la paridad Python ↔ ONNX de la spec §6.3 sobre los mismos vectores.
    """
    if not golden_path.is_file():
        return GoldenResult(cases=0, scored=0, skipped_reason=f"falta {golden_path.name}")
    document = json.loads(golden_path.read_text(encoding="utf-8"))
    meta, cases = document["metadata"], document["cases"]
    if meta["derived_from"]["sha256"] != logreg_sha256 or meta["modelVersion"] != model.model_version:
        return GoldenResult(
            cases=len(cases), scored=0, skipped_reason="los casos dorados son de otro modelo (SHA-256 de logreg.json)"
        )
    order = meta["featureOrder"]
    factor_order = meta["factorOrder"]
    scored = [case for case in cases if "onnx" in case["expected"]]
    vectors = np.asarray([_golden_vector(case, order) for case in scored], dtype=np.float64)
    points, totals = run_onnx(model_bytes, vectors)
    expected_points = np.asarray([case["expected"]["onnx"]["factor_points"] for case in scored], dtype=np.float64)
    expected_totals = np.asarray([case["expected"]["onnx"]["raw_total"] for case in scored], dtype=np.float64)
    rounded_mismatches = score_mismatches = 0
    mismatched: list[str] = []
    for row, case in enumerate(scored):
        result = case["expected"]["result"]
        by_factor = {item["factor"]: item["points"] for item in result["breakdown"]}
        rounded = [backend_factor_points(float(value), model.rounding) for value in points[row]]
        points_ok = rounded == [float(by_factor[name]) for name in factor_order]
        score_ok = backend_match_score(float(totals[row]), model.rounding) == result["matchScore"]
        rounded_mismatches += not points_ok
        score_mismatches += not score_ok
        if not (points_ok and score_ok):
            mismatched.append(case["id"])
    return GoldenResult(
        cases=len(cases),
        scored=len(scored),
        max_abs_diff_factor_points=float(np.abs(points - expected_points).max(initial=0.0)),
        max_abs_diff_raw_total=float(np.abs(totals - expected_totals).max(initial=0.0)),
        rounded_points_mismatches=rounded_mismatches,
        match_score_mismatches=score_mismatches,
        mismatched_ids=tuple(mismatched),
        parity=check_parity("casos dorados", model_bytes, model, vectors),
    )


# ---------------------------------------------------------------------------
# Paquete (spec §6.6)
# ---------------------------------------------------------------------------


def canonical_text(data: bytes) -> bytes:
    """Contenido versionado en git de un archivo de texto: CRLF → LF (`.gitattributes` eol=lf)."""
    return data.replace(b"\r\n", b"\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256sums_text(digests: dict[str, str]) -> bytes:
    """`SHA256SUMS` en formato `sha256sum` (modo texto: dos espacios), LF, en el orden de la spec §6.6."""
    lines = [f"{digests[name]}  {name}\n" for name in CHECKSUMMED_FILES if name in digests]
    return "".join(lines).encode("ascii")


def verify_sha256sums(directory: Path) -> dict[str, bool]:
    """Equivalente a `sha256sum -c SHA256SUMS`: {archivo: coincide} para cada línea del archivo."""
    checks: dict[str, bool] = {}
    for line in (directory / SHA256SUMS).read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        target = directory / name
        checks[name] = target.is_file() and sha256_bytes(target.read_bytes()) == digest
    return checks


def package_sources(paths: ProjectPaths, profile: str) -> dict[str, Path]:
    """Archivos de origen de las copias del paquete (spec §6.6); el perfil `ci` usa sus reportes."""
    _, reports_dir = output_dirs(paths, load_train_params(), profile)
    fixtures = paths.root / FIXTURES_DIR
    return {
        FEATURE_SCHEMA: SCHEMA_SOURCE,
        GOLDEN_CASES: fixtures / LOGREG_GOLDEN_FIXTURE,
        BASELINE_GOLDEN_CASES: fixtures / BASELINE_GOLDEN_FIXTURE,
        MODEL_CARD: reports_dir / MODEL_CARD,
        METRICS: reports_dir / TRAINED_METRICS_JSON,
    }


@dataclass(frozen=True)
class PackageResult:
    profile: str
    directory: Path
    publishable: bool
    reasons: tuple[str, ...]
    digests: dict[str, str]
    golden: GoldenResult
    benchmark: BenchmarkResult | None
    missing: tuple[str, ...]


def latency_bound(result: BenchmarkResult) -> str:
    """Cota estable de la mediana para la model card (las cifras exactas cambian entre corridas)."""
    for bound in LATENCY_BOUNDS_MS:
        if result.median_ms < bound:
            return f"< {bound:g} ms"
    if result.passed:
        return f"< {result.budget_ms:g} ms"
    return f">= {result.budget_ms:g} ms"


def publishability(
    export: ExportResult, golden: GoldenResult, missing: Sequence[str], expected_logreg_sha256: str
) -> tuple[str, ...]:
    """Motivos por los que el paquete NO es publicable (vacío = publicable; spec §6.6)."""
    reasons: list[str] = []
    if export.profile != DEFAULT_PROFILE:
        reasons.append(f"perfil {export.profile} (solo el perfil {DEFAULT_PROFILE} es publicable)")
    if export.logreg_sha256 != expected_logreg_sha256:
        reasons.append("logreg.json no tiene el SHA-256 de la spec ML-06 seccion 1")
    if export.test_pairs is None:
        reasons.append("paridad en los pares de test no ejecutada (faltan los datos de data/processed/)")
    elif not export.test_pairs.passed:
        reasons.append("la paridad en los pares de test falla")
    if not export.grid.passed:
        reasons.append("la paridad en la rejilla fuera del training_range falla")
    if golden.skipped_reason is not None:
        reasons.append(f"casos dorados no ejecutados ({golden.skipped_reason})")
    elif not golden.passed:
        reasons.append("los casos dorados o su paridad fallan")
    if missing:
        reasons.append("faltan archivos de origen: " + ", ".join(missing))
    if export.commit == UNKNOWN_COMMIT:
        reasons.append("commit de git desconocido")
    elif export.commit.endswith(DIRTY_SUFFIX):
        reasons.append("arbol de git sucio (gitCommit con -dirty)")
    return tuple(reasons)


def _parity_row(name: str, result: ParityResult | None) -> str:
    if result is None:
        return f"| {name} | no ejecutado (faltan los datos) | — | — | — | — | **NO EJECUTADO** |"
    verdict = "PASS" if result.passed else "FALLA"
    return (
        f"| {result.name} | {result.rows:,} | {result.max_abs_diff_factor_points:.1e} | "
        f"{result.max_abs_diff_raw_total:.1e} | {result.rounded_points_mismatches} | "
        f"{result.match_score_mismatches} | **{verdict}** |"
    )


def onnx_section(
    export: ExportResult,
    model_bytes: bytes,
    golden: GoldenResult,
    benchmark: BenchmarkResult | None,
    reasons: Sequence[str],
) -> str:
    """Sección «Artefacto ONNX» de la model card del paquete (spec §6.6); determinista."""
    model = onnx.load_from_string(model_bytes)
    props = {prop.key: prop.value for prop in model.metadata_props}
    feature_order = json.loads(props["featureOrder"])
    factor_order = json.loads(props["factorOrder"])
    status = "**SÍ**" if not reasons else "**NO** — " + "; ".join(reasons)
    lines = [
        ONNX_SECTION_TITLE,
        "",
        "> **DATOS SINTÉTICOS.** Sección generada por `python -m thehouse_ml export` (ML-06 T05, spec ML-06 §6.6); "
        "no forma parte de `reports/model_card.md`. Las cifras de paridad comparan implementaciones del mismo modelo, "
        "no miden desempeño.",
        "",
        "| Campo | Valor |",
        "| --- | --- |",
        f"| Paquete | `{PACKAGE_DIRS[export.profile]}` (perfil `{export.profile}`) |",
        f"| Publicable | {status} |",
        f"| Modelo | `{props['modelVersion']}` · esquema de features `{props['featureSchemaVersion']}` |",
        f"| Commit de git (`gitCommit`) | `{export.commit}` |",
        f"| `logreg.json` SHA-256 | `{export.logreg_sha256}` |",
        f"| `model.onnx` SHA-256 | `{export.model_sha256}` |",
        f"| Opset / IR | {model.opset_import[0].version} / {model.ir_version} |",
        f"| Productor | `{model.producer_name}` {model.producer_version} |",
        "| Proveedor de ejecución | onnxruntime `CPUExecutionProvider` (en .NET: `Microsoft.ML.OnnxRuntime`) |",
        "",
        "### Entradas y salidas",
        "",
        "| Nombre | Tipo | Forma | Contenido |",
        "| --- | --- | --- | --- |",
        f"| `features` (entrada) | `tensor(double)` | [N, {len(feature_order)}] | "
        + ", ".join(f"`{name}`" for name in feature_order)
        + " (orden de `featureOrder`; sin recortar: el grafo recorta al `training_range`) |",
        f"| `factor_points` (salida) | `tensor(double)` | [N, {len(factor_order)}] | Puntos sin redondear de "
        + ", ".join(f"`{name}`" for name in factor_order)
        + " |",
        "| `raw_total` (salida) | `tensor(double)` | [N] | Suma sin redondear de `factor_points`, "
        "de izquierda a derecha |",
        "",
        "El ONNX solo puntúa pares que pasaron el perfil incompleto y HC01-HC03; el backend (NV-23) redondea "
        "(round6 y 1 decimal por factor; `matchScore = floor(round6(raw_total) + 0.5)`; "
        "`MidpointRounding.AwayFromZero`), normaliza `-0` a `0` (M-ML-046), arma el JSON y ordena (spec ML-06 §6.1).",
        "",
        "### Metadatos (`metadata_props`)",
        "",
        "| Clave | Valor |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in sorted(props.items())],
        "",
        "### Paridad Python ↔ onnxruntime",
        "",
        f"Tolerancia: \\|dif\\| ≤ {RAW_TOLERANCE:g} sin redondear; 0 diferencias en puntos redondeados y "
        "`matchScore` (spec ML-06 §6.3).",
        "",
        "| Conjunto | Vectores | Máx. \\|dif\\| `factor_points` | Máx. \\|dif\\| `raw_total` | "
        "Dif. redondeados | Dif. `matchScore` | Resultado |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        _parity_row("pares de test", export.test_pairs),
        _parity_row("rejilla fuera del training_range", export.grid),
        _parity_row("casos dorados", golden.parity),
        "",
    ]
    if golden.skipped_reason is not None:
        lines.append(
            f"Casos dorados (`golden_cases.json`) contra `model.onnx`: **NO EJECUTADO** ({golden.skipped_reason})."
        )
    else:
        verdict = "PASS" if golden.passed else "FALLA"
        lines.append(
            f"Casos dorados (`golden_cases.json`, recuento independiente) contra `model.onnx`: {golden.cases} casos, "
            f"{golden.scored} puntuados; máx. \\|dif\\| {golden.max_abs_diff_factor_points:.1e} (`factor_points`) y "
            f"{golden.max_abs_diff_raw_total:.1e} (`raw_total`); {golden.rounded_points_mismatches} diferencias en "
            f"puntos redondeados y {golden.match_score_mismatches} en `matchScore`: **{verdict}**."
        )
    lines += ["", "### Rendimiento", ""]
    if benchmark is None:
        lines.append("**NO EJECUTADO**: faltan los pares de test de `data/processed/` (spec ML-06 §6.5).")
    else:
        env = benchmark.environment
        verdict = "PASS" if benchmark.passed else "FALLA"
        lines += [
            f"onnxruntime CPU, sesión creada una vez (la carga no cuenta), lote de {benchmark.rooms} cuartos "
            f"[{benchmark.rooms}, 10] (los primeros pares de test), {benchmark.warmup_runs} corridas de calentamiento "
            f"y {len(benchmark.timings_ns)} medidas con `time.perf_counter_ns`; hilos intra-op "
            f"{benchmark.intra_op_threads} e inter-op {benchmark.inter_op_threads} (0 = valor por omisión).",
            "",
            "| Campo | Valor |",
            "| --- | --- |",
            f"| Criterio (constitución §13.5) | mediana < {benchmark.budget_ms:g} ms: **{verdict}** |",
            f"| Mediana | {latency_bound(benchmark)} |",
            f"| CPU | {env.get('cpu', '')} ({env.get('logical_cpus', '')} CPUs lógicas) |",
            f"| SO | {env.get('os', '')} |",
            f"| Python / onnxruntime | {env.get('python', '')} / {env.get('onnxruntime', '')} |",
            "",
            "La mediana se da como cota para que el paquete sea reproducible byte a byte; las cifras exactas "
            "(mediana, p95, máximo) varían entre corridas y quedan en la salida de `export` y en "
            "`specs/ml-06-onnx-export/evidence.md` (M-ML-048).",
        ]
    lines += [
        "",
        "### Archivos del paquete",
        "",
        "| Archivo | Origen |",
        "| --- | --- |",
        "| `model.onnx` | Grafo de la spec ML-06 §6.2 construido desde `models/logreg.json` |",
        "| `feature_schema.json` | `src/thehouse_ml/features/feature_schema.json` 2.0.0 |",
        "| `golden_cases.json` | `tests/fixtures/logreg_golden_cases.json` (casos dorados de `logreg-1.0.0`) |",
        "| `baseline_golden_cases.json` | `tests/fixtures/baseline_golden_cases.json` (81 casos, respaldo en C#) |",
        "| `model_card.md` | `reports/model_card.md` más esta sección |",
        "| `metrics.json` | `reports/trained_metrics.json` (DATOS SINTÉTICOS) |",
        "| `SHA256SUMS` | SHA-256 de los 6 archivos anteriores (`sha256sum -c SHA256SUMS`) |",
        "",
        "Las copias son del contenido versionado en git (finales de línea LF). Los enlaces relativos de las "
        "secciones 1-10 apuntan a `reports/` del repositorio `TheHouse-ML`.",
        "",
        "### Desviación conocida ML-05 T05-H1",
        "",
        "Desviación de ML-05 (M-ML-041), no de ML-06: el coeficiente de `price_below_min` es negativo: "
        "por debajo de `budget.min`, el factor presupuesto **sube "
        "con la renta** si `budget.max / budget.min` > 76.2 (subida máxima +0.124 puntos sobre 100; ningún par "
        "sintético observado en esa zona). Se conserva tal cual en este artefacto (M-ML-041) y se corrige en "
        "`logreg-1.1.0`; los casos `t05_h1_zone` de `golden_cases.json` la muestran.",
        "",
        "### Publicación",
        "",
        "El release de GitHub `model-v1.0.0` lo publica el responsable a mano con los 7 archivos, sobre el commit "
        "de `main` que generó el paquete; es inmutable (una corrección es una versión nueva, constitución §13.3). "
        "El backend descarga por tag, verifica `SHA256SUMS` y fija en su repositorio el SHA-256 de `model.onnx`.",
    ]
    return "\n".join(lines) + "\n"


def build_package(
    paths: ProjectPaths,
    export: ExportResult,
    test_features: np.ndarray | None = None,
    expected_logreg_sha256: str = EXPECTED_LOGREG_SHA256,
) -> PackageResult:
    """Escribe el paquete de la spec §6.6 junto a `model.onnx` y decide si es publicable."""
    if export.breakdown is None:
        raise ValueError("build_package necesita el ExportResult de export_model (con breakdown)")
    directory = export.model_path.parent
    for name in (*PACKAGE_FILES[1:], NOT_PUBLISHABLE_MARKER):  # sin restos de corridas anteriores
        (directory / name).unlink(missing_ok=True)
    model_bytes = export.model_path.read_bytes()
    sources = package_sources(paths, export.profile)
    missing = tuple(
        source.relative_to(paths.root).as_posix() if source.is_relative_to(paths.root) else str(source)
        for source in sources.values()
        if not source.is_file()
    )
    golden = check_golden_cases(sources[GOLDEN_CASES], model_bytes, export.breakdown, export.logreg_sha256)
    benchmark = None
    if test_features is not None and len(test_features) >= BENCHMARK_ROOMS:
        benchmark = run_benchmark(model_bytes, benchmark_batch(test_features))
    reasons = publishability(export, golden, missing, expected_logreg_sha256)

    contents: dict[str, bytes] = {MODEL_ONNX: model_bytes}
    for name in (FEATURE_SCHEMA, GOLDEN_CASES, BASELINE_GOLDEN_CASES, METRICS):
        if sources[name].is_file():
            contents[name] = canonical_text(sources[name].read_bytes())
    if sources[MODEL_CARD].is_file():
        card = canonical_text(sources[MODEL_CARD].read_bytes()).rstrip(b"\n") + b"\n\n"
        contents[MODEL_CARD] = card + onnx_section(export, model_bytes, golden, benchmark, reasons).encode("utf-8")
    for name, data in contents.items():
        if name != MODEL_ONNX:
            (directory / name).write_bytes(data)
    digests = {name: sha256_bytes(data) for name, data in contents.items()}
    (directory / SHA256SUMS).write_bytes(sha256sums_text(digests))
    if reasons:
        marker = "Paquete NO publicable (spec ML-06 6.6):\n" + "".join(f"- {reason}\n" for reason in reasons)
        (directory / NOT_PUBLISHABLE_MARKER).write_bytes(marker.encode("utf-8"))
    return PackageResult(
        profile=export.profile,
        directory=directory,
        publishable=not reasons,
        reasons=reasons,
        digests=digests,
        golden=golden,
        benchmark=benchmark,
        missing=missing,
    )


def run_export(paths: ProjectPaths, profile: str | None = None, commit: str | None = None) -> str:
    """Etapa `export`: `model.onnx` verificado, paridad, casos dorados, rendimiento y paquete."""
    profile = profile or resolve_profile()
    models_dir, _ = output_dirs(paths, load_train_params(), profile)
    if not (models_dir / LOGREG_MODEL).is_file():
        return export_skip_message(models_dir, paths.root)
    test_features = load_test_pair_features(paths)
    result = export_model(paths, profile=profile, commit=commit, test_features=test_features)
    if result is None:  # pragma: no cover - comprobado arriba
        return export_skip_message(models_dir, paths.root)
    package = build_package(paths, result, test_features)
    location = result.model_path.relative_to(paths.root).as_posix()
    lines = [
        f"export: {location} (perfil {result.profile}, opset {OPSET}, onnx.checker sin errores, "
        f"thehouse-ml {__version__})",
        f"  logreg.json SHA-256 {result.logreg_sha256} (verificado)",
        f"  model.onnx  SHA-256 {result.model_sha256}",
        f"  commit {result.commit}",
    ]
    lines.append("  paridad Python <-> onnxruntime CPU (spec ML-06, seccion 6.3):")
    if result.test_pairs is None:
        lines.append("    pares de test: omitido; faltan los datos de data/processed/ (ejecute `prepare`)")
    else:
        lines.append(f"    {result.test_pairs.summary()}")
    lines.append(f"    {result.grid.summary()}")
    if package.golden.parity is not None:
        lines.append(f"    {package.golden.parity.summary()}")
    lines.append(f"  {package.golden.summary()}")
    if package.benchmark is None:
        lines.append("  rendimiento: no ejecutado; faltan los pares de test de data/processed/")
    else:
        lines.append(f"  rendimiento: {package.benchmark.summary()}")
    lines.append(f"  paquete {package.directory.relative_to(paths.root).as_posix()}/ (spec ML-06, seccion 6.6):")
    lines += [f"    {package.digests[name]}  {name}" for name in CHECKSUMMED_FILES if name in package.digests]
    if package.publishable:
        lines.append("  PUBLICABLE: si")
    else:
        lines.append(f"  PUBLICABLE: NO (ver {NOT_PUBLISHABLE_MARKER})")
        lines += [f"    - {reason}" for reason in package.reasons]
    return "\n".join(lines)


__all__ = [
    "CHECKSUMMED_FILES",
    "MODEL_ONNX",
    "NOT_PUBLISHABLE_MARKER",
    "PACKAGE_DIRS",
    "PACKAGE_FILES",
    "SHA256SUMS",
    "ExportResult",
    "GoldenResult",
    "PackageResult",
    "build_package",
    "canonical_text",
    "check_golden_cases",
    "export_model",
    "export_skip_message",
    "git_commit",
    "latency_bound",
    "load_test_pair_features",
    "package_dir",
    "publishability",
    "run_export",
    "run_on_grid",
    "run_on_test_pairs",
    "sha256sums_text",
    "verify_sha256sums",
]
