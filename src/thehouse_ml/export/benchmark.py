"""Rendimiento de `model.onnx` en onnxruntime CPU (ML-06 T04, ML-06-R04, spec §6.5).

Medición de la spec §6.5 (constitución §13.5):

- onnxruntime en CPU (`CPUExecutionProvider`); la sesión se crea **una vez**, antes de medir (la
  carga no cuenta en la latencia por solicitud);
- lote de **200 cuartos** `[200, 10]` en double: los primeros 200 pares de test de ML-05, en el
  orden de `data/processed/` (determinista);
- **10 corridas de calentamiento** descartadas y **100 corridas medidas** con
  `time.perf_counter_ns`, cada una alrededor de una sola llamada a `session.run`;
- criterio: **mediana < 20 ms**. Se reportan además p95 (percentil lineal de `numpy`), máximo,
  mínimo y media, la configuración de hilos y el CPU/SO de la máquina.

La medición oficial se hace en la máquina local documentada en la evidencia, sobre el `model.onnx`
que escribe la etapa `export`::

    uv run python -m thehouse_ml export
    uv run python -m thehouse_ml.export.benchmark

En CI corre solo como prueba de humo informativa (sin umbral estricto, spec §6.5).
"""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from thehouse_ml.export import ExportError
from thehouse_ml.export.onnx_graph import FACTOR_OUTPUT, INPUT_NAME, TOTAL_OUTPUT

BENCHMARK_ROOMS = 200
WARMUP_RUNS = 10
MEASURED_RUNS = 100
MEDIAN_BUDGET_MS = 20.0
P95_PERCENTILE = 95.0
FEATURE_COUNT = 10
DEFAULT_THREADS = 0  # 0 = valor por omisión de onnxruntime (un hilo por núcleo físico en intra-op).
NS_PER_MS = 1_000_000


def benchmark_batch(features: np.ndarray, rooms: int = BENCHMARK_ROOMS) -> np.ndarray:
    """Las primeras `rooms` filas de `features` como lote contiguo `[rooms, 10]` en double."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise ExportError(f"el lote debe tener forma [N, {FEATURE_COUNT}]; se recibió {list(matrix.shape)}")
    if rooms < 1 or matrix.shape[0] < rooms:
        raise ExportError(f"se necesitan {rooms} cuartos para el lote; hay {matrix.shape[0]}")
    return np.ascontiguousarray(matrix[:rooms])


def cpu_name() -> str:
    """Nombre comercial del CPU (registro de Windows o `/proc/cpuinfo`); si no, `platform.processor()`."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as handle:
                value, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "desconocido"


def environment(onnxruntime_version: str, providers: Sequence[str]) -> dict[str, str]:
    """CPU/SO, Python y onnxruntime de la máquina que mide (spec §6.5)."""
    return {
        "cpu": cpu_name(),
        "logical_cpus": str(os.cpu_count() or 0),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime_version,
        "providers": ",".join(providers),
    }


@dataclass(frozen=True)
class BenchmarkResult:
    """Tiempos de `session.run` sobre el lote, en ns, y la configuración con la que se midieron."""

    rooms: int
    warmup_runs: int
    timings_ns: tuple[int, ...]
    intra_op_threads: int
    inter_op_threads: int
    environment: Mapping[str, str] = field(default_factory=dict)
    budget_ms: float = MEDIAN_BUDGET_MS

    def _ms(self) -> np.ndarray:
        if not self.timings_ns:
            raise ExportError("no hay corridas medidas")
        return np.asarray(self.timings_ns, dtype=np.float64) / NS_PER_MS

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self._ms().tolist()))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self._ms(), P95_PERCENTILE))

    @property
    def max_ms(self) -> float:
        return float(self._ms().max())

    @property
    def min_ms(self) -> float:
        return float(self._ms().min())

    @property
    def mean_ms(self) -> float:
        return float(self._ms().mean())

    @property
    def passed(self) -> bool:
        """Criterio de la spec §6.5: mediana < `budget_ms` (20 ms)."""
        return bool(self.timings_ns) and self.median_ms < self.budget_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "rooms": self.rooms,
            "warmup_runs": self.warmup_runs,
            "measured_runs": len(self.timings_ns),
            "timer": "time.perf_counter_ns",
            "session_created_once": True,
            "intra_op_num_threads": self.intra_op_threads,
            "inter_op_num_threads": self.inter_op_threads,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "budget_ms": self.budget_ms,
            "criterion": "median < budget",
            "passed": self.passed,
            "environment": dict(self.environment),
        }

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FALLA"
        threads = f"intra_op {self.intra_op_threads}, inter_op {self.inter_op_threads} (0 = por omision)"
        return (
            f"{self.rooms} cuartos, {self.warmup_runs} calentamiento + {len(self.timings_ns)} medidas "
            f"({threads}): mediana {self.median_ms:.4f} ms, p95 {self.p95_ms:.4f} ms, "
            f"max {self.max_ms:.4f} ms; criterio mediana < {self.budget_ms:g} ms: {verdict}"
        )


def run_benchmark(
    model_bytes: bytes,
    batch: np.ndarray,
    *,
    warmup_runs: int = WARMUP_RUNS,
    measured_runs: int = MEASURED_RUNS,
    intra_op_threads: int = DEFAULT_THREADS,
    inter_op_threads: int = DEFAULT_THREADS,
    budget_ms: float = MEDIAN_BUDGET_MS,
) -> BenchmarkResult:
    """Mide `session.run` sobre `batch` con la sesión ya creada (spec §6.5).

    La sesión y el tensor de entrada se preparan antes de medir; cada medida cubre una sola
    llamada a `session.run` con las dos salidas del contrato.
    """
    import onnxruntime  # Diferido, como en `run_onnx` (override de mypy, M-ML-045).

    if warmup_runs < 0 or measured_runs < 1:
        raise ExportError("se necesita al menos 1 corrida medida y un calentamiento no negativo")
    if intra_op_threads < 0 or inter_op_threads < 0:
        raise ExportError("el número de hilos no puede ser negativo")
    feed = np.ascontiguousarray(np.asarray(batch, dtype=np.float64))
    if feed.ndim != 2 or feed.shape[1] != FEATURE_COUNT or feed.shape[0] < 1:
        raise ExportError(f"el lote debe tener forma [N, {FEATURE_COUNT}] con N >= 1; se recibió {list(feed.shape)}")

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    session = onnxruntime.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    outputs = [FACTOR_OUTPUT, TOTAL_OUTPUT]
    inputs = {INPUT_NAME: feed}

    points, total = session.run(outputs, inputs)
    if np.shape(points) != (feed.shape[0], 6) or np.shape(total) != (feed.shape[0],):
        raise ExportError("las salidas del ONNX no tienen la forma del contrato")
    for _ in range(warmup_runs):
        session.run(outputs, inputs)
    timings: list[int] = []
    for _ in range(measured_runs):
        start = time.perf_counter_ns()
        session.run(outputs, inputs)
        timings.append(time.perf_counter_ns() - start)

    return BenchmarkResult(
        rooms=int(feed.shape[0]),
        warmup_runs=warmup_runs,
        timings_ns=tuple(timings),
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
        environment=environment(str(onnxruntime.__version__), session.get_providers()),
        budget_ms=budget_ms,
    )


def main(argv: Sequence[str] | None = None, root: Path | None = None) -> int:
    """Medición oficial sobre `dist/<paquete>/model.onnx` y los pares de test; 0 si pasa, 1 si no."""
    import json

    from thehouse_ml.export.package import DIST_DIR, MODEL_ONNX, PACKAGE_DIRS, load_test_pair_features
    from thehouse_ml.models.train_config import resolve_profile
    from thehouse_ml.paths import ProjectPaths

    args = list(sys.argv[1:] if argv is None else argv)
    threads = int(args[0]) if args else DEFAULT_THREADS
    paths = ProjectPaths(root=root or Path.cwd())
    profile = resolve_profile()
    model_path = paths.root / DIST_DIR / PACKAGE_DIRS[profile] / MODEL_ONNX
    if not model_path.is_file():
        print(f"benchmark: falta {model_path.relative_to(paths.root).as_posix()} (ejecute `export`)")
        return 1
    features = load_test_pair_features(paths)
    if features is None:
        print("benchmark: faltan los pares de test de data/processed/ (ejecute `prepare`)")
        return 1
    result = run_benchmark(model_path.read_bytes(), benchmark_batch(features), intra_op_threads=threads)
    print(f"benchmark: {model_path.relative_to(paths.root).as_posix()} (perfil {profile})")
    print(f"  {result.summary()}")
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=True))
    return 0 if result.passed else 1


__all__ = [
    "BENCHMARK_ROOMS",
    "DEFAULT_THREADS",
    "MEASURED_RUNS",
    "MEDIAN_BUDGET_MS",
    "WARMUP_RUNS",
    "BenchmarkResult",
    "benchmark_batch",
    "cpu_name",
    "environment",
    "main",
    "run_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
