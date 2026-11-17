"""Etapas del pipeline (constitution §9).

Las etapas aún no implementadas informan qué spec las cubre y terminan sin
error. Cada spec posterior reemplaza la suya por la implementación real:
ML-02 T02 implementa ``download``.
"""

from __future__ import annotations

from dataclasses import dataclass

from thehouse_ml.download import Fetcher, run_download
from thehouse_ml.paths import ProjectPaths


@dataclass(frozen=True)
class Stage:
    """Una etapa del pipeline y la spec que la implementa."""

    name: str
    description: str
    spec: str


STAGES: tuple[Stage, ...] = (
    Stage("download", "Descarga las fuentes registradas y verifica su SHA-256.", "ML-02"),
    Stage("prepare", "Limpia, pondera, calibra y genera features y estudiantes sintéticos.", "ML-02"),
    Stage("train", "Entrena el modelo base y los modelos candidatos.", "ML-04"),
    Stage("evaluate", "Evalúa los modelos contra el modelo base y escribe los reportes.", "ML-04"),
    Stage("export", "Exporta el modelo promovido a ONNX con su paquete de entrega.", "ML-06"),
)

STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in STAGES)


def get_stage(name: str) -> Stage:
    """Devuelve la etapa por nombre o lanza `KeyError` si no existe."""
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(name)


def run_stage(name: str, paths: ProjectPaths, source: str | None = None, fetcher: Fetcher | None = None) -> str:
    """Ejecuta una etapa y devuelve el mensaje para la consola.

    ``source`` filtra la descarga por id; ``fetcher`` permite inyectar la red
    en las pruebas (la CLI usa el valor por defecto).
    """
    stage = get_stage(name)
    if name == "download":
        return run_download(paths, source_id=source, fetcher=fetcher)
    del source, fetcher  # Las etapas vacías no leen ni escriben archivos.
    return f"{stage.name}: pendiente de {stage.spec}"
