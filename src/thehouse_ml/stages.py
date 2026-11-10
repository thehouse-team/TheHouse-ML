"""Etapas del pipeline (constitution §9).

En ML-01 todas están vacías: informan qué spec las implementa y terminan
sin error. Cada spec posterior reemplaza la suya por la implementación real.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def run_stage(name: str, paths: ProjectPaths) -> str:
    """Ejecuta una etapa y devuelve el mensaje para la consola.

    `paths` se recibe desde ahora para que las etapas reales no cambien su firma.
    """
    stage = get_stage(name)
    del paths  # Las etapas vacías no leen ni escriben archivos.
    return f"{stage.name}: pendiente de {stage.spec}"
