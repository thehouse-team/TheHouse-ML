"""Etapas del pipeline (constitution §9).

Las etapas aún no implementadas informan qué spec las cubre y terminan sin
error. ML-02 T02 implementa ``download``; T03, T04, T05 y T07 implementan
``prepare`` (bases nacionales, calibración regional, entorno DENUE de los campus
y universidades verificadas); ML-03 T04 agrega los estudiantes sintéticos.
"""

from __future__ import annotations

from dataclasses import dataclass

from thehouse_ml.data_card import run_data_card
from thehouse_ml.denue import run_denue
from thehouse_ml.download import Fetcher, run_download
from thehouse_ml.national import run_prepare
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import run_regional
from thehouse_ml.synthetic.students_stage import run_students
from thehouse_ml.universities import run_universities


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
    if name == "prepare":
        del source, fetcher  # La etapa prepare no usa red ni filtro de fuente.
        return (
            f"{run_prepare(paths)}\n{run_regional(paths)}\n{run_universities(paths)}\n"
            f"{run_denue(paths)}\n{run_students(paths)}\n{run_data_card(paths)}"
        )
    del source, fetcher  # Las etapas vacías no leen ni escriben archivos.
    return f"{stage.name}: pendiente de {stage.spec}"
