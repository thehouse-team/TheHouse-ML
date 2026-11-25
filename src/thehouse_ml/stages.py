"""Etapas del pipeline (constitution §9).

Las etapas aún no implementadas informan qué spec las cubre y terminan sin
error. ML-02 T02 implementa ``download``; T03, T04, T05 y T07 implementan
``prepare`` (bases nacionales, calibración regional, entorno DENUE de los campus
y universidades verificadas); ML-03 T04 agrega los estudiantes sintéticos y
T05 el catálogo sintético de cuartos y T06 el conjunto de pares; ML-04 T04 agrega
las etiquetas sintéticas de relevancia y T06 implementa ``evaluate`` (métricas y
reporte del modelo base). ML-05 T02 implementa ``train`` (validación cruzada y
modelos candidatos) y T03 agrega a ``evaluate`` la medición única en test de los
modelos entrenados y la decisión de promoción. ML-06 T01 implementa ``export``
(verificación de ``logreg.json`` y ``model.onnx``) y T05 completa su paquete de entrega.
"""

from __future__ import annotations

from dataclasses import dataclass

from thehouse_ml.data_card import run_data_card
from thehouse_ml.denue import run_denue
from thehouse_ml.download import Fetcher, run_download
from thehouse_ml.models.baseline_report import EVALUATE_SKIP_MESSAGE, run_evaluate
from thehouse_ml.national import run_prepare
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import run_regional
from thehouse_ml.synthetic.labels_stage import run_labels
from thehouse_ml.synthetic.pairs_stage import run_pairs
from thehouse_ml.synthetic.rooms_stage import run_rooms
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
    Stage("prepare", "Limpia, pondera, calibra y genera estudiantes, cuartos, pares y etiquetas sintéticos.", "ML-02"),
    Stage("train", "Entrena los modelos candidatos con validación cruzada (el modelo base no se entrena).", "ML-05"),
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
    get_stage(name)  # KeyError si la etapa no existe.
    if name == "download":
        return run_download(paths, source_id=source, fetcher=fetcher)
    if name == "prepare":
        del source, fetcher  # La etapa prepare no usa red ni filtro de fuente.
        return (
            f"{run_prepare(paths)}\n{run_regional(paths)}\n{run_universities(paths)}\n"
            f"{run_denue(paths)}\n{run_students(paths)}\n{run_rooms(paths)}\n"
            f"{run_pairs(paths)}\n{run_labels(paths)}\n{run_data_card(paths)}"
        )
    if name == "train":
        del source, fetcher  # La etapa train no usa red ni filtro de fuente.
        from thehouse_ml.models.train import run_train  # Diferido: scikit-learn y LightGBM solo al entrenar.

        return run_train(paths)
    if name == "evaluate":
        del source, fetcher  # La etapa evaluate no usa red ni filtro de fuente.
        baseline = run_evaluate(paths)
        if baseline == EVALUATE_SKIP_MESSAGE:
            return baseline
        # ML-05 T03: medición única en test de los modelos de `train` (diferido: LightGBM solo al evaluarlos).
        from thehouse_ml.models.trained_eval import run_trained_evaluate

        return f"{baseline}\n{run_trained_evaluate(paths)}"
    # export (ML-06 T01): la única etapa que queda; `get_stage` ya rechazó los nombres desconocidos.
    del source, fetcher  # La etapa export no usa red ni filtro de fuente.
    from thehouse_ml.export.package import run_export  # Diferido: onnx y onnxruntime solo al exportar.

    return run_export(paths)
