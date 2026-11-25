"""Línea de comandos: `python -m thehouse_ml <etapa>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from thehouse_ml import __version__
from thehouse_ml.data_card import DataCardError
from thehouse_ml.denue import DenueError
from thehouse_ml.download import DownloadError
from thehouse_ml.export import ExportError
from thehouse_ml.manifests import ManifestError
from thehouse_ml.models.baseline_report import EvaluationError
from thehouse_ml.national import PrepareError
from thehouse_ml.paths import ROOT_ENV_VAR, ProjectPaths
from thehouse_ml.regional import RegionalError
from thehouse_ml.sources import SourceRegistryError
from thehouse_ml.stages import STAGES, run_stage
from thehouse_ml.synthetic.labels import LabelInputError, LabelParamsError
from thehouse_ml.synthetic.labels_stage import LabelsError
from thehouse_ml.synthetic.pairs import PairParamsError, PairSetError
from thehouse_ml.synthetic.pairs_stage import PairsError
from thehouse_ml.synthetic.rooms import RoomParamsError
from thehouse_ml.synthetic.rooms_stage import RoomsError
from thehouse_ml.synthetic.students import StudentParamsError
from thehouse_ml.synthetic.students_stage import StudentsError
from thehouse_ml.universities import OsmError
from thehouse_ml.variables import VariableCheckError


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser con un subcomando por etapa."""
    parser = argparse.ArgumentParser(
        prog="thehouse_ml",
        description="Pipeline de Smart Match: download -> prepare -> train -> evaluate -> export.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"Raíz del proyecto. Por defecto, la variable {ROOT_ENV_VAR} o el directorio actual.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True, metavar="etapa", title="etapas")
    for stage in STAGES:
        sub = subparsers.add_parser(stage.name, help=f"{stage.description} ({stage.spec})")
        if stage.name == "download":
            sub.add_argument(
                "--source",
                metavar="ID",
                default=None,
                help="Descarga y verifica solo la fuente con este id.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida."""
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.resolve(args.root)
    raw_source = getattr(args, "source", None)
    source = raw_source if isinstance(raw_source, str) else None
    try:
        message = run_stage(args.stage, paths, source=source)
    except (
        DataCardError,
        DenueError,
        DownloadError,
        EvaluationError,
        ExportError,
        LabelInputError,
        LabelParamsError,
        LabelsError,
        ManifestError,
        OsmError,
        PairParamsError,
        PairSetError,
        PairsError,
        PrepareError,
        RegionalError,
        RoomParamsError,
        RoomsError,
        SourceRegistryError,
        StudentParamsError,
        StudentsError,
        VariableCheckError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0
