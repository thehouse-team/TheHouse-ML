"""Línea de comandos: `python -m thehouse_ml <etapa>`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from thehouse_ml import __version__
from thehouse_ml.paths import ROOT_ENV_VAR, ProjectPaths
from thehouse_ml.stages import STAGES, run_stage


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
        subparsers.add_parser(stage.name, help=f"{stage.description} ({stage.spec})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida."""
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.resolve(args.root)
    print(run_stage(args.stage, paths))
    return 0
