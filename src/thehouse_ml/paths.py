"""Rutas del proyecto (constitution §6).

La raíz se elige en este orden: argumento explícito, variable de entorno
`THEHOUSE_ML_ROOT` y, si no hay ninguna, el directorio actual. Así cada
etapa lee y escribe rutas configurables y las pruebas usan carpetas
temporales.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT_ENV_VAR = "THEHOUSE_ML_ROOT"


@dataclass(frozen=True)
class ProjectPaths:
    """Ubicaciones de datos, modelos y reportes bajo una raíz."""

    root: Path

    @property
    def data_raw(self) -> Path:
        """Datos descargados (fuera de git)."""
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        """Datos derivados (fuera de git)."""
        return self.root / "data" / "processed"

    @property
    def data_manifests(self) -> Path:
        """Manifiestos con fuente, licencia, fecha y SHA-256 (en git)."""
        return self.root / "data" / "manifests"

    @property
    def models(self) -> Path:
        """Artefactos de modelo (fuera de git)."""
        return self.root / "models"

    @property
    def reports(self) -> Path:
        """Reportes de evaluación, data card y model card (en git)."""
        return self.root / "reports"

    @classmethod
    def resolve(cls, root: Path | None = None, environ: Mapping[str, str] | None = None) -> ProjectPaths:
        """Elige la raíz: argumento, variable de entorno o directorio actual."""
        env = os.environ if environ is None else environ
        if root is None:
            configured = env.get(ROOT_ENV_VAR, "").strip()
            root = Path(configured) if configured else Path.cwd()
        return cls(root=root.resolve())
