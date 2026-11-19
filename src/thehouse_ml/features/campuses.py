"""Coordenadas de los campus para `distance_km` (ML-03 T02, ML-03-R02).

La universidad del estudiante da el punto de origen de la distancia: las
columnas `final_lat` y `final_lon` de `data/universities.csv` (ML-02 T07,
M-ML-017). Este módulo es la única parte de T02 que lee archivos; la función
de features (`thehouse_ml.features.pairs`) recibe los campus ya cargados.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

UNVERIFIED_STATUS = "seed_sin_verificar"
REQUIRED_COLUMNS = ("code", "final_lat", "final_lon", "final_status")


class CampusError(ValueError):
    """`data/universities.csv` no tiene el formato esperado."""


@dataclass(frozen=True)
class Campus:
    """Coordenadas finales de un campus (WGS84)."""

    code: str
    lat: float
    lon: float
    seed_verified: bool = True

    def __post_init__(self) -> None:
        if not self.code:
            raise CampusError("campus sin código")
        if not (math.isfinite(self.lat) and -90.0 <= self.lat <= 90.0):
            raise CampusError(f"{self.code}: latitud fuera de rango ({self.lat})")
        if not (math.isfinite(self.lon) and -180.0 <= self.lon <= 180.0):
            raise CampusError(f"{self.code}: longitud fuera de rango ({self.lon})")


def load_campuses(path: Path) -> dict[str, Campus]:
    """Lee `final_lat`/`final_lon` de `data/universities.csv`, indexado por código."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
            if missing:
                raise CampusError(f"{path}: faltan columnas {missing}")
            rows = list(reader)
    except OSError as error:
        raise CampusError(f"No se pudo leer {path}: {error}") from error
    campuses: dict[str, Campus] = {}
    for row in rows:
        code = (row.get("code") or "").strip()
        if not code:
            continue
        if code in campuses:
            raise CampusError(f"{path}: código duplicado {code!r}")
        try:
            lat, lon = float(row["final_lat"]), float(row["final_lon"])
        except (TypeError, ValueError) as error:
            raise CampusError(f"{path}: {code} sin coordenadas finales numéricas") from error
        campuses[code] = Campus(
            code=code, lat=lat, lon=lon, seed_verified=(row.get("final_status") or "") != UNVERIFIED_STATUS
        )
    return campuses


__all__ = ["UNVERIFIED_STATUS", "Campus", "CampusError", "load_campuses"]
