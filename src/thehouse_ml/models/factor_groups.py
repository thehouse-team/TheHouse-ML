"""Grupos de features por factor del desglose (spec ML-05 §6.6; ML-05 T04 para la ablación, T05 para el desglose).

Los 6 factores son los del baseline `baseline-1.0.0` (constitución §12, M-ML-029), con
los mismos `id`. Cada feature de `feature_order` del esquema 2.0.0 pertenece a un solo
grupo, salvo `pets_compatible`: es constante (1) en los candidatos, no aporta señal y su
término se suma a la constante del modelo (spec §6.6), así que no pertenece a ningún
factor y la ablación nunca la quita.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FactorGroup:
    """Un factor del desglose y las features que le pertenecen."""

    id: str
    label: str
    features: tuple[str, ...]


FACTOR_GROUPS: tuple[FactorGroup, ...] = (
    FactorGroup("budget", "Presupuesto", ("price_to_budget", "price_below_min")),
    FactorGroup("distance", "Distancia", ("distance_km", "effective_distance_km", "distance_ratio")),
    FactorGroup("essential_services", "Servicios esenciales", ("services_coverage",)),
    FactorGroup("place_type", "Tipo de lugar", ("place_type_match",)),
    FactorGroup("property_type", "Tipo de inmueble", ("property_type_match",)),
    FactorGroup("extras", "Extras", ("extra_services_ratio",)),
)
OUTSIDE_GROUPS: dict[str, str] = {
    "pets_compatible": (
        "constante (1) en los candidatos (HC03 excluye los incompatibles): no aporta señal; su término se suma a la "
        "constante del modelo (spec §6.6) y se queda en todos los modelos de la ablación"
    ),
}


class FactorGroupError(ValueError):
    """Los grupos no cubren `feature_order` exactamente una vez."""


def check_groups(feature_order: Sequence[str], groups: Sequence[FactorGroup] = FACTOR_GROUPS) -> None:
    """Cada feature de `feature_order` está en un solo grupo o en `OUTSIDE_GROUPS`; ningún nombre sobra."""
    assigned = [name for group in groups for name in group.features] + list(OUTSIDE_GROUPS)
    repeated = sorted({name for name in assigned if assigned.count(name) > 1})
    missing = sorted(set(feature_order) - set(assigned))
    unknown = sorted(set(assigned) - set(feature_order))
    if repeated or missing or unknown:
        raise FactorGroupError(
            f"grupos de factores inválidos: repetidas {repeated}, sin grupo {missing}, ajenas {unknown}"
        )


def kept_columns(feature_order: Sequence[str], removed: Sequence[str]) -> tuple[int, ...]:
    """Posiciones de `feature_order` que se conservan al quitar `removed` (en el mismo orden)."""
    unknown = sorted(set(removed) - set(feature_order))
    if unknown:
        raise FactorGroupError(f"features a quitar fuera de feature_order: {unknown}")
    return tuple(index for index, name in enumerate(feature_order) if name not in removed)


__all__ = ["FACTOR_GROUPS", "OUTSIDE_GROUPS", "FactorGroup", "FactorGroupError", "check_groups", "kept_columns"]
