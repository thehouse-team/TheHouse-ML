"""Percentiles ponderados con el factor de expansión (ML-02 R03).

Definición adoptada (M-ML-014, T03): para valores ``x`` con factores ``w``,
el percentil ``q`` es el **menor valor x** cuya suma acumulada de factores,
con los valores ordenados de menor a mayor, es ``>= q * sum(w)``. Los valores
iguales se agrupan antes de acumular, así que el resultado no depende del orden
arbitrario de las filas empatadas. Es el inverso de la función de distribución
empírica ponderada (percentil de "menor rango" o *nearest rank*).
"""

from __future__ import annotations

from collections.abc import Sequence

QUANTILES: tuple[tuple[str, float], ...] = (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
)

# Tolerancia relativa para comparar sumas acumuladas de punto flotante.
_RELATIVE_TOLERANCE = 1e-9


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Devuelve el menor valor cuyo peso acumulado alcanza ``q`` del total.

    ``q`` va en [0, 1]. Los factores deben ser no negativos y al menos uno
    positivo. Con ``q = 0`` devuelve el menor valor observado.
    """
    if len(values) != len(weights):
        raise ValueError("values y weights deben tener la misma longitud")
    if len(values) == 0:
        raise ValueError("No hay valores para calcular el percentil")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q debe estar en [0, 1], se recibió {q!r}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Los factores de expansión no pueden ser negativos")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("La suma de factores debe ser positiva")

    grouped: dict[float, float] = {}
    for value, weight in zip(values, weights, strict=True):
        grouped[value] = grouped.get(value, 0.0) + weight

    target = q * total
    tolerance = _RELATIVE_TOLERANCE * max(1.0, target)
    cumulative = 0.0
    for value in sorted(grouped):
        cumulative += grouped[value]
        if cumulative + tolerance >= target:
            return value
    raise ValueError("No se alcanzó el percentil; revise los factores")  # pragma: no cover - q <= 1


def weighted_percentiles(values: Sequence[float], weights: Sequence[float]) -> dict[str, float]:
    """Calcula p10, p25, p50, p75 y p90 con la definición de ``weighted_quantile``."""
    return {name: weighted_quantile(values, weights, q) for name, q in QUANTILES}
