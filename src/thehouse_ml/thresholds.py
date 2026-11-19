"""Umbrales de suficiencia y supresión, con una sola definición (ML-02 T08).

La misma cota de 30 viviendas en muestra se usa para (a) aceptar una
estimación municipal (T04, M-ML-003) y (b) publicar una celda de la data card
(spec §8: no se publican celdas con menos de 30 viviendas).
"""

from __future__ import annotations

MIN_DWELLINGS = 30
MAX_CV = 0.30

__all__ = ["MAX_CV", "MIN_DWELLINGS"]
