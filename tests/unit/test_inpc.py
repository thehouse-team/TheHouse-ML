"""ML-02 T03: solo el índice general y deflactor con INPC inventado."""

from __future__ import annotations

import pytest

from thehouse_ml.inpc import (
    ENIGH_SURVEY_MONTHS,
    ENVI_SURVEY_MONTHS,
    Deflators,
    build_deflators,
    deflactor,
    general_index,
    month_of,
)


def _record(fecha: str, concepto: str, valor: str) -> dict[str, str]:
    return {"FECHA": fecha, "CONCEPTO": concepto, "VALOR": valor}


GENERAL = "Índice nacional de precios al consumidor, Resumen, Precios al Consumidor (INPC)"
SUBYACENTE = "Índice nacional de precios al consumidor, Resumen, Subyacente"


def test_only_the_general_concept_row_is_used() -> None:
    records = [
        _record("2024-07-01", SUBYACENTE, "999.999"),
        _record("2024-07-01", GENERAL, "136.003"),
        _record("2020-10-01", SUBYACENTE, "888.888"),
        _record("2020-10-01", GENERAL, "100.000"),
        _record("2022-08-01", GENERAL + ", extra", "777.777"),
    ]

    index = general_index(records)

    assert index == {"2024-07": 136.003, "2020-10": 100.0}
    assert "2022-08" not in index  # El concepto termina en ", extra": no es el general.


def test_month_of_accepts_day_suffix_and_short_month() -> None:
    assert month_of("2024-07-01") == "2024-07"
    assert month_of("2020-10") == "2020-10"


def test_deflactor_uses_period_average_and_reference_month() -> None:
    index = {
        "2024-07": 136.0,
        "2020-10": 100.0,
        "2020-11": 102.0,
        "2020-12": 104.0,  # promedio 102 -> factor 136/102
        "2022-08": 110.0,
        "2022-09": 112.0,
        "2022-10": 114.0,
        "2022-11": 116.0,  # promedio 113 -> factor 136/113
    }

    result = build_deflators(index)

    assert isinstance(result, Deflators)
    assert result.envi == pytest.approx(136.0 / 102.0)
    assert result.enigh == pytest.approx(136.0 / 113.0)
    assert deflactor(index, ENVI_SURVEY_MONTHS) == pytest.approx(136.0 / 102.0)
    assert deflactor(index, ENIGH_SURVEY_MONTHS) == pytest.approx(136.0 / 113.0)


def test_deflactor_reports_missing_months() -> None:
    with pytest.raises(ValueError, match="Faltan meses"):
        deflactor({"2024-07": 136.0}, ENVI_SURVEY_MONTHS)

    with pytest.raises(ValueError, match="Falta el índice general de 2024-07"):
        deflactor({month: 100.0 for month in ENVI_SURVEY_MONTHS}, ENVI_SURVEY_MONTHS)
