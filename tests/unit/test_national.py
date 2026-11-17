"""ML-02 T03: reglas de exclusión, ponderación y resumen con datos inventados."""

from __future__ import annotations

import pandas as pd
import pytest

from thehouse_ml.national import ENVI_SOURCE, _normalize_enigh, _normalize_envi, _summarize


def test_envi_excludes_special_rent_codes_and_blanks() -> None:
    frame = pd.DataFrame(
        {
            "P5_1": ["1", "1", "1", "1", "1", "1", "1", "2"],
            "P5_2": [
                "000001000",
                "0",
                "000000000",
                "999999888",
                "999999999",
                "",
                "000050001",
                "000005000",
            ],
            "P4_3": ["1"] * 8,
            "ENT": ["13"] * 8,
            "TLOC": ["1"] * 8,
            "FACTOR": ["10"] * 8,
        }
    )

    result = _normalize_envi(frame)

    assert result["rented"].tolist() == [True] * 7 + [False]
    assert result["valid_rent"].tolist() == [True, False, False, False, False, False, False, False]
    assert result.loc[result["valid_rent"], "rent"].tolist() == [1000.0]


def test_envi_accepts_the_upper_bound_of_the_valid_range() -> None:
    frame = pd.DataFrame(
        {
            "P5_1": ["1", "1"],
            "P5_2": ["000050000", "000000099"],
            "P4_3": ["1", "1"],
            "ENT": ["13", "13"],
            "TLOC": ["1", "1"],
            "FACTOR": ["10", "10"],
        }
    )

    assert _normalize_envi(frame)["valid_rent"].tolist() == [True, False]


def test_enigh_uses_only_positive_rents_of_renters() -> None:
    frame = pd.DataFrame(
        {
            "tenencia": ["1", "1", "2", "1"],
            "renta": ["1500", "0", "5000", "2000"],
            "factor": ["10", "10", "10", "10"],
            "ubica_geo": ["13001", "13002", "13003", "15001"],
            "tam_loc": ["1", "1", "1", "2"],
            "tipo_viv": ["1", "1", "1", "2"],
        }
    )

    result = _normalize_enigh(frame)

    assert result["entidad"].tolist() == ["13", "13", "13", "15"]
    assert result["valid_rent"].tolist() == [True, False, False, True]


def test_rented_share_is_weighted_and_between_zero_and_one() -> None:
    frame = pd.DataFrame(
        {
            "entidad": ["13", "13"],
            "tam_loc": ["1", "1"],
            "tipo_viv": ["1", "1"],
            "rented": [True, False],
            "valid_rent": [True, False],
            "rent": [2000.0, 0.0],
            "weight": [10.0, 30.0],
        }
    )

    row = _summarize(frame, ENVI_SOURCE, 2020, "nacional", "nacional", 1.1)

    assert row is not None
    assert 0.0 <= row.rented_share <= 1.0
    assert row.rented_share == pytest.approx(10.0 / 40.0)
    assert row.n_muestra == 1
    assert row.suma_factores == 10.0
    assert row.n_total == 2
    assert row.suma_factores_total == 40.0
    assert row.p50 == 2000.0
    assert row.p50_jul2024 == pytest.approx(2200.0)


def test_groups_without_valid_rent_are_skipped() -> None:
    frame = pd.DataFrame(
        {
            "entidad": ["13"],
            "tam_loc": ["1"],
            "tipo_viv": ["1"],
            "rented": [True],
            "valid_rent": [False],
            "rent": [999999999.0],
            "weight": [10.0],
        }
    )

    assert _summarize(frame, ENVI_SOURCE, 2020, "nacional", "nacional", 1.0) is None
