"""ML-02 T04: proporción ponderada, varianza de Taylor y regla de suficiencia."""

from __future__ import annotations

import math

import pytest

from thehouse_ml.regional import (
    INDICATORS,
    MUNICIPIOS,
    Indicator,
    ProportionEstimate,
    choose_level,
    municipality_key,
    normalize_code,
    weighted_proportion,
)

_RENTADAS = INDICATORS[0]


def _estimate(estimate: float, cv: float, n_muestra: int = 100) -> ProportionEstimate:
    return ProportionEstimate(
        estimate=estimate,
        se=estimate * cv,
        cv=cv,
        n_muestra=n_muestra,
        suma_factores=float(n_muestra) * 10,
        n_total=n_muestra,
        suma_factores_total=float(n_muestra) * 10,
        n_psu=2,
        n_strata=1,
        n_strata_single_psu=0,
    )


def test_weighted_proportion_matches_hand_calculation() -> None:
    result = weighted_proportion(
        estrato=["h1", "h1"],
        upm=["A", "B"],
        weight=[10.0, 30.0],
        code=["2", "1"],
        indicator=_RENTADAS,
    )

    # p = 10 / 40 = 0.25; l_A = 0.1875, l_B = -0.1875;
    # Var = 2 * (0.1875^2 + 0.1875^2) = 0.140625; ee = 0.375; cv = 1.5.
    assert result.estimate == pytest.approx(0.25)
    assert result.suma_factores == pytest.approx(40.0)
    assert result.n_muestra == 2
    assert result.se == pytest.approx(0.375)
    assert result.cv == pytest.approx(1.5)


def test_cv_matches_hand_calculation() -> None:
    result = weighted_proportion(
        estrato=["h1"] * 4,
        upm=["A", "A", "B", "B"],
        weight=[1.0] * 4,
        code=["2", "2", "1", "1"],
        indicator=_RENTADAS,
    )

    # p = 0.5; L_A = 0.25, L_B = -0.25; Var = 2 * (0.25^2 + 0.25^2) = 0.25.
    assert result.estimate == pytest.approx(0.5)
    assert result.se == pytest.approx(0.5)
    assert result.cv == pytest.approx(1.0)


def test_special_codes_are_excluded_from_the_denominator() -> None:
    result = weighted_proportion(
        estrato=["h1", "h1", "h1"],
        upm=["A", "B", "C"],
        weight=[10.0, 20.0, 30.0],
        code=["2", "9", ""],
        indicator=_RENTADAS,
    )

    assert result.n_muestra == 1
    assert result.suma_factores == pytest.approx(10.0)
    assert result.n_total == 3
    assert result.suma_factores_total == pytest.approx(60.0)
    assert result.estimate == pytest.approx(1.0)


def test_codes_with_leading_zeros_are_normalized() -> None:
    assert normalize_code(" 002 ") == "2"
    assert normalize_code("09") == "9"
    assert normalize_code("") == ""


def test_indicator_codes_come_from_the_official_dictionary() -> None:
    codes = {indicator.key: indicator.affirmative for indicator in INDICATORS}
    assert codes["rentadas"] == frozenset({"2"})
    assert codes["internet"] == frozenset({"7"})
    assert codes["electricidad"] == frozenset({"1"})
    assert codes["agua_entubada"] == frozenset({"1", "2"})
    assert codes["agua_dentro"] == frozenset({"1"})
    assert codes["drenaje"] == frozenset({"1", "2", "3", "4"})
    assert codes["gas"] == frozenset({"2"})


def test_null_proportion_has_zero_cv_without_variance() -> None:
    result = weighted_proportion(
        estrato=["h1", "h1"],
        upm=["A", "B"],
        weight=[1.0, 1.0],
        code=["1", "1"],
        indicator=_RENTADAS,
    )

    assert result.estimate == 0.0
    assert result.cv == 0.0


def test_municipality_with_fewer_than_30_renters_falls_back_to_state() -> None:
    choice = choose_level(
        _estimate(0.4, 0.01),
        _estimate(0.3, 0.01),
        None,
        n_rentadas_municipio=29,
        n_rentadas_estado=500,
    )

    assert choice.level == "estado"
    assert choice.source == "Censo"
    assert choice.n_rentadas == 500


def test_municipality_with_high_cv_falls_back_to_state() -> None:
    choice = choose_level(
        _estimate(0.4, 0.31),
        _estimate(0.3, 0.02),
        None,
        n_rentadas_municipio=50,
        n_rentadas_estado=500,
    )

    assert choice.level == "estado"


def test_sufficient_municipality_is_kept() -> None:
    choice = choose_level(
        _estimate(0.4, 0.30),
        _estimate(0.3, 0.01),
        None,
        n_rentadas_municipio=30,
        n_rentadas_estado=500,
    )

    assert choice.level == "municipio"
    assert choice.n_rentadas == 30


def test_nacional_is_the_last_resort() -> None:
    choice = choose_level(
        _estimate(0.4, 0.5),
        _estimate(0.3, 0.4),
        _estimate(0.2, 0.0),
        n_rentadas_municipio=50,
        n_rentadas_estado=500,
    )

    assert choice.level == "nacional"
    assert choice.source == "ENVI"


def test_no_sufficient_estimate_raises() -> None:
    with pytest.raises(RuntimeError, match="suficiencia"):
        choose_level(
            _estimate(0.4, 0.5),
            _estimate(0.3, 0.4),
            None,
            n_rentadas_municipio=50,
            n_rentadas_estado=500,
        )


def test_municipality_key_070_is_tlahuelilpan() -> None:
    assert municipality_key("13", "070") == "13070"
    municipio = next(item for item in MUNICIPIOS if item.clave == "13070")
    assert municipio.nombre == "Tlahuelilpan"


def test_all_seven_municipalities_are_registered() -> None:
    assert len(MUNICIPIOS) == 7
    assert {item.clave for item in MUNICIPIOS} == {
        "13013",
        "13041",
        "13063",
        "13070",
        "13076",
        "15045",
        "15112",
    }
    assert {item.entidad for item in MUNICIPIOS} == {"13", "15"}


def test_single_psu_strata_are_counted_and_do_not_add_variance() -> None:
    result = weighted_proportion(
        estrato=["h1", "h2", "h2"],
        upm=["A", "B", "C"],
        weight=[1.0, 1.0, 1.0],
        code=["2", "2", "2"],
        indicator=Indicator("i", "V", frozenset({"2"}), "i"),
    )

    assert result.n_strata == 2
    assert result.n_strata_single_psu == 1
    assert result.se == pytest.approx(0.0)
    assert result.n_psu == 3
    assert not math.isnan(result.cv)
