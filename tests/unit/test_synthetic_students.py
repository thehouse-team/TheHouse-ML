"""ML-03 T04 (ML-03-R04): generador de estudiantes sintéticos, sin red ni datos fuera de git."""

from __future__ import annotations

import copy
import csv
import io
import json
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from thehouse_ml.features import (
    Campus,
    StudentProfile,
    default_schema,
    load_campuses,
    resolve_student,
    student_from_record,
)
from thehouse_ml.regional import MUNICIPIOS
from thehouse_ml.synthetic import (
    StudentParams,
    StudentParamsError,
    SyntheticStudent,
    generate_students,
    load_params,
    parse_params,
    student_record,
)
from thehouse_ml.synthetic.students import (
    BASIS_PREFIXES,
    OPTIONAL_NULL_PARAMS,
    PARAMS_PACKAGE,
    PARAMS_RESOURCE,
    QUICK_SETUP_FIELDS,
)
from thehouse_ml.synthetic.students_stage import calibration_rows, students_frame

REPO = Path(__file__).resolve().parents[2]
NV11 = json.loads((REPO / "data" / "seed" / "nv11_catalogs.json").read_text(encoding="utf-8"))
# Renta p50 estatal de la ENVI 2020 en pesos de jul-2024, tal como la publica
# reports/regional_calibration.md (en git): Hidalgo $1,872 y Estado de México $2,496 (M-ML-015).
STATE_P50 = {"13": 1872.167721, "15": 2496.223627}
GENDER_WORDS = re.compile(r"gender|genero|género|sex|sexo|mujer|hombre", re.IGNORECASE)


def _raw_params() -> dict[str, Any]:
    from importlib import resources

    text = resources.files(PARAMS_PACKAGE).joinpath(PARAMS_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@pytest.fixture(scope="module")
def params() -> StudentParams:
    return load_params(default_schema())


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(REPO / "data" / "universities.csv")


@pytest.fixture(scope="module")
def rent_p50(params: StudentParams) -> dict[str, float]:
    return {code: STATE_P50[clave[:2]] for code, clave in params.municipio_by_campus.items()}


@pytest.fixture(scope="module")
def students(params: StudentParams, rent_p50: dict[str, float], campuses: dict[str, Campus]) -> list[SyntheticStudent]:
    return generate_students(params, rent_p50, campuses, default_schema())


def _records(students: list[SyntheticStudent]) -> str:
    schema = default_schema()
    return json.dumps([student_record(s, schema) for s in students], ensure_ascii=False, sort_keys=True)


# --- Parámetros -----------------------------------------------------------------


def test_every_parameter_has_a_value_and_a_documented_basis() -> None:
    raw = _raw_params()["parameters"]

    assert raw
    for name, entry in raw.items():
        assert set(entry) == {"value", "basis"}, name
        assert entry["basis"].startswith(BASIS_PREFIXES), name


def test_parameters_are_marked_synthetic_and_approved_as_m_ml_025() -> None:
    metadata = _raw_params()["metadata"]

    assert metadata["synthetic"] is True
    assert metadata["decision"] == "M-ML-025 (APROBADO)"


def test_requested_values_are_in_the_params(params: StudentParams) -> None:
    assert params.n_students == 2000
    assert params.per_campus == 200
    assert params.value("budget.max.round_to") == 100
    assert params.value("budget.max.clip") == [0, 50000]
    assert params.median_factor == pytest.approx(1.3)


def test_campus_municipality_matches_the_seed_city_and_the_regional_calibration(params: StudentParams) -> None:
    names = {municipality.clave: municipality.nombre for municipality in MUNICIPIOS}
    with (REPO / "data" / "seed" / "universities_seed.csv").open(encoding="utf-8", newline="") as stream:
        city = {row["code"]: row["city"] for row in csv.DictReader(stream)}

    for code, clave in params.municipio_by_campus.items():
        assert names[clave] == city[code], code


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("parameters", "seed", "basis"), "porque sí", "basis"),
        (("parameters", "n_students", "value"), 1999, "n_students"),
        (("parameters", "livingWith.probabilities", "value"), {"women": 0.5, "men": 0.5}, "livingWith"),
        (("parameters", "placeType.probabilities", "value", "room"), 0.9, "sumar 1"),
        (("parameters", "budget.max.clip", "value"), [0, 60000], "clip"),
        (("parameters", "pets.null_probability", "value"), 1.5, "pets.null_probability"),
        (("parameters", "budget.min.fraction", "value"), [0.9, 0.1], "fraction"),
        (("parameters", "maxDistanceKm.by_transport", "value", "walk", "values"), [0.1, 1, 2, 3], "rango"),
        (("metadata", "synthetic"), False, "synthetic"),
    ],
)
def test_invalid_parameters_are_rejected(path: tuple[str, ...], value: object, message: str) -> None:
    data = copy.deepcopy(_raw_params())
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(StudentParamsError, match=message):
        parse_params(data, default_schema())


def test_missing_and_unknown_parameters_are_rejected() -> None:
    data = copy.deepcopy(_raw_params())
    del data["parameters"]["pets.true_probability"]
    data["parameters"]["gender.probabilities"] = {"value": 0.5, "basis": "SUPUESTO documentado: no"}

    with pytest.raises(StudentParamsError) as error:
        parse_params(data, default_schema())

    text = str(error.value)
    assert "pets.true_probability" in text
    assert "gender.probabilities" in text


def test_load_params_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "students_params.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(StudentParamsError, match="JSON inválido"):
        load_params(default_schema(), path)


# --- Reproducibilidad -------------------------------------------------------------


def test_same_seed_gives_the_same_students_byte_for_byte(
    params: StudentParams, rent_p50: dict[str, float], campuses: dict[str, Campus], students: list[SyntheticStudent]
) -> None:
    again = generate_students(params, rent_p50, campuses, default_schema())

    assert _records(again) == _records(students)
    first, second = io.BytesIO(), io.BytesIO()
    students_frame(students, default_schema()).to_parquet(first, index=False)
    students_frame(again, default_schema()).to_parquet(second, index=False)
    assert first.getvalue() == second.getvalue()


def test_another_seed_gives_different_students(
    params: StudentParams, rent_p50: dict[str, float], campuses: dict[str, Campus], students: list[SyntheticStudent]
) -> None:
    other = generate_students(params.with_seed(params.seed + 1), rent_p50, campuses, default_schema())

    assert _records(other) != _records(students)
    assert [s.student_id for s in other] == [s.student_id for s in students]


# --- Tamaño, validez y catálogos ----------------------------------------------------


def test_two_thousand_students_two_hundred_per_campus(students: list[SyntheticStudent]) -> None:
    counts = Counter(s.profile.university_code for s in students)

    assert len(students) == 2000
    assert set(counts) == set(default_schema().catalog("university_code"))
    assert set(counts.values()) == {200}


def test_every_student_is_valid_with_nv11_catalogs_and_ranges(students: list[SyntheticStudent]) -> None:
    catalogs, ranges = NV11["catalogs"], NV11["ranges"]
    for student in students:
        record = student_record(student, default_schema())
        profile = student_from_record(
            {key: value for key, value in record.items() if key not in {"student_id", "seed_verified", "synthetic"}}
        )
        assert profile == student.profile
        resolved = resolve_student(profile)
        assert ranges["budget"]["min"] <= resolved.budget_min <= resolved.budget_max <= ranges["budget"]["max"]
        assert resolved.budget_max % 100 == 0
        assert ranges["maxDistanceKm"]["min"] <= resolved.max_distance_km <= ranges["maxDistanceKm"]["max"]
        assert resolved.essential_services <= set(catalogs["essentialServices"])
        assert resolved.living_with in catalogs["livingWith"]
        assert resolved.place_type in catalogs["placeType"]
        assert resolved.transport in catalogs["transport"]
        assert resolved.property_types <= set(catalogs["propertyTypes"])
        assert isinstance(resolved.pets, bool)


def test_unverified_campuses_are_marked(students: list[SyntheticStudent], campuses: dict[str, Campus]) -> None:
    unverified = {s.profile.university_code for s in students if not s.seed_verified}

    assert unverified == {"TESJI", "UMB-VDC", "UCAN"}
    assert unverified == set(default_schema().unverified_university_codes)
    assert all(s.seed_verified == campuses[s.profile.university_code].seed_verified for s in students)


def test_present_property_types_are_never_empty(students: list[SyntheticStudent]) -> None:
    assert all(s.profile.property_types is None or s.profile.property_types for s in students)


# --- Calibración --------------------------------------------------------------------


def test_budget_max_median_per_campus_is_within_the_documented_tolerance(
    students: list[SyntheticStudent], params: StudentParams, rent_p50: dict[str, float]
) -> None:
    from thehouse_ml.synthetic.students_stage import RentAnchor

    anchors = {
        code: RentAnchor(code, clave, clave, "estado", rent_p50[code])
        for code, clave in params.municipio_by_campus.items()
    }
    rows = calibration_rows(students, params, anchors)

    assert len(rows) == 10
    for row in rows:
        assert row.target == pytest.approx(rent_p50[row.code] * 1.3)
        assert abs(row.median / row.target - 1) <= params.calibration_tolerance, row
        assert row.within


# Con objetivos menores a ~1,500 pesos el redondeo a 100 ya pesa ±4 % y la tolerancia de ±10 % deja de
# ser holgada (limitación documentada en evidence.md); los objetivos reales son 2,434 y 3,245 pesos.
@pytest.mark.parametrize("anchor", [1500.0, 6000.0])
def test_calibration_follows_other_anchors(params: StudentParams, campuses: dict[str, Campus], anchor: float) -> None:
    rent = dict.fromkeys(params.municipio_by_campus, anchor)
    generated = generate_students(params, rent, campuses, default_schema())

    for code in rent:
        budgets = sorted(s.profile.budget_max or 0 for s in generated if s.profile.university_code == code)
        median = (budgets[99] + budgets[100]) / 2
        assert abs(median / (anchor * params.median_factor) - 1) <= params.calibration_tolerance


def test_zero_rent_gives_zero_budgets(params: StudentParams, campuses: dict[str, Campus]) -> None:
    rent = dict.fromkeys(params.municipio_by_campus, 0.0)

    generated = generate_students(params, rent, campuses, default_schema())

    assert {s.profile.budget_max for s in generated} == {0}
    assert {s.profile.budget_min for s in generated} <= {0, None}


def test_missing_rent_or_campus_is_an_error(params: StudentParams, campuses: dict[str, Campus]) -> None:
    rent = dict.fromkeys(params.municipio_by_campus, 2000.0)
    del rent["UCAN"]

    with pytest.raises(StudentParamsError, match="UCAN"):
        generate_students(params, rent, campuses, default_schema())


# --- Nulos ----------------------------------------------------------------------------


@pytest.mark.parametrize("field", QUICK_SETUP_FIELDS)
def test_quick_setup_is_never_null(students: list[SyntheticStudent], field: str) -> None:
    frame = students_frame(students, default_schema())

    assert frame[field].notna().all()


@pytest.mark.parametrize("field", sorted(OPTIONAL_NULL_PARAMS))
def test_null_share_of_optional_fields_resembles_the_parameter(
    students: list[SyntheticStudent], params: StudentParams, field: str
) -> None:
    frame = students_frame(students, default_schema())
    expected = float(params.value(OPTIONAL_NULL_PARAMS[field]))

    observed = float(frame[field].isna().mean())

    # Con n = 2,000 el error estándar es ≤ 1.1 puntos; ±3 puntos es ≈ 3 errores estándar.
    assert abs(observed - expected) <= 0.03, (field, observed, expected)
    assert observed > 0


# --- Privacidad y género ---------------------------------------------------------------


def test_no_gender_field_anywhere(students: list[SyntheticStudent], params: StudentParams) -> None:
    frame = students_frame(students, default_schema())
    names = [*frame.columns, *(item.name for item in fields(StudentProfile)), *params.parameters]

    assert not [name for name in names if GENDER_WORDS.search(name)]


def test_ids_are_synthetic_consecutive_and_without_personal_data(students: list[SyntheticStudent]) -> None:
    ids = [s.student_id for s in students]

    assert ids == [f"S{index:04d}" for index in range(1, 2001)]
    assert all(re.fullmatch(r"S\d{4}", item) for item in ids)


def test_frame_has_the_documented_columns_and_synthetic_flag(students: list[SyntheticStudent]) -> None:
    frame = students_frame(students, default_schema())

    assert list(frame.columns)[:3] == ["student_id", "university_code", "seed_verified"]
    assert bool(frame["synthetic"].all())
    assert str(frame["budget.min"].dtype) == "Int64"
    assert isinstance(frame.loc[0, "livingWith"], str)
    assert pd.api.types.is_bool_dtype(frame["seed_verified"])
