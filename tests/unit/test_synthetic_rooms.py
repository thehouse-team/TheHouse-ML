"""ML-03 T05 (ML-03-R05): generador del catálogo sintético de cuartos, sin red ni datos fuera de git.

Los insumos de calibración salen de reportes versionados: la renta p50 estatal y
las proporciones del Censo de `reports/regional_calibration.md` y la forma de la
renta de los histogramas agrupados que publica `reports/data_card.md` (§7).
"""

from __future__ import annotations

import copy
import io
import json
import math
import re
import statistics
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from thehouse_ml.features import (
    Campus,
    default_schema,
    load_campuses,
    room_from_record,
    validate_room,
)
from thehouse_ml.synthetic import (
    CampusInputs,
    HistogramBin,
    RoomParams,
    RoomParamsError,
    SyntheticRoom,
    generate_rooms,
    load_params,
    load_room_params,
    room_record,
)
from thehouse_ml.synthetic.rooms import (
    PLACE_TYPE_RENT_ORDER,
    ROOMS_PARAMS_RESOURCE,
    RentShape,
    destination_point,
    effective_multipliers,
    mixture_normalizer,
    parse_params,
)
from thehouse_ml.synthetic.rooms_stage import (
    CensusShare,
    census_rows,
    census_tolerance,
    median_by_place_type,
    rent_rows,
    rooms_frame,
)
from thehouse_ml.synthetic.students import BASIS_PREFIXES, PARAMS_PACKAGE
from thehouse_ml.synthetic.students_stage import RentAnchor
from thehouse_ml.universities import haversine_m

REPO = Path(__file__).resolve().parents[2]
NV11 = json.loads((REPO / "data" / "seed" / "nv11_catalogs.json").read_text(encoding="utf-8"))
# Renta p50 estatal de la ENVI 2020 en pesos de jul-2024 (reports/regional_calibration.md, M-ML-015).
STATE_P50 = {"13": 1872.167721, "15": 2496.223627}
# Histogramas agrupados (≥ 30 viviendas por celda) que publica reports/data_card.md §7, en pesos de jul-2024.
STATE_HISTOGRAM = {
    "13": (HistogramBin(500, 1500, 32597), HistogramBin(1500, 2000, 16127), HistogramBin(2000, 7500, 32551)),
    "15": (
        HistogramBin(500, 1500, 165766),
        HistogramBin(1500, 2000, 137990),
        HistogramBin(2000, 2500, 129732),
        HistogramBin(2500, 3500, 121903),
        HistogramBin(3500, None, 183989),
    ),
}
CENSUS_COLUMNS = {"wifi": "Internet", "electricity": "Electricidad", "water": "Agua entubada", "gas": "Gas"}
GENDER_WORDS = re.compile(r"gender|genero|género|sex|sexo|mujer|hombre", re.IGNORECASE)
PERSONAL_WORDS = re.compile(r"name|nombre|title|titulo|título|descrip|phone|telefono|email|correo|host|photo|foto")


def _published_census() -> dict[str, dict[str, float]]:
    """Proporciones del Censo por municipio, tal como las publica reports/regional_calibration.md."""
    lines = (REPO / "reports" / "regional_calibration.md").read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("| Clave | Municipio"))
    header = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    result: dict[str, dict[str, float]] = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = dict(zip(header, (cell.strip() for cell in line.strip("|").split("|")), strict=True))
        result[cells["Clave"]] = {
            service: float(cells[column].split("%")[0]) / 100.0 for service, column in CENSUS_COLUMNS.items()
        }
    return result


def _raw_params() -> dict[str, Any]:
    text = resources.files(PARAMS_PACKAGE).joinpath(ROOMS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@pytest.fixture(scope="module")
def params() -> RoomParams:
    return load_room_params(default_schema())


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(REPO / "data" / "universities.csv")


@pytest.fixture(scope="module")
def census() -> dict[str, dict[str, float]]:
    return _published_census()


@pytest.fixture(scope="module")
def inputs(params: RoomParams, census: dict[str, dict[str, float]]) -> dict[str, CampusInputs]:
    return {
        code: CampusInputs(STATE_P50[clave[:2]], STATE_HISTOGRAM[clave[:2]], census[clave])
        for code, clave in params.municipio_by_campus.items()
    }


@pytest.fixture(scope="module")
def rooms(params: RoomParams, inputs: dict[str, CampusInputs], campuses: dict[str, Campus]) -> list[SyntheticRoom]:
    return generate_rooms(params, inputs, campuses, default_schema())


def _parquet_bytes(rooms: list[SyntheticRoom]) -> bytes:
    buffer = io.BytesIO()
    rooms_frame(rooms, default_schema()).to_parquet(buffer, index=False)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------


def test_every_parameter_has_a_value_and_a_documented_basis() -> None:
    raw = _raw_params()["parameters"]
    for name, entry in raw.items():
        assert set(entry) == {"value", "basis"}, name
        assert entry["basis"].startswith(BASIS_PREFIXES), name
        assert len(entry["basis"]) > 40, f"{name}: la base debe explicar el valor"


def test_parameters_are_marked_synthetic_and_approved_as_m_ml_026() -> None:
    metadata = _raw_params()["metadata"]
    assert metadata["synthetic"] is True
    assert metadata["decision"] == "M-ML-026 (APROBADO)"
    assert metadata["spec"] == "ML-03 1.0.0, T05 (ML-03-R05)"


def test_requested_values_are_in_the_params(params: RoomParams) -> None:
    students = load_params(default_schema())
    assert params.n_rooms == 3000
    assert params.per_campus == 300
    assert params.seed != students.seed
    assert params.max_distance_km == 30
    assert params.value("livingWith.probabilities") == {"women": 0.25, "men": 0.25, "any": 0.50}
    assert params.min_living_with_per_campus == 40
    assert dict(params.census_indicators) == {
        "water": "agua_entubada",
        "electricity": "electricidad",
        "gas": "gas",
        "wifi": "internet",
    }
    assert set(params.value("services.priors_by_place_type")) == {
        "hot_water",
        "laundry",
        "kitchen",
        "furnished",
        "parking",
        "security",
        "cleaning",
    }
    multipliers = params.value("rent.place_type_multipliers")
    assert [multipliers[key] for key in PLACE_TYPE_RENT_ORDER] == sorted(multipliers.values())
    assert params.value("rent.round_to") in {50, 100}
    assert params.value("rent.clip") == [0, 50000]
    assert params.calibration_tolerance == 0.10


def test_distance_bands_put_most_rooms_near_with_a_tail_to_thirty_km(params: RoomParams) -> None:
    bands = params.value("location.distance_bands_km")
    near = sum(band["probability"] for band in bands if band["max"] <= 5)
    assert near >= 0.5
    assert bands[-1]["max"] == 30
    assert bands[-1]["probability"] > 0


def test_studio_is_more_likely_to_have_a_kitchen_than_a_room(params: RoomParams) -> None:
    kitchen = params.value("services.priors_by_place_type")["kitchen"]
    assert kitchen["studio"] > kitchen["room"]


def test_campus_municipality_matches_the_students_params(params: RoomParams) -> None:
    assert params.municipio_by_campus == load_params(default_schema()).municipio_by_campus


def _mutated(path: tuple[str, ...], value: object) -> dict[str, Any]:
    data = copy.deepcopy(_raw_params())
    target: Any = data["parameters"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("n_rooms", "value"), 2999, "n_rooms"),
        (("seed", "value"), "x", "seed"),
        (("universities.municipio_clave", "value"), {"UTTT": "13076"}, "municipio_clave"),
        (("location.max_distance_km", "value"), 0, "max_distance_km"),
        (("location.distance_bands_km", "value"), [{"min": 0, "max": 31, "probability": 1.0}], "bandas"),
        (("location.distance_bands_km", "value"), [{"min": 0, "max": 30, "probability": 0.9}], "sumar 1"),
        (("location.distance_bands_km", "value"), [{"min": 0, "max": "x", "probability": 1.0}], "numéricos"),
        (("location.distance_bands_km", "value"), [], "lista"),
        (("location.bearing_degrees", "value"), [0, 180], "rumbo"),
        (("location.coordinate_decimals", "value"), 2, "coordinate_decimals"),
        (
            ("rent.place_type_multipliers", "value"),
            {"shared": 1.0, "room": 0.9, "room_private_bath": 1.15, "studio": 1.5},
            "shared < room",
        ),
        (("rent.place_type_multipliers", "value"), {"room": 1.0}, "multiplicador positivo"),
        (("rent.open_bin_upper", "value"), -1, "open_bin_upper"),
        (("rent.round_to", "value"), 0, "round_to"),
        (("rent.clip", "value"), [0, 60000], "clip"),
        (("rent.calibration_tolerance", "value"), 2, "calibration_tolerance"),
        (("rent.shape", "value"), "", "rent.shape"),
        (("services.census_indicators", "value"), {"sauna": "x"}, "census_indicators"),
        (("services.census_tolerance_se", "value"), 0, "census_tolerance_se"),
        (("services.priors_by_place_type", "value"), {}, "priors_by_place_type"),
        (("services.priors_by_place_type", "value", "laundry", "studio"), 1.5, "laundry"),
        (("livingWith.probabilities", "value"), {"women": 0.5, "men": 0.5, "any": 0.5}, "livingWith"),
        (("livingWith.min_per_campus", "value"), 301, "min_per_campus"),
        (("placeType.probabilities", "value"), {"room": 1.0}, "placeType"),
        (("propertyType.by_place_type", "value"), {"room": {}}, "propertyType.by_place_type"),
        (("propertyType.by_place_type", "value", "studio", "house"), 0.9, "studio"),
        (("pets_allowed.by_property_type", "value"), {"house": 2}, "pets_allowed"),
        (("bedrooms.by_place_type", "value"), {"room": {}}, "bedrooms.by_place_type"),
        (("bedrooms.by_place_type", "value", "studio"), {"values": [11], "weights": [1.0]}, "fuera del rango"),
        (("bedrooms.by_place_type", "value", "studio"), {"values": [1], "weights": [0.5]}, "sumen 1"),
        (("bathrooms.by_place_type", "value", "studio"), {"values": [1, 2], "weights": [1.0]}, "mismo largo"),
        (("bathrooms.by_place_type", "value", "studio"), [1], "{values, weights}"),
        (("area_m2.by_place_type", "value", "studio"), [5, 45], "area_m2"),
        (("area_m2.by_place_type", "value"), {"room": [9, 16]}, "area_m2"),
    ],
)
def test_invalid_parameters_are_rejected(path: tuple[str, ...], value: object, message: str) -> None:
    with pytest.raises(RoomParamsError) as error:
        parse_params(_mutated(path, value), default_schema())
    assert message in str(error.value)


def test_missing_unknown_and_unexplained_parameters_are_rejected() -> None:
    data = copy.deepcopy(_raw_params())
    del data["parameters"]["seed"]
    data["parameters"]["extra"] = {"value": 1, "basis": "SUPUESTO documentado: prueba"}
    data["parameters"]["n_rooms"]["basis"] = "porque sí"
    data["parameters"]["rent.round_to"] = 50
    data["metadata"]["synthetic"] = False
    with pytest.raises(RoomParamsError) as error:
        parse_params(data, default_schema())
    text = str(error.value)
    for fragment in ("faltan ['seed']", "desconocidos ['extra']", "n_rooms.basis", "rent.round_to", "synthetic"):
        assert fragment in text
    with pytest.raises(RoomParamsError, match="raíz"):
        parse_params([], default_schema())
    with pytest.raises(RoomParamsError, match="parameters"):
        parse_params({"metadata": {"synthetic": True}}, default_schema())


def test_load_params_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "rooms_params.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RoomParamsError, match="JSON inválido"):
        load_room_params(default_schema(), path)


def test_load_params_reads_an_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "rooms_params.json"
    path.write_text(json.dumps(_raw_params()), encoding="utf-8")
    assert load_room_params(default_schema(), path).seed == load_room_params(default_schema()).seed


# ---------------------------------------------------------------------------
# Renta, ubicación y generación
# ---------------------------------------------------------------------------


def test_rent_shape_quantile_inverts_the_cdf() -> None:
    shape = RentShape(STATE_HISTOGRAM["15"], 15000)
    for u in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert shape.cdf(shape.quantile(u)) == pytest.approx(u, abs=1e-12)
    assert shape.cdf(0) == 0.0
    assert shape.cdf(20000) == 1.0
    assert shape.cdf(shape.median) == pytest.approx(0.5)


def test_rent_shape_rejects_invalid_histograms() -> None:
    with pytest.raises(RoomParamsError, match="sin frecuencia"):
        RentShape((HistogramBin(0, 500, 0.0),), 15000)
    with pytest.raises(RoomParamsError, match="frecuencia inválida"):
        RentShape((HistogramBin(0, 500, -1.0),), 15000)
    with pytest.raises(RoomParamsError, match="intervalo inválido"):
        RentShape((HistogramBin(500, 400, 1.0),), 15000)


def test_mixture_normalizer_keeps_the_overall_median(params: RoomParams) -> None:
    for entidad, bins in STATE_HISTOGRAM.items():
        shape = RentShape(bins, 15000)
        weights = params.value("placeType.probabilities")
        multipliers = effective_multipliers(params, shape)
        share = sum(weights[key] * shape.cdf(shape.median / multipliers[key]) for key in multipliers)
        assert share == pytest.approx(0.5, abs=1e-9), entidad
    shape = RentShape(STATE_HISTOGRAM["13"], 15000)
    assert mixture_normalizer(shape, {"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0, abs=1e-9)


def test_destination_point_lands_at_the_requested_haversine_distance() -> None:
    for distance in (0.0, 0.3, 5.0, 29.999):
        for bearing in (0.0, 90.0, 225.0, 359.0):
            lat, lon = destination_point(20.05, -99.34, distance, bearing)
            assert haversine_m(20.05, -99.34, lat, lon) / 1000.0 == pytest.approx(distance, abs=1e-6)


def test_same_seed_gives_the_same_rooms_byte_for_byte(
    params: RoomParams, inputs: dict[str, CampusInputs], campuses: dict[str, Campus], rooms: list[SyntheticRoom]
) -> None:
    again = generate_rooms(params, inputs, campuses, default_schema())
    assert again == rooms
    assert _parquet_bytes(again) == _parquet_bytes(rooms)


def test_another_seed_gives_different_rooms(
    params: RoomParams, inputs: dict[str, CampusInputs], campuses: dict[str, Campus], rooms: list[SyntheticRoom]
) -> None:
    other = generate_rooms(params.with_seed(params.seed + 1), inputs, campuses, default_schema())
    assert [item.room for item in other] != [item.room for item in rooms]


def test_three_thousand_rooms_three_hundred_per_campus(rooms: list[SyntheticRoom]) -> None:
    counts = Counter(item.room.university_code for item in rooms)
    assert len(rooms) == 3000
    assert set(counts) == set(default_schema().catalog("university_code"))
    assert set(counts.values()) == {300}


def test_every_room_is_within_thirty_km_of_its_campus(rooms: list[SyntheticRoom], campuses: dict[str, Campus]) -> None:
    for item in rooms:
        campus = campuses[item.room.university_code]
        distance = haversine_m(campus.lat, campus.lon, item.room.lat, item.room.lon) / 1000.0
        assert distance == pytest.approx(item.campus_distance_km, abs=1e-9)
        assert 0.0 <= distance <= 30.0
    assert max(item.campus_distance_km for item in rooms) > 20.0  # la cola llega cerca del tope


def test_every_room_is_valid_with_the_schema_and_nv11(rooms: list[SyntheticRoom]) -> None:
    schema = default_schema()
    catalogs = NV11["catalogs"]
    budget = NV11["ranges"]["budget"]
    ranges = {name: schema.room_field(name).range for name in ("bedrooms", "bathrooms", "area_m2")}
    for item in rooms:
        record = room_record(item, schema)
        room = room_from_record(
            {k: v for k, v in record.items() if k not in {"room_id", "seed_verified", "campus_distance_km"}}
        )
        assert room == item.room
        validate_room(room, schema)
        assert budget["min"] <= room.price_mxn <= budget["max"]
        assert room.price_mxn % 50 == 0
        assert room.services <= set(catalogs["essentialServices"])
        assert room.place_type in catalogs["placeType"]
        assert room.property_type in catalogs["propertyTypes"]
        assert room.living_with in catalogs["livingWith"]
        assert isinstance(room.pets_allowed, bool)
        for name, value in (("bedrooms", room.bedrooms), ("bathrooms", room.bathrooms), ("area_m2", room.area_m2)):
            spec_range = ranges[name]
            assert spec_range is not None and spec_range.contains(float(value)), (name, value)
        assert room.bathrooms <= room.bedrooms + 1
        if room.place_type == "studio":
            assert room.bedrooms == 1
        if room.place_type == "room_private_bath":
            assert room.bathrooms >= 2
        assert room.synthetic is True


def test_unverified_campuses_are_marked(rooms: list[SyntheticRoom], campuses: dict[str, Campus]) -> None:
    unverified = {item.room.university_code for item in rooms if not item.seed_verified}
    assert unverified == {"TESJI", "UMB-VDC", "UCAN"}
    assert unverified == {code for code, campus in campuses.items() if not campus.seed_verified}


def test_rent_median_per_campus_is_within_ten_percent_of_the_regional_p50(
    rooms: list[SyntheticRoom], params: RoomParams
) -> None:
    anchors = {
        code: RentAnchor(code, clave, clave, "estado", STATE_P50[clave[:2]])
        for code, clave in params.municipio_by_campus.items()
    }
    rows = rent_rows(rooms, params, anchors)
    assert len(rows) == 10
    for row in rows:
        assert row.within, (row.code, row.median, row.rent_p50)
        assert abs(row.deviation) <= 0.10
        assert row.p25 < row.median < row.p75


def test_rent_medians_follow_the_place_type_order_in_every_campus(rooms: list[SyntheticRoom]) -> None:
    for code in [None, *default_schema().catalog("university_code")]:
        medians = median_by_place_type(rooms, code)
        values = [medians[key] for key in PLACE_TYPE_RENT_ORDER]
        assert values == sorted(values) and len(set(values)) == len(values), (code, medians)


def test_census_services_follow_the_municipal_census_within_the_documented_tolerance(
    rooms: list[SyntheticRoom], params: RoomParams, census: dict[str, dict[str, float]]
) -> None:
    shares = {
        code: {
            service: CensusShare(indicator, census[clave][service], "municipio")
            for service, indicator in params.census_indicators.items()
        }
        for code, clave in params.municipio_by_campus.items()
    }
    rows = census_rows(rooms, params, shares)
    assert len(rows) == 40
    for row in rows:
        assert row.within, (row.code, row.service, row.generated, row.census, row.tolerance)
    assert census_tolerance(params, 0.45) == pytest.approx(3 * math.sqrt(0.45 * 0.55 / 300))
    assert census_tolerance(params, 0.999) == 0.02


def test_each_campus_has_at_least_forty_women_and_forty_men_rooms(
    rooms: list[SyntheticRoom], params: RoomParams
) -> None:
    for code in default_schema().catalog("university_code"):
        counts = Counter(item.room.living_with for item in rooms if item.room.university_code == code)
        assert counts["women"] >= params.min_living_with_per_campus, code
        assert counts["men"] >= params.min_living_with_per_campus, code
        assert counts["any"] > 0, code


def test_categorical_shares_resemble_the_parameters(rooms: list[SyntheticRoom], params: RoomParams) -> None:
    total = len(rooms)
    for key, probability in params.value("placeType.probabilities").items():
        assert sum(1 for item in rooms if item.room.place_type == key) / total == pytest.approx(probability, abs=0.03)
    for key, probability in params.value("livingWith.probabilities").items():
        assert sum(1 for item in rooms if item.room.living_with == key) / total == pytest.approx(probability, abs=0.03)
    pets = sum(1 for item in rooms if item.room.pets_allowed) / total
    assert 0.2 <= pets <= 0.35


def test_no_personal_or_gender_field_anywhere(rooms: list[SyntheticRoom]) -> None:
    frame = rooms_frame(rooms, default_schema())
    columns = " ".join(frame.columns)
    assert not GENDER_WORDS.search(columns)
    assert not PERSONAL_WORDS.search(columns)
    assert not any(GENDER_WORDS.search(name) or PERSONAL_WORDS.search(name) for name in _raw_params()["parameters"])
    ids = [item.room_id for item in rooms]
    assert ids == [f"R{index:05d}" for index in range(1, 3001)]


def test_frame_has_the_documented_columns_and_synthetic_flag(rooms: list[SyntheticRoom]) -> None:
    frame = rooms_frame(rooms, default_schema())
    assert list(frame.columns) == [
        "room_id",
        "university_code",
        "seed_verified",
        "campus_distance_km",
        "price_mxn",
        "lat",
        "lon",
        "services",
        "placeType",
        "propertyType",
        "pets_allowed",
        "livingWith",
        "bedrooms",
        "bathrooms",
        "area_m2",
        "synthetic",
    ]
    assert bool(frame["synthetic"].all())
    assert frame["price_mxn"].dtype == "int64"
    services_order = list(default_schema().catalog("essentialServices"))
    for services in frame["services"]:
        listed = list(services)
        assert listed == sorted(listed, key=services_order.index)


def test_missing_inputs_are_errors(
    params: RoomParams, inputs: dict[str, CampusInputs], campuses: dict[str, Campus]
) -> None:
    schema = default_schema()
    partial = dict(inputs)
    del partial["UTTT"]
    with pytest.raises(RoomParamsError, match="UTTT"):
        generate_rooms(params, partial, campuses, schema)
    broken = dict(inputs)
    broken["UTTT"] = CampusInputs(inputs["UTTT"].rent_p50, inputs["UTTT"].histogram, {"water": 0.9})
    with pytest.raises(RoomParamsError, match="Censo de electricity"):
        generate_rooms(params, broken, campuses, schema)
    broken["UTTT"] = CampusInputs(0.0, inputs["UTTT"].histogram, inputs["UTTT"].census)
    with pytest.raises(RoomParamsError, match="renta p50"):
        generate_rooms(params, broken, campuses, schema)


def test_rent_is_rounded_and_clipped(params: RoomParams, campuses: dict[str, Campus]) -> None:
    schema = default_schema()
    huge = {
        code: CampusInputs(1_000_000.0, STATE_HISTOGRAM["13"], {service: 0.5 for service in params.census_indicators})
        for code in schema.catalog("university_code")
    }
    generated = generate_rooms(params, huge, campuses, schema)
    prices = [item.room.price_mxn for item in generated]
    assert max(prices) == 50000
    assert statistics.median(prices) == 50000
    frame = pd.DataFrame({"price": prices})
    assert (frame["price"] % 50 == 0).all()
