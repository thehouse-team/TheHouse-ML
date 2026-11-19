"""Generador del catálogo sintético de cuartos (ML-03 T05, ML-03-R05).

Es puro: no lee archivos, no usa red ni el reloj. Los parámetros
(`rooms_params.json`), los insumos de calibración por campus (renta p50, forma
del histograma de renta y proporciones del Censo) y los campus llegan como
argumentos. Cada campus usa su propio `random.Random("rooms:<seed>:<code>")`
y las extracciones siguen un orden fijo, así que la misma semilla produce los
mismos cuartos byte por byte.

Renta: la forma del histograma ENVI del estado del campus (uniforme dentro de
cada intervalo) se escala para que su mediana sea la renta p50 regional; cada
`placeType` aplica un multiplicador y un factor común los normaliza para que
la mediana general del campus siga siendo esa p50. El cuantil se muestrea
estratificado dentro de cada `placeType` (un estrato por cuarto).

Todo lo que sale de aquí es **sintético** (constitución §8): ids sin datos
personales (`R00001`), sin títulos, descripciones ni nombres, y ningún campo de
género (constitución §11): `livingWith` es la política que declara el cuarto.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from thehouse_ml.features.campuses import Campus
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.features.pairs import Room, validate_room
from thehouse_ml.synthetic.students import (
    BASIS_PREFIXES,
    PARAMS_PACKAGE,
    Parameter,
    _check_distribution,
    _is_probability,
    _round_to,
    _weighted,
)
from thehouse_ml.universities import EARTH_RADIUS_M, haversine_m

ROOMS_PARAMS_RESOURCE = "rooms_params.json"
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0
# Margen para que el redondeo de las coordenadas no rebase el tope de distancia (1 m).
DISTANCE_MARGIN_KM = 0.001
# Orden que deben respetar los multiplicadores de renta por placeType (nota para T05).
PLACE_TYPE_RENT_ORDER = ("shared", "room", "room_private_bath", "studio")


class RoomParamsError(ValueError):
    """`rooms_params.json` o los insumos del generador no son válidos; lista todos los problemas."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("rooms_params.json inválido:\n- " + "\n- ".join(self.problems))


REQUIRED_PARAMETERS = (
    "seed",
    "n_rooms",
    "room_id.prefix",
    "room_id.width",
    "universities.per_campus",
    "universities.municipio_clave",
    "location.max_distance_km",
    "location.distance_bands_km",
    "location.bearing_degrees",
    "location.coordinate_decimals",
    "rent.median_anchor",
    "rent.shape",
    "rent.open_bin_upper",
    "rent.place_type_multipliers",
    "rent.round_to",
    "rent.clip",
    "rent.calibration_tolerance",
    "services.census_indicators",
    "services.census_tolerance_se",
    "services.priors_by_place_type",
    "livingWith.probabilities",
    "livingWith.min_per_campus",
    "placeType.probabilities",
    "propertyType.by_place_type",
    "pets_allowed.by_property_type",
    "bedrooms.by_place_type",
    "bathrooms.by_place_type",
    "area_m2.by_place_type",
)


@dataclass(frozen=True)
class RoomParams:
    """Parámetros validados del generador de cuartos."""

    metadata: Mapping[str, Any]
    parameters: Mapping[str, Parameter]

    def value(self, name: str) -> Any:
        return self.parameters[name].value

    @property
    def seed(self) -> int:
        return int(self.value("seed"))

    @property
    def n_rooms(self) -> int:
        return int(self.value("n_rooms"))

    @property
    def per_campus(self) -> int:
        return int(self.value("universities.per_campus"))

    @property
    def municipio_by_campus(self) -> Mapping[str, str]:
        return dict(self.value("universities.municipio_clave"))

    @property
    def max_distance_km(self) -> float:
        return float(self.value("location.max_distance_km"))

    @property
    def calibration_tolerance(self) -> float:
        return float(self.value("rent.calibration_tolerance"))

    @property
    def census_indicators(self) -> Mapping[str, str]:
        return dict(self.value("services.census_indicators"))

    @property
    def min_living_with_per_campus(self) -> int:
        return int(self.value("livingWith.min_per_campus"))

    def with_seed(self, seed: int) -> RoomParams:
        """Copia con otra semilla (para pruebas de reproducibilidad)."""
        parameters = dict(self.parameters)
        parameters["seed"] = Parameter("seed", seed, parameters["seed"].basis)
        return RoomParams(metadata=self.metadata, parameters=parameters)


def parse_params(data: object, schema: FeatureSchema) -> RoomParams:
    """Convierte el JSON decodificado en `RoomParams` y lo valida contra el esquema."""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise RoomParamsError(["raíz: se esperaba un objeto"])
    metadata = data.get("metadata")
    raw = data.get("parameters")
    if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
        problems.append("metadata.synthetic: los cuartos son sintéticos y deben declararlo (constitución §8)")
    if not isinstance(raw, dict):
        raise RoomParamsError([*problems, "parameters: se esperaba un objeto"])
    parameters: dict[str, Parameter] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or "value" not in entry:
            problems.append(f"{name}: se esperaba {{value, basis}}")
            continue
        basis = entry.get("basis")
        if not isinstance(basis, str) or not basis.startswith(BASIS_PREFIXES):
            problems.append(f"{name}.basis: debe empezar con «calibrado:» o «SUPUESTO documentado»")
            continue
        parameters[name] = Parameter(name=name, value=entry["value"], basis=basis)
    missing = [name for name in REQUIRED_PARAMETERS if name not in raw]
    if missing:
        problems.append(f"parameters: faltan {missing}")
    unknown = sorted(set(raw) - set(REQUIRED_PARAMETERS))
    if unknown:
        problems.append(f"parameters: desconocidos {unknown}")
    if not problems:
        problems.extend(_validate_values(parameters, schema))
    if problems:
        raise RoomParamsError(problems)
    return RoomParams(metadata=dict(metadata or {}), parameters=parameters)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_values_weights(name: str, entry: object, low: float, high: float) -> list[str]:
    if not isinstance(entry, dict):
        return [f"{name}: se esperaba {{values, weights}}"]
    values, weights = entry.get("values"), entry.get("weights")
    if not isinstance(values, list) or not isinstance(weights, list) or len(values) != len(weights) or not values:
        return [f"{name}: values y weights del mismo largo"]
    if not all(isinstance(item, int) and not isinstance(item, bool) and low <= item <= high for item in values):
        return [f"{name}: valores enteros fuera del rango SUPUESTO del esquema [{low:g}, {high:g}]"]
    if not all(_is_probability(item) for item in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        return [f"{name}: los pesos deben ser probabilidades que sumen 1"]
    return []


def _validate_location(value: Any) -> list[str]:
    problems: list[str] = []
    max_km = value("location.max_distance_km")
    if not (_is_number(max_km) and max_km > 0):
        return ["location.max_distance_km: debe ser positivo"]
    bands = value("location.distance_bands_km")
    if not isinstance(bands, list) or not bands or not all(isinstance(band, dict) for band in bands):
        return ["location.distance_bands_km: se esperaba una lista de bandas {min, max, probability}"]
    previous = 0.0
    for index, band in enumerate(bands):
        low, high, probability = band.get("min"), band.get("max"), band.get("probability")
        if not (_is_number(low) and _is_number(high) and _is_probability(probability)):
            problems.append(f"location.distance_bands_km[{index}]: min, max y probability numéricos")
        elif not math.isclose(float(low), previous) or not float(low) < float(high) <= float(max_km):
            problems.append(f"location.distance_bands_km[{index}]: bandas contiguas desde 0 y hasta el tope")
        else:
            previous = float(high)
    if not problems and not math.isclose(sum(float(band["probability"]) for band in bands), 1.0, abs_tol=1e-9):
        problems.append("location.distance_bands_km: las probabilidades deben sumar 1")
    if value("location.bearing_degrees") != [0, 360]:
        problems.append("location.bearing_degrees: el rumbo es uniforme en [0, 360]")
    decimals = value("location.coordinate_decimals")
    if not (isinstance(decimals, int) and not isinstance(decimals, bool) and 5 <= decimals <= 10):
        problems.append("location.coordinate_decimals: entero entre 5 y 10")
    return problems


def _validate_rent(value: Any, schema: FeatureSchema) -> list[str]:
    problems: list[str] = []
    place_types = schema.catalog("placeType")
    multipliers = value("rent.place_type_multipliers")
    if (
        not isinstance(multipliers, dict)
        or set(multipliers) != set(place_types)
        or not all(_is_number(item) and item > 0 for item in multipliers.values())
    ):
        problems.append("rent.place_type_multipliers: un multiplicador positivo por placeType del catálogo")
    elif not all(multipliers[low] < multipliers[high] for low, high in itertools.pairwise(PLACE_TYPE_RENT_ORDER)):
        problems.append("rent.place_type_multipliers: se exige shared < room < room_private_bath < studio")
    open_upper = value("rent.open_bin_upper")
    if not (_is_number(open_upper) and open_upper > 0):
        problems.append("rent.open_bin_upper: debe ser positivo")
    step = value("rent.round_to")
    if not (isinstance(step, int) and not isinstance(step, bool) and step > 0):
        problems.append("rent.round_to: entero positivo")
    price = schema.room_field("price_mxn").range
    if price is None or value("rent.clip") != [int(price.min), int(price.max or 0)]:
        problems.append("rent.clip: debe ser el rango de price_mxn del esquema")
    if not _is_probability(value("rent.calibration_tolerance")):
        problems.append("rent.calibration_tolerance: se esperaba una proporción en [0, 1]")
    for name in ("rent.median_anchor", "rent.shape"):
        if not isinstance(value(name), str) or not value(name):
            problems.append(f"{name}: se esperaba un texto que nombre la fuente")
    return problems


def _validate_services(value: Any, schema: FeatureSchema) -> list[str]:
    problems: list[str] = []
    services, place_types = schema.catalog("essentialServices"), schema.catalog("placeType")
    census = value("services.census_indicators")
    if not isinstance(census, dict) or not set(census) <= set(services) or not all(census.values()):
        return ["services.census_indicators: servicio del catálogo → indicador del Censo"]
    priors = value("services.priors_by_place_type")
    expected = set(services) - set(census)
    if not isinstance(priors, dict) or set(priors) != expected:
        problems.append(f"services.priors_by_place_type: debe cubrir exactamente {sorted(expected)}")
    else:
        for service, by_type in priors.items():
            if (
                not isinstance(by_type, dict)
                or set(by_type) != set(place_types)
                or not all(_is_probability(item) for item in by_type.values())
            ):
                problems.append(f"services.priors_by_place_type.{service}: una probabilidad por placeType")
    tolerance = value("services.census_tolerance_se")
    if not (_is_number(tolerance) and tolerance > 0):
        problems.append("services.census_tolerance_se: debe ser positivo")
    return problems


def _validate_values(parameters: Mapping[str, Parameter], schema: FeatureSchema) -> list[str]:
    problems: list[str] = []

    def value(name: str) -> Any:
        return parameters[name].value

    campuses = schema.catalog("university_code")
    n_rooms, per_campus = value("n_rooms"), value("universities.per_campus")
    if not (isinstance(n_rooms, int) and isinstance(per_campus, int) and per_campus * len(campuses) == n_rooms):
        problems.append("n_rooms: debe ser universities.per_campus por el número de campus del esquema")
    if not isinstance(value("seed"), int):
        problems.append("seed: se esperaba un entero")
    if not isinstance(value("universities.municipio_clave"), dict) or set(value("universities.municipio_clave")) != set(
        campuses
    ):
        problems.append("universities.municipio_clave: debe cubrir exactamente el catálogo university_code")
    problems.extend(_validate_location(value))
    problems.extend(_validate_rent(value, schema))
    problems.extend(_validate_services(value, schema))
    place_types, property_types = schema.catalog("placeType"), schema.catalog("propertyTypes")
    problems.extend(_check_distribution("placeType.probabilities", value("placeType.probabilities"), place_types))
    problems.extend(
        _check_distribution("livingWith.probabilities", value("livingWith.probabilities"), schema.catalog("livingWith"))
    )
    minimum = value("livingWith.min_per_campus")
    if not (isinstance(minimum, int) and isinstance(per_campus, int) and 0 <= minimum <= per_campus):
        problems.append("livingWith.min_per_campus: entero entre 0 y universities.per_campus")
    by_place = value("propertyType.by_place_type")
    if not isinstance(by_place, dict) or set(by_place) != set(place_types):
        problems.append("propertyType.by_place_type: una distribución por placeType")
    else:
        for place_type, distribution in by_place.items():
            problems.extend(
                _check_distribution(f"propertyType.by_place_type.{place_type}", distribution, property_types)
            )
    pets = value("pets_allowed.by_property_type")
    if not isinstance(pets, dict) or set(pets) != set(property_types) or not all(map(_is_probability, pets.values())):
        problems.append("pets_allowed.by_property_type: una probabilidad por propertyType")
    for name, field in (("bedrooms.by_place_type", "bedrooms"), ("bathrooms.by_place_type", "bathrooms")):
        bounds = schema.room_field(field).range
        entry = value(name)
        if bounds is None or not isinstance(entry, dict) or set(entry) != set(place_types):
            problems.append(f"{name}: una distribución por placeType")
            continue
        for place_type, distribution in entry.items():
            problems.extend(
                _check_values_weights(f"{name}.{place_type}", distribution, bounds.min, float(bounds.max or 0))
            )
    area = value("area_m2.by_place_type")
    bounds = schema.room_field("area_m2").range
    if bounds is None or not isinstance(area, dict) or set(area) != set(place_types):
        problems.append("area_m2.by_place_type: un intervalo por placeType")
    else:
        for place_type, interval in area.items():
            if not (
                isinstance(interval, list)
                and len(interval) == 2
                and all(_is_number(item) for item in interval)
                and bounds.min <= interval[0] < interval[1] <= float(bounds.max or 0)
            ):
                problems.append(f"area_m2.by_place_type.{place_type}: [min, max] dentro del rango SUPUESTO del esquema")
    return problems


def load_params(schema: FeatureSchema, path: Path | None = None) -> RoomParams:
    """Carga `rooms_params.json` empaquetado (o `path`) y lo valida."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(ROOMS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise RoomParamsError([f"JSON inválido: {error}"]) from error
    return parse_params(data, schema)


# ---------------------------------------------------------------------------
# Renta: forma del histograma, escala y multiplicadores por placeType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistogramBin:
    """Un intervalo del histograma de renta (pesos de jul-2024); `upper = None` es el intervalo abierto."""

    lower: float
    upper: float | None
    weight: float


class RentShape:
    """Distribución continua de la renta: uniforme dentro de cada intervalo del histograma."""

    def __init__(self, bins: Sequence[HistogramBin], open_upper: float) -> None:
        cells: list[tuple[float, float, float]] = []
        for item in sorted(bins, key=lambda b: b.lower):
            if not (math.isfinite(item.weight) and item.weight >= 0):
                raise RoomParamsError([f"histograma: frecuencia inválida en {item.lower:g}"])
            if item.weight == 0:
                continue
            upper = open_upper if item.upper is None else item.upper
            if not (math.isfinite(item.lower) and item.lower >= 0 and upper > item.lower):
                raise RoomParamsError([f"histograma: intervalo inválido [{item.lower:g}, {upper:g})"])
            cells.append((item.lower, upper, item.weight))
        total = sum(cell[2] for cell in cells)
        if not cells or total <= 0:
            raise RoomParamsError(["histograma: sin frecuencia"])
        self._cells = tuple((low, high, weight / total) for low, high, weight in cells)
        self.median = self.quantile(0.5)

    def cdf(self, value: float) -> float:
        """Proporción acumulada hasta `value`."""
        accumulated = 0.0
        for low, high, probability in self._cells:
            if value >= high:
                accumulated += probability
            elif value > low:
                accumulated += probability * (value - low) / (high - low)
        return min(accumulated, 1.0)

    def quantile(self, u: float) -> float:
        """Cuantil `u` en [0, 1] (inversa de la acumulada, lineal dentro del intervalo)."""
        accumulated = 0.0
        for low, high, probability in self._cells:
            if u <= accumulated + probability:
                return low + (high - low) * max(u - accumulated, 0.0) / probability
            accumulated += probability
        return self._cells[-1][1]


def mixture_normalizer(shape: RentShape, multipliers: Mapping[str, float], probabilities: Mapping[str, float]) -> float:
    """Factor común `c` tal que la mediana de la mezcla de `c · m_k · X` sea la mediana de `X`.

    Resuelve `Σ_k w_k · F(mediana / (c · m_k)) = 0.5` por bisección en escala
    logarítmica (determinista). Depende solo de la forma y de los parámetros.
    """

    def share_below(c: float) -> float:
        return sum(
            float(probabilities[key]) * shape.cdf(shape.median / (c * float(multipliers[key]))) for key in multipliers
        )

    low, high = math.log(1e-3), math.log(1e3)
    for _ in range(200):
        middle = (low + high) / 2.0
        if share_below(math.exp(middle)) > 0.5:
            low = middle
        else:
            high = middle
    return math.exp((low + high) / 2.0)


@dataclass(frozen=True)
class CampusInputs:
    """Insumos de calibración de un campus: renta p50, forma de la renta y proporciones del Censo."""

    rent_p50: float
    histogram: tuple[HistogramBin, ...]
    census: Mapping[str, float]


@dataclass(frozen=True)
class SyntheticRoom:
    """Cuarto sintético: id, campus verificado o no, distancia al campus y el cuarto del esquema."""

    room_id: str
    seed_verified: bool
    campus_distance_km: float
    room: Room


def destination_point(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """Punto a `distance_km` y rumbo `bearing_deg` sobre la esfera de 6,371 km (fórmula del gran círculo)."""
    phi1, lambda1 = math.radians(lat), math.radians(lon)
    theta, delta = math.radians(bearing_deg), distance_km / EARTH_RADIUS_KM
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1), math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), (math.degrees(lambda2) + 540.0) % 360.0 - 180.0


def effective_multipliers(params: RoomParams, shape: RentShape) -> dict[str, float]:
    """Multiplicador efectivo por placeType: `c · m_k` (renta del grupo frente a la renta p50)."""
    multipliers = {key: float(item) for key, item in params.value("rent.place_type_multipliers").items()}
    c = mixture_normalizer(shape, multipliers, params.value("placeType.probabilities"))
    return {key: c * item for key, item in multipliers.items()}


def _choose(rng: random.Random, entry: Mapping[str, Sequence[Any]]) -> Any:
    return rng.choices(list(entry["values"]), weights=[float(item) for item in entry["weights"]])[0]


def _campus_rooms(
    rng: random.Random,
    params: RoomParams,
    schema: FeatureSchema,
    campus: Campus,
    inputs: CampusInputs,
) -> list[tuple[float, Room]]:
    value = params.value
    n = params.per_campus
    place_catalog = schema.catalog("placeType")
    shape = RentShape(inputs.histogram, float(value("rent.open_bin_upper")))
    multipliers = effective_multipliers(params, shape)
    # Orden fijo de extracciones: cambiarlo cambia los datos (y la versión de los parámetros).
    # 1) placeType de todos los cuartos del campus.
    place_types = [_weighted(rng, value("placeType.probabilities"), place_catalog) for _ in range(n)]
    # 2) Cuantil de renta estratificado dentro de cada placeType (un estrato por cuarto).
    quantiles = [0.0] * n
    for place_type in place_catalog:
        members = [index for index, item in enumerate(place_types) if item == place_type]
        strata = rng.sample(range(len(members)), len(members))
        for index, stratum in zip(members, strata, strict=True):
            quantiles[index] = (stratum + rng.random()) / len(members)
    # 3) El resto de atributos, cuarto por cuarto.
    low_price, high_price = value("rent.clip")
    step = int(value("rent.round_to"))
    bands = value("location.distance_bands_km")
    services_catalog = schema.catalog("essentialServices")
    census_indicators = params.census_indicators
    priors = value("services.priors_by_place_type")
    decimals = int(value("location.coordinate_decimals"))
    max_km = params.max_distance_km
    rooms: list[tuple[float, Room]] = []
    for place_type, u in zip(place_types, quantiles, strict=True):
        raw_rent = inputs.rent_p50 * multipliers[place_type] * shape.quantile(u) / shape.median
        price = int(min(max(_round_to(raw_rent, step), int(low_price)), int(high_price)))
        property_type = _weighted(rng, value("propertyType.by_place_type")[place_type], schema.catalog("propertyTypes"))
        pets_allowed = rng.random() < float(value("pets_allowed.by_property_type")[property_type])
        band = rng.choices(bands, weights=[float(item["probability"]) for item in bands])[0]
        distance = float(band["min"]) + (float(band["max"]) - float(band["min"])) * rng.random()
        distance = min(distance, max_km - DISTANCE_MARGIN_KM)
        bearing = 360.0 * rng.random()
        services = frozenset(
            item
            for item in services_catalog
            if rng.random()
            < (float(inputs.census[item]) if item in census_indicators else float(priors[item][place_type]))
        )
        living_with = _weighted(rng, value("livingWith.probabilities"), schema.catalog("livingWith"))
        bedrooms = int(_choose(rng, value("bedrooms.by_place_type")[place_type]))
        bathrooms = min(int(_choose(rng, value("bathrooms.by_place_type")[place_type])), bedrooms + 1)
        area_low, area_high = value("area_m2.by_place_type")[place_type]
        area = float(round(rng.uniform(float(area_low), float(area_high))))
        lat, lon = destination_point(campus.lat, campus.lon, distance, bearing)
        lat, lon = round(lat, decimals), round(lon, decimals)
        campus_distance = haversine_m(campus.lat, campus.lon, lat, lon) / 1000.0
        if campus_distance > max_km:
            raise RoomParamsError([f"{campus.code}: cuarto a {campus_distance:.4f} km, más que el tope de {max_km:g}"])
        room = Room(
            price_mxn=price,
            lat=lat,
            lon=lon,
            services=services,
            place_type=place_type,
            property_type=property_type,
            pets_allowed=pets_allowed,
            living_with=living_with,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            area_m2=area,
            university_code=campus.code,
            synthetic=True,
        )
        rooms.append((campus_distance, room))
    return rooms


def generate_rooms(
    params: RoomParams,
    inputs: Mapping[str, CampusInputs],
    campuses: Mapping[str, Campus],
    schema: FeatureSchema,
) -> list[SyntheticRoom]:
    """Genera `n_rooms` cuartos: `per_campus` por campus en el orden del catálogo.

    `inputs` trae por código de campus la renta p50 regional, el histograma del
    estado y las proporciones del Censo (una por servicio de
    `services.census_indicators`). Cada cuarto se valida con `validate_room`.
    """
    codes = schema.catalog("university_code")
    missing = [code for code in codes if code not in inputs or code not in campuses]
    if missing:
        raise RoomParamsError([f"campus sin insumos de calibración o sin coordenadas: {missing}"])
    problems = [
        f"{code}: falta la proporción del Censo de {service} o está fuera de [0, 1]"
        for code in codes
        for service in params.census_indicators
        if not _is_probability(inputs[code].census.get(service))
    ]
    problems += [f"{code}: renta p50 inválida" for code in codes if not inputs[code].rent_p50 > 0]
    if problems:
        raise RoomParamsError(problems)
    prefix, width = str(params.value("room_id.prefix")), int(params.value("room_id.width"))
    rooms: list[SyntheticRoom] = []
    for code in codes:
        rng = random.Random(f"rooms:{params.seed}:{code}")  # noqa: S311 - simulación reproducible, no criptografía
        for distance, room in _campus_rooms(rng, params, schema, campuses[code], inputs[code]):
            validate_room(room, schema)
            room_id = f"{prefix}{len(rooms) + 1:0{width}d}"
            rooms.append(SyntheticRoom(room_id, campuses[code].seed_verified, distance, room))
    return rooms


def room_record(item: SyntheticRoom, schema: FeatureSchema) -> dict[str, Any]:
    """Registro plano con los nombres del esquema; los servicios van en el orden del catálogo."""
    room = item.room
    return {
        "room_id": item.room_id,
        "university_code": room.university_code,
        "seed_verified": item.seed_verified,
        "campus_distance_km": item.campus_distance_km,
        "price_mxn": room.price_mxn,
        "lat": room.lat,
        "lon": room.lon,
        "services": [service for service in schema.catalog("essentialServices") if service in room.services],
        "placeType": room.place_type,
        "propertyType": room.property_type,
        "pets_allowed": room.pets_allowed,
        "livingWith": room.living_with,
        "bedrooms": room.bedrooms,
        "bathrooms": room.bathrooms,
        "area_m2": room.area_m2,
        "synthetic": room.synthetic,
    }


__all__ = [
    "PLACE_TYPE_RENT_ORDER",
    "ROOMS_PARAMS_RESOURCE",
    "CampusInputs",
    "HistogramBin",
    "RentShape",
    "RoomParams",
    "RoomParamsError",
    "SyntheticRoom",
    "destination_point",
    "effective_multipliers",
    "generate_rooms",
    "load_params",
    "mixture_normalizer",
    "parse_params",
    "room_record",
]
