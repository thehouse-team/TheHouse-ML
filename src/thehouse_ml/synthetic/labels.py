"""Simulador de etiquetas sintéticas de relevancia (ML-04 T04, ML-04-R04).

Es puro: no lee archivos, no usa red ni el reloj. Recibe los pares candidatos
con sus features ya calculadas (`pairs.parquet` de ML-03 T06) y los
parámetros (`labels_params.json`), y devuelve la relevancia 0-3 de cada par.

**Las etiquetas no son preferencias reales** (constitución §8.2): son una
utilidad oculta documentada más ruido, y un modelo entrenado con ellas aprende
esa función. Se reemplazan con señales reales en ML-07.

Reglas (spec ML-04 §6.5, M-ML-032):

1. **Pesos ocultos por estudiante:** pesos base de `utility.base_weights`
   (distintos de `baseline-1.0.0`) * multiplicador lognormal `exp(sigma·z)`,
   renormalizados a 100. Con `random.Random("labels:<seed>:<student_id>")`:
   primero una normal estándar por factor (orden de `utility.base_weights`).
2. **Utilidad oculta:** `Σ peso * curva(feature)` (orden de los factores, de
   izquierda a derecha) + ruido gaussiano `N(0, noise.sigma)`, una normal por
   candidato en orden de `room_id`, con el mismo generador. Las curvas de
   `utility.curves` difieren del modelo base en presupuesto y distancia.
3. **Relevancia por posición:** candidatos ordenados por utilidad descendente
   y `room_id` ascendente; la posición `r` (desde 1) recibe 3 si
   `r ≤ ⌈p3·n⌉`, 2 si `r ≤ ⌈(p3+p2)·n⌉`, 1 si `r ≤ ⌈(p3+p2+p1)·n⌉` y 0 en otro
   caso (aritmética exacta con `fractions.Fraction`).

No importa `thehouse_ml.models` (el modelo base): la comparación con el
baseline es solo del reporte. No usa ningún atributo sensible ni de género
(constitución §11): solo `student_id`, `room_id` y las features del par.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import Any, TypeGuard

from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.synthetic.students import BASIS_PREFIXES, PARAMS_PACKAGE, Parameter

LABELS_PARAMS_RESOURCE = "labels_params.json"
FACTORS = ("budget", "distance", "essential_services", "place_type", "property_type", "extras")
GRADES = (3, 2, 1)
SHAPES = ("identity", "one_minus", "exp_decay")
TOTAL_WEIGHT = 100.0
REQUIRED_PARAMETERS = (
    "seed",
    "utility.base_weights",
    "utility.curves",
    "weights.multiplier_sigma",
    "noise.sigma",
    "relevance.bands",
    "check.spearman_range",
)


class LabelParamsError(ValueError):
    """`labels_params.json` no tiene la estructura esperada; lista todos los problemas."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("labels_params.json inválido:\n- " + "\n- ".join(self.problems))


class LabelInputError(ValueError):
    """Los pares de entrada no permiten simular etiquetas (duplicados, features faltantes o fuera de rango)."""


@dataclass(frozen=True)
class UtilityCurve:
    """Curva de un factor de la utilidad oculta: `identity`, `one_minus` o `exp_decay`."""

    feature: str
    shape: str
    scale_km: float | None = None

    def evaluate(self, value: float) -> float:
        """Valor en [0, 1]: `identity` = min(x, 1); `one_minus` = 1 - min(x, 1); `exp_decay` = exp(-x / escala)."""
        if self.shape == "identity":
            return min(value, 1.0)
        if self.shape == "one_minus":
            return 1.0 - min(value, 1.0)
        return math.exp(-value / (self.scale_km or 1.0))


@dataclass(frozen=True)
class LabelParams:
    """Parámetros validados del simulador de etiquetas."""

    metadata: Mapping[str, Any]
    parameters: Mapping[str, Parameter]

    def value(self, name: str) -> Any:
        return self.parameters[name].value

    @property
    def version(self) -> str:
        return str(self.metadata.get("params_version", ""))

    @property
    def seed(self) -> int:
        return int(self.value("seed"))

    @property
    def base_weights(self) -> dict[str, float]:
        weights = self.value("utility.base_weights")
        return {factor: float(weights[factor]) for factor in FACTORS}

    @property
    def curves(self) -> dict[str, UtilityCurve]:
        curves = self.value("utility.curves")
        result: dict[str, UtilityCurve] = {}
        for factor in FACTORS:
            entry = curves[factor]
            scale = entry.get("scale_km")
            result[factor] = UtilityCurve(
                str(entry["feature"]), str(entry["shape"]), None if scale is None else float(scale)
            )
        return result

    @property
    def multiplier_sigma(self) -> float:
        return float(self.value("weights.multiplier_sigma"))

    @property
    def noise_sigma(self) -> float:
        return float(self.value("noise.sigma"))

    @property
    def bands(self) -> dict[int, Fraction]:
        """Fracción de cada grado (3, 2, 1) como fracción exacta del decimal del JSON."""
        bands = self.value("relevance.bands")
        return {grade: Fraction(str(bands[str(grade)])) for grade in GRADES}

    @property
    def spearman_range(self) -> tuple[float, float]:
        low, high = self.value("check.spearman_range")
        return float(low), float(high)

    def replace(self, name: str, value: Any) -> LabelParams:
        """Copia con otro valor de un parámetro (para pruebas); no vuelve a validar."""
        parameters = dict(self.parameters)
        parameters[name] = Parameter(name, value, parameters[name].basis)
        return LabelParams(metadata=self.metadata, parameters=parameters)

    def with_seed(self, seed: int) -> LabelParams:
        return self.replace("seed", seed)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_weights(weights: object) -> list[str]:
    if not isinstance(weights, dict) or list(weights) != list(FACTORS):
        return [f"utility.base_weights: se esperaban exactamente {list(FACTORS)} en ese orden"]
    if not all(_is_number(value) and float(value) > 0 for value in weights.values()):
        return ["utility.base_weights: cada peso debe ser un número > 0"]
    if abs(sum(float(value) for value in weights.values()) - TOTAL_WEIGHT) > 1e-9:
        return [f"utility.base_weights: los pesos deben sumar {TOTAL_WEIGHT:g}"]
    return []


def _validate_curves(curves: object, schema: FeatureSchema | None) -> list[str]:
    if not isinstance(curves, dict) or set(curves) != set(FACTORS):
        return [f"utility.curves: se esperaban exactamente {list(FACTORS)}"]
    problems: list[str] = []
    ranges = {spec.name: spec.range for spec in schema.pair_features} if schema else None
    for factor in FACTORS:
        entry = curves[factor]
        where = f"utility.curves.{factor}"
        if not isinstance(entry, dict) or entry.get("shape") not in SHAPES or not isinstance(entry.get("feature"), str):
            problems.append(f"{where}: se esperaba {{feature, shape}} con shape en {list(SHAPES)}")
            continue
        shape = entry["shape"]
        scale = entry.get("scale_km")
        if shape == "exp_decay" and not (_is_number(scale) and float(scale) > 0):
            problems.append(f"{where}.scale_km: número > 0 para exp_decay")
        if shape != "exp_decay" and "scale_km" in entry:
            problems.append(f"{where}.scale_km: solo aplica a exp_decay")
        if ranges is None:
            continue
        feature = entry["feature"]
        if feature not in ranges:
            problems.append(f"{where}.feature: {feature!r} no está en el esquema")
            continue
        if ranges[feature].min < 0:
            problems.append(f"{where}: {feature} debe ser ≥ 0 en el esquema")
    return problems


def _validate_values(parameters: Mapping[str, Parameter], schema: FeatureSchema | None) -> list[str]:
    problems: list[str] = []
    seed = parameters["seed"].value
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        problems.append("seed: entero no negativo")
    problems.extend(_validate_weights(parameters["utility.base_weights"].value))
    problems.extend(_validate_curves(parameters["utility.curves"].value, schema))
    for name in ("weights.multiplier_sigma", "noise.sigma"):
        value = parameters[name].value
        if not _is_number(value) or float(value) < 0:
            problems.append(f"{name}: número ≥ 0")
    bands = parameters["relevance.bands"].value
    if not isinstance(bands, dict) or set(bands) != {str(grade) for grade in GRADES}:
        problems.append("relevance.bands: se esperaban exactamente las claves '3', '2' y '1'")
    elif not all(_is_number(value) and 0.0 < float(value) < 1.0 for value in bands.values()):
        problems.append("relevance.bands: cada fracción en (0, 1)")
    elif sum(Fraction(str(value)) for value in bands.values()) > 1:
        problems.append("relevance.bands: las fracciones no pueden sumar más de 1")
    spearman = parameters["check.spearman_range"].value
    if (
        not isinstance(spearman, list)
        or len(spearman) != 2
        or not all(_is_number(value) for value in spearman)
        or not -1.0 <= float(spearman[0]) < float(spearman[1]) <= 1.0
    ):
        problems.append("check.spearman_range: [mín, máx] con -1 ≤ mín < máx ≤ 1")
    return problems


def parse_params(data: object, schema: FeatureSchema | None = None) -> LabelParams:
    """Convierte el JSON decodificado en `LabelParams` y lo valida (con `schema`, también las features)."""
    if not isinstance(data, dict):
        raise LabelParamsError(["raíz: se esperaba un objeto"])
    problems: list[str] = []
    metadata = data.get("metadata")
    raw = data.get("parameters")
    if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
        problems.append("metadata.synthetic: las etiquetas son sintéticas y deben declararlo (constitución §8)")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("params_version"), str):
        problems.append("metadata.params_version: se esperaba una versión")
    if not isinstance(raw, dict):
        raise LabelParamsError([*problems, "parameters: se esperaba un objeto"])
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
        raise LabelParamsError(problems)
    return LabelParams(metadata=dict(metadata or {}), parameters=parameters)


def load_params(schema: FeatureSchema | None = None, path: Path | None = None) -> LabelParams:
    """Carga `labels_params.json` empaquetado (o `path`) y lo valida."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(LABELS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LabelParamsError([f"JSON inválido: {error}"]) from error
    return parse_params(data, schema)


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelInput:
    """Un par candidato: ids y features (al menos las de `utility.curves`)."""

    student_id: str
    room_id: str
    features: Mapping[str, float]


@dataclass(frozen=True)
class PairLabel:
    """Etiqueta de un par: utilidad oculta (con ruido) y relevancia 0-3."""

    student_id: str
    room_id: str
    utility: float
    relevance: int


@dataclass(frozen=True)
class StudentWeights:
    """Pesos ocultos de un estudiante (suman 100) y su número de candidatos."""

    student_id: str
    weights: Mapping[str, float]
    candidates: int


@dataclass(frozen=True)
class LabelSet:
    """Etiquetas en orden de `student_id` y `room_id`, y los pesos ocultos por estudiante."""

    labels: tuple[PairLabel, ...]
    students: tuple[StudentWeights, ...]


def grade_cutoffs(n: int, params: LabelParams) -> dict[int, int]:
    """Última posición (desde 1) de cada grado entre `n` candidatos: ⌈p3·n⌉, ⌈(p3+p2)·n⌉ y ⌈(p3+p2+p1)·n⌉."""
    cutoffs: dict[int, int] = {}
    cumulative = Fraction(0)
    for grade in GRADES:
        cumulative += params.bands[grade]
        cutoffs[grade] = min(math.ceil(cumulative * n), n)
    return cutoffs


def relevance_by_position(n: int, params: LabelParams) -> list[int]:
    """Relevancia de las posiciones 1…n (ya ordenadas por utilidad)."""
    cutoffs = grade_cutoffs(n, params)
    grades: list[int] = []
    for position in range(1, n + 1):
        grade = next((grade for grade in GRADES if position <= cutoffs[grade]), 0)
        grades.append(grade)
    return grades


def hidden_weights(rng: random.Random, params: LabelParams) -> dict[str, float]:
    """Pesos base * exp(sigma·z) por factor (una normal por factor, en orden), renormalizados a 100."""
    raw = {
        factor: weight * math.exp(params.multiplier_sigma * rng.gauss(0.0, 1.0))
        for factor, weight in params.base_weights.items()
    }
    total = sum(raw.values())
    return {factor: TOTAL_WEIGHT * value / total for factor, value in raw.items()}


def _feature(pair: LabelInput, name: str) -> float:
    if name not in pair.features:
        raise LabelInputError(f"{pair.student_id}/{pair.room_id}: falta la feature {name!r}")
    value = float(pair.features[name])
    if not math.isfinite(value) or value < 0:
        raise LabelInputError(f"{pair.student_id}/{pair.room_id}: {name} = {value!r} no es un número finito ≥ 0")
    return value


def utility(pair: LabelInput, weights: Mapping[str, float], curves: Mapping[str, UtilityCurve]) -> float:
    """Utilidad oculta sin ruido: Σ peso * curva(feature), en el orden de los factores."""
    total = 0.0
    for factor in FACTORS:
        curve = curves[factor]
        total += weights[factor] * curve.evaluate(_feature(pair, curve.feature))
    return total


def _group(pairs: Sequence[LabelInput]) -> dict[str, list[LabelInput]]:
    seen: set[tuple[str, str]] = set()
    groups: dict[str, list[LabelInput]] = {}
    for pair in pairs:
        key = (pair.student_id, pair.room_id)
        if key in seen:
            raise LabelInputError(f"par duplicado {pair.student_id}/{pair.room_id}")
        seen.add(key)
        groups.setdefault(pair.student_id, []).append(pair)
    return groups


def simulate_labels(pairs: Sequence[LabelInput], params: LabelParams) -> LabelSet:
    """Relevancia 0-3 de cada par candidato; misma semilla → mismas etiquetas.

    Devuelve exactamente una etiqueta por par de entrada, en orden de
    `student_id` y `room_id`. Lanza `LabelInputError` con pares duplicados o
    features faltantes, no finitas o negativas.
    """
    curves = params.curves
    labels: list[PairLabel] = []
    students: list[StudentWeights] = []
    for student_id, group in sorted(_group(pairs).items()):
        rng = random.Random(f"labels:{params.seed}:{student_id}")  # noqa: S311 - simulación reproducible
        weights = hidden_weights(rng, params)
        candidates = sorted(group, key=lambda pair: pair.room_id)
        utilities = {
            pair.room_id: utility(pair, weights, curves) + rng.gauss(0.0, params.noise_sigma) for pair in candidates
        }
        ranked = sorted(candidates, key=lambda pair: (-utilities[pair.room_id], pair.room_id))
        grades = {
            pair.room_id: grade for pair, grade in zip(ranked, relevance_by_position(len(ranked), params), strict=True)
        }
        labels.extend(
            PairLabel(student_id, pair.room_id, utilities[pair.room_id], grades[pair.room_id]) for pair in candidates
        )
        students.append(StudentWeights(student_id, weights, len(candidates)))
    return LabelSet(labels=tuple(labels), students=tuple(students))


__all__ = [
    "FACTORS",
    "GRADES",
    "LABELS_PARAMS_RESOURCE",
    "LabelInput",
    "LabelInputError",
    "LabelParams",
    "LabelParamsError",
    "LabelSet",
    "PairLabel",
    "StudentWeights",
    "UtilityCurve",
    "grade_cutoffs",
    "hidden_weights",
    "load_params",
    "parse_params",
    "relevance_by_position",
    "simulate_labels",
    "utility",
]
