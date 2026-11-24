"""Carga de los datos de entrenamiento de ML-05 (ML-05 T01, ML-05-R01).

Une `data/processed/pairs.parquet` (ML-03 T06, esquema de features 2.0.0) con
`data/processed/labels.parquet` (ML-04 T04, M-ML-032) por (`student_id`,
`room_id`) y entrega, en orden de `student_id` y `room_id`:

- **X**: exactamente las 10 columnas de `feature_order` del esquema, en ese
  orden (lo que recibirá el ONNX y calculará NV-23); nada más;
- **y**: la relevancia sintética graduada 0-3;
- el **grupo** (`student_id`), el `room_id` (desempate del ranking) y la
  **partición** (`split`) de ML-03 por par;
- los **metadatos por estudiante** para los segmentos: `university_code`,
  `seed_verified` y `budget.max` (de `students.parquet`, solo esa columna).

Valida `synthetic = true` en pares y etiquetas, `labels_params_version` 1.0.0
y el esquema 2.0.0. Excluye y cuenta a los estudiantes con menos de 2
candidatos (M-ML-035, herencia 5). **No** genera etiquetas.

Las únicas lecturas de Parquet son las de `TRAINING_INPUTS` (decisión 1 de
ML-05: la utilidad y los pesos ocultos de ML-04 nunca se leen); una prueba lo
verifica. Todo lo demás es puro: `build_training_data` recibe DataFrames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from thehouse_ml.features import default_schema
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models.baseline_report import EvaluationError, join_labels
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.labels_stage import LABELS_PARQUET
from thehouse_ml.synthetic.pairs import SPLITS
from thehouse_ml.synthetic.pairs_stage import PAIRS_PARQUET
from thehouse_ml.synthetic.students_stage import STUDENTS_PARQUET

EXPECTED_SCHEMA_VERSION = "2.0.0"
EXPECTED_LABELS_PARAMS_VERSION = "1.0.0"
MIN_CANDIDATES = 2
RELEVANCE_GRADES = (0, 1, 2, 3)
BUDGET_COLUMN = "budget.max"
TRAINING_INPUTS = (PAIRS_PARQUET, LABELS_PARQUET, STUDENTS_PARQUET)
PAIR_ID_COLUMNS = ("student_id", "room_id", "university_code", "seed_verified", "split", "synthetic")
LABEL_COLUMNS = ("student_id", "room_id", "split", "relevance", "synthetic", "labels_params_version")
STUDENT_COLUMNS = ("student_id", BUDGET_COLUMN)
STUDENT_META_COLUMNS = ("student_id", "split", "university_code", "seed_verified", "budget_max", "candidates")


class TrainingDataError(RuntimeError):
    """Pares, etiquetas o estudiantes que no sirven para entrenar (columna, versión o unión inválidas)."""


@dataclass(frozen=True)
class ExcludedStudent:
    """Estudiante fuera del entrenamiento y la evaluación por tener menos de `MIN_CANDIDATES` candidatos."""

    student_id: str
    split: str
    candidates: int


@dataclass(frozen=True)
class TrainingData:
    """Datos listos para entrenar: una fila por par candidato, alineadas por posición.

    `features`, `relevance`, `group`, `room_id` y `split` comparten el índice
    0..n-1 y el orden (`student_id`, `room_id`). `students` tiene una fila por
    estudiante incluido (en orden de `student_id`) con sus metadatos de segmento.
    """

    feature_order: tuple[str, ...]
    schema_version: str
    labels_params_version: str
    features: pd.DataFrame
    relevance: pd.Series
    group: pd.Series
    room_id: pd.Series
    split: pd.Series
    students: pd.DataFrame
    excluded: tuple[ExcludedStudent, ...]

    def summary(self) -> dict[str, dict[str, int]]:
        """Pares, estudiantes y excluidos por partición (y total), para reportes y evidencia."""
        result: dict[str, dict[str, int]] = {}
        for name in (*SPLITS, "total"):
            mask = self.split == name if name != "total" else self.split.notna()
            result[name] = {
                "pairs": int(mask.sum()),
                "students": int(self.group[mask].nunique()),
                "excluded_students": sum(1 for item in self.excluded if name in ("total", item.split)),
            }
        return result


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TrainingDataError(f"{name}: faltan las columnas {missing}")


def _check_schema(schema: FeatureSchema) -> tuple[str, ...]:
    if schema.version != EXPECTED_SCHEMA_VERSION:
        raise TrainingDataError(
            f"feature_schema.json {schema.version}: ML-05 entrena con el esquema {EXPECTED_SCHEMA_VERSION}"
        )
    return tuple(schema.feature_order)


def _check_pairs(pairs: pd.DataFrame, feature_order: tuple[str, ...]) -> None:
    _require_columns(pairs, [*PAIR_ID_COLUMNS, *feature_order], PAIRS_PARQUET)
    present = [column for column in pairs.columns if column in feature_order]
    if tuple(present) != feature_order:
        raise TrainingDataError(
            f"{PAIRS_PARQUET}: las features no están en el orden de feature_order {list(feature_order)}: {present}"
        )
    if not bool(pairs["synthetic"].all()):
        raise TrainingDataError(f"{PAIRS_PARQUET}: hay pares sin synthetic = true")
    unknown = sorted(set(pairs["split"].astype(str)) - set(SPLITS))
    if unknown:
        raise TrainingDataError(f"{PAIRS_PARQUET}: particiones desconocidas {unknown}")
    values = pairs[list(feature_order)].astype("float64")
    if not bool((values.abs() < math.inf).all().all()):  # NaN < inf es falso: cubre nulos e infinitos
        raise TrainingDataError(f"{PAIRS_PARQUET}: hay features nulas o no finitas")
    for column in ("split", "university_code", "seed_verified"):
        if bool((pairs.groupby("student_id")[column].nunique(dropna=False) > 1).any()):
            raise TrainingDataError(f"{PAIRS_PARQUET}: un estudiante tiene más de un valor de {column}")


def _check_labels(labels: pd.DataFrame) -> None:
    _require_columns(labels, LABEL_COLUMNS, LABELS_PARQUET)
    if not bool(labels["synthetic"].all()):
        raise TrainingDataError(f"{LABELS_PARQUET}: hay etiquetas sin synthetic = true")
    versions = sorted(set(labels["labels_params_version"].astype(str)))
    if versions != [EXPECTED_LABELS_PARAMS_VERSION]:
        raise TrainingDataError(
            f"{LABELS_PARQUET}: labels_params_version {versions}; ML-05 usa {EXPECTED_LABELS_PARAMS_VERSION}"
        )
    grades = set(labels["relevance"].tolist())
    if not grades <= set(RELEVANCE_GRADES):
        raise TrainingDataError(f"{LABELS_PARQUET}: relevancias fuera de 0-3: {sorted(grades - set(RELEVANCE_GRADES))}")


def _budgets(students: pd.DataFrame, student_ids: list[str]) -> Mapping[str, int]:
    _require_columns(students, STUDENT_COLUMNS, STUDENTS_PARQUET)
    if students["student_id"].duplicated().any():
        raise TrainingDataError(f"{STUDENTS_PARQUET}: student_id duplicado")
    table = students.set_index("student_id")[BUDGET_COLUMN]
    missing = [student_id for student_id in student_ids if student_id not in table.index]
    if missing:
        raise TrainingDataError(f"{STUDENTS_PARQUET}: faltan {len(missing)} estudiantes de los pares: {missing[:5]}")
    chosen = table.loc[student_ids]
    if bool(chosen.isna().any()):
        raise TrainingDataError(f"{STUDENTS_PARQUET}: {BUDGET_COLUMN} nulo")
    return {str(key): int(value) for key, value in chosen.items()}


# ---------------------------------------------------------------------------
# Construcción (pura)
# ---------------------------------------------------------------------------


def build_training_data(
    pairs: pd.DataFrame, labels: pd.DataFrame, students: pd.DataFrame, schema: FeatureSchema
) -> TrainingData:
    """Une, valida y separa X, y, grupo, partición y metadatos; excluye estudiantes con < 2 candidatos."""
    feature_order = _check_schema(schema)
    _check_pairs(pairs, feature_order)
    _check_labels(labels)
    try:
        merged = join_labels(pairs, labels)
    except EvaluationError as error:
        raise TrainingDataError(str(error)) from error

    counts = merged.groupby("student_id", sort=True).size()
    few = counts[counts < MIN_CANDIDATES]
    splits = merged.groupby("student_id", sort=True)["split"].first()
    excluded = tuple(ExcludedStudent(str(student_id), str(splits[student_id]), int(n)) for student_id, n in few.items())
    kept = merged[~merged["student_id"].isin(few.index)].reset_index(drop=True)
    if kept.empty:
        raise TrainingDataError(f"ningún estudiante tiene {MIN_CANDIDATES} o más candidatos")

    first = kept.groupby("student_id", sort=True).agg(
        split=("split", "first"),
        university_code=("university_code", "first"),
        seed_verified=("seed_verified", "first"),
        candidates=("room_id", "size"),
    )
    budgets = _budgets(students, [str(item) for item in first.index])
    meta = pd.DataFrame(
        {
            "student_id": [str(item) for item in first.index],
            "split": first["split"].astype(str).tolist(),
            "university_code": first["university_code"].astype(str).tolist(),
            "seed_verified": first["seed_verified"].astype(bool).tolist(),
            "budget_max": [budgets[str(item)] for item in first.index],
            "candidates": first["candidates"].astype("int64").tolist(),
        },
        columns=list(STUDENT_META_COLUMNS),
    ).astype(
        {
            "student_id": "string",
            "split": "string",
            "university_code": "string",
            "seed_verified": "bool",
            "budget_max": "int64",
            "candidates": "int64",
        }
    )

    return TrainingData(
        feature_order=feature_order,
        schema_version=schema.version,
        labels_params_version=EXPECTED_LABELS_PARAMS_VERSION,
        features=kept[list(feature_order)].astype("float64"),
        relevance=kept["relevance"].astype("int64").rename("relevance"),
        group=kept["student_id"].astype("string").rename("student_id"),
        room_id=kept["room_id"].astype("string").rename("room_id"),
        split=kept["split"].astype("string").rename("split"),
        students=meta,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Lectura (única I/O del módulo)
# ---------------------------------------------------------------------------


def _read(paths: ProjectPaths, name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Única lectura de Parquet: solo nombres de `TRAINING_INPUTS`."""
    if name not in TRAINING_INPUTS:
        raise TrainingDataError(f"{name} no es una entrada del entrenamiento")
    path = paths.data_processed / name
    if not path.is_file():
        raise TrainingDataError(f"falta data/processed/{name} (ejecute `prepare`)")
    try:
        return pd.read_parquet(path, columns=None if columns is None else list(columns))
    except (KeyError, ValueError) as error:  # pyarrow.ArrowInvalid hereda de ValueError
        raise TrainingDataError(f"{name}: no se pudieron leer las columnas {list(columns or ())}: {error}") from error


def load_training_data(paths: ProjectPaths, schema: FeatureSchema | None = None) -> TrainingData:
    """Lee pares, etiquetas y `budget.max` de `data/processed/` y construye `TrainingData`."""
    pairs = _read(paths, PAIRS_PARQUET)
    labels = _read(paths, LABELS_PARQUET)
    students = _read(paths, STUDENTS_PARQUET, STUDENT_COLUMNS)
    return build_training_data(pairs, labels, students, schema or default_schema())


__all__ = [
    "EXPECTED_LABELS_PARAMS_VERSION",
    "EXPECTED_SCHEMA_VERSION",
    "MIN_CANDIDATES",
    "TRAINING_INPUTS",
    "ExcludedStudent",
    "TrainingData",
    "TrainingDataError",
    "build_training_data",
    "load_training_data",
]
