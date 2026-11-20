"""Etapa ``prepare`` (conjunto de pares): Parquet y reporte (ML-03 T06, ML-03-R06).

Lee `data/processed/students.parquet` (T04), `data/processed/rooms.parquet`
(T05) y los campus de `data/universities.csv`; construye los pares con
`thehouse_ml.synthetic.pairs` y escribe:

- `data/processed/pairs.parquet` (fuera de git): `student_id`, `room_id`,
  `university_code`, `seed_verified`, `split`, las 7 features en el orden del
  esquema y `synthetic = true`; **sin etiquetas** (ML-04/ML-05);
- `reports/pairs_summary.md` (en git), titulado «DATOS SINTÉTICOS».

Después de escribir, la etapa comprueba el umbral de M-ML-021 (≥ 90 % de los
estudiantes con al menos un cuarto que pasa las restricciones duras, en total
y por campus) y lanza `PairsError` si no se cumple. El reporte no lleva fechas
ni rutas absolutas: correr `prepare` dos veces no cambia ningún archivo
versionado.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from thehouse_ml.features import default_schema
from thehouse_ml.features.campuses import load_campuses
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.features.pairs import ROOM_ATTRIBUTES, STUDENT_ATTRIBUTES, room_from_record, student_from_record
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.pairs import (
    PAIRS_PARAMS_RESOURCE,
    SPLITS,
    Coverage,
    PairParams,
    PairSet,
    StudentCandidates,
    build_pairs,
    coverage,
    load_params,
)
from thehouse_ml.synthetic.rooms import SyntheticRoom
from thehouse_ml.synthetic.rooms_stage import ROOMS_PARQUET
from thehouse_ml.synthetic.students import SyntheticStudent
from thehouse_ml.synthetic.students_stage import STUDENTS_PARQUET, _pct, _row, quartiles

PAIRS_PARQUET = "pairs.parquet"
PAIRS_REPORT = "pairs_summary.md"
PAIRS_SKIP_MESSAGE = (
    f"pairs: omitido; falta data/processed/{STUDENTS_PARQUET}, data/processed/{ROOMS_PARQUET} "
    "o data/universities.csv (ejecute `prepare` con los insumos de ML-02)"
)
ID_COLUMNS = ("student_id", "room_id", "university_code", "seed_verified", "split")
TOTAL = "**Total**"
TOTAL_VERIFIED = "**Total sin `seed_sin_verificar`**"


class PairsError(RuntimeError):
    """Un insumo de los pares es inválido o no se cumple el umbral de cobertura de M-ML-021."""


# ---------------------------------------------------------------------------
# Parquet ↔ objetos
# ---------------------------------------------------------------------------


def _native(value: Any) -> Any:
    """Valor de pandas/numpy a Python: nulos → None, arreglos → lista, escalares numpy → nativos."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def students_from_frame(frame: pd.DataFrame) -> list[SyntheticStudent]:
    """Estudiantes de `students.parquet` (T04) con el perfil NV-11 tal como se generó (nulos incluidos)."""
    students: list[SyntheticStudent] = []
    for raw in frame.to_dict("records"):
        record = {key: _native(value) for key, value in raw.items()}
        profile = student_from_record({name: record[name] for name in STUDENT_ATTRIBUTES if record[name] is not None})
        students.append(SyntheticStudent(str(record["student_id"]), bool(record["seed_verified"]), profile))
    return students


def rooms_from_frame(frame: pd.DataFrame) -> list[SyntheticRoom]:
    """Cuartos de `rooms.parquet` (T05)."""
    rooms: list[SyntheticRoom] = []
    for raw in frame.to_dict("records"):
        record = {key: _native(value) for key, value in raw.items()}
        room = room_from_record({name: record[name] for name in ROOM_ATTRIBUTES})
        rooms.append(
            SyntheticRoom(
                str(record["room_id"]), bool(record["seed_verified"]), float(record["campus_distance_km"]), room
            )
        )
    return rooms


def pair_columns(schema: FeatureSchema) -> tuple[str, ...]:
    """Columnas de `pairs.parquet`: ids, partición, features en el orden del esquema y `synthetic`."""
    return (*ID_COLUMNS, *schema.feature_order, "synthetic")


def pair_dtypes(schema: FeatureSchema) -> dict[str, str]:
    """Tipos fijos: las features `integer` del esquema son int64 y las `number`, float64."""
    kinds = {feature.name: feature.type for feature in schema.pair_features}
    dtypes = {column: "string" for column in ID_COLUMNS}
    dtypes["seed_verified"] = "bool"
    for name in schema.feature_order:
        dtypes[name] = "int64" if kinds[name] == "integer" else "float64"
    dtypes["synthetic"] = "bool"
    return dtypes


def pairs_frame(pair_set: PairSet, schema: FeatureSchema) -> pd.DataFrame:
    """Tabla de pares (una fila por par, sin etiquetas)."""
    rows = [
        {
            "student_id": pair.student_id,
            "room_id": pair.room_id,
            "university_code": pair.university_code,
            "seed_verified": pair.seed_verified,
            "split": pair.split,
            **{name: pair.features[name] for name in schema.feature_order},
            "synthetic": True,
        }
        for pair in pair_set.pairs
    ]
    frame = pd.DataFrame(rows, columns=list(pair_columns(schema)))
    return frame.astype(pair_dtypes(schema))


def failing_coverage(pair_set: PairSet, params: PairParams, schema: FeatureSchema) -> list[Coverage]:
    """Campus, total y total sin las `seed_sin_verificar` por debajo del umbral de M-ML-021."""
    rows = coverage(pair_set.students, params, schema)
    verified_total = coverage(pair_set.students, params, schema, verified_only=True)[-1]
    rows.append(
        Coverage(
            "Total sin seed_sin_verificar",
            verified_total.students,
            verified_total.with_candidates,
            verified_total.min_share,
        )
    )
    return [row for row in rows if not row.meets]


def run_pairs(paths: ProjectPaths) -> str:
    """Genera `pairs.parquet` y `reports/pairs_summary.md` y comprueba el umbral de cobertura."""
    students_path = paths.data_processed / STUDENTS_PARQUET
    rooms_path = paths.data_processed / ROOMS_PARQUET
    if not (students_path.is_file() and rooms_path.is_file() and paths.universities_csv.is_file()):
        return PAIRS_SKIP_MESSAGE
    schema = default_schema()
    params = load_params()
    students = students_from_frame(pd.read_parquet(students_path))
    rooms = rooms_from_frame(pd.read_parquet(rooms_path))
    campuses = load_campuses(paths.universities_csv)
    pair_set = build_pairs(students, rooms, campuses, params, schema)

    parquet_path = paths.data_processed / PAIRS_PARQUET
    pairs_frame(pair_set, schema).to_parquet(parquet_path, index=False)
    report_path = paths.reports / PAIRS_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(pair_set, params, schema), encoding="utf-8", newline="\n")
    failing = failing_coverage(pair_set, params, schema)
    if failing:
        detail = ", ".join(f"{row.label} {row.share * 100:.1f} %" for row in failing)
        raise PairsError(
            f"umbral de M-ML-021 no cumplido (≥ {params.min_share * 100:.0f} % de estudiantes con al menos un "
            f"cuarto que pasa las restricciones duras): {detail}; ver {report_path.as_posix()}"
        )
    return (
        f"pairs: {len(pair_set.pairs)} pares sintéticos de {len(pair_set.students)} estudiantes -> "
        f"{parquet_path.as_posix()}; reporte {report_path.as_posix()}"
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _groups(pair_set: PairSet, schema: FeatureSchema) -> list[tuple[str, list[StudentCandidates]]]:
    """Por campus (orden del catálogo), total y total sin las `seed_sin_verificar`."""
    students = list(pair_set.students)
    groups = [
        (code, [item for item in students if item.university_code == code])
        for code in schema.catalog("university_code")
    ]
    groups = [(code, group) for code, group in groups if group]
    return [*groups, (TOTAL, students), (TOTAL_VERIFIED, [item for item in students if item.seed_verified])]


def _verified_cell(label: str, group: Sequence[StudentCandidates]) -> str:
    if label in {TOTAL, TOTAL_VERIFIED}:
        return "—"
    return "sí" if group[0].seed_verified else "**no**"


def _table(
    pair_set: PairSet,
    schema: FeatureSchema,
    header: Sequence[str],
    cells: Callable[[list[StudentCandidates]], Sequence[object]],
) -> list[str]:
    lines = [_row(["Campus", "Verificado", *header]), _row(["---", "---", *(["---:"] * len(header))])]
    for label, group in _groups(pair_set, schema):
        lines.append(_row([label, _verified_cell(label, group), *cells(group)]))
    return lines


def _split_cells(group: list[StudentCandidates]) -> list[object]:
    students = [sum(1 for item in group if item.split == split) for split in SPLITS]
    pairs = [sum(len(item.room_ids) for item in group if item.split == split) for split in SPLITS]
    return [f"{count:,}" for count in [*students, *pairs]]


def _candidate_cells(params: PairParams) -> Callable[[list[StudentCandidates]], Sequence[object]]:
    def cells(group: list[StudentCandidates]) -> list[object]:
        counts = [float(len(item.room_ids)) for item in group]
        p25, median, p75 = quartiles(counts)
        return [
            f"{len(group):,}",
            _num(min(counts)),
            _num(p25),
            _num(median),
            _num(p75),
            _num(max(counts)),
            f"{max(item.compatible for item in group):,}",
            f"{sum(1 for item in group if item.sampled):,}",
            f"{sum(1 for item in group if not item.compatible):,}",
        ]

    return cells


def _exclusion_cells(schema: FeatureSchema) -> Callable[[list[StudentCandidates]], Sequence[object]]:
    ids = [constraint.id for constraint in schema.hard_constraints]

    def cells(group: list[StudentCandidates]) -> list[object]:
        evaluated = sum(item.campus_rooms for item in group)
        excluded = sum(item.excluded for item in group)
        by_id = [sum(item.exclusions[constraint_id] for item in group) for constraint_id in ids]
        return [
            f"{evaluated:,}",
            f"{excluded:,} ({_pct(excluded, evaluated)})",
            *(f"{count:,} ({_pct(count, evaluated)})" for count in by_id),
        ]

    return cells


def _coverage_cells(params: PairParams) -> Callable[[list[StudentCandidates]], Sequence[object]]:
    def cells(group: list[StudentCandidates]) -> list[object]:
        row = Coverage("", len(group), sum(1 for item in group if item.compatible), params.min_share)
        return [
            f"{row.students:,}",
            f"{row.with_candidates:,}",
            f"{row.share * 100:.1f} %",
            "sí" if row.meets else "**no**",
        ]

    return cells


def _rooms_in_several_splits(pair_set: PairSet) -> tuple[int, int]:
    splits_by_room: dict[str, set[str]] = {}
    for pair in pair_set.pairs:
        splits_by_room.setdefault(pair.room_id, set()).add(pair.split)
    return sum(1 for splits in splits_by_room.values() if len(splits) > 1), len(splits_by_room)


def render_report(pair_set: PairSet, params: PairParams, schema: FeatureSchema) -> str:
    """Reporte Markdown determinista del conjunto de pares."""
    summaries = pair_set.students
    total_pairs = len(pair_set.pairs)
    all_rows = coverage(summaries, params, schema)
    verified_rows = coverage(summaries, params, schema, verified_only=True)
    meets = all(row.meets for row in all_rows) and all(row.meets for row in verified_rows)
    no_candidates = sum(1 for item in summaries if not item.compatible)
    sampled = sum(1 for item in summaries if item.sampled)
    beyond = sum(item.beyond_max_distance for item in summaries)
    shared_rooms, rooms_with_pairs = _rooms_in_several_splits(pair_set)
    unverified = sorted({item.university_code for item in summaries if not item.seed_verified})
    threshold = f"{params.min_share * 100:.0f} %"
    fractions = params.fractions
    constraints = [constraint.id for constraint in schema.hard_constraints]
    lines = [
        "# DATOS SINTÉTICOS — Conjunto de pares estudiante-cuarto (ML-03 T06)",
        "",
        "> **DATOS SINTÉTICOS.** Los pares combinan estudiantes sintéticos (T04) con cuartos sintéticos (T05); ninguno",
        "> es una persona ni un anuncio real (constitución §8). Las métricas que se calculen con ellos no son",
        "> desempeño real.",
        "",
        "Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.pairs_stage`). Parámetros:",
        f"`src/thehouse_ml/synthetic/{PAIRS_PARAMS_RESOURCE}` versión {params.metadata.get('params_version', '—')}"
        f" (decisión {params.metadata.get('decision', '—')}).",
        f"Entradas: `data/processed/{STUDENTS_PARQUET}` y `data/processed/{ROOMS_PARQUET}`. Salida:"
        f" `data/processed/{PAIRS_PARQUET}` (fuera de git, columna `synthetic = true`). Esquema de features"
        f" {schema.version}.",
        "",
        "## 1. Resumen",
        "",
        f"- Estudiantes: **{len(summaries):,}**; pares: **{total_pairs:,}** (semilla {params.seed}).",
        "- **Sin etiquetas:** el conjunto no lleva relevancia ni puntaje (ML-04 y ML-05). Columnas: "
        + ", ".join(f"`{column}`" for column in pair_columns(schema))
        + ".",
        f"- Umbral de M-ML-021 (≥ {threshold} de los estudiantes con al menos un cuarto de su campus que pasa las"
        " restricciones duras, en total y por campus, con y sin las `seed_sin_verificar`): "
        + ("**cumple**." if meets else "**NO CUMPLE**."),
        f"- Estudiantes sin candidatos (no generan pares): **{no_candidates:,}**.",
        f"- Estudiantes con más de {params.max_candidates} cuartos compatibles (muestra uniforme de"
        f" {params.max_candidates}): **{sampled:,}**.",
        f"- Pares con `distance_km` mayor que el `maxDistanceKm` resuelto del estudiante: **{beyond:,}**"
        f" ({_pct(beyond, total_pairs)}). La distancia no es restricción dura (M-ML-023): esos cuartos sí son"
        " candidatos y ML-04 aprende a penalizarlos.",
        "- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017; distancias orientativas): "
        + (", ".join(f"`{code}`" for code in unverified) or "ninguno")
        + ". Cada tabla trae el total con y sin ellos.",
        "",
        "## 2. Regla de candidatos (M-ML-027)",
        "",
        "Para cada estudiante, los candidatos son los cuartos **de su campus** que pasan las restricciones duras",
        f"({', '.join(f'`{item}`' for item in constraints)}; `filter_rooms`, M-ML-024), hasta"
        f" {params.max_candidates}. Si hay más,",
        'se toma una muestra uniforme sin reemplazo con `random.Random("pairs:candidates:<seed>:<student_id>")`.',
        "Los cuartos que no pasan las restricciones no entran. Las 7 features se calculan con",
        "`compute_pair_features` (T02, una sola implementación) en el orden de `feature_order` del esquema.",
        "",
        "## 3. Partición por estudiante",
        "",
        f"Entrenamiento {fractions['train'] * 100:.0f} %, validación {fractions['validation'] * 100:.0f} % y prueba"
        f" {fractions['test'] * 100:.0f} %, estratificada por campus (cuotas por mayor residuo) con",
        '`random.Random("pairs:split:<seed>:<code>")`. Cada estudiante está en una sola partición (sin fuga). La',
        "partición no depende de los candidatos: un estudiante sin candidatos conserva su partición y no aporta pares.",
        f"Los cuartos son el catálogo, como en la app, y **sí se repiten entre particiones**: {shared_rooms:,} de"
        f" {rooms_with_pairs:,} cuartos con algún par aparecen en más de una partición.",
        "",
    ]
    split_header = [*(f"Estudiantes `{split}`" for split in SPLITS), *(f"Pares `{split}`" for split in SPLITS)]
    lines += _table(pair_set, schema, split_header, _split_cells)
    lines += [
        "",
        "## 4. Candidatos por estudiante",
        "",
        f"Número de candidatos (después del máximo de {params.max_candidates}); percentiles con interpolación lineal.",
        "«Compatibles máx.» es el mayor número de cuartos que pasan las restricciones antes del máximo.",
        "",
    ]
    candidate_header = [
        "Estudiantes",
        "Mín.",
        "p25",
        "Mediana",
        "p75",
        "Máx.",
        "Compatibles máx.",
        f"Con muestra (> {params.max_candidates})",
        "Sin candidatos",
    ]
    lines += _table(pair_set, schema, candidate_header, _candidate_cells(params))
    lines += [
        "",
        "## 5. Exclusiones por restricción dura",
        "",
        "Pares evaluados = estudiantes por cuartos de su campus. Un cuarto que viola varias restricciones cuenta en",
        "cada una, así que la suma por restricción puede ser mayor que los excluidos.",
        "",
    ]
    lines += _table(
        pair_set,
        schema,
        ["Pares evaluados", "Excluidos", *(f"`{item}`" for item in constraints)],
        _exclusion_cells(schema),
    )
    lines += [
        "",
        "## 6. Estudiantes con al menos un candidato",
        "",
        f"Umbral de M-ML-021: ≥ {threshold} en total y en cada campus.",
        "",
    ]
    lines += _table(
        pair_set, schema, ["Estudiantes", "Con ≥ 1 candidato", "Porcentaje", f"≥ {threshold}"], _coverage_cells(params)
    )
    lines += [
        "",
        "## 7. Features de los pares",
        "",
        "Valores calculados sobre todos los pares (sin etiqueta). `training_range` del esquema: `distance_km` y",
        "`effective_distance_km` 0-30 km y `price_to_budget` 0-1 (los pares pasan HC01).",
        "",
        _row(["Feature", "Tipo", "Mín.", "Mediana", "Máx.", "Media"]),
        _row(["---", "---", "---:", "---:", "---:", "---:"]),
    ]
    kinds = {feature.name: feature.type for feature in schema.pair_features}
    for name in schema.feature_order:
        values = [float(pair.features[name]) for pair in pair_set.pairs]
        if values:
            cells: list[object] = [
                f"{min(values):.3f}",
                f"{statistics.median(values):.3f}",
                f"{max(values):.3f}",
                f"{statistics.fmean(values):.3f}",
            ]
        else:
            cells = ["—"] * 4
        lines.append(_row([f"`{name}`", kinds[name], *cells]))
    lines += [
        "",
        "## 8. Privacidad y limitaciones",
        "",
        "- Ids sintéticos (`S…`, `R…`) sin datos personales (constitución §7); sin género, edad ni origen",
        "  (constitución §11): `livingWith` es la elección del estudiante y la política del cuarto.",
        "- Los candidatos son solo cuartos que pasan las restricciones duras: el conjunto no trae negativos por",
        "  restricción y ML-04 decide el puntaje dentro de los compatibles.",
        "- Las distribuciones de estudiantes y cuartos son supuestos documentados (M-ML-025 y M-ML-026); validar con",
        "  el equipo y reemplazar con señales reales en ML-07 (riesgo RK02).",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "PAIRS_PARQUET",
    "PAIRS_REPORT",
    "PAIRS_SKIP_MESSAGE",
    "PairsError",
    "failing_coverage",
    "pair_columns",
    "pair_dtypes",
    "pairs_frame",
    "render_report",
    "rooms_from_frame",
    "run_pairs",
    "students_from_frame",
]
