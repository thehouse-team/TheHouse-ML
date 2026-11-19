"""Etapa ``prepare`` (estudiantes sintéticos): Parquet y reporte (ML-03 T04, ML-03-R04).

Lee la renta p50 regional de `data/processed/regional_calibration.parquet`
(ML-02 T04) y los campus de `data/universities.csv`, genera los estudiantes con
`thehouse_ml.synthetic.students` y escribe:

- `data/processed/students.parquet` (fuera de git) con la columna `synthetic = true`;
- `reports/students_synthetic.md` (en git), titulado «DATOS SINTÉTICOS».

El reporte no lleva fechas ni rutas absolutas: correr `prepare` dos veces no
cambia ningún archivo versionado.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from thehouse_ml.features import default_schema
from thehouse_ml.features.campuses import load_campuses
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.students import (
    OPTIONAL_NULL_PARAMS,
    PARAMS_RESOURCE,
    QUICK_SETUP_FIELDS,
    StudentParams,
    SyntheticStudent,
    budget_targets,
    generate_students,
    load_params,
    student_record,
)

REGIONAL_PARQUET = "regional_calibration.parquet"
STUDENTS_PARQUET = "students.parquet"
STUDENTS_REPORT = "students_synthetic.md"
RENT_INDICATOR = "renta"
STUDENTS_SKIP_MESSAGE = (
    f"students: omitido; falta data/processed/{REGIONAL_PARQUET} o data/universities.csv "
    "(ejecute `download` y `prepare` de ML-02)"
)
COLUMNS = (
    "student_id",
    "university_code",
    "seed_verified",
    "budget.min",
    "budget.max",
    "maxDistanceKm",
    "essentialServices",
    "livingWith",
    "placeType",
    "pets",
    "transport",
    "propertyTypes",
    "synthetic",
)
DTYPES: Mapping[str, str] = {
    "student_id": "string",
    "university_code": "string",
    "seed_verified": "bool",
    "budget.min": "Int64",
    "budget.max": "int64",
    "maxDistanceKm": "Float64",
    "livingWith": "string",
    "placeType": "string",
    "pets": "boolean",
    "transport": "string",
    "synthetic": "bool",
}


class StudentsError(RuntimeError):
    """Falta una renta p50 regional o un campus para generar los estudiantes."""


@dataclass(frozen=True)
class RentAnchor:
    """Renta p50 regional de un campus (pesos de jul-2024) y su procedencia."""

    code: str
    municipio_clave: str
    municipio_nombre: str
    level: str
    p50: float


def rent_anchors(regional: pd.DataFrame, municipio_by_campus: Mapping[str, str]) -> dict[str, RentAnchor]:
    """Renta p50 del municipio de cada campus (con el nivel de respaldo de M-ML-015)."""
    rent = regional.loc[regional["indicador"] == RENT_INDICATOR]
    anchors: dict[str, RentAnchor] = {}
    for code, clave in municipio_by_campus.items():
        rows = rent.loc[rent["municipio_clave"].astype(str) == clave]
        if len(rows) != 1 or pd.isna(rows.iloc[0]["p50"]) or float(rows.iloc[0]["p50"]) <= 0:
            raise StudentsError(f"{code}: sin una renta p50 válida para el municipio {clave} en {REGIONAL_PARQUET}")
        row = rows.iloc[0]
        anchors[code] = RentAnchor(
            code=code,
            municipio_clave=clave,
            municipio_nombre=str(row["municipio_nombre"]),
            level=str(row["level"]),
            p50=float(row["p50"]),
        )
    return anchors


def students_frame(students: Sequence[SyntheticStudent], schema: FeatureSchema) -> pd.DataFrame:
    """Tabla de estudiantes con los nombres del esquema y tipos fijos."""
    frame = pd.DataFrame([student_record(student, schema) for student in students], columns=list(COLUMNS))
    return frame.astype(dict(DTYPES))


def run_students(paths: ProjectPaths) -> str:
    """Genera `students.parquet` y `reports/students_synthetic.md`."""
    regional_path = paths.data_processed / REGIONAL_PARQUET
    if not regional_path.is_file() or not paths.universities_csv.is_file():
        return STUDENTS_SKIP_MESSAGE
    schema = default_schema()
    params = load_params(schema)
    anchors = rent_anchors(pd.read_parquet(regional_path), params.municipio_by_campus)
    campuses = load_campuses(paths.universities_csv)
    missing = [code for code in schema.catalog("university_code") if code not in campuses]
    if missing:
        raise StudentsError(f"data/universities.csv: faltan campus {missing}")
    rent_p50 = {code: anchor.p50 for code, anchor in anchors.items()}
    students = generate_students(params, rent_p50, campuses, schema)

    paths.data_processed.mkdir(parents=True, exist_ok=True)
    parquet_path = paths.data_processed / STUDENTS_PARQUET
    students_frame(students, schema).to_parquet(parquet_path, index=False)
    report_path = paths.reports / STUDENTS_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(students, params, anchors, schema), encoding="utf-8", newline="\n")
    return (
        f"students: {len(students)} estudiantes sintéticos -> {parquet_path.as_posix()}; "
        f"reporte {report_path.as_posix()}"
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """p25, p50 y p75 con interpolación lineal (método inclusivo)."""
    if len(values) == 1:
        return (float(values[0]),) * 3
    p25, p50, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return p25, p50, p75


@dataclass(frozen=True)
class CalibrationRow:
    """Calibración de un campus: renta p50 regional frente a la mediana de `budget.max`."""

    code: str
    municipio: str
    level: str
    rent_p50: float
    target: float
    median: float
    deviation: float
    within: bool


def calibration_rows(
    students: Sequence[SyntheticStudent], params: StudentParams, anchors: Mapping[str, RentAnchor]
) -> list[CalibrationRow]:
    """Por campus: renta p50 regional, objetivo, mediana de `budget.max` y si cumple la tolerancia."""
    targets = budget_targets(params, {code: anchor.p50 for code, anchor in anchors.items()})
    rows: list[CalibrationRow] = []
    for code, anchor in anchors.items():
        budgets = [s.profile.budget_max for s in students if s.profile.university_code == code]
        median = float(statistics.median(b for b in budgets if b is not None))
        deviation = median / targets[code] - 1.0
        rows.append(
            CalibrationRow(
                code=code,
                municipio=anchor.municipio_nombre,
                level=anchor.level,
                rent_p50=anchor.p50,
                target=targets[code],
                median=median,
                deviation=deviation,
                within=abs(deviation) <= params.calibration_tolerance,
            )
        )
    return rows


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f} %" if total else "—"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _by_campus(students: Sequence[SyntheticStudent], schema: FeatureSchema) -> list[tuple[str, list[SyntheticStudent]]]:
    return [
        (code, [s for s in students if s.profile.university_code == code]) for code in schema.catalog("university_code")
    ]


def _category_table(
    students: Sequence[SyntheticStudent],
    schema: FeatureSchema,
    catalog: str,
    attribute: str,
    *,
    with_null: bool,
) -> list[str]:
    values = list(schema.catalog(catalog))
    header = ["Campus", *(f"`{value}`" for value in values), *(["nulo"] if with_null else [])]
    lines = [_row(header), _row(["---", *(["---:"] * (len(header) - 1))])]
    groups = [*_by_campus(students, schema), ("**Total**", list(students))]
    for code, group in groups:
        observed = [getattr(s.profile, attribute) for s in group]
        cells = [_pct(sum(1 for item in observed if item == value), len(group)) for value in values]
        if with_null:
            cells.append(_pct(sum(1 for item in observed if item is None), len(group)))
        lines.append(_row([code, *cells]))
    return lines


def render_report(
    students: Sequence[SyntheticStudent],
    params: StudentParams,
    anchors: Mapping[str, RentAnchor],
    schema: FeatureSchema,
) -> str:
    """Reporte Markdown determinista de los estudiantes sintéticos."""
    total = len(students)
    tolerance = params.calibration_tolerance
    calibration = calibration_rows(students, params, anchors)
    lines = [
        "# DATOS SINTÉTICOS — Estudiantes sintéticos (ML-03 T04)",
        "",
        "> **DATOS SINTÉTICOS.** Ningún perfil de este reporte corresponde a una persona real. Los estudiantes se",
        "> generan con distribuciones documentadas y semilla fija (constitución §8); las métricas que se calculen",
        "> con ellos no son desempeño real.",
        "",
        "Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.students_stage`). Parámetros:",
        f"`src/thehouse_ml/synthetic/{PARAMS_RESOURCE}` versión {params.metadata.get('params_version', '—')}"
        f" (decisión {params.metadata.get('decision', '—')}).",
        f"Salida: `data/processed/{STUDENTS_PARQUET}` (fuera de git, columna `synthetic = true`).",
        f"Esquema de features {schema.version}.",
        "",
        "## 1. Resumen",
        "",
        f"- Estudiantes: **{total:,}** ({params.per_campus} por campus, {len(anchors)} campus; semilla {params.seed}).",
        "- Ids sintéticos sin datos personales (`"
        + (f"{students[0].student_id}` a `{students[-1].student_id}`" if students else "—`")
        + ").",
        "- Sin ningún campo de género (constitución §11): `livingWith` es la elección del estudiante.",
        f"- Quick setup ({', '.join(f'`{name}`' for name in QUICK_SETUP_FIELDS)}) nunca nulo; "
        "los opcionales llevan nulos para ejercitar las reglas para faltantes (M-ML-022).",
        "- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017): "
        + (
            ", ".join(
                f"`{code}`" for code, group in _by_campus(students, schema) if group and not group[0].seed_verified
            )
            or "ninguno"
        )
        + ".",
        "- Presupuestos en pesos de julio de 2024 (M-ML-014), igual que la renta regional.",
        "",
        "## 2. Calibración del presupuesto",
        "",
        f"Mediana objetivo de `budget.max` = renta p50 regional por {params.median_factor:g}"
        f" (`budget.max.median_factor`); tolerancia ±{tolerance * 100:.0f} %. La renta p50 sale de",
        f"`data/processed/{REGIONAL_PARQUET}` (indicador `renta`, ENVI 2020 deflactada); el nivel indica el",
        "respaldo de M-ML-015 (`estado` = entidad completa).",
        "",
        _row(
            [
                "Campus",
                "Municipio",
                "Nivel",
                "Renta p50 regional",
                "Objetivo",
                "Mediana budget.max",
                "Desviación",
                "Dentro",
            ]
        ),
        _row(["---", "---", "---", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in calibration:
        lines.append(
            _row(
                [
                    row.code,
                    row.municipio,
                    row.level,
                    _money(row.rent_p50),
                    _money(row.target),
                    _money(row.median),
                    f"{row.deviation * 100:+.1f} %",
                    "sí" if row.within else "**no**",
                ]
            )
        )
    lines += [
        "",
        "## 3. Presupuesto por universidad",
        "",
        _row(
            [
                "Campus",
                "Verificado",
                "n",
                "budget.max p25",
                "budget.max p50",
                "budget.max p75",
                "budget.min p50",
                "budget.min nulo",
            ]
        ),
        _row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for code, group in _by_campus(students, schema):
        budgets = [float(s.profile.budget_max or 0) for s in group]
        minimums = [float(s.profile.budget_min) for s in group if s.profile.budget_min is not None]
        p25, p50, p75 = quartiles(budgets)
        lines.append(
            _row(
                [
                    code,
                    "sí" if group[0].seed_verified else "**no**",
                    len(group),
                    _money(p25),
                    _money(p50),
                    _money(p75),
                    _money(statistics.median(minimums)) if minimums else "—",
                    _pct(len(group) - len(minimums), len(group)),
                ]
            )
        )
    lines += ["", "## 4. Transporte", "", *_category_table(students, schema, "transport", "transport", with_null=True)]
    lines += ["", "## 5. Con quién vivir (`livingWith`)", ""]
    lines += _category_table(students, schema, "livingWith", "living_with", with_null=False)
    lines += ["", "## 6. Tipo de lugar (`placeType`)", ""]
    lines += _category_table(students, schema, "placeType", "place_type", with_null=False)
    lines += ["", "## 7. Servicios esenciales", ""]
    services = list(schema.catalog("essentialServices"))
    lines += [
        "Porcentaje de estudiantes que marca cada servicio como esencial (sobre el total del campus; los nulos",
        "cuentan como no marcado y se resuelven a conjunto vacío).",
        "",
        _row(["Campus", *(f"`{item}`" for item in services), "nulo"]),
        _row(["---", *(["---:"] * (len(services) + 1))]),
    ]
    for code, group in [*_by_campus(students, schema), ("**Total**", list(students))]:
        cells = [
            _pct(
                sum(1 for s in group if s.profile.essential_services and item in s.profile.essential_services),
                len(group),
            )
            for item in services
        ]
        cells.append(_pct(sum(1 for s in group if s.profile.essential_services is None), len(group)))
        lines.append(_row([code, *cells]))
    lines += [
        "",
        "## 8. Nulos",
        "",
        "Proporción de nulos por campo frente a la probabilidad de `students_params.json`. Los del quick setup",
        "deben ser 0.",
        "",
        _row(["Campo", "Probabilidad (parámetro)", "Observado", "Regla para faltantes"]),
        _row(["---", "---:", "---:", "---"]),
    ]
    attributes = {
        "budget.min": "budget_min",
        "budget.max": "budget_max",
        "maxDistanceKm": "max_distance_km",
        "essentialServices": "essential_services",
        "livingWith": "living_with",
        "placeType": "place_type",
        "pets": "pets",
        "transport": "transport",
        "propertyTypes": "property_types",
    }
    for field, attribute in attributes.items():
        observed = sum(1 for s in students if getattr(s.profile, attribute) is None)
        expected = params.value(OPTIONAL_NULL_PARAMS[field]) if field in OPTIONAL_NULL_PARAMS else 0.0
        rule = schema.student_field(field).missing
        lines.append(
            _row(
                [
                    f"`{field}`",
                    f"{float(expected) * 100:.1f} %",
                    _pct(observed, total),
                    rule.description if rule else "—",
                ]
            )
        )
    limited = sum(1 for s in students if s.profile.living_with != "any")
    lines += [
        "",
        "## 9. Hallazgo: estudiantes limitados por `livingWith`",
        "",
        f"{limited:,} de {total:,} estudiantes ({_pct(limited, total)}) piden `women` o `men`. Con HC02",
        "(M-ML-024) solo ven cuartos con el mismo `livingWith`: **no ven cuartos `any`**. Por eso la",
        "distribución de `livingWith` de los cuartos (T05) debe tener suficientes cuartos `women` y `men`",
        "en cada campus para el umbral de ≥ 90 % de T06; la recomendación está en `evidence.md` de ML-03 (T04).",
        "",
        "## 10. Privacidad",
        "",
        "- Sin nombres, correos, teléfonos ni identificadores reales; los ids son consecutivos (constitución §7).",
        "- Sin género, edad ni origen (constitución §11).",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "STUDENTS_PARQUET",
    "STUDENTS_REPORT",
    "CalibrationRow",
    "RentAnchor",
    "StudentsError",
    "calibration_rows",
    "quartiles",
    "render_report",
    "rent_anchors",
    "run_students",
    "students_frame",
]
