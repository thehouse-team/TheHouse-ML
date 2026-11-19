"""Datos sintéticos de ML-03 (constitución §8): estudiantes (T04).

`students_params.json` vive junto a este módulo y se empaqueta con él.
"""

from __future__ import annotations

from thehouse_ml.synthetic.students import (
    Parameter,
    StudentParams,
    StudentParamsError,
    SyntheticStudent,
    budget_targets,
    generate_students,
    load_params,
    parse_params,
    student_record,
)

__all__ = [
    "Parameter",
    "StudentParams",
    "StudentParamsError",
    "SyntheticStudent",
    "budget_targets",
    "generate_students",
    "load_params",
    "parse_params",
    "student_record",
]
