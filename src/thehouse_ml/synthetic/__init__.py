"""Datos sintéticos de ML-03 (constitución §8): estudiantes (T04) y catálogo de cuartos (T05).

`students_params.json` y `rooms_params.json` viven junto a este módulo y se
empaquetan con él.
"""

from __future__ import annotations

from thehouse_ml.synthetic.rooms import (
    CampusInputs,
    HistogramBin,
    RoomParams,
    RoomParamsError,
    SyntheticRoom,
    generate_rooms,
    room_record,
)
from thehouse_ml.synthetic.rooms import load_params as load_room_params
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
    "CampusInputs",
    "HistogramBin",
    "Parameter",
    "RoomParams",
    "RoomParamsError",
    "StudentParams",
    "StudentParamsError",
    "SyntheticRoom",
    "SyntheticStudent",
    "budget_targets",
    "generate_rooms",
    "generate_students",
    "load_params",
    "load_room_params",
    "parse_params",
    "room_record",
    "student_record",
]
