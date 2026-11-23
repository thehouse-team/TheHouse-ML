"""Datos sintéticos (constitución §8): estudiantes, cuartos y pares (ML-03 T04-T06) y etiquetas (ML-04 T04).

`students_params.json`, `rooms_params.json`, `pairs_params.json` y `labels_params.json` viven junto a
este módulo y se empaquetan con él.
"""

from __future__ import annotations

from thehouse_ml.synthetic.labels import (
    LabelInput,
    LabelParams,
    LabelParamsError,
    LabelSet,
    simulate_labels,
)
from thehouse_ml.synthetic.labels import load_params as load_label_params
from thehouse_ml.synthetic.pairs import (
    PairParams,
    PairParamsError,
    PairSet,
    PairSetError,
    build_pairs,
)
from thehouse_ml.synthetic.pairs import load_params as load_pair_params
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
    "LabelInput",
    "LabelParams",
    "LabelParamsError",
    "LabelSet",
    "PairParams",
    "PairParamsError",
    "PairSet",
    "PairSetError",
    "Parameter",
    "RoomParams",
    "RoomParamsError",
    "StudentParams",
    "StudentParamsError",
    "SyntheticRoom",
    "SyntheticStudent",
    "budget_targets",
    "build_pairs",
    "generate_rooms",
    "generate_students",
    "load_label_params",
    "load_pair_params",
    "load_params",
    "load_room_params",
    "parse_params",
    "room_record",
    "simulate_labels",
    "student_record",
]
