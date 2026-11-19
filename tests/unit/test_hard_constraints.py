"""ML-03 T03 (R03): restricciones duras con la implementación de referencia.

Los casos por tabla viven en `tests/fixtures/hard_constraint_cases.json` para
reutilizarse como casos dorados del backend (ML-06). Sus salidas se escribieron
a mano, sin este código.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import (
    ConstraintResult,
    FeatureSchema,
    HardConstraint,
    IncompleteProfileError,
    PairInputError,
    ResolvedStudent,
    Room,
    StudentProfile,
    check_hard_constraints,
    constraints,
    default_schema,
    filter_rooms,
    implemented_constraints,
    pairs,
    room_from_record,
    student_from_record,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
CASES: dict[str, Any] = json.loads((FIXTURES / "hard_constraint_cases.json").read_text(encoding="utf-8"))
PAIR_CASES: dict[str, Any] = json.loads((FIXTURES / "pair_feature_cases.json").read_text(encoding="utf-8"))
LIVING_WITH = ("any", "women", "men")
GENDER_WORDS = re.compile(r"gender|genero|género|sex|sexo", re.IGNORECASE)


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


def _case(case_id: str) -> dict[str, Any]:
    for case in CASES["cases"]:
        if case["id"] == case_id:
            return dict(case)
    raise KeyError(case_id)


def _student(**changes: Any) -> StudentProfile:
    return dataclasses.replace(student_from_record(_case("budget_rent_below")["input"]["student"]), **changes)


def _room(**changes: Any) -> Room:
    return dataclasses.replace(room_from_record(_case("budget_rent_below")["input"]["room"]), **changes)


# --- Casos por tabla -------------------------------------------------------


@pytest.mark.parametrize("case", CASES["cases"], ids=[case["id"] for case in CASES["cases"]])
def test_table_case(case: dict[str, Any], schema: FeatureSchema) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    if "expected_error" in case:
        assert case["expected_error"]["type"] == "IncompleteProfileError"
        with pytest.raises(IncompleteProfileError) as raised:
            check_hard_constraints(student, room, schema)
        assert list(raised.value.missing) == case["expected_error"]["missing"]
        return
    result = check_hard_constraints(student, room, schema)
    expected = case["expected"]
    assert result == ConstraintResult(allowed=expected["allowed"], violated=tuple(expected["violated"]))


def test_the_table_covers_every_requested_scenario() -> None:
    ids = {case["id"] for case in CASES["cases"]}
    assert {"budget_rent_below", "budget_rent_equal", "budget_rent_above"} <= ids
    assert {f"living_with_student_{s}_room_{r}" for s in LIVING_WITH for r in LIVING_WITH} <= ids
    assert {f"pets_student_{s}_room_{r}" for s in ("false", "true") for r in ("false", "true")} <= ids
    assert "pets_null_room_false" in ids
    multiple = [case for case in CASES["cases"] if len(case.get("expected", {}).get("violated", [])) > 1]
    assert {len(case["expected"]["violated"]) for case in multiple} == {2, 3}
    incomplete = {case["id"] for case in CASES["cases"] if "expected_error" in case}
    assert incomplete == {
        "incomplete_budget_max",
        "incomplete_livingWith",
        "incomplete_placeType",
        "incomplete_all_quick_setup",
    }
    assert "pets" not in _case("pets_null_room_false")["input"]["student"]


def test_the_table_metadata_matches_the_schema(schema: FeatureSchema) -> None:
    metadata = CASES["metadata"]
    assert metadata["schema_version"] == schema.version
    assert metadata["synthetic"] is True
    assert metadata["hard_constraints"] == [constraint.id for constraint in schema.hard_constraints]


def test_living_with_matrix_of_the_spec() -> None:
    """Spec §7 regla 3: válido si el estudiante acepta `any` o si coinciden."""
    for student_value in LIVING_WITH:
        for room_value in LIVING_WITH:
            result = check_hard_constraints(_student(living_with=student_value), _room(living_with=room_value))
            expected = student_value == "any" or student_value == room_value
            assert result.allowed is expected, (student_value, room_value)


# --- Reglas -----------------------------------------------------------------


def test_rent_equal_to_budget_passes_and_one_peso_more_is_excluded() -> None:
    assert check_hard_constraints(_student(budget_max=4000), _room(price_mxn=4000)).allowed
    assert check_hard_constraints(_student(budget_max=4000), _room(price_mxn=4001)).violated == ("HC01_budget",)


def test_null_pets_follows_the_missing_rule_of_the_schema(schema: FeatureSchema) -> None:
    rule = schema.student_field("pets").missing
    assert rule is not None and rule.action == "default" and rule.value is False
    assert check_hard_constraints(_student(pets=None), _room(pets_allowed=False)).allowed


def test_distance_is_not_a_hard_constraint() -> None:
    far = _case("distance_over_max_and_30_km_not_excluded")["input"]
    student = student_from_record(far["student"])
    room = room_from_record(far["room"])
    assert student.max_distance_km == 5
    assert check_hard_constraints(student, room).allowed
    result = filter_rooms(student, [room])
    assert result.allowed == (room,)
    assert result.excluded == 0


def test_every_violated_constraint_is_reported_in_schema_order(schema: FeatureSchema) -> None:
    student = _student(living_with="women", pets=True)
    room = _room(price_mxn=9000, living_with="men", pets_allowed=False)
    assert check_hard_constraints(student, room, schema).violated == ("HC01_budget", "HC02_living_with", "HC03_pets")
    reversed_schema = dataclasses.replace(schema, hard_constraints=tuple(reversed(schema.hard_constraints)))
    assert check_hard_constraints(student, room, reversed_schema).violated == (
        "HC03_pets",
        "HC02_living_with",
        "HC01_budget",
    )


def test_constraints_are_read_from_the_schema(schema: FeatureSchema) -> None:
    only_pets = dataclasses.replace(
        schema, hard_constraints=tuple(c for c in schema.hard_constraints if c.id == "HC03_pets")
    )
    student = _student(living_with="women", pets=True)
    room = _room(price_mxn=9000, living_with="men", pets_allowed=False)
    assert check_hard_constraints(student, room, only_pets).violated == ("HC03_pets",)


def test_every_schema_constraint_has_an_implementation(schema: FeatureSchema) -> None:
    assert [constraint.id for constraint in schema.hard_constraints] == list(implemented_constraints())


def test_the_implemented_rules_are_the_ones_written_in_the_schema(schema: FeatureSchema) -> None:
    """El código liga cada id a su predicado; si cambia `excludes_when`, esta prueba obliga a revisarlo."""
    assert {constraint.id: constraint.excludes_when for constraint in schema.hard_constraints} == {
        "HC01_budget": "room.price_mxn > student.budget.max",
        "HC02_living_with": 'student.livingWith != "any" and room.livingWith != student.livingWith',
        "HC03_pets": "student.pets = true and room.pets_allowed = false",
    }


def test_a_schema_constraint_without_implementation_fails(schema: FeatureSchema) -> None:
    extra = HardConstraint(
        id="HC99_noise",
        description="Restricción sin implementación.",
        excludes_when="room.bedrooms > 3",
        fields=("room.bedrooms",),
    )
    broken = dataclasses.replace(schema, hard_constraints=(*schema.hard_constraints, extra))
    with pytest.raises(PairInputError, match="HC99_noise"):
        check_hard_constraints(_student(), _room(), broken)
    with pytest.raises(PairInputError, match="HC99_noise"):
        filter_rooms(_student(), [_room()], broken)


def test_invalid_inputs_are_rejected_like_in_t02() -> None:
    with pytest.raises(PairInputError, match="livingWith"):
        check_hard_constraints(_student(living_with="mixed"), _room())
    with pytest.raises(PairInputError, match=r"room.livingWith"):
        check_hard_constraints(_student(), _room(living_with="mixed"))
    with pytest.raises(PairInputError, match=r"room.price_mxn"):
        filter_rooms(_student(), [_room(), _room(price_mxn=-1)])


def test_missing_rules_are_the_single_implementation_of_t02(monkeypatch: pytest.MonkeyPatch) -> None:
    assert constraints.resolve_student is pairs.resolve_student
    assert constraints.validate_room is pairs.validate_room
    calls: list[StudentProfile] = []
    original = pairs.resolve_student

    def spy(student: StudentProfile, schema: FeatureSchema | None = None) -> ResolvedStudent:
        calls.append(student)
        return original(student, schema)

    monkeypatch.setattr(constraints, "resolve_student", spy)
    student = _student()
    check_hard_constraints(student, _room())
    filter_rooms(student, [_room(), _room(), _room()])
    assert calls == [student, student]  # una vez por par y una vez por filtro


# --- Filtro -----------------------------------------------------------------


def test_filter_returns_allowed_rooms_in_order_and_exclusions_per_constraint(schema: FeatureSchema) -> None:
    student = _student(budget_max=4000, living_with="women", pets=True)
    ok_1 = _room(price_mxn=3000, living_with="women", pets_allowed=True)
    over_budget = _room(price_mxn=4500, living_with="women", pets_allowed=True)
    wrong_living = _room(price_mxn=3000, living_with="any", pets_allowed=True)
    everything = _room(price_mxn=4500, living_with="men", pets_allowed=False)
    ok_2 = _room(price_mxn=4000, living_with="women", pets_allowed=True)
    result = filter_rooms(student, [ok_1, over_budget, wrong_living, everything, ok_2], schema)
    assert result.allowed == (ok_1, ok_2)
    assert result.excluded == 3
    assert dict(result.exclusions) == {"HC01_budget": 2, "HC02_living_with": 2, "HC03_pets": 1}
    assert list(result.exclusions) == [constraint.id for constraint in schema.hard_constraints]


def test_filter_agrees_with_the_pair_function() -> None:
    rooms = [room_from_record(case["input"]["room"]) for case in CASES["cases"] if "expected" in case]
    student = _student(living_with="men", pets=True)
    result = filter_rooms(student, rooms)
    assert result.allowed == tuple(room for room in rooms if check_hard_constraints(student, room).allowed)


def test_filter_with_no_rooms_reports_zero_for_every_constraint(schema: FeatureSchema) -> None:
    result = filter_rooms(_student(), [], schema)
    assert result.allowed == ()
    assert result.excluded == 0
    assert dict(result.exclusions) == dict.fromkeys(implemented_constraints(), 0)


def test_filter_with_incomplete_profile_raises_the_t02_error() -> None:
    with pytest.raises(IncompleteProfileError) as raised:
        filter_rooms(_student(budget_max=None, budget_min=None, living_with=None), [_room()])
    assert raised.value.missing == ("budget.max", "livingWith")


# --- Coherencia con T02 -----------------------------------------------------


def test_pairs_that_pass_the_constraints_have_price_to_budget_at_most_one_and_compatible_pets() -> None:
    checked = 0
    for case in PAIR_CASES["cases"]:
        if "expected" not in case:
            continue
        student = student_from_record(case["input"]["student"])
        room = room_from_record(case["input"]["room"])
        if not check_hard_constraints(student, room).allowed:
            continue
        checked += 1
        assert case["expected"]["price_to_budget"] <= 1.0, case["id"]
        assert case["expected"]["pets_compatible"] == 1, case["id"]
    assert checked >= 20  # la prueba no es vacía


def test_pair_cases_excluded_by_t02_signals_violate_the_matching_constraint() -> None:
    for case_id, constraint_id in (
        ("price_over_budget", "HC01_budget"),
        ("budget_max_zero_price_positive", "HC01_budget"),
        ("pets_student_true_room_false", "HC03_pets"),
    ):
        case = next(case for case in PAIR_CASES["cases"] if case["id"] == case_id)
        result = check_hard_constraints(
            student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"])
        )
        assert constraint_id in result.violated, case_id


# --- Privacidad y pureza ------------------------------------------------------


def test_no_gender_field_exists_in_the_profile_or_is_used_by_the_constraints(schema: FeatureSchema) -> None:
    """Constitution §11: el género del estudiante no existe; HC02 solo compara livingWith."""
    profile_fields = [field.name for cls in (StudentProfile, ResolvedStudent) for field in dataclasses.fields(cls)]
    assert not [name for name in profile_fields if GENDER_WORDS.search(name)]
    assert not [name for name in pairs.STUDENT_ATTRIBUTES if GENDER_WORDS.search(name)]
    assert not [spec.name for spec in schema.student if GENDER_WORDS.search(spec.name)]
    hc02 = next(constraint for constraint in schema.hard_constraints if constraint.id == "HC02_living_with")
    assert set(hc02.fields) == {"student.livingWith", "room.livingWith"}
    tree = ast.parse(Path(constraints.__file__).read_text(encoding="utf-8"))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    identifiers |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    assert not [name for name in identifiers if GENDER_WORDS.search(name)]
    student_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "student"
    }
    assert student_attributes == {"budget_max", "living_with", "pets"}


def test_the_constraints_module_has_no_io_or_randomness() -> None:
    tree = ast.parse(Path(constraints.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported |= {
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"random", "numpy", "os", "pathlib", "io", "csv", "json", "urllib", "time", "datetime"}
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "open" not in called


def test_the_function_is_deterministic_and_does_not_mutate_inputs() -> None:
    student = _student(living_with="women", pets=True)
    rooms = [_room(), _room(price_mxn=5000, living_with="women")]
    before = (dataclasses.asdict(student), [dataclasses.asdict(room) for room in rooms])
    assert check_hard_constraints(student, rooms[1]) == check_hard_constraints(student, rooms[1])
    assert filter_rooms(student, rooms) == filter_rooms(student, rooms)
    assert (dataclasses.asdict(student), [dataclasses.asdict(room) for room in rooms]) == before
