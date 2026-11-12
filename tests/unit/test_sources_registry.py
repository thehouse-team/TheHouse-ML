"""T01 (ML-02-R01): el registro de fuentes debe ser completo."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = REPO_ROOT / "data" / "sources.toml"
REQUIRED_FIELDS = (
    "url",
    "edition",
    "date",
    "license",
    "attribution",
    "variables_used",
    "dictionary_url",
    "variables_status",
)
ALLOWED_VARIABLES_STATUS = {"pending", "confirmed"}


def _load_sources() -> list[dict[str, object]]:
    with SOURCES_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    sources = data.get("sources")
    assert isinstance(sources, list), "sources.toml debe definir [[sources]]"
    assert sources, "sources.toml debe registrar al menos una fuente"
    return sources


def test_sources_toml_exists() -> None:
    assert SOURCES_PATH.is_file(), f"No existe {SOURCES_PATH}"


def test_each_source_has_required_fields() -> None:
    for source in _load_sources():
        source_id = source.get("id", "<sin id>")
        for field in REQUIRED_FIELDS:
            assert field in source, f"La fuente {source_id!r} no tiene el campo {field!r}"
            value = source[field]
            if isinstance(value, str):
                assert value.strip(), f"La fuente {source_id!r} tiene {field!r} vacío"
            elif isinstance(value, list):
                assert value, f"La fuente {source_id!r} tiene {field!r} vacío"
                assert all(isinstance(item, str) and item.strip() for item in value), (
                    f"La fuente {source_id!r} tiene valores inválidos en {field!r}"
                )
            else:
                raise AssertionError(f"La fuente {source_id!r} tiene tipo inválido para {field!r}: {type(value)!r}")


def test_sources_have_unique_ids() -> None:
    ids = [source.get("id") for source in _load_sources()]
    assert all(isinstance(source_id, str) and source_id.strip() for source_id in ids), (
        "Cada fuente debe tener un id de texto no vacío"
    )
    assert len(ids) == len(set(ids)), "Los ids de fuentes deben ser únicos"


def test_variables_status_is_valid() -> None:
    for source in _load_sources():
        source_id = source.get("id", "<sin id>")
        status = source.get("variables_status")
        assert status in ALLOWED_VARIABLES_STATUS, (
            f"La fuente {source_id!r} tiene variables_status inválido: {status!r}"
        )
