"""ML-02 T08 (unitaria): data card con datos inventados y sin red."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from thehouse_ml.data_card import DataCardInputs, University, render_data_card
from thehouse_ml.manifests import Manifest, ManifestFile
from thehouse_ml.national import ENIGH_SOURCE, ENVI_SOURCE
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import CENSO_SOURCE
from thehouse_ml.rent_histogram import build_bins, rows_to_frame
from thehouse_ml.sources import SourceSpec

SECTION_TITLES = (
    "## 1. Propósito y alcance",
    "## 2. Fuentes",
    "## 3. Atribuciones",
    "## 4. Ponderación y deflactación",
    "## 5. Niveles de respaldo",
    "## 6. Tamaños de muestra",
    "## 7. Histogramas",
    "## 8. Universidades",
    "## 9. Transformaciones",
    "## 10. Limitaciones",
    "## 11. Privacidad",
)


def _source(
    source_id: str,
    attribution: str,
    variables: tuple[str, ...],
    *,
    status: str = "confirmed",
    reference: str | None = None,
) -> SourceSpec:
    return SourceSpec(
        id=source_id,
        name=f"Producto {source_id}",
        license=f"Licencia {source_id}",
        attribution=attribution,
        variables_status=status,
        variables_used=variables,
        download_url=(),
        status_reference=reference,
    )


def _inputs() -> DataCardInputs:
    sources = (
        _source(CENSO_SOURCE, "Fuente: INEGI, Censo 2020", ("FACTOR", "ESTRATO", "UPM")),
        _source(ENVI_SOURCE, "Fuente: INEGI, ENVI 2020", ("FACTOR", "EST_DIS", "UPM_DIS")),
        _source(ENIGH_SOURCE, "Fuente: INEGI, ENIGH 2022", ("factor", "est_dis", "upm")),
        _source("inegi_denue_vigente", "Fuente: INEGI, DENUE 05/2026", ("id", "codigo_act")),
        _source("inegi_inpc", "Fuente: INEGI, INPC", ("FECHA", "CONCEPTO", "VALOR")),
        _source("openstreetmap", "© OpenStreetMap contributors (ODbL).", ("osm_id",)),
        _source(
            "inside_airbnb_cdmx_2026_09_23",
            "Fuente: Inside Airbnb, listings (CC BY 4.0)",
            ("price",),
            status="discarded",
            reference="M-ML-019",
        ),
    )
    manifest = Manifest(
        source_id=ENVI_SOURCE,
        files=(
            ManifestFile(
                url="https://example.test/envi.zip",
                filename="envi.zip",
                downloaded_at="2026-11-17",
                size_bytes=123456,
                sha256="a" * 64,
                license="Libre uso",
            ),
        ),
    )
    baselines = pd.DataFrame(
        [
            _baseline(ENVI_SOURCE, "nacional", "nacional", 100, 10_000.0),
            _baseline(ENVI_SOURCE, "entidad", "13", 50, 2_500.0),
            _baseline(ENVI_SOURCE, "entidad", "15", 60, 3_000.0),
            _baseline(ENVI_SOURCE, "tipo_viv", "6", 5, 99_999.0),
            _baseline(ENIGH_SOURCE, "nacional", "nacional", 120, 12_000.0),
            _baseline(ENIGH_SOURCE, "tipo_viv", "&", 40, 12_345.0),
        ]
    )
    regional = pd.DataFrame(
        [
            _regional("13013", "Axtla", "13", "renta", "estado", "ENVI", 1_500.0, 50, float("nan")),
            _regional("13013", "Axtla", "13", "rentadas", "municipio", "Censo", 0.16, 45, 0.10),
            _regional("13013", "Axtla", "13", "internet", "municipio", "Censo", 0.55, 5, 0.20),
        ]
    )
    histogram = []
    for scope, n, weight in (("nacional", 100, 100.0), ("13", 50, 50.0), ("15", 60, 50.0)):
        histogram.extend(build_bins(scope, [1000.0] * n, [weight] * n, 1.0))
    campus = pd.DataFrame(
        [
            {"code": "AAA", "radius_m": 2000, "category": "total", "count": 50},
            {"code": "BBB", "radius_m": 2000, "category": "total", "count": 100},
            {"code": "AAA", "radius_m": 1000, "category": "total", "count": 10},
        ]
    )
    universities = (
        University("AAA", "Universidad A", "seed_sin_verificar", "seed", "20.0", "-99.0"),
        University("BBB", "Universidad B", "ok_osm", "osm:way/1", "20.1", "-99.1"),
    )
    return DataCardInputs(
        sources=sources,
        source_meta={
            source.id: {"edition": f"Edición {source.id}", "url": f"https://example.test/{source.id}"}
            for source in sources
        },
        manifests={ENVI_SOURCE: manifest},
        baselines=baselines,
        regional=regional,
        campus=campus,
        rent_histogram=rows_to_frame(histogram),
        universities=universities,
    )


def _baseline(source: str, level: str, value: str, n: int, expanded: float) -> dict[str, object]:
    return {
        "source": source,
        "year": 2020,
        "level": level,
        "level_value": value,
        "n_muestra": n,
        "suma_factores": expanded,
        "deflactor": 1.25,
    }


def _regional(
    clave: str,
    nombre: str,
    entidad: str,
    indicador: str,
    level: str,
    source: str,
    estimate: float,
    rented: int,
    cv: float,
) -> dict[str, object]:
    return {
        "municipio_clave": clave,
        "municipio_nombre": nombre,
        "entidad": entidad,
        "indicador": indicador,
        "level": level,
        "source": source,
        "estimate": estimate,
        "n_rentadas": rented,
        "cv": cv,
    }


def _section(text: str, number: int) -> str:
    start = text.index(f"## {number}.")
    end = text.index(f"## {number + 1}.") if number < len(SECTION_TITLES) else len(text)
    return text[start:end]


def test_data_card_has_all_eleven_sections() -> None:
    text = render_data_card(_inputs())

    for title in SECTION_TITLES:
        assert title in text


def test_discarded_source_appears_as_discarded_but_not_as_an_attribution() -> None:
    text = render_data_card(_inputs())

    assert "DESCARTADA (M-ML-019)" in _section(text, 2)
    assert "Inside Airbnb" not in _section(text, 3)
    assert "inside_airbnb_cdmx_2026_09_23" in _section(text, 10)


def test_every_inegi_product_and_osm_are_attributed_without_endorsement() -> None:
    inputs = _inputs()
    text = render_data_card(inputs)
    attributions = _section(text, 3)

    for source in inputs.sources:
        if source.variables_status != "discarded" and "INEGI" in source.attribution:
            assert source.attribution in attributions
    assert "© OpenStreetMap contributors (ODbL)." in attributions
    assert "INEGI no respalda este proyecto" in attributions


def test_source_figures_come_from_the_manifest() -> None:
    text = render_data_card(_inputs())

    assert "123,456 B" in _section(text, 2)
    assert "a" * 64 in text


def test_unspecified_tipo_viv_code_is_labeled_in_the_sample_table() -> None:
    section = _section(render_data_card(_inputs()), 6)

    assert "| `inegi_enigh_2022` | tipo de vivienda no especificado (&) | 40 | 12,345 |" in section
    assert "| tipo de vivienda & |" not in section


def test_cv_note_is_below_the_levels_table() -> None:
    section = _section(render_data_card(_inputs()), 5)
    note = (
        "El CV se muestra solo para proporciones (linealización de Taylor con ESTRATO/UPM). "
        "La renta es una mediana ponderada; su CV no se calcula y se toma del nivel estatal "
        "porque ningún municipio alcanza la suficiencia."
    )

    assert note in section
    assert section.index("| Municipio | indicador |") < section.index(note)


def test_rounding_note_uses_the_registered_deflactor() -> None:
    section = _section(render_data_card(_inputs()), 7)

    assert "Las rentas se reportan en cifras redondas (2,000, 2,500…)" in section
    assert "(factor 1.250000) se desplazan de intervalo (2,500 → 3,125)" in section
    assert "No es un error de procesamiento" in section


def test_cells_with_fewer_than_thirty_dwellings_are_suppressed() -> None:
    text = render_data_card(_inputs())

    samples = _section(text, 6)
    assert "| `inegi_envi_2020` | tipo de vivienda 6 | suprimido | suprimido |" in samples
    assert "99,999" not in samples
    levels = _section(text, 5)
    assert "internet | municipio | Censo | suprimido | suprimido | suprimido |" in levels
    assert "| 55.0 % |" not in levels


def test_rent_histogram_adds_up_to_the_expanded_total() -> None:
    inputs = _inputs()
    text = render_data_card(inputs)

    blocks = re.findall(r"```text\n(.*?)```", _section(text, 7), re.DOTALL)
    assert len(blocks) == 4
    for block, expected in zip(blocks[:3], (10_000.0, 2_500.0, 3_000.0), strict=True):
        bars = [line for line in block.splitlines() if " (n=" in line]
        rendered_total = re.search(r"Total expandido\s*\|\s*([\d,]+)", block)
        assert rendered_total is not None
        assert sum(int(re.search(r"([\d,]+)\s+\(n=", line).group(1).replace(",", "")) for line in bars) == int(expected)
        assert int(rendered_total.group(1).replace(",", "")) == int(expected)


def test_denue_histogram_shows_the_total_category_at_the_configured_radius() -> None:
    text = render_data_card(_inputs())

    denue = _section(text, 7).split("### Negocios totales")[1]
    assert "AAA" in denue
    assert "BBB" in denue
    assert "100" in denue
    assert "```text" in denue


def test_unverified_universities_are_warned_about() -> None:
    text = render_data_card(_inputs())
    section = _section(text, 8)

    assert "`AAA`" in section
    assert "`seed_sin_verificar`" in section
    assert "Universidad A" in section


def test_privacy_section_names_never_read_columns_and_aggregates_only() -> None:
    text = render_data_card(_inputs())
    section = _section(text, 11)

    for column in ("nom_estab", "telefono", "correoelec", "tipo_vial"):
        assert f"`{column}`" in section
    assert "conteos agregados" in section
    assert "data/universities.csv" in section


def test_all_inputs_are_read_from_disk(tmp_path: Path) -> None:
    from thehouse_ml.data_card import collect_inputs

    inputs = _inputs()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    inputs.baselines.to_parquet(tmp_path / "data" / "processed" / "national_baselines.parquet", index=False)
    inputs.regional.to_parquet(tmp_path / "data" / "processed" / "regional_calibration.parquet", index=False)
    inputs.campus.to_parquet(tmp_path / "data" / "processed" / "campus_context.parquet", index=False)
    inputs.rent_histogram.to_parquet(tmp_path / "data" / "processed" / "rent_histogram.parquet", index=False)
    (tmp_path / "data" / "manifests" / f"{ENVI_SOURCE}.json").write_text(
        '{"source_id": "inegi_envi_2020", "files": []}', encoding="utf-8"
    )
    (tmp_path / "data" / "universities.csv").write_text(
        "code,name,final_status,final_source,final_lat,final_lon\nAAA,A,ok_osm,osm:way/1,20,-99\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "sources.toml").write_text(
        '[[sources]]\nid = "inegi_envi_2020"\nname = "ENVI"\nlicense = "L"\nattribution = "Fuente: INEGI, ENVI 2020"\n'
        'variables_status = "confirmed"\nvariables_used = ["FACTOR"]\n',
        encoding="utf-8",
    )

    collected = collect_inputs(ProjectPaths(root=tmp_path))

    assert len(collected.universities) == 1
    assert collected.manifests[ENVI_SOURCE].source_id == ENVI_SOURCE
    assert "INEGI" in render_data_card(collected)
