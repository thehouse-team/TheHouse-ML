"""ML-06 T06 (ML-06-R06): coherencia de `docs/backend-integration.md` con el contrato.

La guía de integración para NV-23 cita nombres, órdenes, reglas y cifras que deben coincidir con
`feature_schema.json`, el grafo ONNX y los casos dorados. Aquí:

1. sin el paquete (corren siempre, también en CI): orden de las features = `feature_order` del
   esquema; nombres de entrada y salida del grafo; orden de factores y sus features, `maxPoints`,
   conteos, tolerancias y ejemplos de redondeo = `tests/fixtures/logreg_golden_cases.json`;
   HC01 a HC03 y factores de transporte = esquema; regla de redondeo (`AwayFromZero`, round6, `-0`);
   JSON de ejemplo = caso dorado `perfect_100`; SHA-256 de `baseline_params.json` = el del commit
   del release (si el commit está en el clon);
2. con el paquete del release (`dist/model-v1.0.0/` generado desde el commit del release, fuera de
   git): entradas, salidas, tipos, formas y `metadata_props` de `model.onnx`, `maxPointsRaw` y las
   líneas de `SHA256SUMS` citadas en la guía, y el SHA-256 de `model.onnx` fijado. Sin el paquete
   (p. ej. en CI, ML-06-RK05) se omiten con motivo explícito: un control no ejecutado no es PASS.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto

from thehouse_ml.export.onnx_graph import FACTOR_OUTPUT, INPUT_NAME, IR_VERSION, OPSET, TOTAL_OUTPUT
from thehouse_ml.export.package import CHECKSUMMED_FILES, MODEL_ONNX, PACKAGE_DIRS, SHA256SUMS

REPO = Path(__file__).resolve().parents[2]
GUIDE_PATH = REPO / "docs" / "backend-integration.md"
SCHEMA_PATH = REPO / "src" / "thehouse_ml" / "features" / "feature_schema.json"
LOGREG_CASES_PATH = REPO / "tests" / "fixtures" / "logreg_golden_cases.json"
BASELINE_CASES_PATH = REPO / "tests" / "fixtures" / "baseline_golden_cases.json"
PACKAGE_DIR = REPO / "dist" / PACKAGE_DIRS["full"]

RELEASE_TAG = "model-v1.0.0"
RELEASE_COMMIT = "59f07bcfe1fa6cf4a0f183b568c56913980175ee"
BASELINE_PARAMS = "src/thehouse_ml/models/baseline_params.json"
METADATA_KEYS = (
    "modelVersion",
    "featureSchemaVersion",
    "featureOrder",
    "factorOrder",
    "maxPointsRaw",
    "roundingRule",
    "logregSha256",
    "gitCommit",
    "profile",
    "synthetic",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


GUIDE = GUIDE_PATH.read_text(encoding="utf-8")
SCHEMA: dict[str, Any] = _load_json(SCHEMA_PATH)
LOGREG: dict[str, Any] = _load_json(LOGREG_CASES_PATH)
BASELINE: dict[str, Any] = _load_json(BASELINE_CASES_PATH)


def section(number: int) -> str:
    """Texto de la sección `## <number>. …` de la guía, hasta la siguiente sección de nivel 2."""
    match = re.search(rf"^## {number}\. .*?(?=^## |\Z)", GUIDE, re.M | re.S)
    assert match is not None, f"la guía no tiene la sección {number}"
    return match.group(0)


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def round_factor(value: float) -> float:
    """Regla de la guía §8: round6 y 1 decimal, mitad hacia arriba (AwayFromZero para x ≥ 0)."""
    round6 = Decimal(repr(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return float(round6.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def case(case_id: str) -> dict[str, Any]:
    matches = [item for item in LOGREG["cases"] if item["id"] == case_id]
    assert len(matches) == 1, case_id
    result: dict[str, Any] = matches[0]
    return result


def table_rows(text: str, columns: int) -> list[list[str]]:
    """Filas de tablas Markdown con `columns` celdas (sin la fila de separadores)."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        if len(cells) == columns:
            rows.append(cells)
    return rows


# 1. Sin el paquete: esquema, grafo y casos dorados versionados.


def test_feature_table_follows_schema_feature_order() -> None:
    rows = re.findall(r"^\| (\d) \| `([a-z_]+)` \|", section(6), re.M)
    assert [int(index) for index, _ in rows] == list(range(10))
    assert [name for _, name in rows] == SCHEMA["feature_order"]
    assert compact(SCHEMA["feature_order"]) in GUIDE


def test_feature_ranges_cited_match_schema() -> None:
    rows = {cells[1].strip("`"): cells for cells in table_rows(section(6), 5) if cells[0].isdigit()}
    for feature in SCHEMA["pair_features"]:
        cells = rows[feature["name"]]
        training = feature.get("training_range")
        expected = "—" if training is None else f"[{training[0]}, {training[1]}]"
        assert cells[4] == expected, feature["name"]
        if feature["range"]["max"] is None:
            assert cells[3] == "[0, sin máximo]", feature["name"]


def test_graph_input_and_output_names_types_and_shapes() -> None:
    contract = section(7)
    assert f"| `{INPUT_NAME}` | Entrada | `tensor(double)` | `[N, 10]` |" in contract
    assert f"| `{FACTOR_OUTPUT}` | Salida | `tensor(double)` | `[N, 6]` |" in contract
    assert f"| `{TOTAL_OUTPUT}` | Salida | `tensor(double)` | `[N]` |" in contract
    assert f'CreateFromTensor("{INPUT_NAME}"' in GUIDE
    assert f'r.Name == "{FACTOR_OUTPUT}"' in GUIDE and f'r.Name == "{TOTAL_OUTPUT}"' in GUIDE
    assert f"opset {OPSET}, IR {IR_VERSION}" in GUIDE


def test_factor_order_and_factor_features_match_golden_cases() -> None:
    factors = LOGREG["metadata"]["factors"]
    assert [item["factor"] for item in factors] == LOGREG["metadata"]["factorOrder"]
    assert compact(LOGREG["metadata"]["factorOrder"]) in GUIDE
    rows = {cells[0].strip("`"): cells[1] for cells in table_rows(section(9), 2) if cells[0].startswith("`")}
    assert list(rows) == LOGREG["metadata"]["factorOrder"]
    for item in factors:
        assert rows[item["factor"]] == ", ".join(f"`{name}`" for name in item["features"]), item["factor"]
    assert LOGREG["metadata"]["fixedTermFeatures"] == ["pets_compatible"]
    assert "`pets_compatible` no pertenece a ningún factor" in section(9)


def test_max_points_table_matches_golden_cases() -> None:
    rows = {cells[0].strip("`"): cells for cells in table_rows(section(8), 3) if cells[0].startswith("`")}
    assert list(rows) == LOGREG["metadata"]["factorOrder"]
    for item in LOGREG["metadata"]["factors"]:
        cells = rows[item["factor"]]
        assert float(cells[2]) == item["maxPoints"], item["factor"]
        assert round_factor(float(cells[1])) == item["maxPoints"], item["factor"]


def test_hard_constraints_required_fields_and_transport_factors_match_schema() -> None:
    constraints = section(5)
    for constraint in SCHEMA["hard_constraints"]:
        assert f"| `{constraint['id']}` | `{constraint['excludes_when']}` |" in constraints
    required = LOGREG["metadata"]["required_fields"]
    assert required == ["budget.max", "livingWith", "placeType"]
    assert "en el orden " + ", ".join(f"`{name}`" for name in required) in constraints
    factors = SCHEMA["pair_features"][1]["parameters"]["transport_factors"]
    cited = ", ".join(f"`{mode}` {factor}" for mode, factor in factors.items())
    assert cited in section(6)
    assert "6371 km" in section(6) and "6371 km" in SCHEMA["pair_features"][0]["formula"]


def test_rounding_rule_is_away_from_zero_with_round6() -> None:
    rounding = section(8)
    assert "floor(round6(raw_total) + 0.5)" in rounding
    assert LOGREG["metadata"]["rounding"]["matchScore"] == "floor(round_half_up(raw_total, 6) + 0.5)"
    assert "MidpointRounding.AwayFromZero" in LOGREG["metadata"]["rounding"]["mode"]
    assert "private const MidpointRounding Mode = MidpointRounding.AwayFromZero;" in rounding
    for line in GUIDE.splitlines():
        if "Math.Round(" in line:
            assert any(token in line for token in ("AwayFromZero", "Mode", "ToEven")), line


def test_to_even_counterexamples_cited_match_golden_cases() -> None:
    budget = case("factor_round_budget_4_65")["expected"]["result"]["breakdown"][0]
    assert budget["factor"] == "budget" and budget["points"] == 4.7
    assert case("logreg_total_x5000004_up")["expected"]["result"]["matchScore"] == 81
    rounding = section(8)
    assert "| `factor_round_budget_4_65` | presupuesto 4.64999994 → round6 4.65 | **4.7** | 4.6 |" in rounding
    assert "| `logreg_total_x5000004_up` | total 80.5000004 → round6 80.5 | **81** | 80 |" in rounding


def test_negative_zero_rule_and_case_are_cited() -> None:
    assert "normalizar -0 a 0" in LOGREG["metadata"]["rounding"]["negative_zero"]
    assert "### 8.2 Normalizar `-0` a `0`" in GUIDE
    assert "return r == 0.0 ? 0.0 : r;" in GUIDE
    assert case("factor_minimum_place_type")["expected"]["result"]["breakdown"][3]["points"] == 0.0
    assert "`factor_minimum_place_type`" in section(8)


def test_example_json_is_the_perfect_100_golden_case() -> None:
    match = re.search(r"```json\n(.*?)```", section(9), re.S)
    assert match is not None
    assert json.loads(match.group(1)) == case("perfect_100")["expected"]["result"]


def test_golden_case_counts_and_tolerances_match_fixtures() -> None:
    cases = LOGREG["metadata"]
    golden = section(11)
    assert len(LOGREG["cases"]) == cases["case_count"] == 109
    assert f"{cases['case_count']} casos" in golden
    for category, count in cases["category_counts"].items():
        assert f"`{category}` {count}" in golden, category

    def kind(item: dict[str, Any]) -> str:
        result = item["expected"]["result"]
        if result.get("excluded"):
            return "excluded"
        if result.get("incomplete"):
            return "incomplete"
        return "vector" if "features" in item["input"] else "scored"

    logreg_kinds = [kind(item) for item in LOGREG["cases"]]
    for label, name in (("scored", "Entradas crudas"), ("vector", "Solo vector"), ("excluded", "Excluidos")):
        assert re.search(rf"^\| {name}[^|]*\| {logreg_kinds.count(label)} \|", golden, re.M), label
    assert re.search(rf"^\| Perfil incompleto \| {logreg_kinds.count('incomplete')} \|", golden, re.M)

    baseline_kinds = [kind(item) for item in BASELINE["cases"]]
    baseline_text = (
        f"{len(BASELINE['cases'])} casos ({baseline_kinds.count('scored')} puntuados, "
        f"{baseline_kinds.count('excluded')} excluidos, {baseline_kinds.count('incomplete')} incompletos)"
    )
    assert baseline_text in golden

    tolerance = cases["tolerance"]

    def sci(value: float) -> str:  # 1e-06 → 1e-6, como en la guía
        return f"{value:.0e}".replace("e-0", "e-")

    assert f"| `expected.features` y `breakdown[].value` | {sci(tolerance['features'])} (absoluta) |" in golden
    assert f"| `expected.onnx` (`factor_points`, `raw_total` sin redondear) | {sci(tolerance['onnx'])} |" in golden
    assert tolerance["points"] == tolerance["matchScore"] == 0


def test_t05_h1_cases_cited_exist_in_golden_cases() -> None:
    cited = re.findall(r"`(t05_h1_rent_\d+)`", section(12))
    zone = [item["id"] for item in LOGREG["cases"] if item["category"] == "t05_h1_zone"]
    assert cited == zone
    assert "76.2" in section(12) and "+0.124 puntos" in section(12)


def test_release_files_and_performance_criterion_are_cited() -> None:
    files = section(3)
    for name in (*CHECKSUMMED_FILES, SHA256SUMS):
        assert f"| `{name}` |" in files, name
    assert f"releases/tag/{RELEASE_TAG}" in GUIDE and f"TAG={RELEASE_TAG}" in GUIDE
    assert "**< 20 ms** de mediana" in section(13)


def test_baseline_params_pinned_to_release_commit() -> None:
    url = f"https://raw.githubusercontent.com/thehouse-team/TheHouse-ML/{RELEASE_COMMIT}/{BASELINE_PARAMS}"
    assert url in GUIDE
    pinned = re.search(r"SHA-256 de `baseline_params.json` en ese commit \| `([0-9a-f]{64})` \|", GUIDE)
    assert pinned is not None
    git = shutil.which("git")
    if git is None:
        pytest.skip("git no está disponible para leer el commit del release")
    blob = subprocess.run(
        [git, "show", f"{RELEASE_COMMIT}:{BASELINE_PARAMS}"], cwd=REPO, capture_output=True, check=False
    )
    if blob.returncode != 0:
        pytest.skip(f"el commit del release {RELEASE_COMMIT[:7]} no está en este clon (clon superficial)")
    assert hashlib.sha256(blob.stdout).hexdigest() == pinned.group(1)
    assert f"{pinned.group(1)}  baseline_params.json" in GUIDE


# 2. Con el paquete del release (fuera de git).


@pytest.fixture(scope="module")
def release_model() -> onnx.ModelProto:
    path = PACKAGE_DIR / MODEL_ONNX
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} ausente (fuera de git; se genera con `export`, ML-06-RK05)")
    model = onnx.load(str(path))
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    if metadata.get("gitCommit") != RELEASE_COMMIT or metadata.get("profile") != "full":
        pytest.skip(f"el paquete local no es el del release {RELEASE_TAG} (gitCommit {metadata.get('gitCommit')})")
    return model


def _shape(value: onnx.ValueInfoProto) -> list[str | int]:
    return [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]


def test_release_model_inputs_and_outputs_match_guide(release_model: onnx.ModelProto) -> None:
    graph = release_model.graph
    assert [value.name for value in graph.input] == [INPUT_NAME]
    assert [value.name for value in graph.output] == [FACTOR_OUTPUT, TOTAL_OUTPUT]
    for value in (*graph.input, *graph.output):
        assert value.type.tensor_type.elem_type == TensorProto.DOUBLE, value.name
    assert _shape(graph.input[0]) == ["N", 10]
    assert _shape(graph.output[0]) == ["N", 6]
    assert _shape(graph.output[1]) == ["N"]
    assert release_model.ir_version == IR_VERSION
    assert [(item.domain, item.version) for item in release_model.opset_import] == [("", OPSET)]


def test_release_metadata_cited_in_guide(release_model: onnx.ModelProto) -> None:
    metadata = {prop.key: prop.value for prop in release_model.metadata_props}
    assert set(metadata) == set(METADATA_KEYS)
    loading = section(4)
    for key in METADATA_KEYS:
        assert f"| `{key}` |" in loading, key
        if key != "roundingRule":
            assert f"`{metadata[key]}`" in loading, key
    rule = json.loads(metadata["roundingRule"])
    assert rule["mode"] == "half_up" and "MidpointRounding.AwayFromZero" in rule["dotnet"]
    assert json.loads(metadata["featureOrder"]) == SCHEMA["feature_order"]
    assert json.loads(metadata["factorOrder"]) == LOGREG["metadata"]["factorOrder"]

    max_points_raw = json.loads(metadata["maxPointsRaw"])
    rows = [cells for cells in table_rows(section(8), 3) if cells[0].startswith("`")]
    assert [cells[0].strip("`") for cells in rows] == json.loads(metadata["factorOrder"])
    assert [float(cells[1]) for cells in rows] == max_points_raw
    assert [round_factor(value) for value in max_points_raw] == [float(cells[2]) for cells in rows]


def test_release_checksums_cited_and_model_hash_pinned(release_model: onnx.ModelProto) -> None:
    del release_model  # solo para omitir sin el paquete del release
    lines = (PACKAGE_DIR / SHA256SUMS).read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in lines] == list(CHECKSUMMED_FILES)
    for line in lines:
        assert line in section(3), line
    model_bytes = (PACKAGE_DIR / MODEL_ONNX).read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    assert lines[0] == f"{digest}  {MODEL_ONNX}"
    assert f'public const string ExpectedModelSha256 =\n        "{digest}";' in GUIDE
    assert f"| SHA-256 de `model.onnx` | `{digest}` |" in GUIDE
    assert f"de {len(model_bytes):,} bytes" in GUIDE
