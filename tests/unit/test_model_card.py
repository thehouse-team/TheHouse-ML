"""ML-05 T06 (unitaria): model card generada desde los JSON versionados de `reports/` (sin datos ni red)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_report import EvaluationError
from thehouse_ml.models.model_card import (
    INPUT_REPORTS,
    MODEL_CARD,
    MODEL_CARD_SKIP_MESSAGE,
    SECTION_TITLES,
    UV_LOCK,
    ModelCardError,
    ModelCardInputs,
    collect_inputs,
    library_versions,
    render_model_card,
    run_model_card,
)
from thehouse_ml.paths import ProjectPaths

ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths(root=ROOT)


@pytest.fixture(scope="module")
def inputs() -> ModelCardInputs:
    return collect_inputs(PATHS)


@pytest.fixture(scope="module")
def card(inputs: ModelCardInputs) -> str:
    return render_model_card(inputs)


def _json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))
    return data


def _section(card: str, index: int) -> str:
    start = card.index(SECTION_TITLES[index])
    end = card.index(SECTION_TITLES[index + 1]) if index + 1 < len(SECTION_TITLES) else len(card)
    return card[start:end]


def _ci(stat: dict[str, float], signed: bool = False) -> str:
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"{fmt.format(stat['mean'])} [{fmt.format(stat['ci_low'])}, {fmt.format(stat['ci_high'])}]"


def test_the_versioned_model_card_is_the_generated_one(card: str) -> None:
    """`reports/model_card.md` coincide byte a byte con lo que generan los JSON versionados."""
    assert (ROOT / "reports" / MODEL_CARD).read_text(encoding="utf-8") == card


def test_has_the_ten_sections_in_order(card: str) -> None:
    headings = [line for line in card.splitlines() if line.startswith("## ")]
    assert tuple(headings) == SECTION_TITLES
    assert len(SECTION_TITLES) == 10


def test_title_and_banner_say_synthetic_data(card: str) -> None:
    title = card.splitlines()[0]
    assert title.startswith("# ") and "DATOS SINTÉTICOS" in title
    assert "logreg-1.0.0" in title
    assert card.splitlines()[2].startswith("> **DATOS SINTÉTICOS")
    for index in (2, 4, 5):
        assert "DATOS SINTÉTICOS" in _section(card, index)


@pytest.mark.parametrize("finding", ["T05-H1", "T05-H2", "T05-H3", "T04-H2", "T04-H3", "TheHouse-GenAI"])
def test_mentions_the_required_findings(card: str, finding: str) -> None:
    assert finding in card


def test_summary_matches_the_promotion_decision(card: str) -> None:
    trained = _json("trained_metrics.json")
    summary = _section(card, 0)
    assert trained["promotion"]["decision"]["outcome"] == "logreg_promoted"
    assert "**PROMOVIDO**" in summary and "M-ML-041" in summary and "2026-11-24" in summary
    assert "`baseline-1.0.0`" in summary and "2.0.0" in summary
    assert _ci(trained["test"]["logreg"]["ndcg@10"]) in summary
    assert _ci(trained["paired_difference_vs_baseline"]["logreg"]["ndcg@10"], signed=True) in summary


def test_intended_use_lists_the_campuses_and_the_exclusions(card: str) -> None:
    section = _section(card, 1)
    campuses = [item for item in _json("ablation_metrics.json")["segments"]["items"] if item["kind"] == "campus"]
    assert f"**{len(campuses)} universidades**" in section
    for item in campuses:
        assert f"`{item['id'].split(':', 1)[1]}`" in section
    for text in ("No decide", "livingWith", "HC01-HC03", "género"):
        assert text in section


def test_data_section_has_split_counts_and_circularity(card: str) -> None:
    section = _section(card, 2)
    summary = _json("train_cv_metrics.json")["data"]["summary"]
    for split, item in summary.items():
        assert f"| {split} | {item['pairs']:,} | {item['students']:,} |" in section
    assert "M-ML-032" in section and "data_card.md" in section and "circularidad" in section and "ML-07" in section


def test_training_section_matches_the_grid_and_hashes(card: str) -> None:
    section = _section(card, 3)
    cv = _json("train_cv_metrics.json")
    for item in [*cv["logreg_grid"], *cv["lightgbm_grid"]]:
        assert f"| `{item['config_id']}` | {item['ndcg@10_mean']:.4f} | {item['ndcg@10_std']:.4f} |" in section
    for entry in cv["final_models"]["files"].values():
        assert entry["sha256"] in section
    assert str(cv["folds"]["seed"]) in section


def test_fold_and_test_metrics_match_the_json(card: str) -> None:
    section = _section(card, 4)
    folds = _json("ablation_metrics.json")["cross_validation"]["metrics"]
    trained = _json("trained_metrics.json")
    labels = {"ndcg@10": "NDCG@10", "ndcg@5": "NDCG@5", "precision@5": "Precision@5", "map": "MAP"}
    for model, block in folds.items():
        for metric, label in labels.items():
            values = " | ".join(f"{fold[metric]:.4f}" for fold in block["by_fold"])
            mean_std = f"{block['mean'][metric]:.4f} ± {block['std'][metric]:.4f}"
            assert f"| {label} | {values} | {mean_std} |" in section, (model, metric)
        test_row = " | ".join(_ci(trained["test"][model][m]) for m in labels)
        assert test_row in section
    for model in ("logreg", "lightgbm"):
        diff_row = " | ".join(_ci(trained["paired_difference_vs_baseline"][model][m], signed=True) for m in labels)
        assert diff_row in section
    gap_test = trained["test"]["lightgbm"]["ndcg@10"]["mean"] - trained["test"]["logreg"]["ndcg@10"]["mean"]
    assert f"{gap_test:+.4f} en test" in section and "condición 3" in section


def test_ablation_segments_and_hypothesis_match_the_json(card: str) -> None:
    section = _section(card, 5)
    ablation = _json("ablation_metrics.json")
    for group in ablation["ablation"]["logreg"]["groups"]:
        assert f"{group['delta_ndcg@10_mean']:+.4f} ± {group['delta_ndcg@10_std']:.4f}" in section
    for item in ablation["segments"]["items"]:
        assert f"| {item['label']} | {item['students']:,} | {_ci(item['baseline']['ndcg@10'])} |" in section
        assert _ci(item["diff_logreg_minus_baseline"]["ndcg@10"], signed=True) in section
    assert "`seed_sin_verificar`" in section
    hyp = ablation["hypothesis_budget_gt_3500"]
    assert hyp["verdict"] == "confirmed" and "CONFIRMADA" in section
    assert _ci(hyp["delta_ndcg@10_by_budget_range"][hyp["segment"]], signed=True) in section
    assert "**T04-H2:**" in section and "**T04-H3:**" in section
    assert f"{hyp['budget_ablation_delta_ndcg@10_mean']:+.4f}" in section


def test_breakdown_matches_the_calibration_and_baseline_weights(card: str) -> None:
    section = _section(card, 6)
    calibration = _json("trained_metrics.json")["breakdown"]["calibration"]
    weights = default_config().weights
    for factor in calibration["factors"]:
        assert (
            f"| {factor['label']} (`{factor['factor']}`) | {factor['maxPoints']:.1f} | {weights[factor['factor']]:g} |"
            in section
        )
    for item in _json("trained_metrics.json")["breakdown"]["monotonicity"]["coefficients"]:
        assert f"| `{item['feature']}` |" in section
        assert f"{item['coef_standardized']:+.6f}" in section
    assert f"S = {calibration['total_span_logit']:.6f}" in section
    for text in ("T05-H2", "no es importancia", "TheHouse-GenAI", "trained_breakdown.md", "cuartos cercanos"):
        assert text in section


def test_limitations_cover_the_required_risks(card: str) -> None:
    section = _section(card, 7)
    monotonicity = _json("trained_metrics.json")["breakdown"]["monotonicity"]
    assert "T05-H1" in section and f"{monotonicity['budget_ratio_threshold']:.1f}" in section
    assert f"{monotonicity['budget']['worst_increase_points']:+.3f}" in section
    for text in (
        "ENVI 2020",
        "ENIGH 2022",
        "estatal",
        "3 universidades sin verificar",
        "S1078",
        "ML-06",
        "NV-23",
        "NV-12",
        "T05-H3",
    ):
        assert text in section, text


def test_future_work_and_traceability(card: str, inputs: ModelCardInputs) -> None:
    future = _section(card, 8)
    for text in ("logreg-1.1.0", "price_below_min", "maxPoints", "pruebas lentas", "LightGBM", "ML-07"):
        assert text in future
    trace = _section(card, 9)
    for name, version in inputs.libraries.items():
        assert f"| `{name}` (`uv.lock`) | {version} |" in trace
    for model in _json("trained_metrics.json")["models"].values():
        assert model["sha256"] in trace
    for pr in range(30, 37):
        assert f"#{pr}" in trace
    assert "| T06 model card (M-ML-042) | #36 | `docs/ml-05-model-card` |" in trace  # T06-H2
    assert "| T07 pruebas, evidencia y cierre (M-ML-043) | pendiente | `feat/ml-05-cierre` |" in trace
    for number in range(36, 44):
        assert f"M-ML-0{number}" in trace


def test_distribution_and_examples_are_read_from_trained_metrics(card: str) -> None:
    """ML-05 T07 (T06-H1): la ficha copia la distribución y los ejemplos de `trained_metrics.json`."""
    section = _section(card, 6)
    breakdown = _json("trained_metrics.json")["breakdown"]
    score = breakdown["distribution"]["matchScore"]
    row = " | ".join([str(score["min"]), *(f"{value:g}" for value in score["percentiles"].values()), str(score["max"])])
    assert f"| {row} | {score['mean']:.1f} |" in section
    assert "| Pares | " + " | ".join(f"{item['count']:,}" for item in score["histogram"]) + " |" in section
    for item in breakdown["distribution"]["factor_points"]:
        assert f"| {item['label']} (`{item['factor']}`) | {item['maxPoints']:g} | {item['mean']:.2f} |" in section
    assert f"mediana de `matchScore` {score['percentiles']['p50']:g}" in section
    for example in breakdown["examples"]:
        assert f"`{example['student_id']}` — {example['university_code']}" in section
        for item in example["top"]:
            assert f"| `{item['room_id']}` | **{item['matchScore']}** |" in section
    first = breakdown["examples"][0]["first_place"]["output"]
    assert json.dumps(first, ensure_ascii=False, indent=2) in section
    assert "no se copian aquí" not in card


def test_a_changed_distribution_or_example_changes_the_card(inputs: ModelCardInputs) -> None:
    trained = copy.deepcopy(dict(inputs.trained))
    trained["breakdown"]["distribution"]["matchScore"]["percentiles"]["p50"] = 42.0
    trained["breakdown"]["examples"][0]["student_id"] = "S9999"
    text = render_model_card(replace(inputs, trained=trained))
    assert "mediana de `matchScore` 42" in text
    assert "`S9999` —" in text


@pytest.mark.parametrize("block", ["distribution", "examples"])
def test_without_distribution_or_examples_the_card_is_an_error(inputs: ModelCardInputs, block: str) -> None:
    trained = copy.deepcopy(dict(inputs.trained))
    del trained["breakdown"][block]
    with pytest.raises(ModelCardError, match=f"breakdown.{block}"):
        render_model_card(replace(inputs, trained=trained))


def test_without_examples_the_card_says_so(inputs: ModelCardInputs) -> None:
    trained = copy.deepcopy(dict(inputs.trained))
    trained["breakdown"]["examples"] = []
    assert "Ningún estudiante cumple la regla." in render_model_card(replace(inputs, trained=trained))


def test_library_versions_come_from_uv_lock() -> None:
    versions = library_versions(ROOT / UV_LOCK)
    lock = (ROOT / UV_LOCK).read_text(encoding="utf-8")
    for name, version in versions.items():
        assert f'name = "{name}"\nversion = "{version}"' in lock


def test_no_figure_is_hardcoded_a_changed_json_value_changes_the_card(inputs: ModelCardInputs) -> None:
    trained = copy.deepcopy(dict(inputs.trained))
    trained["test"]["logreg"]["ndcg@10"] = {"mean": 0.111111, "ci_low": 0.101, "ci_high": 0.121}
    trained["breakdown"]["calibration"]["factors"][1]["maxPoints"] = 55.5
    text = render_model_card(replace(inputs, trained=trained))
    assert "0.111 [0.101, 0.121]" in text
    assert "| 55.5 |" in text and "55.5 % de la decisión" in text


@pytest.mark.parametrize(
    ("document", "key", "value"),
    [
        ("trained", "synthetic", False),
        ("trained", "profile", "ci"),
        ("ablation", "official", False),
        ("cv", "profile", "ci"),
    ],
)
def test_only_the_official_synthetic_run_is_accepted(
    inputs: ModelCardInputs, document: str, key: str, value: object
) -> None:
    changed: dict[str, Any] = dict(getattr(inputs, document))
    changed[key] = value
    overrides: dict[str, Any] = {document: changed}
    with pytest.raises(ModelCardError):
        render_model_card(replace(inputs, **overrides))
    assert issubclass(ModelCardError, EvaluationError)  # la CLI termina con código 1


def test_model_hashes_must_agree_between_reports(inputs: ModelCardInputs) -> None:
    trained = copy.deepcopy(dict(inputs.trained))
    trained["models"]["logreg"]["sha256"] = "0" * 64
    with pytest.raises(ModelCardError, match="SHA-256"):
        render_model_card(replace(inputs, trained=trained))


def _project(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "reports").mkdir()
    for name in INPUT_REPORTS:
        shutil.copyfile(ROOT / "reports" / name, tmp_path / "reports" / name)
    shutil.copyfile(ROOT / UV_LOCK, tmp_path / UV_LOCK)
    return ProjectPaths(root=tmp_path)


def test_two_generations_have_the_same_sha256(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    digests = []
    for _ in range(2):
        message = run_model_card(paths)
        assert message.startswith("model card (ML-05 T06): reports/model_card.md")  # mensaje sin cambios en T07
        message.encode("cp1252")
        digests.append(hashlib.sha256((paths.reports / MODEL_CARD).read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    assert digests[0] == hashlib.sha256((ROOT / "reports" / MODEL_CARD).read_bytes()).hexdigest()


def test_output_is_deterministic_and_has_no_absolute_paths_or_personal_data(card: str) -> None:
    assert str(ROOT) not in card and "C:\\" not in card
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", card)  # sin correos
    assert card.endswith("\n") and not card.endswith("\n\n")


def test_without_inputs_it_is_skipped_and_writes_nothing(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    assert run_model_card(paths) == MODEL_CARD_SKIP_MESSAGE
    assert not (tmp_path / "reports" / MODEL_CARD).exists()
