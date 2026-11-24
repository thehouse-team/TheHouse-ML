"""ML-03 T07: reglas del workflow de CI (constitution §14 y §17).

Sin dependencias nuevas: el YAML se revisa como texto.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"
RULESET_JOBS = ("quality", "tests-windows", "dependency-audit", "secrets-gitleaks")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _ci_text() -> str:
    return CI.read_text(encoding="utf-8")


def _job_blocks(text: str) -> dict[str, str]:
    """Bloques de texto de cada job (claves con dos espacios bajo `jobs:`)."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\s*$", body, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    uses = re.findall(r"uses:\s*([^\s#]+)", _ci_text())
    assert uses
    for reference in uses:
        action, _, ref = reference.partition("@")
        assert FULL_SHA.match(ref), f"{action} no está fijada a un SHA completo: {ref!r}"


def test_the_jobs_required_by_the_main_ruleset_keep_their_names() -> None:
    blocks = _job_blocks(_ci_text())
    for job in RULESET_JOBS:
        assert job in blocks, f"Falta el job {job}"
        assert re.search(rf"^\s+name:\s*{re.escape(job)}\s*$", blocks[job], flags=re.MULTILINE)


def test_quality_caches_data_raw_keyed_by_the_sources_and_manifests_before_the_cli() -> None:
    quality = _job_blocks(_ci_text())["quality"]
    assert "uses: actions/cache@" in quality
    assert "path: data/raw/" in quality
    assert "hashFiles('data/sources.toml', 'data/manifests/**')" in quality
    assert quality.index("actions/cache@") < quality.index("for stage in download prepare")


def test_quality_runs_train_with_the_ci_profile_in_the_cli_step() -> None:
    """ML-05 T02 (M-ML-038, T02-H1): el paso de la CLI usa el perfil `ci` para caber en el timeout."""
    quality = _job_blocks(_ci_text())["quality"]
    step = quality[quality.index("- name: CLI con las cinco etapas") :]
    step = step[: step.index("- name:", 1)]
    assert re.search(r"^\s+THEHOUSE_ML_TRAIN_PROFILE:\s*ci\s*$", step, flags=re.MULTILINE)
    assert "for stage in download prepare train evaluate export" in step
    assert re.search(r"^\s+timeout-minutes:\s*15\s*$", quality, flags=re.MULTILINE)
