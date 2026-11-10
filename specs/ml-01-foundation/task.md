# ML-01 — Base del proyecto — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-01 |
| Documento | `task.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-10 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-10 («Apruebo y EJECUTA»: ML-01 1.0.0 con licencia MIT; tasks T01–T05; rutas `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `src/`, `tests/`, `.github/workflows/`, `README.md`, `INSTALACION.txt` y `specs/ml-01-foundation/`; ruleset de `main`) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01–T05 autorizadas** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 5 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F1 — Diseño y contrato

- [ ] **T01 · Entorno reproducible** — Requisito: ML-01-R01 · Aceptación: `uv sync --frozen` instala el entorno desde cero en Windows y Linux; `python -m thehouse_ml --help` lista las cinco etapas.

## F2 — Implementación

- [ ] **T02 · Calidad automática** — Requisito: ML-01-R02 · Aceptación: ruff, mypy (modo estricto en `src`) y pytest con cobertura corren localmente y en CI sin errores.

## F3 — Calidad, seguridad y datos

- [ ] **T03 · Exclusión de datos y modelos** — Requisito: ML-01-R03 · Aceptación: `.gitignore` excluye `data/raw`, `data/processed` y `models`; una prueba de CI falla si alguno aparece en el árbol de git.
- [ ] **T04 · CI y ruleset** — Requisito: ML-01-R04 · Aceptación: Workflow con los checks de la §6 en verde en un PR; ruleset de `main` activo con checks obligatorios y una aprobación.

## F4 — Evidencia

- [ ] **T05 · Pruebas y evidencia** — Requisito: ML-01-R05 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F3 | ABIERTA |
| T04 | R04 | F3 | ABIERTA |
| T05 | R05 | F4 | ABIERTA |
