# ML-01 — Base del proyecto — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-01 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-10 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-10 («Apruebo y EJECUTA»: ML-01 1.0.0 con licencia MIT; tasks T01–T05; rutas `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `src/`, `tests/`, `.github/workflows/`, `README.md`, `INSTALACION.txt` y `specs/ml-01-foundation/`; ruleset de `main`) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01–T05 autorizadas** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero el entorno y la CLI; después calidad y pruebas; al final CI y ruleset (con permiso para crear el repositorio y su configuración).

## F1 — Diseño y contrato

- **T01 · Entorno reproducible** (R01): `uv sync --frozen` instala el entorno desde cero en Windows y Linux; `python -m thehouse_ml --help` lista las cinco etapas.

## F2 — Implementación

- **T02 · Calidad automática** (R02): ruff, mypy (modo estricto en `src`) y pytest con cobertura corren localmente y en CI sin errores.

## F3 — Calidad, seguridad y datos

- **T03 · Exclusión de datos y modelos** (R03): `.gitignore` excluye `data/raw`, `data/processed` y `models`; una prueba de CI falla si alguno aparece en el árbol de git.
- **T04 · CI y ruleset** (R04): Workflow con los checks de la §6 en verde en un PR; ruleset de `main` activo con checks obligatorios y una aprobación.

## F4 — Evidencia

- **T05 · Pruebas y evidencia** (R05): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
