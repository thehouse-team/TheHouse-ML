# ML-01 — Evidencia

| Campo | Valor |
| --- | --- |
| ID | ML-01 |
| Documento | `evidence.md` |
| Versión de la spec | 1.0.0 |
| Fecha | 2026-11-10 |
| Autorización | «Apruebo y EJECUTA» — Victor Manuel Jose Jose, 2026-11-10: ML-01 1.0.0 con licencia MIT, tasks T01–T05 y ruleset de `main` (M-ML-009) |
| Rama y commit | `feat/ml-01-foundation` (implementación `325f440`) → PR #2 |
| Resultado | **✅ PASS**: 5/5 tasks completas; CI del PR en verde (4/4 checks) y ruleset de `main` activo |

## 1. Cambios

| Área | Archivo | Cambio |
| --- | --- | --- |
| Entorno | `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.12; sin dependencias de ejecución; grupo `dev` con ruff, mypy, pytest, pytest-cov y pip-audit; `exclude-newer = 2026-10-27` para reproducibilidad; backend de build `uv_build` |
| Paquete | `src/thehouse_ml/__init__.py`, `__main__.py`, `py.typed` | Versión desde los metadatos; `python -m thehouse_ml` |
| CLI | `src/thehouse_ml/cli.py` | `argparse` con las etapas `download`, `prepare`, `train`, `evaluate`, `export`; `--root` y `--version` |
| Rutas | `src/thehouse_ml/paths.py` | Raíz por argumento, `THEHOUSE_ML_ROOT` o directorio actual; rutas de la constitution §6 |
| Etapas | `src/thehouse_ml/stages.py` | Catálogo de etapas vacías: informan su spec y salen con código 0 |
| Pruebas | `tests/unit/`, `tests/integration/`, `tests/test_repo_hygiene.py` | 34 pruebas |
| CI | `.github/workflows/ci.yml` | Jobs `quality`, `tests-windows`, `dependency-audit`, `secrets-gitleaks`; acciones fijadas por SHA; uv 0.13.7 |
| Licencia | `LICENSE` | MIT; aclara que no cubre datos de terceros |
| Documentos | `README.md`, `INSTALACION.txt` | Uso rápido y estado actual |
| GitHub | Ruleset `Main Proteccion` (id 26958489) | `main` protegida: PR con 1 aprobación, 4 checks obligatorios y actualizados, sin borrado ni force-push, sin bypass |

## 2. Comandos y resultados

Local (Windows 11, Python 3.12.10, uv 0.13.7, 2026-11-10):

| Comando | Resultado |
| --- | --- |
| `uv lock` / `uv sync --frozen` | 42 paquetes resueltos; entorno creado desde el lock |
| `uv run python -m thehouse_ml --help` | Lista las 5 etapas en orden del pipeline |
| `uv run python -m thehouse_ml download` | `download: pendiente de ML-02`, código 0 |
| `uv run python -m thehouse_ml nope` | Error de uso, código 2 |
| `uv run ruff format --check .` | Sin cambios |
| `uv run ruff check .` | All checks passed! |
| `uv run mypy src` | Success: no issues found in 5 source files |
| `uv run pytest --cov=thehouse_ml` | 34/34; cobertura 100 % (líneas y ramas) |
| `uv run pip-audit --skip-editable` | No known vulnerabilities found |

CI del PR #2 (run 42850072521, commit `325f440`):

| Check | Resultado |
| --- | --- |
| `quality` (Ubuntu) | ✅ formato, lint, mypy, 34/34 pruebas, cobertura 100 % (líneas y ramas), las 5 etapas ejecutadas |
| `tests-windows` | ✅ 34/34 |
| `dependency-audit` | ✅ sin vulnerabilidades conocidas |
| `secrets-gitleaks` | ✅ sin secretos en todo el historial |

## 3. Resultados esperados (spec §6.1)

| Salida | ¿Existe? | Verificación |
| --- | --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Sí | `uv sync --frozen` en Windows y Linux (CI) |
| `.github/workflows/ci.yml` | Sí | 4/4 checks en verde |
| `src/thehouse_ml/` (`__init__`, `__main__`, `cli`, `paths`, `stages`) | Sí | Pruebas de CLI, rutas y etapas |
| `tests/test_cli.py`, `test_paths.py`, `test_repo_hygiene.py` | Sí, con `test_cli` y `test_paths` en `tests/unit/` | 34 pruebas |
| Comandos de uso de la §6.1 | Sí | Salidas de la §2 |

## 4. Trazabilidad requisito → prueba

| Requisito | Pruebas |
| --- | --- |
| R01 Entorno reproducible | `uv sync --frozen` en CI (Linux y Windows); `test_help_lists_the_five_stages_in_pipeline_order`, `test_each_empty_stage_reports_its_spec_and_exits_zero`, `test_python_dash_m_lists_the_stages`, `test_console_script_runs_a_stage`, `test_module_entry_point_runs_main` |
| R02 Calidad automática | Pasos de ruff, mypy y pytest del job `quality`; `fail_under = 80` en `pyproject.toml` |
| R03 Exclusión de datos y modelos | `test_no_data_or_model_files_are_tracked`, `test_gitignore_excludes_data_and_models`, `test_detector_flags_forbidden_paths`, `test_detector_allows_versioned_paths` |
| R04 CI y ruleset | Run 42850072521 en verde; ruleset 26958489 activo |
| R05 Pruebas y evidencia | Cobertura 100 %, ruff y mypy sin errores; este documento |

## 5. Datos y licencias

- ML-01 no descarga ni procesa datos.
- `test_no_data_or_model_files_are_tracked` corre en cada PR.
- Licencia del código: MIT; los datos de terceros conservan la suya.

## 6. Hallazgos y desviaciones

| Hallazgo | Impacto | Decisión |
| --- | --- | --- |
| La flecha «→» en la ayuda lanzaba `UnicodeEncodeError` en consolas de Windows con cp1252 (riesgo ML-01-RK01) | `--help` fallaba en PowerShell/cmd | Se cambió por `->` y se agregó `test_cli_text_is_printable_on_a_windows_cp1252_console`; dentro del alcance de T01 |
| `pip-audit --strict` falla porque el propio paquete está instalado en modo editable | El job fallaría sin haber vulnerabilidades | Se usa `--skip-editable` sin `--strict`; todas las dependencias de terceros se auditan |
| Las pruebas de CLI y rutas quedaron en `tests/unit/` y no en `tests/` como decía la §6.1 | Ninguno; sigue la organización de `CONTRIBUTING.md` | Ajuste de ubicación sin cambio de alcance |
| Se agregaron `tests/unit/test_stages.py` y `tests/integration/test_installed_package.py`, no listadas en la §6.1 | Más cobertura del R01 | Dentro de las rutas autorizadas (`tests/`) |

## 7. Revisión de coherencia y seguridad

- **Constitution:**
  - §5: solo herramientas de la tabla.
  - §6: estructura cumplida.
  - §9: lock y `exclude-newer` para reproducibilidad.
  - §14: acciones fijadas por SHA y pip-audit.
  - §16: cobertura ≥ 80 %.
  - §17: CI con los checks obligatorios.
  - §19: ruleset activo.
- **AGENTS:** solo se tocaron las tasks y rutas autorizadas, más el registro de la aprobación en `memoria.md` y en el índice de specs.
- **Secretos:** el CI no usa secretos y gitleaks revisa todo el historial.
- **Datos personales:** no hay.

## 8. CI y merge

| Paso | Resultado |
| --- | --- |
| CI del PR (run 42850072521) | 4/4 en verde |
| Revisión y merge | Pendiente: aprobación de otra persona (ruleset) y merge en `main` |
