# ML-02 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-02 1.0.1 |
| Task ejecutada | T02 · Descarga verificable (R02) |
| Autorización | «EJECUTA ML-02 1.0.1, task T02, en la rama feat/ml-02-descarga» — César Yael López Trujillo, 2026-11-17 |
| Rama | `feat/ml-02-descarga` |
| Commit | Ninguno: el responsable pidió explícitamente no hacer commit |
| Fecha de ejecución | 2026-11-17 |

## Qué se hizo

1. `data/sources.toml`: campo `download_url` por fuente con URL, nombre, tamaño, SHA-256, tipo, tabla CSV dentro del ZIP y codificación. ENVI y ENIGH quedaron con `variables_status = "confirmed"` y comentario de verificación del 2026-11-17; `P5_8` excluida. Censo, DENUE e INPC quedaron con `download_url = []`.
2. `src/thehouse_ml/`: módulos `sources.py` (registro), `manifests.py` (manifiestos), `download.py` (descarga, modo manual, verificación de hash) y `variables.py` (encabezados de CSV dentro de ZIP y reporte). La CLI acepta `download --source <id>` y devuelve código 1 con mensaje claro si falla un hash o una variable confirmada.
3. `memoria.md`: decisión M-ML-013.
4. Pruebas sin red con fixtures sintéticos en `tests/fixtures/`.

## Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run python -m thehouse_ml download --source inegi_envi_2020` | `inegi_envi_2020: descargado - 2 archivo(s) verificados` |
| `uv run python -m thehouse_ml download --source inegi_enigh_2022` | `inegi_enigh_2022: descargado - 1 archivo(s) verificados` |
| `uv run python -m thehouse_ml download` | 2 fuentes en modo `manual` (sin red, solo verificación) y 6 con `download_url pendiente`; termina con código 0 |
| `uv run ruff format --check .` | 57 archivos ya formateados |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 9 source files |
| `uv run pytest --cov=thehouse_ml` | 86 passed; cobertura total 91.30 % (mínimo 80 %) |

## Evidencia de datos

`data/manifests/inegi_envi_2020.json` y `data/manifests/inegi_enigh_2022.json` registran los tres archivos con fecha 2026-11-17, tamaño y SHA-256. Los valores coinciden con los verificados por el responsable:

| Archivo | Tamaño | SHA-256 |
| --- | --- | --- |
| `envi_2020_base_de_datos_csv.zip` | 7,322,530 | `36b8e3d04f37722974e0d48edb2b4708b4a5e55c9c3faa551b9b75fda95cc40b` |
| `envi_2020_fd.xlsx` | 242,100 | `208fc3d2c70a4cc2b346479813fd98c0481fa5c490d5c8bc2ca8970694befbcf` |
| `enigh2022_ns_viviendas_csv.zip` | 1,712,569 | `6ebd189feb30657dda1ebc95ab8815d13717dac4022d95672ec17ccf56095bd4` |

`reports/variables_check.md` muestra, sobre los archivos oficiales, las 8 variables confirmadas de ENVI y las 11 de ENIGH, sin faltantes. Las fuentes sin `download_url` aparecen como `verificado: no` y no se cuentan como fallo.

## Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R02 · descarga y manifiesto | `tests/unit/test_download.py`, `tests/integration/test_download_stage.py` |
| R02 · hash distinto falla | `test_wrong_sha256_in_registry_fails_clearly`, `test_manifest_hash_mismatch_fails_clearly`, `test_cli_fails_when_the_hash_does_not_match` |
| R02 · fuente sin URL se salta | `test_source_with_empty_download_url_is_skipped`, `test_confirmed_source_without_url_is_skipped_not_failed` |
| R02 · modo manual | `test_manual_mode_uses_local_file_without_fetching`, `test_second_run_uses_manual_mode_without_network` |
| R02 · variable faltante | `tests/unit/test_variables_check.py`, `test_cli_fails_when_a_confirmed_variable_is_missing` |
| R01 · registro ampliado | `tests/unit/test_sources_loader.py` |

## Hallazgos y desviaciones

1. **openpyxl no se agregó.** No está en la tabla de stack de la constitution §5. T02 solo lee encabezados de CSV dentro de ZIP; el XLSX de ENVI se descarga y se verifica por SHA-256, sin leerse. No hizo falta solicitar autorización para agregarlo.
2. **Inside Airbnb está `confirmed` desde T01 pero no tiene `download_url` verificado.** La primera corrida completa marcó sus variables como faltantes; se ajustó el verificador para no fallar fuentes omitidas (`verificado: no`). Debe recibir URL y hash antes de usar sus variables.
3. Censo 2020, DENUE, INPC, Inside Airbnb, OpenStreetMap y seed siguen sin descarga: fuera del alcance de esta autorización o a la espera de URL/hash verificados. El Censo se descargará solo para Hidalgo (13) y Estado de México (15), según M-ML-013.
4. Datos crudos quedaron en `data/raw/` (ignorado por git); solo manifiestos y reporte son versionables. Sin commit ni PR por instrucción del responsable.

## Revisión

- **Constitución:** se respetan §5 (sin dependencias nuevas), §7 (licencia INEGI registrada, sin datos personales en salidas, SHA-256 en manifiesto), §9 (descarga reproducible) y §16/§17 (ruff, mypy, pytest con cobertura).
- **AGENTS / memoria:** autorización EJECUTA vigente (M-ML-011), fuentes aprobadas en M-ML-003, verificación automática de M-ML-012 y variables confirmadas en M-ML-013.
- **Seguridad y licencias:** sin secretos, sin scraping, sin servicios de pago; solo productos públicos de INEGI citados como fuente, sin sugerir respaldo.

## Resultado

**PASS WITH WARNINGS.** T02 cumple su criterio de aceptación para las fuentes con URL verificada. Quedan pendientes los `download_url` de las demás fuentes y la revisión de otro humano en el pull request (no se hizo commit ni PR).
