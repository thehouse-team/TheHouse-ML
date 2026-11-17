# ML-02 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-02 1.0.1 |
| Tareas ejecutadas | T02 · Descarga verificable (R02); **T03 · Bases nacionales ponderadas (R03)** |
| Autorización T02 | «EJECUTA ML-02 1.0.1, task T02, en la rama feat/ml-02-descarga» — César Yael López Trujillo, 2026-11-17 |
| Autorización T03 | «Apruebo M-ML-014 y EJECUTA ML-02 1.0.1, task T03, en la rama feat/ml-02-bases» — César Yael López Trujillo, 2026-11-17 |
| Rama | `feat/ml-02-bases` |
| Commit | Ninguno: el responsable pidió explícitamente no hacer commit |
| Fecha de ejecución T03 | 2026-11-17 |

## T03 · Bases nacionales ponderadas

### Qué se hizo

1. `data/sources.toml`: `inegi_inpc` con `download_url` verificado (33157 bytes, SHA-256 `dc1aeb86…24af`, tabla `conjunto_de_datos/conjunto_de_datos_inpc_mensual.csv`, UTF-8), `variables_used = ["FECHA", "CONCEPTO", "VALOR"]` y `variables_status = "confirmed"`. Se agregaron `P4_3` a ENVI y `tipo_viv` a ENIGH.
2. `memoria.md`: decisión M-ML-014 (pesos de julio 2024, deflactor del INPC general, alcance de la T03 frente a la T04).
3. `src/thehouse_ml/`: `percentiles.py` (percentil ponderado documentado), `inpc.py` (índice general y deflactores), `csvio.py` (CSV dentro de ZIP sin perder ceros a la izquierda) y `national.py` (normalización ENVI/ENIGH, niveles, Parquet y reporte). `prepare` quedó conectado a la CLI y `PrepareError` se reporta con código 1.
4. `pyproject.toml` / `uv.lock`: `uv add pandas pyarrow` (únicas dependencias directas nuevas, permitidas por la constitution §5).
5. Pruebas sin red: `tests/unit/test_percentiles.py`, `test_inpc.py`, `test_national.py` y `tests/integration/test_prepare_stage.py` con ZIPs inventados dentro de `tmp_path`.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv add pandas pyarrow` | `+ pandas==3.1.0`, `+ pyarrow==26.0.0` (transitivas: numpy 2.5.4, python-dateutil, six, tzdata) |
| `uv run python -m thehouse_ml download --source inegi_inpc` | `inegi_inpc: descargado - 1 archivo(s) verificados` |
| `uv run python -m thehouse_ml prepare` | `prepare: .../data/processed/national_baselines.parquet (87 filas) - .../reports/national_baselines.md` |
| `uv run ruff format --check .` | 67 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 13 source files |
| `uv run pytest --cov=thehouse_ml` | **110 passed**; cobertura total **89.26 %** (mínimo 80 %) |

### Evidencia de datos reales

`data/processed/national_baselines.parquet` (fuera de git) tiene 87 filas: 44 de ENVI 2020 y 43 de ENIGH 2022. Niveles: 2 nacional, 64 entidad, 8 tamaño de localidad y 13 clase/tipo de vivienda. Ningún grupo quedó sin fila por falta de renta válida o de peso.

Índice general INPC 2024-07 = 136.003. Promedio oct-dic 2020 = 108.967 → deflactor ENVI = 1.248112. Promedio ago-nov 2022 = 124.912 → deflactor ENIGH = 1.088793.

Medianas de la renta mensual (pondéradas con factor de expansión):

| Fuente | Ámbito | p50 en pesos del año | p50 en pesos de jul-2024 | % rentadas | n_muestra | Σ factores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ENVI 2020 | Nacional | 2,000 | 2,496.224 | 16.4 % | 8,170 | 5,619,414 |
| ENVI 2020 | Hidalgo (13) | 1,500 | 1,872.168 | 9.9 % | 164 | 81,275 |
| ENVI 2020 | Estado de México (15) | 2,000 | 2,496.224 | 16.3 % | 271 | 739,380 |
| ENIGH 2022 | Nacional | 2,000 | 2,177.585 | 15.1 % | 11,437 | 5,566,173 |
| ENIGH 2022 | Hidalgo (13) | 2,000 | 2,177.585 | 9.9 % | 213 | 88,536 |
| ENIGH 2022 | Estado de México (15) | 2,000 | 2,177.585 | 14.7 % | 387 | 706,980 |

La tabla completa (p10, p25, p50, p75 y p90 en ambos pesos) está en `reports/national_baselines.md`.

Definición aplicada: ordenar por renta, agrupar empatados y tomar el menor valor cuya suma acumulada de factores alcanza `q * (suma total de factores)`. `n_muestra` y `Σ factores` corresponden a las viviendas rentadas con renta válida; la proporción de rentadas usa como denominador todas las viviendas del grupo, también ponderadas.

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R03 · percentil contra cálculo a mano | `tests/unit/test_percentiles.py::test_weighted_quantile_matches_hand_calculation`, `test_tied_values_are_grouped_before_accumulating` |
| R03 · orden p10 ≤ … ≤ p90 | `test_weighted_percentiles_are_ordered`, `tests/integration/test_prepare_stage.py::test_percentiles_are_ordered_and_shares_are_in_range` |
| R03 · exclusión de códigos especiales | `tests/unit/test_national.py::test_envi_excludes_special_rent_codes_and_blanks`, `test_envi_accepts_the_upper_bound_of_the_valid_range`, `test_enigh_uses_only_positive_rents_of_renters` |
| R03 · proporciones en [0, 1] | `test_rented_share_is_weighted_and_between_zero_and_one`, `test_groups_without_valid_rent_are_skipped` |
| R03 · deflactor con INPC inventado | `tests/unit/test_inpc.py::test_deflactor_uses_period_average_and_reference_month`, `test_deflactor_reports_missing_months` |
| R03 · fila del índice general | `test_only_the_general_concept_row_is_used`, `test_inpc_registers_the_verified_general_index_file` |
| R03 · esquema del Parquet | `tests/integration/test_prepare_stage.py::test_prepare_writes_parquet_with_documented_schema`, `test_national_and_state_medians_use_the_invented_data` |
| R03 · etapa y CLI | `test_cli_prepare_returns_zero`, `test_cli_prepare_fails_clearly_when_a_file_is_missing`, `test_run_stage_dispatches_prepare`, `test_prepare_report_has_reference_month_note_and_summary_rows` |
| R01/R02 · registro ampliado | `tests/unit/test_sources_loader.py` (14 pruebas, incluye INPC y las variables nuevas) |

## T02 · Descarga verificable (registro histórico)

- `data/sources.toml` quedó con `download_url`, nombre, tamaño, SHA-256, tipo, tabla y codificación por archivo; ENVI y ENIGH en `confirmed`, `P5_8` excluida.
- `download` verifica el hash y el encabezado de cada CSV dentro del ZIP, escribe manifiestos y `reports/variables_check.md`.
- Manifiestos verificados el 2026-11-17: `envi_2020_base_de_datos_csv.zip` (7,322,530 B, `36b8e3d0…40b`), `envi_2020_fd.xlsx` (242,100 B, `208fc3d2…fbcf`), `enigh2022_ns_viviendas_csv.zip` (1,712,569 B, `6ebd189f…5bd4`). El de INPC se agregó en T03 (`data/manifests/inegi_inpc.json`).
- En T02: 86 pruebas, cobertura 91.30 %, ruff y mypy sin errores.

## Hallazgos y desviaciones

1. **Servicios en T04, no en T03.** La redacción de R03 en la spec menciona «servicios ponderados», pero el responsable acotó la T03 a renta y proporción de viviendas rentadas y asignó la prevalencia de servicios por municipio al Censo 2020 de la T04 (M-ML-014(c) y autorización de ejecución). No se reporta como PASS una salida de servicios que no se generó.
2. **pandas sin stubs de tipo.** `pandas` no publica `py.typed` y `pandas-stubs` no está en la tabla de la constitution §5, así que no se agregó. Se configuró en `pyproject.toml` un override de mypy `ignore_missing_imports` **solo para `pandas`/`pandas.*`**; el resto del código sigue en modo estricto.
3. **INPC: valor semántico.** El diccionario del ZIP describe `VALOR` como «variación porcentual anual», pero el responsable verificó que son niveles del índice base 2Q jul 2018 = 100 (2024-07 = 136.003). Se documentó en `data/sources.toml`, el reporte y el código; se filtra solo el concepto general.
4. **Definición de `n_muestra`.** Se adoptó como el conteo sin ponderar de viviendas rentadas con renta válida (la muestra que produce los percentiles); la proporción de rentadas usa todas las viviendas del grupo. Queda explicado en el reporte.
5. **Sin commit ni PR** por instrucción explícita del responsable. La aprobación de otro humano en GitHub sigue pendiente.
6. Censo 2020, DENUE, Inside Airbnb, OpenStreetMap y seed siguen sin descarga: fuera de esta autorización (T04–T09).

## Revisión

- **Constitución:** §5 (solo pandas y pyarrow, añadidos como dependencias directas), §6 y §7 (Parquet fuera de git; INEGI citado como fuente sin sugerir respaldo), §7 (SHA-256 verificado antes de usar el ZIP; sin datos personales en las salidas), §9 (misma entrada → mismo Parquet), §16 (pruebas sin red, cobertura 89.26 %).
- **AGENTS / memoria:** «EJECUTA» vigente de M-ML-011 y M-ML-014; fuentes aprobadas en M-ML-003; decisiones M-ML-010, M-ML-012, M-ML-013 aplicadas; M-ML-014 registrada con aprobación del responsable.
- **Seguridad y licencias:** sin secretos, sin scraping, sin servicios de pago; solo productos públicos de INEGI citados como autor; ningún dato crudo o procesado versionado (verificado por `tests/test_repo_hygiene.py`).

## Resultado

**PASS WITH WARNINGS.** T03 cumple su criterio de aceptación con evidencia reproducible. Advertencias: los servicios quedan para T04 por decisión del responsable (hallazgo 1), el override de mypy para pandas (hallazgo 2) y la ausencia de commit/PR por instrucción (hallazgo 5).
