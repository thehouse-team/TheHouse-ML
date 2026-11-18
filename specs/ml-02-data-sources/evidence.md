# ML-02 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-02 1.0.1 |
| Tareas ejecutadas | T02 · Descarga verificable (R02); T03 · Bases nacionales ponderadas (R03); **T04 · Calibración regional con respaldo (R04)** |
| Autorización T02 | «EJECUTA ML-02 1.0.1, task T02, en la rama feat/ml-02-descarga» — César Yael López Trujillo, 2026-11-17 |
| Autorización T03 | «Apruebo M-ML-014 y EJECUTA ML-02 1.0.1, task T03, en la rama feat/ml-02-bases» — César Yael López Trujillo, 2026-11-17 |
| Autorización T04 | «Apruebo M-ML-015 y EJECUTA ML-02 1.0.1, task T04, en la rama feat/ml-02-regional» — César Yael López Trujillo, 2026-11-17 |
| Rama | `feat/ml-02-regional` |
| Commit | Ninguno: el responsable pidió explícitamente no hacer commit |
| Fecha de ejecución T03 | 2026-11-17 |
| Fecha de ejecución T04 | 2026-11-17 |

## T04 · Calibración regional con respaldo

### Qué se hizo

1. `data/sources.toml`: `inegi_censo_2020_viviendas_ca` pasó de `download_url = []` y `variables_status = "pending"` a cuatro archivos verificados (dos ZIP de datos para Hidalgo 13 y Estado de México 15, tablas `Viviendas13.CSV` y `Viviendas15.CSV`, latin-1; el diccionario ampliado y el catálogo de clasificaciones, ambos `kind = "dictionary"`), `variables_status = "confirmed"` y `dictionary_url` apuntando al diccionario oficial. Se retiró `AGUA_ENT` (inexistente) y se agregaron `AGUA_ENTUBADA`, `ESTRATO` y `UPM`.
2. `memoria.md`: decisión M-ML-015 (APROBADA) con los cuatro archivos y hashes, los códigos confirmados con el diccionario oficial, las claves de los 7 municipios y el origen estatal de la renta.
3. `src/thehouse_ml/regional.py` (nuevo): catálogo de los 7 municipios y de los 7 indicadores con sus códigos afirmativos; proporción ponderada; varianza por linealización de Taylor para razón con UPM con reemplazo dentro del estrato; regla de suficiencia municipio → estado → nacional; copia de los percentiles de renta desde `national_baselines.parquet`; Parquet y reporte. `prepare` quedó conectado a la CLI y `RegionalError` se reporta con código 1.
4. Pruebas sin red: `tests/unit/test_regional.py` (14 pruebas), `tests/integration/test_regional_stage.py` (4 pruebas) y actualización de las fixtures de `test_prepare_stage.py`, `test_sources_loader.py` y `test_sources_registry.py`.

### Fórmula documentada

Con pesos `w`, validez `x` (1 si el código no es 9/blanco) y respuesta afirmativa `y`:

`p = Σ(wᵢ yᵢ) / Σ(wᵢ xᵢ)`; `lᵢ = wᵢ (yᵢ − p xᵢ) / Σ(wⱼ xⱼ)`; `L_hj = Σ lᵢ` en el estrato `h` y la UPM `j`;

`Var(p) = Σ_h [ n_h / (n_h − 1) · Σ_j (L_hj − media(L_h*))² ]`; `ee = √Var(p)`; `cv = ee / p`.

Los estratos con una sola UPM no aportan varianza y se cuentan en `n_strata_single_psu`.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run python -m thehouse_ml download --source inegi_censo_2020_viviendas_ca` | `inegi_censo_2020_viviendas_ca: descargado - 4 archivo(s) verificados` |
| `uv run python -m thehouse_ml download` | 4 fuentes verificadas (Censo, ENVI, ENIGH, INPC) y 4 pendientes sin `download_url`; `variables_check` sin fallas |
| `uv run python -m thehouse_ml prepare` | `national_baselines.parquet (87 filas)` y `regional_calibration.parquet (56 filas)` |
| `uv run ruff format --check .` | 71 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 14 source files |
| `uv run pytest --cov=thehouse_ml` | **129 passed**; cobertura total **89.79 %** (mínimo 80 %), `regional.py` 91 % |

### Evidencia de datos reales

`data/processed/regional_calibration.parquet` (fuera de git) tiene 56 filas: 7 municipios × 7 indicadores del Censo más 7 filas de renta. Los 49 indicadores salieron del **municipio** (todos tienen entre 94 y 385 viviendas rentadas en la muestra y CV entre 0.1 % y 21.8 %, por debajo del 30 %); ninguna fila necesitó respaldo estatal o nacional. Las 7 filas de renta salieron del **estado** (`level = estado`, `source = ENVI`).

Tabla de los 7 municipios (porcentaje ponderado con `FACTOR`, nivel entre paréntesis; renta mediana en pesos de julio-2024 de ENVI estatal):

| Clave | Municipio | Rentadas | Internet | Electricidad | Agua entubada | Agua dentro | Drenaje | Gas | Renta p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13013 | Atotonilco de Tula | 11.6 % (mun) | 44.7 % (mun) | 99.9 % (mun) | 98.2 % (mun) | 81.6 % (mun) | 99.2 % (mun) | 94.8 % (mun) | $1,872 (edo) |
| 13041 | Mixquiahuala de Juárez | 8.6 % (mun) | 43.0 % (mun) | 97.4 % (mun) | 96.4 % (mun) | 76.5 % (mun) | 95.8 % (mun) | 91.2 % (mun) | $1,872 (edo) |
| 13063 | Tepeji del Río de Ocampo | 12.5 % (mun) | 37.6 % (mun) | 99.5 % (mun) | 98.8 % (mun) | 60.6 % (mun) | 97.8 % (mun) | 92.6 % (mun) | $1,872 (edo) |
| 13070 | Tlahuelilpan | 9.8 % (mun) | 42.6 % (mun) | 99.5 % (mun) | 99.5 % (mun) | 80.1 % (mun) | 99.4 % (mun) | 96.6 % (mun) | $1,872 (edo) |
| 13076 | Tula de Allende | 11.1 % (mun) | 46.9 % (mun) | 99.3 % (mun) | 98.6 % (mun) | 80.7 % (mun) | 98.5 % (mun) | 95.5 % (mun) | $1,872 (edo) |
| 15045 | Jilotepec | 7.8 % (mun) | 35.1 % (mun) | 97.8 % (mun) | 93.0 % (mun) | 65.5 % (mun) | 90.9 % (mun) | 90.2 % (mun) | $2,496 (edo) |
| 15112 | Villa del Carbón | 5.0 % (mun) | 18.8 % (mun) | 97.5 % (mun) | 97.7 % (mun) | 41.1 % (mun) | 87.6 % (mun) | 72.7 % (mun) | $2,496 (edo) |

La tabla completa con `n`, `Σ w`, viviendas rentadas, `ee` y `cv` por indicador está en `reports/regional_calibration.md`.

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R04 · proporción ponderada a mano | `tests/unit/test_regional.py::test_weighted_proportion_matches_hand_calculation`, `tests/integration/test_regional_stage.py::test_weights_of_valid_dwellings_are_the_denominator` |
| R04 · exclusión de 9/blanco | `test_special_codes_are_excluded_from_the_denominator`, `normalize_code` en `test_codes_with_leading_zeros_are_normalized` |
| R04 · suficiencia (< 30 rentadas → estado) | `test_municipality_with_fewer_than_30_renters_falls_back_to_state` |
| R04 · suficiencia (CV > 30 % → estado) | `test_municipality_with_high_cv_falls_back_to_state`, `test_sufficient_municipality_is_kept`, `test_nacional_is_the_last_resort` |
| R04 · CV contra caso a mano | `test_cv_matches_hand_calculation` |
| R04 · claves de municipio | `test_municipality_key_070_is_tlahuelilpan`, `test_all_seven_municipalities_are_registered` |
| R04 · códigos del diccionario | `test_indicator_codes_come_from_the_official_dictionary` |
| R04 · 7 municipios presentes | `tests/integration/test_regional_stage.py::test_run_regional_writes_schema_and_all_seven_municipalities` |
| R04 · renta copiada de ENVI estatal | `test_rent_rows_copy_state_envi_percentiles` |
| R04 · esquema y etapa | `test_run_regional_writes_schema_and_all_seven_municipalities`, `tests/integration/test_prepare_stage.py` (7 pruebas con Censo inventado) |
| R01/R02 · registro del Censo | `tests/unit/test_sources_loader.py::test_censo_registers_four_verified_files_and_confirmed_variables`, `tests/unit/test_sources_registry.py::test_censo_dictionary_url_points_to_the_official_amplified_dictionary` |

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

1. **Servicios en T04, no en T03 (resuelto).** La redacción de R03 en la spec menciona «servicios ponderados», pero el responsable acotó la T03 a renta y proporción de viviendas rentadas y asignó la prevalencia de servicios por municipio al Censo 2020 de la T04 (M-ML-014(c)). La T04 generó esos siete indicadores para los 7 municipios; no se reportó como PASS una salida de servicios de T03.
2. **pandas sin stubs de tipo.** `pandas` no publica `py.typed` y `pandas-stubs` no está en la tabla de la constitution §5, así que no se agregó. Se configuró en `pyproject.toml` un override de mypy `ignore_missing_imports` **solo para `pandas`/`pandas.*`**; el resto del código sigue en modo estricto.
3. **INPC: valor semántico.** El diccionario del ZIP describe `VALOR` como «variación porcentual anual», pero el responsable verificó que son niveles del índice base 2Q jul 2018 = 100 (2024-07 = 136.003). Se documentó en `data/sources.toml`, el reporte y el código; se filtra solo el concepto general.
4. **Definición de `n_muestra`.** Se adoptó como el conteo sin ponderar de viviendas rentadas con renta válida (la muestra que produce los percentiles); la proporción de rentadas usa todas las viviendas del grupo. Queda explicado en el reporte.
5. **Sin commit ni PR** por instrucción explícita del responsable. La aprobación de otro humano en GitHub sigue pendiente.
6. **Fuentes pendientes.** El Censo 2020 se descargó y verificó en T04; DENUE, Inside Airbnb, OpenStreetMap y el seed siguen sin descarga por estar fuera de esta autorización (T05–T09).
7. **Renta sin `ee`/`cv` y respaldo nacional limitado.** Las filas de renta copian los percentiles de la ENVI 2020 de T03, que no publica varianza; sus columnas `se` y `cv` quedan como `NaN` y se marcan `—` en el reporte. La suficiencia no se aplica a la renta porque nunca es una estimación municipal. El respaldo `nacional` de ENVI/ENIGH de T03 solo está implementado para la proporción de viviendas rentadas; T03 no publica prevalencias de servicios (M-ML-014(c)), por lo que para los seis indicadores de servicios la entidad completa es el último nivel disponible y una entidad insuficiente daría un error explícito en vez de una estimación no documentada. En los datos reales no se presentó ese caso.
8. **Estratos con una sola UPM.** En el diseño con reemplazo no tienen varianza definida; no aportan a la varianza y se cuentan en `n_strata_single_psu`. En los datos reales ningún municipio de los 7 quedó sin estimación por este motivo.
9. **Reescritura de manifiestos y reportes.** Al ejecutar `download` y `prepare` con datos reales se reescribieron los manifiestos y `reports/national_baselines.md` (solo diferencias de fin de línea CRLF/LF en Windows; el contenido es idéntico) y se creó `data/manifests/inegi_censo_2020_viviendas_ca.json`.

## Revisión

- **Constitución:** §5 (solo pandas y pyarrow, añadidos como dependencias directas; T04 sin dependencias nuevas), §6 y §7 (Parquet fuera de git; INEGI citado como fuente sin sugerir respaldo), §7 (SHA-256 de los cuatro archivos del Censo verificado antes de usarlos; sin datos personales en las salidas), §9 (misma entrada → mismo Parquet), §16 (pruebas sin red, cobertura 89.79 %).
- **AGENTS / memoria:** «EJECUTA» vigente de M-ML-011, M-ML-014 y M-ML-015; fuentes aprobadas en M-ML-003; decisiones M-ML-010, M-ML-012, M-ML-013, M-ML-014 y M-ML-015 aplicadas; la T04 no modifica `constitution.md` ni `AGENTS.md`.
- **Seguridad y licencias:** sin secretos, sin scraping, sin servicios de pago; solo productos públicos de INEGI citados como autor; ningún dato crudo o procesado versionado (verificado por `tests/test_repo_hygiene.py`); el Censo se descargó solo para las dos entidades autorizadas.

## Resultado

**PASS WITH WARNINGS.** T04 cumple su criterio de aceptación con evidencia reproducible: los 7 municipios están presentes, cada indicador registra su nivel de respaldo y la regla de suficiencia está probada. Advertencias: `ee`/`cv` no disponibles en la renta de T03 (hallazgo 7), estratos con una sola UPM (hallazgo 8), reescritura de manifiestos por fin de línea (hallazgo 9), el override de mypy para pandas (hallazgo 2) y la ausencia de commit/PR por instrucción (hallazgo 5). T03 mantiene su PASS WITH WARNINGS previo con el hallazgo 1 resuelto en T04.
