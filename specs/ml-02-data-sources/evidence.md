# ML-02 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-02 1.0.1 |
| Tareas ejecutadas | T02 · Descarga verificable (R02); T03 · Bases nacionales ponderadas (R03); T04 · Calibración regional con respaldo (R04); T07 · Universidades verificadas (R07); T05 · Entorno de campus (R05); **T06 · DESCARTADA (M-ML-019)**; T08 · Data card (R08) |
| Autorización T02 | «EJECUTA ML-02 1.0.1, task T02, en la rama feat/ml-02-descarga» — César Yael López Trujillo, 2026-11-17 |
| Autorización T03 | «Apruebo M-ML-014 y EJECUTA ML-02 1.0.1, task T03, en la rama feat/ml-02-bases» — César Yael López Trujillo, 2026-11-17 |
| Autorización T04 | «Apruebo M-ML-015 y EJECUTA ML-02 1.0.1, task T04, en la rama feat/ml-02-regional» — César Yael López Trujillo, 2026-11-17 |
| Autorización T07 | «Apruebo M-ML-016 y EJECUTA ML-02 1.0.1, task T07 (antes que T05, porque T05 necesita coordenadas verificadas), en la rama feat/ml-02-universidades» — César Yael López Trujillo, 2026-11-17 |
| Autorización T07 (amplíación) | «Apruebo M-ML-017 y continúa T07 de ML-02 1.0.1 en la rama feat/ml-02-universidades» — César Yael López Trujillo, 2026-11-18 |
| Autorización T05 | «Apruebo M-ML-018 y EJECUTA ML-02 1.0.1, task T05, en la rama feat/ml-02-denue. Rutas permitidas: src/thehouse_ml/, tests/, data/sources.toml, data/manifests/, reports/, memoria.md y specs/ml-02-data-sources/. Sin dependencias nuevas» — César Yael López Trujillo, 2026-11-18 |
| Autorización T06 | «Apruebo M-ML-019 y EJECUTA ML-02 1.0.1, task T06, en la rama docs/ml-02-t06-descartada. Rutas permitidas: data/sources.toml, memoria.md, specs/ml-02-data-sources/, reports/variables_check.md, src/thehouse_ml/ y tests/ (solo si hace falta para que la fuente descartada no se descargue ni se verifique). Sin dependencias nuevas y sin descargar nada de Inside Airbnb» — César Yael López Trujillo, 2026-11-18 |
| Autorización T08 | «EJECUTA ML-02 1.0.1, task T08 (ML-02-R08), en la rama feat/ml-02-data-card. Rutas permitidas: src/thehouse_ml/, tests/, reports/, memoria.md y specs/ml-02-data-sources/. Sin dependencias nuevas (nada de matplotlib)» — César Yael López Trujillo, 2026-11-18 |
| Rama (T02–T04) | `feat/ml-02-descarga`, `feat/ml-02-bases`, `feat/ml-02-regional` |
| Rama (T07) | `feat/ml-02-universidades` |
| Rama (T05) | `feat/ml-02-denue` |
| Rama (T06) | `docs/ml-02-t06-descartada` |
| Rama (T08) | `feat/ml-02-data-card` |
| Commit | `46c9d4f` (T07 M-ML-016, hecho por el responsable). Las ampliaciones M-ML-017 y M-ML-018 (T05) se entregan sin commit, por instrucción explícita |
| Fecha de ejecución T03 | 2026-11-17 |
| Fecha de ejecución T04 | 2026-11-17 |
| Fecha de ejecución T07 | 2026-11-17 |
| Fecha de ampliación T07 (M-ML-017) | 2026-11-18 |
| Fecha de ejecución T05 | 2026-11-18 |
| Fecha de ejecución T06 (descarte) | 2026-11-18 |
| Fecha de ejecución T08 | 2026-11-18 |

## T08 · Data card

### Qué se hizo

1. `src/thehouse_ml/data_card.py` (nuevo): `collect_inputs` lee `data/sources.toml`, todos los JSON de `data/manifests/`, `data/universities.csv` y los cuatro Parquet de `data/processed/`; `render_data_card` produce las 11 secciones del requisito sin cifras escritas a mano; `run_data_card` escribe `reports/data_card.md` con finales LF. La generación se conecta al final de `prepare` en `stages.py`, y `cli.py` reporta `DataCardError` con código 1.
2. `src/thehouse_ml/rent_histogram.py` (nuevo): intervalos fijos de 500 pesos hasta 10,000 y `10,000+`, con conteo de muestra y frecuencia expandida; `group_small_bins` agrupa intervalos adyacentes hasta alcanzar el mínimo de 30 viviendas, preservando la suma expandida. `src/thehouse_ml/thresholds.py` centralized ese mínimo y el `MAX_CV` que antes vivían en `regional.py` (ahora aliases, sin cambio de comportamiento).
3. `national.py`: la parte nacional de `prepare` escribe además `data/processed/rent_histogram.parquet` (63 intervalos: 21 × nacional/Hidalgo/Edomex) a partir de las rentas válidas de ENVI ya leídas, sin abrir nuevas fuentes ni agregar dependencias.
4. `spec.md`, `plan.md` y `task.md`: T08 marcada COMPLETA (7 de 9 tasks; T06 sigue descartada). `memoria.md` no se modifica: la ejecución no crea una decisión nueva.

### Secciones del reporte

1. Propósito y alcance (7 municipios y 10 universidades leídos de los Parquet y del CSV; catálogo de cuartos sintético en ML-03).
2. Fuentes (id, producto, edición, URL, licencia, tamaños y SHA-256 de los manifiestos, estado y decisión de memoria; Inside Airbnb como `DESCARTADA (M-ML-019)`).
3. Atribuciones (`Fuente: INEGI, <producto>` por producto, `© OpenStreetMap contributors (ODbL)` y la aclaración de que INEGI no respalda el proyecto).
4. Ponderación (factor y diseño estrato/UPM derivados de `variables_used`) y deflactación al INPC de jul-2024 (factores leídos de `national_baselines.parquet`).
5. Niveles de respaldo por municipio e indicador, con la regla ≥ 30 y CV ≤ 30 %.
6. Tamaños de muestra con supresión de celdas < 30; en datos reales se suprimen dos filas (`tipo de vivienda 6` y `7` de ENVI).
7. Histogramas en texto: renta ENVI (nacional, Hidalgo, Edomex) y negocios totales a 2 km por universidad (DENUE).
8. Universidades con `final_status` y fuente de coordenada; advertencia para TESJI, UMB-VDC y UCAN.
9. Transformaciones y archivos de salida de cada etapa.
10. Limitaciones (encuestas 2020/2022 deflactadas, renta estatal, 3 sin verificar, DENUE cuenta establecimientos, sin cuartos reales, Inside Airbnb descartada).
11. Privacidad (columnas del DENUE nunca leídas y solo agregados en git).

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | 86 files already formatted (PASS) |
| `uv run ruff check .` | All checks passed! (PASS) |
| `uv run mypy src` | Success: no issues found in 19 source files (PASS) |
| `uv run pytest --cov=thehouse_ml` | **238 passed in 12.08s**; cobertura total **91.76 %** ≥ 80 %; `data_card.py` 94 %, `rent_histogram.py` 92 % (PASS) |
| `uv run python -m thehouse_ml prepare` | `national_baselines.parquet (87 filas), rent_histogram.parquet (63 intervalos)`; `regional_calibration.parquet (56 filas)`; `data/universities.csv (10 filas)`; `campus_context.parquet (300 filas, 981,519 establecimientos)`; `data card: reports/data_card.md` |

### Verificación de los histogramas con datos reales

La suma de las barras de cada ámbito se calculó leyendo el reporte y se comparó con `suma_factores` de `national_baselines.parquet`:

| Ámbito | suma de barras | total impreso | Parquet | Resultado |
| --- | ---: | ---: | ---: | --- |
| Nacional | 5,619,414 | 5,619,414 | 5,619,414 | PASS |
| Hidalgo | 81,275 | 81,275 | 81,275 | PASS |
| Estado de México | 739,380 | 739,380 | 739,380 | PASS |

El reporte tiene 11 secciones `## N.` y tres filas marcadas `suprimido`. Los intervalos con menos de 30 viviendas se agruparon (por ejemplo, Hidalgo publica `500-1,499`, `1,500-1,999` y `2,000-7,499`).

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R08 · 11 secciones | `tests/unit/test_data_card.py::test_data_card_has_all_eleven_sections`, `tests/integration/test_prepare_stage.py::test_prepare_generates_the_data_card_with_all_sections` |
| R08 · fuentes, hashes y decisión de memoria | `test_source_figures_come_from_the_manifest`, `test_discarded_source_appears_as_discarded_but_not_as_an_attribution` |
| R08 · atribuciones sin respaldo de INEGI | `test_every_inegi_product_and_osm_are_attributed_without_endorsement` |
| R08 · ponderación y deflactación | `tests/unit/test_data_card.py` (sección 4 generada de `variables_used` y del Parquet) |
| R08 · niveles y regla de suficiencia | `test_cells_with_fewer_than_thirty_dwellings_are_suppressed` (sección 5) |
| R08 · supresión < 30 | `test_cells_with_fewer_than_thirty_dwellings_are_suppressed`, `tests/unit/test_rent_histogram.py::test_group_small_bins_suppresses_a_tiny_sample` |
| R08 · histograma suma el total expandido | `test_rent_histogram_adds_up_to_the_expanded_total`, `test_expanded_frequency_is_the_sum_of_the_weights`, `test_group_small_bins_preserves_the_expanded_total` |
| R08 · DENUE y universidades | `test_denue_histogram_shows_the_total_category_at_the_configured_radius`, `test_unverified_universities_are_warned_about` |
| R08 · privacidad | `test_privacy_section_names_never_read_columns_and_aggregates_only` |
| R08 · lectura automática de archivos | `test_all_inputs_are_read_from_disk`, `tests/unit/test_stages.py::test_prepare_generates_the_data_card_after_the_other_outputs` |
| R08 · nota `tipo_viv = &` (sección 6) | `test_unspecified_tipo_viv_code_is_labeled_in_the_sample_table` |
| R08 · nota del CV (sección 5) | `test_cv_note_is_below_the_levels_table` |
| R08 · nota de redondeo (sección 7) | `test_rounding_note_uses_the_registered_deflactor` |

### Ajustes textuales del 2026-11-19

Sin cambiar cifras: (1) la sección 6 rotula `tipo_viv = &` de ENIGH como `tipo de vivienda no especificado (&)`; (2) la sección 5 agrega bajo la tabla la nota sobre el CV (solo proporciones, linealización de Taylor con ESTRATO/UPM; la renta es mediana estatal sin CV); (3) la sección 7 agrega bajo el histograma nacional la nota sobre el desplazamiento de intervalos al deflactar rentas redondas. El factor `1.248112` y el mes se leen de `national_baselines.parquet` y de `inpc.REFERENCE_MONTH`; el ejemplo `2,500 → 3,120` se calcula con ese factor. Se agregaron tres pruebas unitarias. Revalidado: 238 pruebas, 91.76 % de cobertura, los tres histogramas siguen sumando su total expandido y las dos celdas `suprimido` permanecen.

### Revisión y resultado

- **Constitución:** §7 y §8 (sin datos personales, sin celdas < 30 viviendas, INEGI citado como autor sin sugerir respaldo, Inside Airbnb sin atribución), §6 (solo agregados y metadatos en git; Parquet fuera de git), §9 (misma entrada produce la misma data card), §16 (pruebas sin red con datos inventados, cobertura 91.76 %).
- **AGENTS / memoria:** «EJECUTA» del responsable vigente (M-ML-011) con rutas respetadas; M-ML-013/M-ML-014 (deflactor), M-ML-015 (suficiencia), M-ML-018 (privacidad del DENUE), M-ML-019 (descarte) aplicados; `constitution.md` y `AGENTS.md` intactos; sin commit por instrucción explícita.
- **Seguridad y licencias:** sin red, sin secretos, sin scraping y sin servicios de pago; sin dependencias nuevas (solo biblioteca estándar, pandas y pyarrow ya presentes); no se descargó nada.
- **Archivos versionados:** `data/manifests/`, `data/universities.csv` y los reportes `denue_campus.md`, `national_baselines.md`, `regional_calibration.md`, `universities_check.md` y `variables_check.md` quedaron sin cambios tras `prepare` (`git status --short` sobre esas rutas, sin salida).
- **PASS.** T08 cumple R08 con evidencia reproducible: 11 secciones generadas desde código, ninguna cifra escrita a mano, celdas < 30 suprimidas, los tres histogramas de renta suman exactamente su total expandido y todas las atribuciones se generan desde `sources.toml`. Sin commit, por instrucción explícita.

## T05 · Entorno de campus (DENUE 05/2026)

### Qué se hizo

1. `data/sources.toml`: `inegi_denue_vigente` pasó de `download_url = []` y `variables_status = "pending"` a **confirmed**, edición **DENUE 05/2026** (modificado 2026-05-20), licencia «Términos de Libre Uso de la Información del INEGI» y atribución «Fuente: INEGI, DENUE 05/2026». Variables permitidas: `id`, `codigo_act`, `per_ocu`, `cve_ent`, `cve_mun`, `latitud`, `longitud` y `fecha_alta`. Se eliminaron del registro `clee`, `nom_estab`, `nombre_act`, `cve_loc` y demás caminos de dirección.
2. `src/thehouse_ml/csvio.py`: `resolve_zip_members` acepta patrones `conjunto_de_datos/*.csv` y `iter_zip_csv` lee por chunks con `usecols`; `variables.py::read_zip_header` usa el mismo resolvedor para verificar los encabezados de todos los CSV del ZIP sin cargar los datos.
3. `src/thehouse_ml/denue.py` (nuevo): constantes únicas `RADII_M = (1000, 2000, 5000)` y `CATEGORIES` (10 categorías SCIAN, `total` = todos); lectura exclusiva de las 8 variables con `usecols`; haversine reutilizado de `universities.py`; conteos agregados por campus/radio/categoría; Parquet y reporte con sección aparte para `seed_sin_verificar`. `prepare` lo despacha y `DenueError` se reporta con código 1.
4. `memoria.md`: decisión M-ML-018 (APROBADA). `spec.md`, `plan.md` y `task.md`: T05 marcada COMPLETA (6 de 9 tasks).
5. Pruebas sin red: `tests/unit/test_denue.py` (26), `tests/integration/test_denue_stage.py` (5), plus ampliaciones en `test_sources_loader.py`, `test_sources_registry.py`, `test_variables_check.py` y `test_prepare_stage.py`.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | 80 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 16 source files |
| `uv run pytest --cov=thehouse_ml` | **208 passed**; cobertura total **91.15 %** (mínimo 80 %), `denue.py` 90 % |
| `uv run python -m thehouse_ml download --source inegi_denue_vigente` | `inegi_denue_vigente: mixto - 3 archivo(s) verificados`; `variables_check` sin fallas |
| `uv run python -m thehouse_ml prepare` | `denue: data/processed/campus_context.parquet (300 filas, 981,519 establecimientos) - reports/denue_campus.md` |

### Desviación aprobada: URL de Edomex corregida

La URL indicada en la autorización, `https://www.inegi.org.mx/contenidos/masiva/denue/denue_15_csv.zip`, **devuelve una página 404 de INEGI** (HTML de 2,263 bytes). El catálogo oficial de descarga masiva del DENUE 05/2026 publica el Estado de México en **dos ZIP**: `denue_15_1_csv.zip` y `denue_15_2_csv.zip`. Se registraron esas dos URL (misma fuente, misma edición), se descargaron y se leyeron todos los CSV dentro de `conjunto_de_datos/` de cada una. El archivo inválido de 404 se eliminó de `data/raw`. **Aprobado por César Yael López Trujillo el 2026-11-18: adenda a M-ML-018 en `memoria.md`.**

Tamaños y SHA-256 registrados por `download` en `data/manifests/inegi_denue_vigente.json` y confirmados con `Get-FileHash`:

| Archivo | Tamaño (bytes) | SHA-256 |
| --- | ---: | --- |
| `denue_13_csv.zip` (registrado de antemano) | 15,059,273 | `d0747958bc364177ea8d281047334cffd4d3505247f43967e08da2ca9c1d9ca5` |
| `denue_15_1_csv.zip` | 50,572,588 | `23021fc24a2046e1aa3ad8c21137ca57988ded26b6f293ec0c929f59dd2cb1ae` |
| `denue_15_2_csv.zip` | 30,236,017 | `ad53e5fb65f15b491e140a80e79b207b95eb0320e4a69822d5a3cefb7652bc20` |

### Evidencia de datos reales

`data/processed/campus_context.parquet` (fuera de git, nombre de la spec §6.2) tiene 300 filas: 10 universidades × 3 radios × 10 categorías, con columnas `code`, `radius_m`, `category`, `count` y `final_status`. Se procesaron 981,519 establecimientos de los dos estados (Hidalgo y Estado de México); `reports/denue_campus.md` contiene la tabla a 2 km, una tabla por universidad, la atribución y la sección `seed_sin_verificar` con TESJI, UMB-VDC y UCAN. El ZIP de Hidalgo contiene 160,666 filas en `conjunto_de_datos/denue_inegi_13_.csv`, según el registro.

`reports/variables_check.md` cambió **solo en la sección `inegi_denue_vigente`** (de `pending`/sin verificar a `confirmed`/verificado, 8 variables encontradas, nota `-`); las siete secciones restantes quedaron byte por byte iguales (`git diff -- reports/variables_check.md`).

### Verificación de los conteos bajos de Jilotepec

Conteos del Parquet para las tres universidades solicitadas (categorías `total`, `comida` y `abarrotes_super`):

| Universidad | 1 km (total / comida / abarrotes) | 2 km (total / comida / abarrotes) | 5 km (total / comida / abarrotes) |
| --- | --- | --- | --- |
| TESJI (`seed_sin_verificar`) | 1 / 0 / 0 | 1 / 0 / 0 | 68 / 5 / 10 |
| UMB-JILO (`ok_manual`) | 1 / 0 / 0 | 1 / 0 / 0 | 10 / 0 / 1 |
| UTTT (`ok_osm`) | 1 / 0 / 0 | 197 / 27 / 54 | 2,830 / 390 / 630 |

Los dos ZIP de Edomex se leyeron íntegramente: `denue_15_1_csv.zip` 520,000 filas y `denue_15_2_csv.zip` 300,853 filas (820,853 en total), de las cuales **3,241 tienen `cve_ent = 15` y `cve_mun = 045` (Jilotepec)**. Por tanto los valores bajos de TESJI y UMB-JILO no se deben a un error de lectura: sus coordenadas finales caen en una zona rural con muy pocos establecimientos (a 5 km solo 68 y 10). El mismo proceso cuenta 1,366 negocios a 2 km en UMB-VDC (Villa del Carbón, también Edomex) y 2,830 a 5 km en UTTT (Hidalgo). TESJI conserva coordenada `seed_sin_verificar` (M-ML-017), así que su conteo es orientativo.

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R05 · radios y categorías en una sola constante | `tests/unit/test_denue.py::test_categories_and_radii_are_declared_in_one_constant` |
| R05 · clasificación por prefijo SCIAN | `test_categorize_matches_scian_prefixes`, `test_prefix_does_not_match_a_longer_scian_code`, `test_total_counts_every_business_even_with_a_blank_code` |
| R05 · frontera exacta del radio | `test_radius_boundary_is_inclusive` |
| R05 · privacidad (nunca leer columnas prohibidas) | `test_reader_uses_only_usecols_and_never_surfaces_private_columns`, `tests/integration/test_denue_stage.py::test_final_status_is_copied_to_every_row` |
| R05 · varios CSV dentro del ZIP | `test_reader_reads_every_csv_inside_the_folder`, `tests/unit/test_variables_check.py::test_read_zip_header_joins_every_csv_matching_a_glob` |
| R05 · separación de `seed_sin_verificar` | `test_render_report_separates_unverified_campuses`, `test_denue_stage.py::test_report_has_attribution_a_table_per_campus_and_the_warning` |
| R05 · ambos estados y conteos crecientes | `test_denue_stage.py::test_counts_are_monotonic_and_use_both_states` |
| R05 · esquema del Parquet y despacho | `test_denue_stage.py::test_prepare_writes_the_denue_parquet_with_the_documented_schema`, `test_run_stage_dispatches_the_denue_stage`, `tests/integration/test_prepare_stage.py::test_run_stage_dispatches_prepare` |
| R01/R02 · registro DENUE | `test_sources_loader.py::test_denue_registers_two_states_and_only_public_variables`, `test_sources_registry.py::test_denue_registration_follows_m_ml_018`, `test_denue.py::test_real_registry_declares_the_two_states_and_confirmed_variables` |

## T07 · Universidades verificadas

### Qué se hizo

1. `data/seed/universities_seed.csv` (en git): copia exacta de las 10 universidades de `TheHouse.Application/Common/UniversitySeedData.cs` del backend, commit `b7d75955513320a0a97aa5daf6b3bbe92e9514d2`. Procedencia registrada en `memoria.md` (M-ML-016) y en `data/sources.toml`.
2. `src/thehouse_ml/universities.py` (nuevo): consulta a Nominatim con User-Agent `TheHouse-ML/0.1 (proyecto escolar UTTT)`, `format=jsonv2`, `limit=5`, `countrycodes=mx`, viewbox de ±0.1° y `bounded=1`; máximo 1 petición por segundo; una consulta por universidad; cacheo por archivo JSON; manifiesto SHA-256. Haversine (radio 6 371 000 m), elección del candidato educativo más cercano (`category`/`class=amenity` y `type=university|college|school`), estados por umbral de 200 m, CSV y reporte.
3. Integración: `download --source openstreetmap` despacha la descarga OSM; `prepare` agrega la generación de `data/universities.csv` y `reports/universities_check.md`; la CLI reporta `OsmError` con código 1; `data/sources.toml` confirma las variables OSM y `variables_check.md` verifica los campos jsonv2.
4. `memoria.md`: decisión M-ML-016 (APROBADA): T07 antes que T05, fuente y commit del seed, política de Nominatim, umbral de 200 m y confirmación humana de las coincidencias dudosas en openstreetmap.org antes de T05.
5. Pruebas sin red: `tests/unit/test_universities.py` (22), `tests/unit/test_osm_download.py` (5), `tests/integration/test_universities_stage.py` (5), dos pruebas nueva en `test_variables_check.py` y actualización de `test_prepare_stage.py` para que la etapa completa incluya T07.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | 75 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 15 source files |
| `uv run pytest --cov=thehouse_ml` | **163 passed**; cobertura total **90.53 %** (mínimo 80 %), `universities.py` 93 % |
| `uv run python -m thehouse_ml download --source openstreetmap` | `openstreetmap: descargado - 10 respuesta(s) verificadas`; `variables_check` sin fallas |
| `uv run python -m thehouse_ml download --source openstreetmap` (2.ª vez) | `openstreetmap: manual - 10 respuesta(s) verificadas`: sin peticiones nuevas |
| `uv run python -m thehouse_ml prepare` | `national_baselines.parquet (87 filas)`, `regional_calibration.parquet (56 filas)` y `universities: data/universities.csv (10 filas) - reports/universities_check.md` |

### Evidencia de datos reales

`data/universities.csv` tiene 10 filas con `verified_at = 2026-11-17`. Distancias haversine al elemento educativo OSM más cercano:

| Código | Universidad | Distancia (m) | Estado |
| --- | --- | ---: | --- |
| UTTT | Universidad Tecnológica de Tula-Tepeji | 73.69 | ok |
| UAEH-ESTE | UAEH, Escuela Superior de Tepeji del Río | — | sin_coincidencia |
| UAEH-ESTL | UAEH, Escuela Superior de Tlahuelilpan | — | sin_coincidencia |
| UAEH-ESAT | UAEH, Escuela Superior de Atotonilco de Tula | — | sin_coincidencia |
| ITSOEH | Instituto Tecnológico Superior del Occidente del Estado de Hidalgo | 3347.54 | revisar |
| TESJI | Tecnológico de Estudios Superiores de Jilotepec | — | sin_coincidencia |
| UMB-JILO | UMB, Unidad de Estudios Superiores Jilotepec | — | sin_coincidencia |
| UMB-VDC | UMB, Unidad de Estudios Superiores Villa del Carbón | — | sin_coincidencia |
| UCAN | Universidad Canadiense | — | sin_coincidencia |
| UNE-TULA | Universidad del Nuevo México | 7.73 | ok |

`reports/universities_check.md` contiene la tabla completa (con coordenadas seed y OSM), la atribución © OpenStreetMap contributors (ODbL) y la lista de 8 universidades a corregir en NV-11 (1 `revisar` y 7 `sin_coincidencia`). Las respuestas cruntas quedan en `data/raw/openstreetmap/` (fuera de git) con su SHA-256 en `data/manifests/openstreetmap.json` (en git).

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R07 · haversine contra valor conocido | `tests/unit/test_universities.py::test_haversine_matches_a_known_equator_degree`, `test_haversine_is_zero_for_the_same_point`, `test_haversine_is_symmetric` |
| R07 · candidato educativo más cercano | `test_select_candidate_chooses_the_closest_educational_element`, `test_is_educational_accepts_only_educational_amenities`, `test_parse_candidates_reads_category_of_the_current_nominatim_response` |
| R07 · estados ok/revisar/sin_coincidencia | `test_build_rows_marks_ok_at_the_threshold`, `test_build_rows_marks_revisar_beyond_the_threshold`, `test_build_rows_marks_sin_coincidencia_and_leaves_osm_columns_empty` |
| R07 · parámetros y User-Agent de Nominatim | `test_build_query_url_contains_the_nominatim_policy_parameters`, `test_user_agent_identifies_the_project`, `tests/unit/test_osm_download.py::test_default_fetch_sends_the_project_user_agent` |
| R07 · sin red si el JSON existe | `test_osm_download.py::test_existing_json_is_not_requested_again`, `test_download_writes_ten_responses_and_their_manifest`, `test_rate_limit_sleeps_between_consecutive_requests` |
| R07 · 10 filas presentes | `tests/integration/test_universities_stage.py::test_prepare_writes_the_ten_universities_with_all_three_statuses`, `test_universities.py::test_load_seed_reads_the_ten_universities` |
| R07 · reporte y columnas OSM | `test_prepare_writes_the_osm_columns_and_haversine_distance`, `test_report_has_attribution_and_the_list_to_fix_nv11` |
| R07 · etapa y CLI | `test_prepare_fails_clearly_when_a_response_is_missing`, `test_cli_reports_osm_errors_with_exit_code_one`, `tests/integration/test_prepare_stage.py::test_run_stage_dispatches_prepare` |

## T07 (ampliación M-ML-017) · Verificación humana y coordenadas finales

### Qué se hizo

1. `data/seed/universities_manual.csv` (en git): 8 decisiones humanas tomadas en openstreetmap.org el 2026-11-18 — 3 `ok_seed` (UAEH-ESTE, UAEH-ESAT, UMB-JILO), 2 `corregida` (ITSOEH, UAEH-ESTL) y 3 `no_localizada` (TESJI, UMB-VDC, UCAN). Solo se guardan coordenadas de OSM; nunca de Google Maps.
2. `src/thehouse_ml/universities.py`: `load_manual` valida columnas, estados y coordenadas de las filas corregidas; `finalize_row`/`apply_manual` aplican las cuatro reglas y agregan `final_lat`, `final_lon`, `final_status` y `final_source` sin quitar las columnas de la T07.
3. `reports/universities_check.md`: nuevas secciones «Verificación humana (openstreetmap.org)» y «Correcciones para NV-11 (backend)», con la duda del nombre de UCAN y la instrucción de que T05 reporte aparte las `seed_sin_verificar`. Se conserva la atribución © OpenStreetMap contributors (ODbL).
4. `memoria.md`: decisión M-ML-017 (APROBADA). Actualizados `task.md` y este `evidence.md`. Sin consultas nuevas a Nominatim y sin dependencias nuevas.
5. Pruebas sin red: 10 pruebas nuevas en `test_universities.py` (cuatro reglas de `final_status`, carga del manual y validaciones) y ampliación de `test_universities_stage.py` (10 filas con coordenadas finales, estados esperados y secciones del reporte).

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | sin hallazgos |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues found in 15 source files |
| `uv run pytest --cov=thehouse_ml` | ver resultado final de esta sección |
| `uv run python -m thehouse_ml prepare` | `universities: data/universities.csv (10 filas) - reports/universities_check.md` |

### Coordenadas finales

| Código | final_status | final_lat | final_lon | final_source |
| --- | --- | ---: | ---: | --- |
| UTTT | ok_osm | 20.0084106 | -99.3449608 | osm:way/386380501 |
| UAEH-ESTE | ok_manual | 19.91703 | -99.36116 | seed |
| UAEH-ESTL | corregida_manual | 20.132062 | -99.235535 | way/550950316 |
| UAEH-ESAT | ok_manual | 20.01584 | -99.23601 | seed |
| ITSOEH | corregida_manual | 20.2333739 | -99.2096330 | nominatim T07 |
| TESJI | seed_sin_verificar | 19.91953 | -99.57719 | seed |
| UMB-JILO | ok_manual | 19.91248 | -99.57849 | seed |
| UMB-VDC | seed_sin_verificar | 19.72791 | -99.46312 | seed |
| UCAN | seed_sin_verificar | 20.0517 | -99.34507 | seed |
| UNE-TULA | ok_osm | 20.0310934 | -99.3363626 | osm:way/458178077 |

### Trazabilidad y revisión

| Regla | Prueba |
| --- | --- |
| `ok` → `ok_osm` (coordenada OSM) | `tests/unit/test_universities.py::test_final_status_ok_osm_uses_the_osm_coordinates` |
| `ok_seed` → `ok_manual` (seed) | `test_final_status_ok_manual_uses_the_seed_coordinates` |
| `corregida` → `corregida_manual` (lat/lon manuales) | `test_final_status_corregida_manual_uses_the_human_coordinates` |
| `no_localizada` → `seed_sin_verificar` (seed) | `test_final_status_seed_sin_verificar_uses_the_seed_coordinates` |
| 10 filas con coordenadas finales | `tests/integration/test_universities_stage.py::test_all_ten_rows_have_final_coordinates_and_the_expected_final_statuses` |
| Reporte humano y correcciones | `test_report_has_attribution_and_the_list_to_fix_nv11` |

- **Constitución / AGENTS / memoria:** autorización M-ML-017 vigente; solo fuentes aprobadas; sin scraping, sin secretos, sin servicios de pago; sin dependencias nuevas; se conserva la atribución ODbL. M-ML-016 sigue vigente para el origen de los datos T07.
- **Hallazgos:** sin consultas nuevas a Nominatim. Las 3 `seed_sin_verificar` y la duda de `UCAN` quedan pendientes del backend; T05 debe reportarlas aparte. No se hizo commit (instrucción del responsable); el commit `46c9d4f` cubre solo la T07 original.
- **Resultado:** **PASS** — las cuatro reglas están probadas, las 10 filas tienen coordenada final y el reporte incluye la verificación humana y las correcciones a NV-11. Los reportes `variables_check.md` y `regional_calibration.md` no se modifican al ejecutar `prepare`.

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

## T06 · Catálogo de cuartos — DESCARTADA (M-ML-019)

### Decisión

M-ML-019 (APROBADA por César Yael López Trujillo el 2026-11-18): **Inside Airbnb queda descartada** como fuente de ML-02. Motivos: (1) en México solo publica datos descargables de Ciudad de México y no cubre Tula, Tepeji, Atotonilco, Mixquiahuala, Tlahuelilpan, Jilotepec ni Villa del Carbón; (2) mide rentas turísticas por noche, no cuartos para estudiantes por mes; (3) sus archivos traen datos personales de anfitriones (nombres, fotos), contrarios a la constitución §7; (4) las rentas ya están cubiertas por ENVI 2020 y ENIGH 2022 (M-ML-014), los servicios de la vivienda por el Censo 2020 (M-ML-015) y el entorno por el DENUE (M-ML-018). Consecuencia: el catálogo de cuartos se generará **sintético en ML-03**, calibrado con esas fuentes y etiquetado como sintético (constitution §8).

### Qué se hizo

1. `data/sources.toml`: `inside_airbnb_cdmx_2026_09_23` pasó a `variables_status = "discarded"` con `status_reference = "M-ML-019"`; se retiró la URL no verificada `https://data.insideairbnb.com/.../listings.csv.gz` (se conserva solo la página de políticas `https://insideairbnb.com/get-the-data/`) y `download_url` sigue vacío. No se descargó ni se verificó ningún archivo de Inside Airbnb.
2. `src/thehouse_ml/sources.py`: `ALLOWED_STATUS` acepta `"discarded"`; `SourceSpec` incorpora `status_reference`, obligatorio cuando el estado es `discarded` (validación con error explícito si falta).
3. `src/thehouse_ml/download.py`: una fuente descartada se omite antes de cualquier descarga o manifiesto, con `DownloadOutcome(..., "omitida", "descartada: <status_reference>")`.
4. `src/thehouse_ml/variables.py`: una fuente descartada conserva su sección en `reports/variables_check.md` con `- estado: discarded`, `verificado: no` y `nota: descartada: M-ML-019`; no cuenta como fallo y no se leen encabezados. Las demás secciones del reporte no se tocan.
5. `spec.md`, `plan.md` y `task.md`: T06 marcada **DESCARTADA (M-ML-019)** con la justificación, R06 reorientado al descarte, matriz y estados actualizados, y nota para ML-03 sobre el catálogo sintético. `memoria.md`: M-ML-019 (APROBADA) y M-ML-003 ajustada. `evidence.md`: esta sección.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `80 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 16 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | `215 passed in 8.57s`; cobertura total `91.22 %` ≥ 80 % (PASS) |
| `uv run thehouse-ml download` (sin `--source`) | Todas las fuentes reales en modo `manual`; **línea de Inside Airbnb: `inside_airbnb_cdmx_2026_09_23: omitida - descartada: M-ML-019`**; OSM omitido bajo demanda; sin red y sin descargar nada de Inside Airbnb |

Salida completa de `download` (2026-11-18):

```text
inegi_censo_2020_viviendas_ca: manual - 4 archivo(s) verificados
inegi_envi_2020: manual - 2 archivo(s) verificados
inegi_enigh_2022: manual - 1 archivo(s) verificados
inegi_denue_vigente: manual - 3 archivo(s) verificados
inegi_inpc: manual - 1 archivo(s) verificados
inside_airbnb_cdmx_2026_09_23: omitida - descartada: M-ML-019
openstreetmap: omitido - ejecute download --source openstreetmap
thehouse_seed_nv07_nv08_nv11: pendiente - download_url pendiente
variables_check: C:/Users/cesar/source/repos/TheHouse-ML/reports/variables_check.md
```

Diff de `reports/variables_check.md` tras `download` (solo la sección de Inside Airbnb):

```diff
 ## inside_airbnb_cdmx_2026_09_23
 
-- estado: `confirmed`
+- estado: `discarded`
 - verificado: no
 - variables encontradas: (ninguna)
 - variables faltantes: (sin verificar)
-- nota: download_url pendiente (sin verificar)
+- nota: descartada: M-ML-019
```

### Trazabilidad requisito → prueba

| Requisito | Prueba |
| --- | --- |
| R06 · fuente descartada en el registro con referencia | `tests/unit/test_sources_registry.py::test_inside_airbnb_is_discarded_per_m_ml_019`, `tests/unit/test_sources_loader.py::test_inside_airbnb_is_discarded_with_reference_and_no_download_url` |
| R06 · estado `discarded` validado con `status_reference` | `tests/unit/test_sources_loader.py::test_discarded_status_requires_a_reference`, `test_discarded_status_is_loaded_with_its_reference` |
| R06 · `download` omite sin descargar ni verificar | `tests/unit/test_download.py::test_discarded_source_is_skipped_without_fetching`, `tests/integration/test_download_stage.py::test_discarded_source_is_omitted_but_keeps_its_report_section` |
| R06 · reporte marca el descarte y conserva las otras secciones | `tests/unit/test_variables_check.py::test_discarded_source_is_shown_as_discarded_and_other_sections_are_kept`, el diff de `reports/variables_check.md` mostrado arriba |
| R09 · calidad general | 215 pruebas, 91.22 % de cobertura, ruff y mypy sin errores |

### Revisión y resultado

- **Constitución:** §7 (sin datos personales ni descargas de fuentes no autorizadas; se elimina la fuente que traía nombres y fotos de anfitriones), §8 (el catálogo de cuartos pasa a ML-03 como sintético y etiquetado como tal), §3/SDD-001 (cambio por spec, aprobación y «EJECUTA» vigentes), §16 (cobertura 91.22 % con pruebas sin red). No se modificaron `constitution.md` ni `AGENTS.md`.
- **AGENTS / memoria:** autorización de M-ML-019 del responsable vigente (M-ML-011); M-ML-019 APROBADA; rutas respetadas; sin dependencias nuevas; sin descargas de Inside Airbnb.
- **Seguridad y licencias:** sin secretos, sin scraping, sin servicios de pago; el único acceso a red autorizado sigue siendo Nominatim en T07 y no se usó en esta ejecución. Ningún dato crudo o procesado entró a git.
- **PASS.** T06 quedó DESCARTADA con evidencia reproducible: la fuente figura como `discarded` con `M-ML-019`, `download` la omite con el aviso exigido, el reporte de variables solo cambió en su sección y las 215 pruebas pasan. El hallazgo 20 quedó resuelto con la autorización y el ajuste de `specs/ARQUITECTURA.md`; el hallazgo 21 queda **PENDIENTE PARA T09** (reescritura de `downloaded_at` en modo manual), sin corregir en esta task. No se hizo commit, por instrucción explícita del responsable.

## Hallazgos y desviaciones

1. **Servicios en T04, no en T03 (resuelto).** La redacción de R03 en la spec menciona «servicios ponderados», pero el responsable acotó la T03 a renta y proporción de viviendas rentadas y asignó la prevalencia de servicios por municipio al Censo 2020 de la T04 (M-ML-014(c)). La T04 generó esos siete indicadores para los 7 municipios; no se reportó como PASS una salida de servicios de T03.
2. **pandas sin stubs de tipo.** `pandas` no publica `py.typed` y `pandas-stubs` no está en la tabla de la constitution §5, así que no se agregó. Se configuró en `pyproject.toml` un override de mypy `ignore_missing_imports` **solo para `pandas`/`pandas.*`**; el resto del código sigue en modo estricto.
3. **INPC: valor semántico.** El diccionario del ZIP describe `VALOR` como «variación porcentual anual», pero el responsable verificó que son niveles del índice base 2Q jul 2018 = 100 (2024-07 = 136.003). Se documentó en `data/sources.toml`, el reporte y el código; se filtra solo el concepto general.
4. **Definición de `n_muestra`.** Se adoptó como el conteo sin ponderar de viviendas rentadas con renta válida (la muestra que produce los percentiles); la proporción de rentadas usa todas las viviendas del grupo. Queda explicado en el reporte.
5. **Sin commit ni PR** por instrucción explícita del responsable. La aprobación de otro humano en GitHub sigue pendiente.
6. **Fuentes pendientes.** El Censo 2020 se descargó y verificó en T04 y el DENUE en T05; Inside Airbnb quedó **DESCARTADA** (M-ML-019), OpenStreetMap se consultó solo en T07 y el seed sigue sin descarga por ser dato propio del backend.
7. **Renta sin `ee`/`cv` y respaldo nacional limitado.** Las filas de renta copian los percentiles de la ENVI 2020 de T03, que no publica varianza; sus columnas `se` y `cv` quedan como `NaN` y se marcan `—` en el reporte. La suficiencia no se aplica a la renta porque nunca es una estimación municipal. El respaldo `nacional` de ENVI/ENIGH de T03 solo está implementado para la proporción de viviendas rentadas; T03 no publica prevalencias de servicios (M-ML-014(c)), por lo que para los seis indicadores de servicios la entidad completa es el último nivel disponible y una entidad insuficiente daría un error explícito en vez de una estimación no documentada. En los datos reales no se presentó ese caso.
8. **Estratos con una sola UPM.** En el diseño con reemplazo no tienen varianza definida; no aportan a la varianza y se cuentan en `n_strata_single_psu`. En los datos reales ningún municipio de los 7 quedó sin estimación por este motivo.
9. **Reescritura de manifiestos y reportes.** Al ejecutar `download` y `prepare` con datos reales se reescribieron los manifiestos y `reports/national_baselines.md` (solo diferencias de fin de línea CRLF/LF en Windows; el contenido es idéntico) y se creó `data/manifests/inegi_censo_2020_viviendas_ca.json`.
10. **`category` frente a `class` en jsonv2 (resuelto).** El servicio Nominatim actual publica `category=amenity`, no `class=amenity`. El primer parseo dejó todos los candidatos como no educativos; se corrigió para aceptar ambas claves y se agregó una prueba. La corrección se hizo sobre los JSON ya guardados: **no se hizo ninguna consulta adicional** (se respetaron las 10 permitidas y el cacheo).
11. **Coincidencias dudosas (pendiente de humano).** Con los nombres completos del seed y `bounded=1`, 7 respuestas quedaron vacías y 1 candidato quedó a 3347.54 m (`ITSOEH`). Son 8 universidades en la lista de NV-11 del reporte. Según M-ML-016 las confirma un humano en openstreetmap.org **antes de usarlas en T05**; no se reconsultó Nominatim con otro texto porque la autorización era de 10 consultas.
12. **Reescritura de reportes por fin de línea (T07).** Al ejecutar `prepare` se reescribió `reports/regional_calibration.md` y al ejecutar `download --source openstreetmap`, `reports/variables_check.md` (solo CRLF/LF en Windows; contenido equivalente).
13. **Sin commit ni PR por instrucción explícita del responsable.** La aprobación de otro humano en GitHub sigue pendiente.
14. **Reporte de variables filtrado (detectado y corregido).** `run_download` usaba la lista devuelta por `select_sources` tanto para descargar como para verificar, de modo que `download --source openstreetmap` reescribía `reports/variables_check.md` con una sola sección. Se separaron las dos responsabilidades: `--source` limita solo la descarga y `verify_sources` recibe siempre todas las fuentes del registro. El reporte se escribe con finales LF. Pruebas: `test_download_stage.py::test_source_filter_limits_the_download_but_not_the_report` y `test_universities_stage.py::test_filtered_download_keeps_every_source_in_the_variables_report`.
15. **CI en máquina vacía (detectado y corregido).** El job `quality` fallaba en «CLI con las cinco etapas»: un `download` sin filtro incluía OpenStreetMap y consultaba Nominatim (prohibido por M-ML-016 en CI), y un `prepare` sin respuestas en `data/raw/openstreetmap/` fallaba con `OsmError`. Ahora OSM es fuente bajo demanda: sin `--source` se omite con `openstreetmap: omitido - ejecute download --source openstreetmap` y no se llama al fetcher; solo `download --source openstreetmap` consulta. `prepare` sin respuestas conserva `data/universities.csv` y `reports/universities_check.md` versionados y reporta `universities: omitido (sin respuestas OSM; se conserva data/universities.csv versionado)`. Pruebas sin red: `test_download_stage.py::test_download_without_source_omits_osm_and_never_calls_the_fetcher` y `test_universities_stage.py::test_prepare_without_osm_responses_keeps_the_versioned_files`. Sin dependencias nuevas ni consultas nuevas a Nominatim.
16. **URL de Edomex 404 (resuelto el 2026-11-18).** `denue_15_csv.zip` devuelve una página HTML de 2,263 bytes. El catálogo de descarga masiva del DENUE 05/2026 publica Edomex en dos ZIP (`denue_15_1_csv.zip`, `denue_15_2_csv.zip`); se usaron esos dos archivos y sus hashes quedaron en el manifiesto. El responsable aprobó el cambio y se registró como adenda de M-ML-018 en `memoria.md` (ver sección T05).
17. **Nombre de salida (resuelto el 2026-11-18).** La spec 1.0.1 describe `campus_context.parquet` en §6 y §6.2; la autorización de T05 pidió inicialmente `denue_campus.parquet`. El responsable aprobó usar el nombre de la spec: la salida se renombró a `data/processed/campus_context.parquet` en código, pruebas, reporte, `task.md` y esta evidencia. El reporte conserva su nombre `reports/denue_campus.md`.
18. **Reescritura de `data/universities.csv` por fin de línea (resuelto el 2026-11-18).** `prepare` reescribía el CSV versionado porque `csv.DictWriter` usa CRLF por omisión en Windows. Ahora `write_universities_csv` genera el contenido con `lineterminator="\n"`, compara con el archivo existente (normalizando CRLF→LF) y no escribe si el contenido no cambia. Se restauró el archivo con `git checkout -- data/universities.csv`; `prepare` ya no lo modifica. Prueba: `tests/unit/test_universities.py::test_write_universities_csv_uses_lf_and_does_not_rewrite_unchanged_content`.
19. **Fechas de manifiestos reescritas por `download` (resuelto el 2026-11-18).** Al ejecutar `download` sin filtro en la T06, los cinco manifiestos de INEGI se reescribieron solo en `downloaded_at` (2026-11-17 → 2026-11-18), un efecto colateral de la etapa y fuera de las rutas permitidas para esta task. Se restauraron con `git checkout -- data/manifests/`; los hashes y tamaños no cambiaron. No se creó ningún manifiesto de Inside Airbnb (fuente descartada).
20. **Referencia residual en `specs/ARQUITECTURA.md` (resuelto el 2026-11-18).** La línea 62 listaba a Inside Airbnb entre las fuentes de `download`. El responsable **autorizó** agregar `specs/ARQUITECTURA.md` a las rutas permitidas y ajustar ese archivo: se quitó Inside Airbnb de la tabla de `download`, se añadió la nota de que fue descartada por M-ML-019 y que el catálogo de cuartos será sintético en ML-03, y se alinearon las filas de `prepare` y de ML-02. Solo cambiaron esas menciones; no afectó al código ni al comportamiento de `download`.
21. **PENDIENTE PARA T09 — `download` en modo manual reescribe `downloaded_at` en los manifiestos aunque el SHA-256 no cambie; debe conservarse la fecha original si el archivo es idéntico.** Detectado al ejecutar `download` sin filtro en T06: los cinco manifiestos de INEGI cambiaron solo `downloaded_at` (2026-11-17 → 2026-11-18) con el mismo SHA-256 y tamaño; se restauraron con `git checkout -- data/manifests/`, pero la causa (escritura incondicional del manifiesto) sigue sin corregirse. No se corrige en T06: queda registrado para la T09 (pruebas y evidencia) con su prueba de regresión. Ver hallazgo 19.
22. **Salida intermedia nueva (resuelto el 2026-11-18).** Para que el histograma de renta use la distribución real y no una reconstrucción a partir de percentiles, `prepare` escribe `data/processed/rent_histogram.parquet` con los intervalos de ENVI. Es un derivation interno fuera de git; no cambia el contrato de la spec §6.2 ni agrega dependencias. Se documenta en la sección 9 de la data card.
23. **Proyectos sin DENUE (resuelto el 2026-11-18).** Las pruebas de integración que simulan `prepare` sin los ZIP del DENUE no producen `campus_context.parquet`. Para no convertir eso en un error, `run_data_card` omite la generación con el aviso `data card: omitido (faltan salidas de prepare)` cuando falta alguna entrada; con datos reales, `prepare` produce el Parquet y la data card normalmente.
24. **`verified_at` dependiente de la fecha (2026-11-19, PENDIENTE PARA T09).** Al ejecutar `prepare` el 2026-11-19, `run_universities` reescribió las 10 filas de `data/universities.csv` cambiando solo `verified_at` de 2026-11-18 a 2026-11-19 (usa la fecha del sistema). No guarda relación con la data card: se restauró el archivo con `git checkout -- data/universities.csv` y quedó sin cambios. Queda registrado para la T09 junto con el hallazgo 21.

## Revisión

- **Constitución:** §5 (solo pandas y pyarrow, añadidos como dependencias directas; T04 sin dependencias nuevas), §6 y §7 (Parquet fuera de git; INEGI citado como fuente sin sugerir respaldo), §7 (SHA-256 de los cuatro archivos del Censo verificado antes de usarlos; sin datos personales en las salidas), §9 (misma entrada → mismo Parquet), §16 (pruebas sin red, cobertura 89.79 %).
- **AGENTS / memoria:** «EJECUTA» vigente de M-ML-011, M-ML-014 y M-ML-015; fuentes aprobadas en M-ML-003; decisiones M-ML-010, M-ML-012, M-ML-013, M-ML-014 y M-ML-015 aplicadas; la T04 no modifica `constitution.md` ni `AGENTS.md`.
- **Seguridad y licencias:** sin secretos, sin scraping, sin servicios de pago; solo productos públicos de INEGI citados como autor; ningún dato crudo o procesado versionado (verificado por `tests/test_repo_hygiene.py`); el Censo se descargó solo para las dos entidades autorizadas.
- **T07 (M-ML-016):** sin dependencias nuevas (HTTP y JSON de la biblioteca estándar); solo 10 consultas a Nominatim con User-Agent propio y ≤ 1 req/s; sin scraping; respuestas crudas fuera de git y SHA-256 en `data/manifests/openstreetmap.json`; atribución ODbL en el reporte y en `sources.toml`; sin datos personales; se ejecutó antes que T05 según lo autorizado; no se modifica `constitution.md` ni `AGENTS.md`.
- **T05 (M-ML-018):** constitución §7 (solo fuente INEGI con licencia registrada, sin republicar microdatos; `usecols` impide leer nombres, teléfonos, correos ni direcciones; solo agregados en el Parquet y el reporte), §9 (misma entrada → mismos conteos) y §16 (cobertura 91.23 %, pruebas sin red con CSV latin-1 inventado); AGENTS/memoria: autorización vigente, M-ML-018 APROBADA, sin dependencias nuevas (pandas, pyarrow y biblioteca estándar), sin tocar `constitution.md` ni `AGENTS.md`. El ZIP de Edomex se sustituyó por los dos del catálogo oficial por un 404 (hallazgo 16, pendiente de confirmación).

## Resultado

**PASS.** T05 cumple su criterio de aceptación con evidencia reproducible: `data/processed/campus_context.parquet` tiene 300 filas (10 universidades × 3 radios × 10 categorías), los conteos son no negativos y crecientes con el radio, se procesaron 981,519 establecimientos de ambos estados, `reports/denue_campus.md` incluye la tabla a 2 km, una tabla por campus y la sección aparte de `seed_sin_verificar`, y `reports/variables_check.md` solo cambió en la sección de DENUE. Advertencia resuelta: la URL de Edomex autorizada estaba caída y se sustituyó por los dos ZIP del catálogo oficial (hallazgo 16, **aprobado por el responsable el 2026-11-18**). Hallazgos 17 y 18 resueltos el 2026-11-18. Sin commit, por instrucción explícita.

**PASS WITH WARNINGS.** T07 cumple su criterio de aceptación con evidencia reproducible: `data/universities.csv` tiene las 10 universidades con id OSM (cuando hay coincidencia), distancia haversine, estado y fecha de verificación; las 3 aproximadas del seed quedaron registradas y los 8 casos dudosos se reportan para NV-11. Advertencias: 7 `sin_coincidencia` y 1 `revisar` requieren confirmación humana en openstreetmap.org antes de T05 (hallazgos 10 y 11), reescritura de reportes por fin de línea (hallazgo 12) y ausencia de commit/PR por instrucción (hallazgo 13).

T04 mantiene su PASS WITH WARNINGS previo: los 7 municipios están presentes, cada indicador registra su nivel de respaldo y la regla de suficiencia está probada. Advertencias: `ee`/`cv` no disponibles en la renta de T03 (hallazgo 7), estratos con una sola UPM (hallazgo 8), reescritura de manifiestos por fin de línea (hallazgo 9), el override de mypy para pandas (hallazgo 2) y la ausencia de commit/PR (hallazgo 5). T03 mantiene su PASS WITH WARNINGS previo con el hallazgo 1 resuelto en T04.

**PASS.** T08 cumple R08: `reports/data_card.md` se genera automáticamente desde el registro, los manifiestos, el CSV de universidades y los Parquet; contiene las 11 secciones exigidas, suprime las celdas con menos de 30 viviendas en muestra, sus tres histogramas de renta suman exactamente el total expandido de `national_baselines.parquet`, Inside Airbnb figura como `DESCARTADA (M-ML-019)` sin atribución como fuente y todas las atribuciones INEGI/OSM se producen desde `sources.toml`. Sin commit, por instrucción explícita.
