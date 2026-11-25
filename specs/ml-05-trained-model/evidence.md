# ML-05 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-05 1.0.0 |
| Tareas ejecutadas | T01 · Dependencias y carga de datos (R01); T02 · Validación cruzada y entrenamiento reproducible (R02); T03 · Evaluación contra el baseline y decisión de promoción (R03); T04 · Ablación y segmentos (R04); T05 · Calibración 0-100 y desglose (R05); T06 · Model card (R06); T07 · Pruebas y evidencia (R07) con el resumen final. Cada task con sección propia abajo |
| Estado | **✅ COMPLETA — T01–T07 (7/7); cierre M-ML-043** |
| Autorización (T01) | «EJECUTA ML-05 1.0.0, task T01 (ML-05-R01), en la rama feat/ml-05-deps-data. Rutas permitidas: pyproject.toml, uv.lock, src/thehouse_ml/ (un módulo de carga de datos de entrenamiento, por ejemplo en models/ o evaluation/ según ARQUITECTURA.md), tests/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). No modifiques feature_schema.json ni ningún params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-23 |
| Rama (T01) | `feat/ml-05-deps-data` (commit base `2bcb71e`, merge del PR #30); mezclada en `main` con el PR #31 (`386fd1d`) |
| Autorización (T02) | «EJECUTA ML-05 1.0.0, task T02 (ML-05-R02), en la rama feat/ml-05-train. Rutas permitidas: src/thehouse_ml/ (models/ para el entrenamiento; evaluation/ y features/ solo para reutilizar), tests/, reports/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No modifiques feature_schema.json, baseline_params.json ni los params de ML-03/ML-04. Los modelos entrenados van a data/processed/ o models/ (fuera de git), nunca con pickle» — César Yael López Trujillo, 2026-11-23 |
| Rama (T02) | `feat/ml-05-train` (commit base `386fd1d`, merge del PR #31) |
| Aprobación (T02) | «Apruebo M-ML-038 (César Yael López Trujillo, 2026-11-24) con sus 9 decisiones y la elección de C=0.01 por la regla registrada (T02-H3)»; en el mismo mensaje, por T02-H1: «Autorizo .github/workflows/ci.yml solo para esto: agrega un perfil de entrenamiento "ci" […] que use 1 combinación por modelo (las elegidas) y 2 pliegues, escriba sus salidas en una ubicación que no sobrescriba reports/training_report.md ni reports/train_cv_metrics.json versionados, y deje la corrida completa como la oficial. No cambies nombres de jobs ni timeouts» |
| Autorización (T03) | «EJECUTA ML-05 1.0.0, task T03 (ML-05-R03), en la rama feat/ml-05-eval. Rutas permitidas: src/thehouse_ml/ (models/ y evaluation/; features/ solo para reutilizar), tests/, reports/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No modifiques train_params.json, los modelos entrenados, baseline_params.json, baseline_eval_params.json ni los params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-24 |
| Rama (T03) | `feat/ml-05-eval` (commit base `971beaf`, merge del PR #32) |
| Aprobación (T03) | «Apruebo M-ML-039 (César Yael López Trujillo, 2026-11-24)»; en el mismo mensaje: «Autorizo retroactivamente el cambio en src/thehouse_ml/stages.py (T03-H6). T03-H2: que T04 registre las 4 métricas (NDCG@10, NDCG@5, Precision@5 y MAP) por pliegue al reentrenar para la ablación, y que T04 complete la tabla por pliegue de trained_report.md; anótalo en evidence.md como traspaso a T04. Con eso, T03 queda en PASS» |
| Autorización (T04) | «EJECUTA ML-05 1.0.0, task T04 (ML-05-R04), en la rama feat/ml-05-ablation. Rutas permitidas: src/thehouse_ml/ (models/ y evaluation/; features/ solo para reutilizar; stages.py solo si hace falta para integrar la etapa), tests/, reports/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No modifiques train_params.json, los modelos oficiales, baseline_params.json, baseline_eval_params.json ni los params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-24 |
| Rama (T04) | `feat/ml-05-ablation` (commit base `03a7b41`, merge del PR #33) |
| Aprobación (T04) | «Apruebo M-ML-040 (César Yael López Trujillo, 2026-11-24): agrupamiento de features, regla de la hipótesis y perfil ci reducido»; en el mismo mensaje: «Deja la hipótesis > $3,500 como CONFIRMADA con las salvedades T04-H2 y T04-H3 documentadas, para que la model card (T06) las incluya. T04-H1 se cierra cuando el CI del PR confirme el tiempo en ubuntu. Marca T04 en PASS, T03-H2 resuelto» |
| Autorización (T05) | «EJECUTA ML-05 1.0.0, task T05 (ML-05-R05), en la rama feat/ml-05-breakdown. Rutas permitidas: src/thehouse_ml/ (models/ y evaluation/; features/ solo para reutilizar; stages.py solo si hace falta), tests/ (incluido tests/fixtures/), reports/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No modifiques train_params.json, los modelos oficiales, feature_schema.json, baseline_params.json ni los params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-24 |
| Rama (T05) | `feat/ml-05-breakdown` (commit base `8dd9e6e`, merge del PR #34) |
| Aprobación (T05) | «Decisión sobre T05-H1: opción 1. La condición (c) de la spec §6.5 aprobada es "desglose aditivo que suma el total" y se cumple; la verificación de monotonía del presupuesto fue una protección adicional que yo agregué en el «EJECUTA», no parte del criterio de la spec. Acepto la no monotonía de price_below_min como desviación conocida […] Promueve logreg-1.0.0 como modelo de ML-05 (baseline-1.0.0 queda como respaldo) y vuelve a correr evaluate […] Apruebo M-ML-041 (César Yael López Trujillo, 2026-11-24) con eso. Marca T05 completa en PASS WITH WARNINGS» |
| Autorización (T06) | «EJECUTA ML-05 1.0.0, task T06 (ML-05-R06), en la rama docs/ml-05-model-card. Rutas permitidas: reports/model_card.md, src/thehouse_ml/ (solo si la model card se genera automáticamente desde código, como data_card.md), tests/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No cambies modelos, métricas ni params» — César Yael López Trujillo, 2026-11-24 |
| Rama (T06) | `docs/ml-05-model-card` (commit base `9c49499`, merge del PR #35) |
| Aprobación (T06) | «Apruebo M-ML-042 (César Yael López Trujillo, 2026-11-24). T06-H1 se traspasa a T07: agregar a trained_metrics.json la distribución de matchScore y los ejemplos explicados, y que la model card los tome de ahí. T06-H2 aceptado. Marca T06 en PASS» |
| Autorización (T07) | «EJECUTA ML-05 1.0.0, task T07 (ML-05-R07), en la rama feat/ml-05-cierre. Rutas permitidas: src/thehouse_ml/, tests/, reports/, memoria.md, specs/ml-05-trained-model/, specs/README.md y specs/ARQUITECTURA.md (solo lo de ML-05). Sin dependencias nuevas. No cambies los modelos entrenados, train_params.json ni ningún params» — César Yael López Trujillo, 2026-11-24 |
| Rama (T07) | `feat/ml-05-cierre` (commit base `44a89ee`, merge del PR #36) |
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | T01: 2026-11-23 · T02: 2026-11-23 y 2026-11-24 (retomada tras un apagado del equipo) · T03: 2026-11-24 · T04: 2026-11-24 · T05: 2026-11-24 · T06: 2026-11-24 · T07: 2026-11-24 |
| Decisiones | M-ML-036 (APROBADO, decisiones 1 y 6); M-ML-037 (**APROBADO**, César Yael López Trujillo, 2026-11-23: versiones, override de mypy, ubicación, tercera entrada `students.parquet`, validación del esquema, reutilización de `join_labels`, formato de salida, prueba de humo y protección sin pickle); M-ML-038 (**APROBADO**, César Yael López Trujillo, 2026-11-24: semilla, pliegues, rejilla, selección, `num_threads`, persistencia, salidas y perfil `ci`); M-ML-039 (**APROBADO**, César Yael López Trujillo, 2026-11-24: ubicación y etapa, verificaciones, referencia reproducida, bootstrap pareado, condiciones 1 y 3, tabla por pliegue y salidas; `stages.py` autorizado retroactivamente; T03-H2 traspasado a T04); M-ML-040 (**APROBADO**, César Yael López Trujillo, 2026-11-24: grupos de features, pliegues y modelos de la validación cruzada, 4 métricas por pliegue, ablación, segmentos, regla de la hipótesis > $3,500, integración y perfil `ci`); M-ML-041 (**APROBADO**, César Yael López Trujillo, 2026-11-24: calibración afín, `maxPoints`, `pets_compatible` al término fijo, formato de `breakdown`, `modelVersion` `logreg-1.0.0`, regla de monotonía con la desviación aceptada T05-H1; `logreg-1.0.0` promovida); M-ML-042 (**APROBADO**, César Yael López Trujillo, 2026-11-24: generación de la model card desde los JSON de `reports/`, constantes de trazabilidad, distribución y ejemplos enlazados, solo perfil `full`; T06-H1 traspasado a T07, T06-H2 aceptado); M-ML-043 (**APROBADO**, César Yael López Trujillo, 2026-11-24: cierre de ML-05 y herencia para ML-06) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§5 stack y §13 sin pickle) → `AGENTS.md` → `memoria.md` (M-ML-032 a M-ML-036) → `spec.md` (sobre todo R01 y §10 decisiones 1 y 6), `plan.md` y `task.md` de ML-05 → `specs/ARQUITECTURA.md` → `src/thehouse_ml/synthetic/labels_stage.py`, `src/thehouse_ml/evaluation/` y `src/thehouse_ml/models/baseline_report.py` (para reutilizar) → `specs/ml-04-baseline-model/evidence.md` (formato).

## T01 · Dependencias y carga de datos

### Qué se hizo

1. **Dependencias** con `uv add scikit-learn lightgbm` (uv 0.13.7), que respeta `exclude-newer = "2026-10-27T00:00:00Z"` de `pyproject.toml`: `pyproject.toml` (`lightgbm>=4.7.0`, `scikit-learn>=1.9.1`) y `uv.lock` (+103 líneas, 55 paquetes). No se fijó ninguna versión a mano.

   | Paquete | Versión exacta (`uv.lock`) | Tipo | Licencia | Publicado (PyPI) |
   | --- | --- | --- | --- | --- |
   | `scikit-learn` | **1.9.1** | directa (constitución §5) | BSD-3-Clause | 2026-09-10 |
   | `lightgbm` | **4.7.0** | directa (constitución §5) | MIT | 2026-07-18 |
   | `scipy` | 1.18.1 | transitiva (ambas) | BSD-3-Clause | 2026-08-21 |
   | `joblib` | 1.6.0 | transitiva (scikit-learn) | BSD-3-Clause | 2026-08-31 |
   | `threadpoolctl` | 3.7.0 | transitiva (scikit-learn) | BSD-3-Clause | 2026-09-15 |
   | `narwhals` | 2.27.1 | transitiva (ambas) | MIT | 2026-10-10 |
   | `cloudpickle` | 3.1.2 | transitiva (`joblib`) | BSD-3-Clause | 2025-11-03 |

   `numpy` 2.5.4 ya estaba (vía pandas). Todas las fechas son anteriores a `exclude-newer`. Hay wheels para `cp312`/`py3` en `win_amd64` y `manylinux_2_28_x86_64` (los dos sistemas del CI).

2. **mypy:** `scikit-learn` no publica `py.typed` (comprobado con un módulo de sondeo en el scratchpad: `import-untyped` en `sklearn`, `sklearn.linear_model` y `sklearn.model_selection`); se agregó `[[tool.mypy.overrides]]` con `ignore_missing_imports` para `sklearn` y `sklearn.*`, como el de pandas. `lightgbm` publica `py.typed` y no lleva excepción. Con el override el sondeo pasa en modo estricto y no aparece aviso de sección sin uso.

3. **`src/thehouse_ml/models/training_data.py`** (nuevo; `ARQUITECTURA.md` §4: `models/` = ML-05):
   - `build_training_data(pairs, labels, students, schema)` (puro) y `load_training_data(paths)` (única I/O, con la lista blanca `TRAINING_INPUTS = (pairs.parquet, labels.parquet, students.parquet)`; de `students.parquet` solo se leen `student_id` y `budget.max`).
   - Valida: esquema empaquetado **2.0.0**; columnas de `pairs.parquet` (ids, `synthetic` y las 10 de `feature_order`) con las features **en el orden** del esquema; `synthetic = true` en pares y etiquetas; `labels_params_version` = **1.0.0** en todas las filas; relevancias en 0–3; features finitas; un solo `split`, `university_code` y `seed_verified` por estudiante; `budget.max` presente y no nulo. Cada fallo da `TrainingDataError` con el archivo y la columna o versión.
   - Une con `join_labels` de `models/baseline_report.py` (uno a uno por `student_id` y `room_id`, sin duplicados, partición coincidente: las mismas reglas que la evaluación de ML-04).
   - Excluye y cuenta a los estudiantes con < 2 candidatos (`excluded`, con su partición).
   - Devuelve `TrainingData`: `features` (X: exactamente `feature_order`, float64), `relevance` (y), `group` (`student_id`), `room_id`, `split`, `students` (metadatos por estudiante: `split`, `university_code`, `seed_verified`, `budget_max`, `candidates`), `excluded` y `summary()`. **No genera etiquetas** y no importa scikit-learn ni LightGBM.

4. **Pruebas** (sin red):
   - `tests/unit/test_training_data.py` (43 casos, DataFrames construidos en la prueba): unión correcta con etiquetas barajadas, exclusión y recuento, columnas de entrada exactamente `feature_order` aunque los insumos traigan `utility` o `weight_*`, metadatos de segmento, error claro por cada columna faltante de pares (15), de etiquetas (4) y de `budget.max`, features fuera de orden, cambio de `labels_params_version`, cambio de versión del esquema, filas no sintéticas, par sin etiqueta, partición distinta o desconocida, relevancia fuera de 0–3, NaN e infinito, estudiante con dos campus o dos `seed_verified`, estudiante ausente o duplicado en `students.parquet`, `budget.max` nulo, ningún estudiante con 2 candidatos, lista blanca de lecturas y archivo faltante, y equivalencia `load` = `build` leyendo solo `budget.max`. Cobertura del módulo: **100 %**.
   - `tests/integration/test_training_data_stage.py` (3 casos, marcados `integration`): sobre un proyecto temporal con 2 campus (UTTT verificado y UCAN sin verificar) generado con `run_pairs` y `run_labels` (que escribe también los dos archivos ocultos), la **prueba obligatoria** de la decisión 1, las entradas del modelo y la cobertura de todos los pares.
   - `tests/unit/test_ml_dependencies.py` (4 casos): versiones instaladas = `uv.lock`, dependencias declaradas, **LightGBM LambdaRank** con restricción monotónica, `deterministic = true` y un hilo (texto nativo idéntico en dos ajustes, sin pickle) y **regresión logística** en `Pipeline` con `StandardScaler` y `StratifiedGroupKFold` sin estudiantes compartidos. Corre en los dos jobs del CI.
   - `tests/test_repo_hygiene.py` (protección sin pickle, M-ML-037 decisión 9, pedida al aprobar): `pickle_uses` analiza el AST de cada módulo y `test_no_module_of_the_package_uses_pickle` falla si algún archivo de `src/thehouse_ml/` importa `pickle`, `cloudpickle` o `dill` o usa `joblib.dump`/`joblib.load`, con alias, `from … import`, `__import__` o `importlib.import_module`. **Ampliación** (pedida por el responsable después de aprobar M-ML-037): también falla con los métodos de pandas que serializan con pickle, `read_pickle` y `to_pickle`, como atributo de cualquier objeto (`pandas.read_pickle`, `pd.read_pickle`, `df.to_pickle`, `frame["x"].to_pickle`, incluso una referencia sin llamar como `reader = pd.read_pickle`) o importados con `from pandas import read_pickle` (con o sin alias). 21 casos prohibidos y 10 permitidos (p. ej. `to_parquet`, `from pandas import read_parquet` y la mención en un texto) prueban el detector.

### Prueba obligatoria (decisión 1): nunca se leen los archivos ocultos

`test_training_data_stage.py::test_loader_never_reads_the_hidden_label_files`, igual que `test_evaluate_never_reads_the_hidden_label_files` de ML-04 T06:

1. carga los datos con los dos archivos ocultos presentes;
2. espía `pd.read_parquet`, **borra** `labels_hidden_utility.parquet` y `labels_hidden_weights.parquet` y vuelve a cargar (los restaura al final);
3. comprueba que las lecturas fueron exactamente `pairs.parquet`, `labels.parquet` y `students.parquet`, que ningún archivo oculto se leyó, que las dos salidas son **idénticas** (`assert_frame_equal`/`assert_series_equal` en X, y, grupo, `room_id`, partición y metadatos, e igualdad de excluidos) y que el código de `training_data.py` no contiene `HIDDEN_`, `labels_hidden` ni los dos nombres de archivo.

`test_model_inputs_are_exactly_feature_order_without_utility_or_weights` y su versión unitaria comprueban que las columnas de X son exactamente `feature_order` (10, en orden), sin `utility` ni `weight_*`.

### Resultado con los datos de ML-03 y ML-04 (DATOS SINTÉTICOS)

`load_training_data(ProjectPaths(root=.))` sobre `data/processed/` (fuera de git), ~2 s:

| Partición | Pares | Estudiantes | Excluidos (< 2 candidatos) |
| --- | ---: | ---: | ---: |
| `train` | 135,992 | 1,399 | 1 (`S1078`, 1 candidato) |
| `validation` | 29,857 | 300 | 0 |
| `test` | 30,755 | 300 | 0 |
| **Total** | **196,604** | **1,999** | **1** |

- Coincide con ML-03/ML-04: 196,605 pares y 2,000 estudiantes menos el par y el estudiante `S1078` (M-ML-035, herencia 5); train 1,400 → 1,399 estudiantes.
- Relevancia de los pares cargados: 3 · 2 · 1 · 0 = 20,361 · 19,588 · 19,743 · 136,912 (M-ML-032 menos la relevancia 3 de `S1078`).
- Columnas de X: `distance_km`, `effective_distance_km`, `price_to_budget`, `services_coverage`, `place_type_match`, `property_type_match`, `pets_compatible`, `distance_ratio`, `price_below_min`, `extra_services_ratio` (el `feature_order` del esquema 2.0.0).

### Compatibilidad Windows y Linux

| Sistema | Comprobación | Resultado |
| --- | --- | --- |
| Windows 11 (local, `win_amd64`) | Importación y ajuste real de LightGBM y scikit-learn (`test_ml_dependencies.py`, 4 casos) | **PASS** |
| Linux (`manylinux_2_28_x86_64`) | Wheel de `lightgbm` 4.7.0 de `uv.lock` descargado a un directorio vacío del scratchpad, SHA-256 verificado contra `uv.lock`; sección dinámica ELF de `lightgbm/lib/lib_lightgbm.so` leída sin extraer: `libdl.so.2`, `libstdc++.so.6`, `libm.so.6`, **`libgomp.so.1`**, `libgcc_s.so.1`, `libpthread.so.0`, `libc.so.6`, `ld-linux-x86-64.so.2`. El wheel **no** incluye `libgomp` (no hay carpeta `*.libs`) | Análisis estático hecho; **ejecución en Linux no realizada** (Docker no estaba activo y no se arrancó). Ver T01-H1 |
| Linux (CI `ubuntu-latest`) | `test_ml_dependencies.py` corre en el job `quality` | **PASS** en el PR #31 (4 checks en verde, según el responsable; T01-H1 resuelto el 2026-11-24) |

`scikit-learn` y `scipy` traen sus bibliotecas nativas dentro del wheel (`manylinux`); la única dependencia de sistema identificada es `libgomp.so.1` de LightGBM (OpenMP de GCC).

### Trazabilidad R01 → prueba

| Criterio | Prueba o comando |
| --- | --- |
| `scikit-learn` y `lightgbm` en `pyproject.toml` y `uv.lock` | `test_ml_dependencies.py::test_dependencies_are_declared_in_pyproject`, `::test_installed_versions_are_the_locked_ones`; `uv lock --check` |
| pip-audit sin vulnerabilidades | `uv run pip-audit` |
| Une `pairs.parquet` y `labels.parquet` | `test_training_data.py::test_join_aligns_each_pair_with_its_label_even_if_labels_come_shuffled`, `::test_pair_without_label_is_an_error`, `::test_split_mismatch_between_pairs_and_labels_is_an_error`; `test_training_data_stage.py::test_every_candidate_pair_is_loaded_except_students_with_one_candidate` |
| Valida `synthetic = true` | `::test_non_synthetic_rows_are_an_error[pairs]`, `[labels]` |
| Valida `labels_params_version` 1.0.0 | `::test_changed_labels_params_version_is_an_error`, `::test_missing_label_column_is_a_clear_error[labels_params_version]` |
| Valida esquema 2.0.0 y las 10 columnas de `feature_order` | `::test_changed_feature_schema_version_is_an_error`, `::test_missing_pair_column_is_a_clear_error[*]`, `::test_features_out_of_schema_order_is_an_error`, `::test_inputs_are_exactly_feature_order_without_hidden_or_extra_columns` |
| Excluye y cuenta < 2 candidatos | `::test_students_with_fewer_than_two_candidates_are_excluded_and_counted`; integración `::test_every_candidate_pair_is_loaded_except_students_with_one_candidate`; datos reales (tabla anterior) |
| Metadatos de segmento | `::test_student_metadata_for_segments`, `::test_load_reads_only_budget_from_students_and_matches_the_pure_build` |
| **Nunca lee los archivos ocultos** | `test_training_data_stage.py::test_loader_never_reads_the_hidden_label_files`, `::test_model_inputs_are_exactly_feature_order_without_utility_or_weights`, `test_training_data.py::test_only_training_inputs_can_be_read` |
| Sin generación de etiquetas | `training_data.py` no importa el simulador (`synthetic.labels`) ni escribe archivos; solo lee |

### Comandos y resultados (T01)

| Comando | Resultado |
| --- | --- |
| `uv add scikit-learn lightgbm` | `Resolved 55 packages`; instala `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `joblib==1.6.0`, `threadpoolctl==3.7.0`, `narwhals==2.27.1`, `cloudpickle==3.1.2` |
| `uv lock --check` | `Resolved 55 packages in 1ms`, código 0 (PASS) |
| `uv run pip-audit` | `No known vulnerabilities found` (solo omite el paquete local `thehouse-ml`, que no está en PyPI) (PASS) |
| `uv run ruff format --check .` | `142 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS; repetido tras la protección sin pickle: `142 files already formatted`, `All checks passed!`) |
| `uv run mypy src` | `Success: no issues found in 40 source files` (PASS) |
| `uv run pytest tests/unit/test_training_data.py tests/integration/test_training_data_stage.py tests/unit/test_ml_dependencies.py` (afectadas) | **50 passed** en 17.6 s (43 + 3 + 4); `models/training_data.py` **100 %** de líneas y ramas (PASS) |
| `uv run pytest tests/test_repo_hygiene.py tests/unit/test_training_data.py tests/integration/test_training_data_stage.py tests/unit/test_ml_dependencies.py` (afectadas tras la protección sin pickle) | **90 passed** en 10.1 s (PASS) |
| `uv run ruff format --check .` y `uv run ruff check .` (tras agregar `read_pickle`/`to_pickle`) | `142 files already formatted`, `All checks passed!` (PASS) |
| `uv run pytest tests/test_repo_hygiene.py` (tras agregar `read_pickle`/`to_pickle`) | **50 passed** en 0.7 s; ningún módulo de `src/thehouse_ml/` usa `read_pickle` ni `to_pickle` (PASS) |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing` (suite completa) | **1728 passed** en 11 min 48 s (Windows); cobertura total **96 %**; `models/training_data.py` **100 %** (PASS) |

### Tabla de coherencia (T01)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §5 | `scikit-learn` y LightGBM están en la tabla de stack; ninguna dependencia directa fuera de ella | PASS |
| Constitución §9 y §14.2 | Versiones fijadas en `uv.lock` dentro de `exclude-newer`; `uv lock --check` y pip-audit limpios | PASS |
| Constitución §13.1 | Sin pickle: el cargador no persiste nada; la prueba de humo guarda LightGBM solo como texto nativo; `cloudpickle` (transitiva) no se usa; `test_repo_hygiene.py::test_no_module_of_the_package_uses_pickle` lo impide en todo `src/thehouse_ml/` | PASS |
| Constitución §8 y §20.3 | Solo datos sintéticos; el resumen de esta evidencia está marcado «DATOS SINTÉTICOS» | PASS |
| Constitución §11.2 y spec §7 regla 4 | X = las 10 features de `feature_order`; sin ids, campus, presupuesto ni atributos sensibles | PASS |
| Spec §6.1 y decisión 1 (M-ML-036) | Única fuente de etiquetas `labels.parquet`; archivos ocultos nunca leídos (prueba obligatoria); < 2 candidatos excluidos y contados | PASS |
| Decisión 6 (M-ML-036) | Dependencias en T01 con pip-audit limpio | PASS |
| M-ML-032 y M-ML-035 | Recuentos iguales a los publicados (196,605 pares; `S1078` excluido) | PASS |
| `specs/ARQUITECTURA.md` §4 | Módulo en `models/` (ML-05) | PASS |
| AGENTS: rutas permitidas | `pyproject.toml`, `uv.lock`, `src/thehouse_ml/models/training_data.py`, `tests/`, `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05); `feature_schema.json` y params de ML-03/ML-04 sin cambios | PASS |
| AGENTS: agregar dependencias | Constitución §5 (tabla de stack) y decisión 6 de M-ML-036 | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La ejecución en Linux no se hizo: queda como advertencia (T01-H1), no como PASS | PASS |

### Hallazgos (T01)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T01-H1 | `lib_lightgbm.so` del wheel de Linux necesita `libgomp.so.1` del sistema. `ubuntu-latest` incluye GCC (y con él `libgomp1`), pero no se ha ejecutado en el runner. **Cambio propuesto al CI** (no aplicado; `ci.yml` fuera de las rutas): en el job `quality`, antes de «Entorno reproducible (uv.lock)», un paso `run: sudo apt-get update && sudo apt-get install -y --no-install-recommends libgomp1`. El job `windows-latest` no lo necesita (verificado localmente) y `dependency-audit` no importa LightGBM. | ADVERTENCIA → resuelta | **RESUELTO** (2026-11-24): el PR #31 pasó los 4 checks del CI, incluida la prueba de humo de LightGBM (`test_ml_dependencies.py`) en el job `quality` de `ubuntu-latest`, sin el paso `libgomp1` (según el responsable; `gh` no está instalado en este equipo para verificarlo de forma independiente, y `git log` muestra el merge `386fd1d`). `ci.yml` sin cambios |
| T01-H7 | Protección sin pickle en `tests/test_repo_hygiene.py`, pedida por el responsable al aprobar M-ML-037; ampliada después, también a pedido del responsable, a `read_pickle` y `to_pickle` de pandas. | INFORMATIVO | Implementada (M-ML-037 decisión 9, APROBADO; la ampliación no está en el texto de M-ML-037) |
| T01-H2 | R01 nombra `pairs.parquet` y `labels.parquet`; el cargador lee también `students.parquet` (solo `student_id` y `budget.max`) porque los segmentos de presupuesto lo necesitan y `pairs.parquet` no lo trae. Pedido en el «EJECUTA». | INFORMATIVO | M-ML-037 decisión 4 (APROBADO) |
| T01-H3 | `pairs.parquet` no guarda la versión del esquema: se valida el esquema empaquetado (2.0.0) más el nombre y el orden de las 10 columnas. | INFORMATIVO | M-ML-037 decisión 5 (APROBADO) |
| T01-H4 | `cloudpickle` entra como transitiva de `joblib`. No se usa; la persistencia de ML-05 es JSON y texto nativo de LightGBM (constitución §13.1). | INFORMATIVO | M-ML-037 decisión 1 (APROBADO) |
| T01-H5 | Override de mypy para `sklearn` (sin `py.typed`). | INFORMATIVO | M-ML-037 decisión 2 (APROBADO) |
| T01-H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T01)

- **Seguridad:** sin secretos, sin red en las pruebas, sin servicios de pago (COST-001); la única descarga fue el wheel de LightGBM para Linux (fijado en `uv.lock`, SHA-256 verificado) a un directorio vacío del scratchpad, analizado sin ejecutar su código.
- **Datos y licencias:** solo datos sintéticos de `data/processed/` (fuera de git); ningún dato personal en salidas; licencias de las dependencias nuevas BSD-3-Clause y MIT.
- **Coherencia:** tabla anterior.

**Resultado T01: PASS WITH WARNINGS** — aceptación de R01 cumplida y verificada; advertencia T01-H1 (ejecución en Linux pendiente del CI); M-ML-037 APROBADO (César Yael López Trujillo, 2026-11-23).

**Actualización (2026-11-24):** T01-H1 queda **RESUELTO** — el PR #31 pasó los 4 checks, incluida la prueba de humo de LightGBM en el job de ubuntu (`libgomp.so.1` presente en `ubuntu-latest`); no hace falta cambiar `ci.yml`. Con eso T01 no tiene advertencias abiertas.

## T02 · Validación cruzada y entrenamiento reproducible

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§9, §10, §13) → `AGENTS.md` → `memoria.md` (M-ML-035 a M-ML-037) → `spec.md` (R02, §6 y §10 decisiones 2, 3 y 5), `plan.md`, `task.md` y esta evidencia → `specs/ARQUITECTURA.md` → `src/thehouse_ml/models/training_data.py`, `evaluation/ranking.py` y `models/baseline_*` (para reutilizar).

La ejecución se interrumpió por un apagado del equipo (2026-11-23, después de las dos corridas completas, mientras se revisaba el empate de la regresión logística). Al retomarla (2026-11-24) se comprobó con `git status`, las fechas de modificación y `sha256sum` que el código en disco es el que produjo las salidas (último cambio de código 22:52; corridas 23:02–23:15) y que los 5 archivos de salida conservan los SHA-256 de las dos corridas. No se volvió a correr `train` completo.

### Qué se hizo

1. **`src/thehouse_ml/models/train_params.json` 1.0.0** (en git; `metadata.decision = "M-ML-038 (APROBADO)"`): 21 parámetros (20 de la rejilla y la reproducibilidad más `profiles.ci`, agregado al aprobar), cada uno con `value` y `basis` («calibrado: …» o «SUPUESTO documentado»). El umbral `relevance ≥ 2` y el mínimo de 2 candidatos no se repiten: se leen de `baseline_eval_params.json` (M-ML-034).

   | Parámetro | Valor | Base |
   | --- | --- | --- |
   | `seed` | 20261124 | SUPUESTO (spec §6.3) |
   | `folds.n_splits` · `folds.splits` · `folds.stratify_by` | 5 · train + validation · `university_code` | calibrado (spec §6.2) |
   | `selection.metric` | NDCG@10 medio; desempate: menor desviación muestral, después la primera de la rejilla | calibrado (spec §6.3) |
   | `logreg.C` | 0.01, 0.1, 1, 10 | SUPUESTO (spec §6.3) |
   | `logreg.transforms` | `standard` (`StandardScaler`), `quantile_bins` (`KBinsDiscretizer`, 5 cuantiles, one-hot) | SUPUESTO (spec §6.3) |
   | `logreg.solver` · `max_iter` · `tol` | lbfgs · 2000 · 1e-4 | SUPUESTO |
   | `logreg.clip` | `training_range` (o `range`) del esquema 2.0.0 | calibrado (spec §6.6, decisión 5) |
   | `lightgbm.num_leaves` · `learning_rate` · `n_estimators` | 15, 31 · 0.05, 0.1 · 100, 300 | SUPUESTO (spec §6.3) |
   | `lightgbm.min_data_in_leaf` | 50 (fijo) | SUPUESTO (spec §6.3) |
   | `lightgbm.label_gain` | 0, 1, 3, 7 (2^rel − 1) | calibrado (M-ML-034) |
   | `lightgbm.monotone_decreasing` | `price_to_budget`, `distance_km`, `effective_distance_km`, `distance_ratio` | calibrado (decisión 3) |
   | `lightgbm.num_threads` · `deterministic` | 1 · true (con `force_row_wise`) | SUPUESTO · calibrado (spec §6.3) |
   | `profiles.ci` | 2 pliegues; LR `standard` C=0.01; LightGBM 15 hojas, 0.05, 300 árboles; salidas en `models/ci` | calibrado (M-ML-038 decisión 10, T02-H1) |

2. **`src/thehouse_ml/models/train_config.py`**: lectura y validación de `train_params.json` (`TrainParamsError` con todos los problemas: base sin prefijo, parámetros desconocidos, test en los pliegues, `num_threads` ≠ 1, `deterministic` falso, `label_gain` distinto de 2^rel − 1, etc.) y rejillas en orden (la más simple primero).
3. **`src/thehouse_ml/models/train.py`** (etapa `train`):
   - `assign_folds`: `StratifiedGroupKFold(5, shuffle=True, random_state=20261124)` sobre **una fila por estudiante** de train + validación (grupo `student_id`, estrato campus). `build_cv_data` descarta todos los pares de test antes de cualquier ajuste.
   - `cross_validate`: por pliegue, ajusta con los otros 4 y puntúa solo los pares retenidos; `ndcg_by_student` ordena por puntaje descendente y `room_id` ascendente y usa `ndcg_at_k` de `evaluation/ranking.py` (mismas definiciones que el baseline; una prueba lo compara con `rank`).
   - Regresión logística pointwise (`relevance ≥ 2`): `np.clip` al `training_range` → `Pipeline(StandardScaler | KBinsDiscretizer, LogisticRegression)`. `LogRegModel` guarda recorte, transformación (media y escala, o cortes internos), coeficientes e intercepto en JSON y reproduce `Pipeline.decision_function` **bit a bit** (`assert_array_equal`).
   - LightGBM LambdaRank con `lightgbm.train`, grupos contiguos por estudiante, las 4 restricciones −1, un hilo, `deterministic` y `force_row_wise`; se guarda con `Booster.save_model` (texto nativo).
   - Baseline por pliegue con `score_features` y `rank` de ML-04 (sin entrenar), sobre los mismos estudiantes retenidos.
   - Modelos finales con los hiperparámetros elegidos sobre todo train + validación; verificación monotónica del LightGBM final (200 pares equiespaciados × 21 valores por feature restringida).
   - Salidas: `models/logreg.json`, `models/lightgbm_lambdarank.txt` y `models/train_manifest.json` (SHA-256 y bytes de los modelos, SHA-256 de las entradas y asignación estudiante → pliegue), fuera de git; `reports/training_report.md` y `reports/train_cv_metrics.json` en git, deterministas (sin fechas, tiempos ni rutas absolutas) y marcados «DATOS SINTÉTICOS».
4. **`src/thehouse_ml/models/train_report.py`**: render del reporte y del JSON de métricas.
5. **`src/thehouse_ml/stages.py`**: `train` llama a `run_train` (importación diferida de scikit-learn y LightGBM); sin `data/processed/` devuelve el mensaje de omisión, como `evaluate`. Descripción de la etapa corregida («el modelo base no se entrena»).
6. **Perfil de entrenamiento `ci`** (M-ML-038 decisión 10, pedido al aprobar por T02-H1): `THEHOUSE_ML_TRAIN_PROFILE` = `full` (por defecto; vacía o ausente) o `ci`; otro valor es un error. `TrainParams.for_profile("ci")` reduce la rejilla a las combinaciones de `profiles.ci` (validadas contra la rejilla al leer el archivo) y a 2 pliegues, con la misma semilla, datos y reglas; `output_dirs` manda modelos, manifiesto y los dos reportes a `models/ci/` (fuera de git; el parser exige una subcarpeta relativa de `models/`). El perfil queda en el manifiesto, en el JSON de métricas, en el reporte y en el mensaje de la etapa (`train (perfil ci): …`). **`.github/workflows/ci.yml`**: solo un `env` con `THEHOUSE_ML_TRAIN_PROFILE: ci` (y un comentario) en el paso «CLI con las cinco etapas» del job `quality`; sin cambios en nombres de jobs, timeouts ni acciones.
7. **Pruebas** (sin red):
   - `tests/unit/test_train_config.py` (39): valores de la spec y decisión APROBADO, base documentada en cada parámetro, rejillas de 8 y 8 con ids únicos, restricciones en el orden de `feature_order`, 17 valores inválidos y, del perfil: **el perfil por defecto es el completo** (`full`, 8 + 8 combinaciones y 5 pliegues), el perfil `ci` (1 + 1 combinaciones, 2 pliegues, mismas reglas), sus combinaciones = las elegidas en `reports/train_cv_metrics.json` versionado (perfil `full`), perfiles desconocidos rechazados y 11 `profiles.ci` inválidos (combinación fuera de la rejilla, más pliegues que la rejilla, salida en `reports`, en `models` mismo, con `..` o absoluta, estructura incompleta).
   - `tests/unit/test_train.py` (18, datos construidos en la prueba): pliegues sin estudiantes compartidos y sin test, estratificados y reproducibles (otra semilla cambia la asignación); `build_cv_data` sin pares de test; grupos contiguos; **la búsqueda solo usa los pliegues** (ajuste y retenidos disjuntos; la unión es train + validación); **test no influye**: con las features y la relevancia de test alteradas, rejillas, baseline y modelos son idénticos; desempate por `room_id` igual que `rank`; regla de selección; **LR reconstruida desde su JSON = mismas predicciones** (las dos transformaciones); recorte efectivo; pipeline solo con `StandardScaler`, `KBinsDiscretizer` y `LogisticRegression`; JSON de formato desconocido rechazado; **restricciones monotónicas efectivas** (0 violaciones con restricciones en datos donde la relevancia sube con el precio; > 0 sin ellas, para mostrar que la verificación detecta); **reproducibilidad** (JSON y texto del modelo idénticos en dos ajustes).
   - `tests/integration/test_train_stage.py` (7, proyecto temporal de 2 campus y rejilla pequeña): **el perfil `ci` no modifica los reportes versionados ni los modelos oficiales** (la etapa con `THEHOUSE_ML_TRAIN_PROFILE=ci` deja idénticos los SHA-256 de `reports/` y `models/` y escribe los 5 archivos en `models/ci/` con 2 pliegues y 1 combinación por modelo); **sin la variable, el perfil es `full`** y escribe `reports/`; además, dos corridas de la etapa con los **mismos SHA-256** de los 3 archivos de `models/` y los 2 reportes; **prueba obligatoria de la decisión 1** (espía `pd.read_parquet`, borra los dos archivos ocultos: lecturas = `TRAINING_INPUTS` y salidas idénticas; los módulos `train*.py` no nombran los archivos ocultos); modelos JSON y texto nativo que se recargan sin pickle; test fuera de los pliegues y de los modelos finales; omisión sin entradas.
   - `tests/test_ci_workflow.py` (4, +1): el paso «CLI con las cinco etapas» del job `quality` pone `THEHOUSE_ML_TRAIN_PROFILE: ci` y el job conserva `timeout-minutes: 15`; las 3 pruebas previas (acciones fijadas a SHA completo, nombres de los jobs del ruleset, caché de `data/raw/`) siguen en verde.
   - Actualizadas porque `train` dejó de ser una etapa vacía: `tests/unit/test_stages.py`, `tests/unit/test_cli.py` (nueva prueba de omisión de `train`) y `tests/integration/test_installed_package.py` (el script de consola se prueba con `export`).
   - `tests/test_repo_hygiene.py::test_no_module_of_the_package_uses_pickle` cubre los módulos nuevos sin cambios.

### Resultado con los datos de ML-03 y ML-04 (DATOS SINTÉTICOS)

Pliegues: **1,699 estudiantes** y **165,849 pares** de train + validación (`S1078` excluido por tener 1 candidato); test no participa. Cada pliegue tiene 34 estudiantes por campus (P3: 33 de TESJI): 340 · 340 · 340 · 339 · 340 estudiantes y 32,234 · 33,999 · 36,237 · 33,014 · 30,365 pares.

**Rejilla — regresión logística** (NDCG@10 medio de los estudiantes retenidos):

| Configuración | P0 | P1 | P2 | P3 | P4 | Media | Desv. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard`, C=0.01 **(elegida)** | 0.8664 | 0.8570 | 0.8616 | 0.8626 | 0.8640 | 0.862325 | 0.0035 |
| `standard`, C=0.1 | 0.8671 | 0.8576 | 0.8622 | 0.8617 | 0.8629 | 0.862293 | 0.0034 |
| `standard`, C=1 | 0.8667 | 0.8579 | 0.8619 | 0.8612 | 0.8629 | 0.862113 | 0.0032 |
| `standard`, C=10 | 0.8667 | 0.8576 | 0.8618 | 0.8612 | 0.8631 | 0.862067 | 0.0033 |
| `quantile_bins`, C=0.01 | 0.7145 | 0.7084 | 0.7069 | 0.7098 | 0.7158 | 0.711095 | 0.0039 |
| `quantile_bins`, C=0.1 | 0.7124 | 0.7061 | 0.7047 | 0.7084 | 0.7146 | 0.709257 | 0.0042 |
| `quantile_bins`, C=1 | 0.7125 | 0.7058 | 0.7048 | 0.7082 | 0.7153 | 0.709326 | 0.0045 |
| `quantile_bins`, C=10 | 0.7125 | 0.7059 | 0.7047 | 0.7086 | 0.7154 | 0.709415 | 0.0045 |

**Rejilla — LightGBM LambdaRank:**

| Configuración | P0 | P1 | P2 | P3 | P4 | Media | Desv. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leaves 15, lr 0.05, 100 árboles | 0.8658 | 0.8522 | 0.8556 | 0.8571 | 0.8593 | 0.858012 | 0.0051 |
| leaves 15, lr 0.05, 300 árboles **(elegida)** | 0.8733 | 0.8597 | 0.8651 | 0.8661 | 0.8684 | 0.866530 | 0.0049 |
| leaves 15, lr 0.1, 100 árboles | 0.8741 | 0.8576 | 0.8626 | 0.8635 | 0.8681 | 0.865178 | 0.0062 |
| leaves 15, lr 0.1, 300 árboles | 0.8725 | 0.8582 | 0.8637 | 0.8635 | 0.8679 | 0.865158 | 0.0053 |
| leaves 31, lr 0.05, 100 árboles | 0.8682 | 0.8559 | 0.8593 | 0.8602 | 0.8606 | 0.860853 | 0.0045 |
| leaves 31, lr 0.05, 300 árboles | 0.8726 | 0.8586 | 0.8638 | 0.8640 | 0.8671 | 0.865209 | 0.0051 |
| leaves 31, lr 0.1, 100 árboles | 0.8725 | 0.8605 | 0.8638 | 0.8643 | 0.8675 | 0.865719 | 0.0045 |
| leaves 31, lr 0.1, 300 árboles | 0.8702 | 0.8576 | 0.8604 | 0.8625 | 0.8662 | 0.863360 | 0.0049 |

**Comparación por pliegue con el baseline** (mismos estudiantes retenidos; insumo de la condición 1 de la §6.5, que decide T03):

| Pliegue | Estudiantes | Baseline | Regresión logística | Δ | LightGBM | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 340 | 0.5396 | 0.8664 | +0.3267 | 0.8733 | +0.3336 |
| P1 | 340 | 0.5470 | 0.8570 | +0.3100 | 0.8597 | +0.3128 |
| P2 | 340 | 0.5453 | 0.8616 | +0.3163 | 0.8651 | +0.3198 |
| P3 | 339 | 0.5241 | 0.8626 | +0.3386 | 0.8661 | +0.3420 |
| P4 | 340 | 0.5345 | 0.8640 | +0.3295 | 0.8684 | +0.3339 |
| **Media ± desv.** | | 0.5381 ± 0.0093 | 0.8623 ± 0.0035 | +0.3242 | 0.8665 ± 0.0049 | +0.3284 |

Los dos modelos superan al baseline en los 5 pliegues. Es **sintético**: el modelo aprende la utilidad oculta del simulador (circularidad, ML-05-RK01); no es desempeño real, y la decisión de promoción es de T03.

**Hiperparámetros elegidos:** regresión logística `standard`, **C = 0.01** (NDCG@10 0.8623 ± 0.0035); LightGBM **`num_leaves` 15, `learning_rate` 0.05, `n_estimators` 300** (0.8665 ± 0.0049). Avisos de no convergencia: 0 en la búsqueda y 0 en el modelo final. Verificación monotónica del LightGBM final: **0 violaciones** en las 4 features restringidas (200 pares × 21 valores).

**Empate aparente C = 0.01 contra C = 0.1** (revisado al retomar): con 4 decimales ambos muestran 0.8623; con precisión completa las medias son 0.8623253678 y 0.8622928605, una diferencia de **3.25 × 10⁻⁵** (C = 0.1 gana en P0–P2 y C = 0.01 en P3–P4). **No es un empate exacto**: la regla registrada (mayor media) elige C = 0.01 y el desempate por desviación no entra. La diferencia es ~100 veces menor que la desviación entre pliegues (0.0035), así que las 4 configuraciones `standard` son prácticamente equivalentes (rango 2.6 × 10⁻⁴); C = 0.01 es además la más regularizada, la que elegiría también un criterio de equivalencia con preferencia por la más simple. La regla no se cambió después de ver los resultados (sería ajustar la selección a los datos). Ver T02-H3.

**Modelos finales** (1,699 estudiantes, 165,849 pares; fuera de git):

| Archivo | Bytes | SHA-256 |
| --- | ---: | --- |
| `models/logreg.json` | 1,853 | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` |
| `models/lightgbm_lambdarank.txt` | 543,515 | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` |
| `models/train_manifest.json` | 28,148 | `39e0b2b1f863746bf3f3d8ad76176216ae1b7432a174b83a626d75eca675c7dc` |

Coeficientes de la regresión logística final (sobre features estandarizadas; intercepto −4.133224): `distance_km` −3.350, `effective_distance_km` −1.870, `price_to_budget` −0.996, `services_coverage` +0.566, `place_type_match` +1.006, `property_type_match` +0.360, `pets_compatible` 0 (constante, escala 1), `distance_ratio` −0.101, `price_below_min` −0.012, `extra_services_ratio` +0.439.

### Reproducibilidad: dos corridas completas

`uv run python -m thehouse_ml train` dos veces seguidas en Windows 11 (CPU local, un hilo para LightGBM):

| Corrida | Duración | Salida |
| --- | ---: | --- |
| 1 | **536 s (8 min 56 s)** | `logreg:standard:C=0.01 NDCG@10 0.862, lightgbm:leaves=15:lr=0.05:trees=300 NDCG@10 0.867 (DATOS SINTÉTICOS)` |
| 2 | **801 s (13 min 21 s)** | idéntica |

| Archivo | SHA-256 (corrida 1 = corrida 2) |
| --- | --- |
| `models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` |
| `models/lightgbm_lambdarank.txt` | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` |
| `models/train_manifest.json` | `39e0b2b1f863746bf3f3d8ad76176216ae1b7432a174b83a626d75eca675c7dc` |
| `reports/training_report.md` | `6164aca2af2581bc1b1fe69c59bb12b4e6976c40de0960ab41bf16f2e43175c8` |
| `reports/train_cv_metrics.json` | `8c85967a571b5d3a2623812b50239adb289e134c4e249c44483173f4032b5b7f` |

`diff` de los dos listados de SHA-256: **idénticos**. La diferencia de duración es del equipo (mismo código y mismos datos); domina el ajuste de LightGBM (≈ 9–30 s por ajuste; 40 ajustes de la rejilla más el final).

### Corrida oficial con M-ML-038 APROBADO y perfil `ci` (2026-11-24)

Tras aprobar M-ML-038 se corrió una vez el perfil `ci` (para medirlo) y una vez la corrida completa oficial para regenerar los reportes (decisión `APROBADO` y perfil `full`):

| Comando | Duración | Resultado |
| --- | ---: | --- |
| `THEHOUSE_ML_TRAIN_PROFILE=ci uv run python -m thehouse_ml train` | **41 s** | `train (perfil ci): 1699 estudiantes de train+validation en 2 pliegues; logreg:standard:C=0.01 NDCG@10 0.862, lightgbm:leaves=15:lr=0.05:trees=300 NDCG@10 0.865 (DATOS SINTÉTICOS)`; los SHA-256 de `models/*` oficiales y de `reports/` antes y después son idénticos (`diff` vacío): **no tocó las salidas oficiales** |
| `uv run python -m thehouse_ml train` (perfil `full`) | **499 s (8 min 19 s)** | `train (perfil full): … en 5 pliegues; logreg:standard:C=0.01 NDCG@10 0.862, lightgbm:leaves=15:lr=0.05:trees=300 NDCG@10 0.867 (DATOS SINTÉTICOS)` |

SHA-256 después de la corrida oficial:

| Archivo | SHA-256 | Cambio |
| --- | --- | --- |
| `models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` | igual |
| `models/lightgbm_lambdarank.txt` | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` | igual |
| `models/train_manifest.json` | `040c82b428ac9bc507f81bdafd3d46e40ac7ce4899600d6ac8bf7453a148009a` | nuevo campo `profile` |
| `reports/training_report.md` | `5d49f7fbd26d06aa0510937d135fd8a2228aa91fb4e4ca9daa4cc4f3e883690b` | `M-ML-038 (APROBADO)` y línea de perfil |
| `reports/train_cv_metrics.json` | `288d3d9409f2deac0c268eaaa0ba152dbb6b6dba500dc2516a5205d0fd82b932` | `train_params_decision` y `profile` |

Las métricas no cambian (mismos NDCG@10 por pliegue y configuración; LR 0.8623253677677383 y LightGBM 0.8665295318253534 de media; baseline igual). Perfil `ci` (DATOS SINTÉTICOS, 2 pliegues, solo prueba de humo): LR 0.8639 · 0.8607, LightGBM 0.8665 · 0.8632, baseline 0.5363 · 0.5399. Sus modelos finales (`models/ci/logreg.json`, `models/ci/lightgbm_lambdarank.txt`) tienen **los mismos SHA-256** que los oficiales: el modelo final se entrena con la misma configuración sobre todo train + validación y no depende de los pliegues. `models/ci/` está ignorado por git (`.gitignore:4:/models/`).

### Trazabilidad R02 → prueba

| Criterio | Prueba o comando |
| --- | --- |
| 5 pliegues por estudiante y estratificados por campus sobre train + validación | `test_train.py::test_folds_have_no_shared_students_and_never_include_test`, `::test_folds_are_stratified_by_campus_and_reproducible`; corrida completa (recuento por pliegue) |
| Sin estudiantes compartidos; recuento por pliegue y campus | `::test_folds_have_no_shared_students_and_never_include_test`, `::test_cv_data_drops_every_test_pair_and_keeps_pairs_contiguous`; `reports/training_report.md` §Pliegues |
| Test nunca aparece | `::test_training_ignores_the_test_partition_entirely`; `test_train_stage.py::test_test_partition_is_not_in_folds_or_final_models` |
| Rejilla solo con los pliegues | `test_train.py::test_cross_validation_fits_on_other_folds_and_scores_only_held_out_students`; `test_train_config.py::test_invalid_values_are_rejected` (test en `folds.splits`) |
| Modelo final en train + validación | `test_train.py::test_final_models_are_reproducible_and_monotone`; `test_train_stage.py::test_test_partition_is_not_in_folds_or_final_models` |
| 4 restricciones monotónicas efectivas | `test_train.py::test_lightgbm_monotone_constraints_are_effective`; verificación del modelo final (0 violaciones) |
| LR reconstruida desde JSON con las mismas predicciones | `test_train.py::test_logreg_rebuilt_from_json_gives_the_same_predictions[standard]`, `[quantile_bins]` |
| Mismos SHA-256 en dos corridas | `test_train_stage.py::test_two_runs_produce_identical_models_and_reports`; dos corridas completas (tabla anterior) |
| Sin pickle | `test_repo_hygiene.py::test_no_module_of_the_package_uses_pickle`; `test_train_stage.py::test_models_are_json_and_native_text_and_reload_without_pickle` |
| Nunca lee los archivos ocultos (decisión 1) | `test_train_stage.py::test_train_never_reads_the_hidden_label_files` |
| Parámetros documentados | `test_train_config.py` (39 casos) |
| Perfil `ci` sin tocar los reportes versionados; perfil por defecto = completo | `test_train_stage.py::test_ci_profile_does_not_modify_the_versioned_reports_or_official_models`, `::test_default_profile_is_full_and_writes_the_versioned_reports`; `test_train_config.py::test_default_profile_is_the_full_official_grid`, `::test_ci_profile_uses_one_combination_per_model_and_two_folds`, `::test_ci_profile_combinations_are_the_ones_chosen_by_the_official_run`; corrida real (41 s, `diff` vacío) |
| CI con el perfil `ci`, mismos jobs y timeouts | `test_ci_workflow.py` (4) |

### Comandos y resultados (T02)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `149 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 43 source files` (PASS) |
| `uv run pytest` de las afectadas (`test_train.py`, `test_train_config.py`, `test_train_stage.py`, `test_stages.py`, `test_installed_package.py`, `test_repo_hygiene.py`, `test_training_data.py`, `test_training_data_stage.py`, `test_ml_dependencies.py`, `test_cli.py`) con cobertura de los módulos nuevos | Primera pasada al retomar: 166 passed, **1 failed** (`test_cli.py::test_each_empty_stage_reports_its_spec_and_exits_zero[train]`, que seguía tratando `train` como etapa vacía y no se había corrido antes). Tras corregir la prueba (T02-H6): `test_cli.py` + `test_stages.py` **18 passed**. Cobertura: `train.py` 96 %, `train_config.py` 98 %, `train_report.py` 86 %, `stages.py` 96 % (95.3 % en conjunto) |
| `uv run python -m thehouse_ml train` × 2 | 536 s y 801 s; SHA-256 idénticos (tabla anterior) (PASS) |
| Tras la aprobación: `THEHOUSE_ML_TRAIN_PROFILE=ci … train` y `… train` (oficial) | 41 s y 499 s (tabla anterior) (PASS) |
| Tras la aprobación: `uv run pytest` de las afectadas (`test_train_stage.py`, `test_train.py`, `test_stages.py`, `test_cli.py`, `test_ci_workflow.py`, `test_repo_hygiene.py`) | **97 passed** en 30.8 s; `test_train_config.py` **39 passed** tras regenerar el reporte oficial (antes fallaba solo `test_ci_profile_combinations_are_the_ones_chosen_by_the_official_run`, que necesita el campo `profile` del reporte regenerado) (PASS) |
| Tras la aprobación: `ruff format --check .`, `ruff check .`, `mypy src` y `uv run pytest --cov=thehouse_ml` (suite completa) | `149 files already formatted`, `All checks passed!`, `Success: no issues found in 43 source files`; **1826 passed** en 984.9 s (16 min 25 s, Windows); cobertura total **96 %**: `train.py` 96 %, `train_config.py` 98 %, `train_report.py` 86 %, `stages.py` 100 % (PASS) |
| `uv run pytest --cov=thehouse_ml` (suite completa, antes de la aprobación) | **1808 passed** en 795.9 s (13 min 15 s, Windows), incluida la corrección de `test_cli.py`; cobertura total **96 %** (`stages.py` 100 %, `training_data.py` 100 %, `train_report.py` 86 %) (PASS) |

### Tabla de coherencia (T02)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §9 | Semilla fija, un comando (`train`), mismas salidas byte a byte en dos corridas; los modelos registran versión de datos (SHA-256 de las entradas), hiperparámetros y métricas. El commit no se registra en los archivos del modelo (T02-H8) | PASS |
| Constitución §10.1–§10.3 | Comparación con el baseline por pliegue; partición por estudiante sin fuga con validación cruzada; media ± desviación y valor de cada pliegue | PASS |
| Constitución §13.1 | Sin pickle: JSON y texto nativo de LightGBM; protección del repositorio en verde | PASS |
| Constitución §8 y §20.3 | «DATOS SINTÉTICOS» en el reporte, el JSON de métricas, los modelos (`synthetic: true`) y el mensaje de la etapa | PASS |
| Spec §6.1 y decisión 1 | Solo `TRAINING_INPUTS`; prueba obligatoria para `train` | PASS |
| Spec §6.2 y decisión 2 | 5 pliegues agrupados y estratificados en train + validación; test sin usar | PASS |
| Spec §6.3 y decisión 3 | Rejilla de la spec en `train_params.json`, LR pointwise `relevance ≥ 2`, LambdaRank con las 4 restricciones, un hilo y `deterministic` | PASS |
| Spec §6.6 y decisión 5 | Recorte al `training_range` antes de puntuar (fuera del `Pipeline`, guardado en el JSON; T02-H2, M-ML-038 decisión 5) | PASS |
| Spec §7 regla 4 y constitución §11.2 | Entradas = las 10 de `feature_order`; sin atributos sensibles | PASS |
| `specs/ARQUITECTURA.md` §3–§4 | `train` en `models/`; modelos en `models/` (fuera de git) | PASS |
| AGENTS: rutas permitidas | `src/thehouse_ml/models/` (4 archivos nuevos), `src/thehouse_ml/stages.py`, `tests/`, `reports/` (2 nuevos), `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05) y `.github/workflows/ci.yml` (solo el perfil `ci`, autorizado al aprobar); `evaluation/` y `features/` solo reutilizados; `feature_schema.json`, `baseline_params.json`, `baseline_eval_params.json` y params de ML-03/ML-04 sin cambios | PASS |
| AGENTS: sin dependencias nuevas | `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: cambios en GitHub / constitución §17 | `.github/workflows/ci.yml` solo con el `env` del perfil `ci` en el paso de la CLI, autorizado explícitamente; jobs, nombres, timeouts y acciones fijadas sin cambios (`test_ci_workflow.py`); el CI sigue corriendo las 5 etapas | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La conversión real con skl2onnx no se ejecutó (no está instalado): queda para T05/ML-06 (T02-H2); los checks del PR #31 no se verificaron con `gh` (no instalado; T01-H1) | PASS |

### Hallazgos (T02)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T02-H1 | El job `quality` del CI corre `prepare` con datos reales y después las 5 etapas; con `timeout-minutes: 15` y una corrida completa de 9–13 min, el CI habría fallado. **Solución aprobada:** perfil `ci` (`THEHOUSE_ML_TRAIN_PROFILE=ci` en el paso de la CLI; 1 combinación por modelo, 2 pliegues, salidas en `models/ci/`): **41 s** en este equipo, sin tocar los reportes versionados. La corrida oficial sigue siendo el perfil `full`, que se corre a mano y cuyos reportes se versionan. El tiempo en el runner de ubuntu se confirma en el CI del PR. | ADVERTENCIA → resuelta | **RESUELTO** (M-ML-038 decisión 10, 2026-11-24) |
| T02-H2 | El recorte al `training_range` va antes del `Pipeline` (`np.clip`) y se guarda en el JSON, porque scikit-learn no tiene un transformador de recorte con convertidor en skl2onnx; en ML-06 se exporta como nodo `Clip` de ONNX. La prueba de T02 verifica las clases del pipeline (`StandardScaler`, `KBinsDiscretizer`, `LogisticRegression`), no la conversión real: `skl2onnx` no está instalado (ML-06). | INFORMATIVO (aprobado en M-ML-038 decisión 5; la verificación corresponde por spec a T05 y ML-06) | Para T05 (lista verificada) y ML-06 (conversión) |
| T02-H3 | Empate aparente C = 0.01 contra C = 0.1 (0.8623 con 4 decimales; diferencia real 3.25 × 10⁻⁵): se aplica la regla registrada sin cambiarla. El reporte muestra 4 decimales y puede leerse como empate; el JSON trae la precisión completa. Mostrar 6 decimales exigiría regenerar el reporte (una corrida de ~9–13 min); no se hizo porque no cambia ningún resultado. | INFORMATIVO | Cerrado (documentado aquí) |
| T02-H4 | `quantile_bins` rinde mucho menos que `standard` (0.71 contra 0.86): con 5 cuantiles, los cuartos de un estudiante caen en pocas celdas y hay muchos empates que se desempatan por `room_id`. Es un resultado de la rejilla, no un error. | INFORMATIVO | Para T05 (la calibración usa la transformación elegida, `standard`) |
| T02-H5 | El «EJECUTA» proponía `min_data_in_leaf` «en pocos valores» como ejemplo; se dejó fijo en 50 como dice la spec aprobada (8 configuraciones de LightGBM, no 16). | INFORMATIVO | M-ML-038 decisión 3 (PROPUESTO) |
| T02-H6 | Las pruebas que esperaban `train: pendiente de ML-05` (`test_stages.py`, `test_cli.py`, `test_installed_package.py`) se actualizaron; `test_cli.py` se detectó al retomar (no estaba en la primera lista de afectadas). | INFORMATIVO | Corregido |
| T02-H7 | Al aprobar, `train_params.json` pasó a `(APROBADO)` y los reportes se regeneraron con una corrida oficial: cambiaron los SHA-256 de los 2 reportes y también el de `train_manifest.json` (la previsión decía que no, pero el perfil agregó el campo `profile`); `logreg.json` y `lightgbm_lambdarank.txt` no cambiaron. | INFORMATIVO | Cerrado (2026-11-24) |
| T02-H8 | El commit no se registra en los archivos de `models/` (la rama no tiene commit de T02 y el árbol de trabajo está sucio). La constitución §9.3 lo pide para el artefacto publicado: ML-06 lo registra al exportar. | INFORMATIVO | Para ML-06 |
| T02-H9 | La ejecución se interrumpió por un apagado; se retomó sin repetir el entrenamiento completo, tras verificar que el código y las salidas no habían cambiado. | INFORMATIVO | Cerrado |
| T02-H10 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |
| T02-H11 | Los modelos finales del perfil `ci` son idénticos byte a byte a los oficiales (el modelo final usa la misma configuración sobre todo train + validación; los pliegues solo cambian las métricas de validación cruzada). El CI comprueba así de paso que la etapa produce los mismos modelos en Linux; los números de 2 pliegues no sustituyen a los oficiales de 5. | INFORMATIVO | Cerrado |

### Revisión y resultado (T02)

- **Seguridad:** sin secretos, sin red en las pruebas, sin servicios de pago (COST-001: CPU local); sin pickle.
- **Datos y licencias:** solo datos sintéticos de `data/processed/` (fuera de git); modelos en `models/` (fuera de git); los reportes en git solo tienen agregados por pliegue, sin datos personales (los `student_id` sintéticos de la asignación de pliegues van solo en `models/train_manifest.json`, fuera de git). Sin dependencias nuevas.
- **Coherencia:** tabla anterior.

**Resultado T02: PASS** — aceptación de R02 cumplida y verificada; M-ML-038 **APROBADO** (César Yael López Trujillo, 2026-11-24) con el perfil `ci` que resuelve T02-H1; T02-H2 queda como informativo para T05/ML-06, que lo verifican por spec. T02 **completa** (ML-05 en 2/7). Sin commit, por instrucción del responsable.

## T03 · Evaluación contra el baseline y decisión de promoción

Lecturas previas (T03), en el orden de `AGENTS.md`: `constitution.md` (§10 evaluación y §12 explicabilidad) → `AGENTS.md` → `memoria.md` (M-ML-034 a M-ML-038) → `spec.md` (R03, §6.2, §6.4, §6.5 y §10 decisiones 4 y 5), `plan.md` y `task.md` → `specs/ARQUITECTURA.md` (§3: `evaluate` = ML-04 · ML-05) → `reports/baseline_metrics.json`, `reports/train_cv_metrics.json` → `src/thehouse_ml/models/train.py`, `train_config.py`, `train_report.py`, `training_data.py`, `baseline_report.py` y `src/thehouse_ml/evaluation/ranking.py`.

### Qué se hizo

- **`src/thehouse_ml/models/trained_eval.py`** (nuevo): verificación de los modelos finales de T02 por SHA-256 contra `models/train_manifest.json` (y de las entradas, del perfil y de que `train_cv_metrics.json` es de la misma corrida); carga sin pickle (`LogRegModel.from_dict` y `lightgbm.Booster(model_file=…)`), **sin reentrenar**; selección de **solo** la partición test (`select_split`); puntaje de cada modelo, ranking por puntaje descendente con desempate por `room_id` y métricas con `thehouse_ml.evaluation.ranking.student_metrics` (`ranked_metrics`); recálculo del baseline con `score_features` y `rank` y verificación contra `per_student_baseline` (`check_baseline_reference`) y contra el intervalo del total (`check_baseline_intervals`); bootstrap pareado (`paired_intervals`); tabla por pliegue (`fold_series`); condiciones de la spec §6.5 y decisión (`promotion`); etapa `run_trained_evaluate` con perfiles `full` y `ci`.
- **`src/thehouse_ml/models/trained_report.py`** (nuevo): `reports/trained_metrics.json` y `reports/trained_report.md`, deterministas y marcados «DATOS SINTÉTICOS».
- **`src/thehouse_ml/stages.py`**: `evaluate` corre el baseline (ML-04) y después la medición de ML-05 (importación diferida de LightGBM); si el baseline se omite por falta de datos, ML-05 también (T03-H6).
- **Pruebas** (sin red): `tests/unit/test_trained_eval.py` (20) y `tests/integration/test_trained_eval_stage.py` (12).
- **Salidas en git:** `reports/trained_metrics.json` y `reports/trained_report.md`.
- **Decisiones nuevas:** M-ML-039 (**APROBADO**, César Yael López Trujillo, 2026-11-24).

### Medición única en test

**Test se evaluó una sola vez:** los modelos finales de T02 (`logreg:standard:C=0.01` y `lightgbm:leaves=15:lr=0.05:trees=300`) se cargaron con su SHA-256 verificado y se midieron en test sin cambiar nada. Ningún hiperparámetro, feature, calibración, umbral ni decisión se ajustó con estos resultados (spec §6.2 y §7 regla 2); `train_params.json`, los modelos y los params de ML-03/ML-04 tienen los mismos SHA-256 antes y después. La etapa se corrió 3 veces con el perfil oficial (la primera escribió los archivos pero el mensaje de consola falló, T03-H1) y las 3 dieron los mismos bytes: repetir es reproducibilidad, no una nueva medición con ajustes (T03-H5). T04–T06 reutilizan esta medición.

| Modelo (fuera de git) | SHA-256 verificado contra `train_manifest.json` |
| --- | --- |
| `models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` |
| `models/lightgbm_lambdarank.txt` | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` |

### Resultado en test (DATOS SINTÉTICOS)

Partición test: **300 estudiantes** (los mismos de `per_student_baseline`) y **30,755 pares**; el baseline recalculado reproduce `per_student_baseline` y su intervalo del total. Bootstrap pareado por estudiante: 1,000 remuestreos, percentil 95 %, `bootstrap:20261122:total`.

| Modelo | NDCG@10 [IC 95 %] | NDCG@5 [IC 95 %] | Precision@5 [IC 95 %] | MAP [IC 95 %] |
| --- | ---: | ---: | ---: | ---: |
| Baseline `baseline-1.0.0` | 0.527 [0.503, 0.550] | 0.535 [0.507, 0.560] | 0.586 [0.557, 0.615] | 0.503 [0.485, 0.522] |
| Regresión logística | 0.862 [0.852, 0.873] | 0.874 [0.861, 0.886] | 0.916 [0.900, 0.931] | 0.831 [0.821, 0.839] |
| LightGBM | 0.863 [0.853, 0.872] | 0.877 [0.864, 0.890] | 0.911 [0.894, 0.927] | 0.835 [0.825, 0.844] |

| Diferencia pareada contra el baseline | Δ NDCG@10 [IC 95 %] | Δ NDCG@5 | Δ Precision@5 | Δ MAP |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | **+0.335 [+0.312, +0.360]** | +0.339 [+0.313, +0.367] | +0.330 [+0.299, +0.361] | +0.327 [+0.307, +0.347] |
| LightGBM | +0.336 [+0.313, +0.360] | +0.342 [+0.315, +0.370] | +0.325 [+0.295, +0.358] | +0.331 [+0.312, +0.350] |

Validación cruzada (NDCG@10, de `reports/train_cv_metrics.json`, media ± desviación muestral): baseline 0.5381 ± 0.0093; regresión logística 0.8623 ± 0.0035 (5/5 pliegues por encima, Δ mínima +0.3100); LightGBM 0.8665 ± 0.0049 (5/5, Δ mínima +0.3128). Tabla completa por pliegue en `reports/trained_report.md`.

### Decisión de promoción (spec §6.5)

| Condición | Regresión logística | LightGBM |
| --- | --- | --- |
| 1. NDCG@10 > baseline en los 5 pliegues | sí (5/5) | sí (5/5) |
| 2. IC 95 % de Δ NDCG@10 pareada > 0 en test | sí (límite inferior +0.312) | sí (+0.313) |
| 3. Desglose aditivo que suma el total | pendiente (T05) | no (no aditivo) |
| Resultado | **candidata a promoción, sujeta a T05** | referencia (no se promueve en v1, decisión 5) |

**Decisión:** la regresión logística cumple 1 y 2 y queda **candidata a promoción, sujeta a T05** (condición 3). `baseline-1.0.0` sigue vigente hasta que T05 verifique el desglose; la decisión final va en la model card (T06).

### Reproducibilidad y perfil `ci`

| Comando | Duración | Resultado |
| --- | ---: | --- |
| `uv run python -m thehouse_ml evaluate` (1.ª, perfil `full`) | 14 s | Escribió `reports/trained_metrics.json` y `reports/trained_report.md`; **exit 1** al imprimir el mensaje en la consola cp1252 («Δ», T03-H1) |
| `uv run python -m thehouse_ml evaluate` (2.ª y 3.ª, tras corregir el mensaje) | 15 s y 14 s | exit 0; `diff` de los SHA-256 de `models/*`, `src/thehouse_ml/models/*.json`, `reports/*.json` y `reports/*.md` entre las 3 corridas: **vacío** |
| `THEHOUSE_ML_TRAIN_PROFILE=ci uv run python -m thehouse_ml evaluate` (con `models/ci/` de T02) | — | exit 0; escribe `models/ci/trained_metrics.json` y `models/ci/trained_report.md` (`profile: ci`, `official: false`, 2 pliegues, decisión `baseline_stays` porque la condición 1 exige 5; T03-H3); `diff` de los SHA-256 de los archivos versionados: **vacío** |

| Archivo | SHA-256 (las 3 corridas) |
| --- | --- |
| `reports/trained_metrics.json` | `288258c212d7550fab127008b098d365156a13db28b57199e60001c53ce2ce68` |
| `reports/trained_report.md` | `32e0b90f02bafe3436c4a0581205a2a786c2eb1411ccebdb8584c7fcc59c766b` |

Respecto del estado previo a T03, el único cambio en esos archivos es la aparición de los 2 reportes nuevos: `reports/baseline_*`, `reports/train*`, `models/*` y los params (`train_params.json`, `baseline_params.json`, `baseline_eval_params.json`, params de ML-03/ML-04) tienen los mismos SHA-256.

### Trazabilidad R03 → prueba

| Criterio | Prueba o comando |
| --- | --- |
| SHA-256 de los modelos verificado (falla si no coincide) | `test_trained_eval.py::test_verify_models_accepts_the_files_of_the_manifest`, `::test_verify_models_fails_if_a_model_changed[logreg.json]`, `[lightgbm_lambdarank.txt]`, `::test_verify_models_fails_if_a_model_or_its_digest_is_missing`; `test_trained_eval_stage.py::test_a_model_that_does_not_match_its_sha256_is_an_error` (2), `::test_inputs_different_from_the_training_ones_are_an_error` |
| Solo se puntúa test | `test_trained_eval.py::test_select_split_keeps_only_the_test_pairs`; `test_trained_eval_stage.py::test_only_test_pairs_are_scored` (espía de los dos modelos), `::test_only_the_test_students_of_per_student_baseline_are_evaluated` |
| Métricas iguales a `ranking.py` (orden puntaje desc, `room_id` asc) | `test_trained_eval.py::test_ranked_metrics_order_by_score_then_room_id_and_use_ranking_py`; `test_trained_eval_stage.py::test_metrics_match_an_independent_recomputation_with_ranking_py` |
| Mismos 300 estudiantes que `per_student_baseline` y baseline reproducido | `test_trained_eval.py::test_baseline_reference_must_have_the_same_students_and_metrics`; `test_trained_eval_stage.py::test_only_the_test_students_of_per_student_baseline_are_evaluated` (incluye el intervalo del total); verificación en la etapa con los datos reales |
| Comparación pareada correcta (ejemplo a mano) | `test_trained_eval.py::test_paired_difference_matches_a_hand_computed_example`, `::test_paired_intervals_use_the_same_resamples_as_the_baseline_bootstrap` |
| Las 3 condiciones, «sujeta a T05» y «se queda el baseline» | `test_trained_eval.py::test_cv_condition_needs_every_required_fold_strictly_above`, `::test_holdout_condition_needs_the_lower_bound_above_zero` (3), `::test_logreg_meeting_cv_and_test_is_a_candidate_subject_to_t05`, `::test_only_lightgbm_meeting_the_conditions_keeps_the_baseline`, `::test_without_a_candidate_meeting_cv_and_test_the_baseline_stays` (3) |
| Tabla por pliegue de la misma corrida | `test_trained_eval.py::test_fold_series_reads_the_selected_models_of_the_same_run` |
| Reportes sintéticos, medición única, decisión | `test_trained_eval_stage.py::test_outputs_are_synthetic_and_record_the_single_measurement_and_decision` |
| Reproducibilidad (SHA-256 iguales) | `test_trained_eval_stage.py::test_running_twice_does_not_change_any_file`; 3 corridas reales (tabla anterior) |
| Nunca lee los archivos ocultos (decisión 1) | `test_trained_eval_stage.py::test_evaluate_never_reads_the_hidden_label_files` |
| Etapa `evaluate` y perfil `ci` sin tocar los reportes versionados | `test_trained_eval_stage.py::test_evaluate_stage_runs_the_baseline_and_then_the_trained_models`, `::test_ci_profile_does_not_modify_the_versioned_reports`, `::test_without_models_the_stage_is_skipped`; corrida real con `THEHOUSE_ML_TRAIN_PROFILE=ci` |

### Comandos y resultados (T03)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `154 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 45 source files` (PASS) |
| `uv run pytest` de las afectadas (`test_trained_eval.py`, `test_trained_eval_stage.py`, `test_stages.py`, `test_cli.py`, `test_installed_package.py`, `test_evaluate_stage.py`, `test_train_stage.py`, `test_repo_hygiene.py`, `test_ci_workflow.py`, `test_baseline_report.py`) con cobertura de los módulos nuevos | **164 passed** en 113.9 s; cobertura: `trained_eval.py` 93 %, `trained_report.py` 100 %, `stages.py` 97 % (94.9 % en conjunto) (PASS) |
| `uv run python -m thehouse_ml evaluate` × 3 y con `THEHOUSE_ML_TRAIN_PROFILE=ci` | SHA-256 idénticos; versionados intactos con `ci` (tabla anterior) (PASS) |
| Suite completa (`uv run pytest --cov=thehouse_ml`) | **No ejecutada** en T03 (se pidieron las afectadas); corresponde a T07 |

### Tabla de coherencia (T03)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §10.1 | Los dos modelos se comparan contra `baseline-1.0.0`; sin la condición 3 (T05) no hay promoción definitiva | PASS |
| Constitución §10.2–§10.3 | NDCG@10, NDCG@5, Precision@5 y MAP por estudiante; partición por estudiante sin fuga; intervalos por bootstrap en test y media ± desviación entre pliegues | PASS |
| Constitución §12 | La condición 3 queda pendiente de T05 para la LR y no se cumple para LightGBM; ningún modelo sin desglose se promueve | PASS |
| Constitución §8 y §20.3 | «DATOS SINTÉTICOS» y advertencia de ML-07 en el JSON, el reporte y el mensaje de la etapa | PASS |
| Constitución §9 | Un comando (`evaluate`), mismas salidas byte a byte; versiones y SHA-256 de entradas y modelos en el JSON | PASS |
| Constitución §13.1 | Modelos cargados de JSON y texto nativo; sin pickle (`test_repo_hygiene.py` en verde) | PASS |
| Spec §6.2 y §7 regla 2 | Test medido una vez con los modelos finales, sin ajustar nada; constancia en la evidencia, el JSON y el reporte | PASS |
| Spec §6.4 | Funciones de `ranking.py`, umbral 2, desempate `room_id`; bootstrap con `baseline_eval_params.json` 1.0.0; prueba de los 300 estudiantes y del baseline reproducido | PASS |
| Spec §6.5, §6.6 y decisiones 4 y 5 | 3 condiciones por candidato; LR «sujeta a T05»; LightGBM referencia | PASS |
| Spec R03 | Tabla por pliegue solo con NDCG@10 (la de `train_cv_metrics.json`, como pidió el «EJECUTA»); R03 nombra también NDCG@5, Precision@5 y MAP por pliegue: traspasado a T04 por el responsable (T03-H2) | PASS |
| Spec §6.1 y decisión 1 | Solo `TRAINING_INPUTS`; prueba obligatoria para la etapa de T03 | PASS |
| `specs/ARQUITECTURA.md` §3–§4 | Integrado en `evaluate`; módulos en `models/`; reportes en `reports/` | PASS |
| AGENTS: rutas permitidas | `src/thehouse_ml/models/` (2 nuevos), `src/thehouse_ml/stages.py` (T03-H6), `tests/` (2 nuevos), `reports/` (2 nuevos), `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05); `evaluation/` y `features/` solo reutilizados | PASS |
| AGENTS: archivos protegidos por el «EJECUTA» | `train_params.json`, modelos, `baseline_params.json`, `baseline_eval_params.json` y params de ML-03/ML-04 con los mismos SHA-256 | PASS |
| AGENTS: sin dependencias nuevas | `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La suite completa y el CI del PR no se ejecutaron (T07 / PR); la condición 3 queda «pendiente», no «sí» | PASS |

### Hallazgos (T03)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T03-H1 | La primera corrida oficial escribió los dos reportes y después falló al imprimir el mensaje en la consola de Windows (cp1252 no tiene «Δ»). Se cambió solo el texto del mensaje («dif.») y se agregó la comprobación `encode("cp1252")` en la prueba de integración; los reportes de las 3 corridas son idénticos. | ADVERTENCIA → corregida | Corregido |
| T03-H2 | La spec R03 pide la tabla por pliegue de NDCG@10, NDCG@5, Precision@5 y MAP; `train_cv_metrics.json` (T02) solo guarda NDCG@10 y el «EJECUTA» pidió usar esa tabla sin reentrenar. Las condiciones de promoción solo usan NDCG@10, así que la decisión no cambia. **Propuesta:** que T04, que ya reentrena por pliegue para la ablación, registre también las 4 métricas por pliegue de los modelos completos y del baseline; o aceptar NDCG@10 como suficiente para R03. | ADVERTENCIA → traspasada | **Traspaso a T04** (M-ML-039 decisión 7, APROBADO 2026-11-24): T04 registra las 4 métricas por pliegue al reentrenar para la ablación y completa la tabla por pliegue de `reports/trained_report.md`. **RESUELTO en T04** (2026-11-24): ver la sección T04, «Traspaso T03-H2: las 4 métricas por pliegue» |
| T03-H3 | Con el perfil `ci` (2 pliegues) la condición 1 nunca se cumple y la salida dice `baseline_stays`; es intencional (la condición exige los 5 pliegues oficiales) y la salida va a `models/ci/` marcada «no oficial». | INFORMATIVO | Cerrado |
| T03-H4 | En test la LR y LightGBM quedan prácticamente empatados (Δ NDCG@10 +0.335 contra +0.336, intervalos casi iguales); LightGBM es algo mejor en validación cruzada (0.8665 contra 0.8623). No cambia la decisión (LightGBM no se promueve en v1). | INFORMATIVO | Para la model card (T06) |
| T03-H5 | «Test una sola vez»: la etapa se corrió 3 veces con el perfil oficial y 1 con `ci`, sin cambiar modelos, parámetros ni código de cálculo entre corridas (solo el texto del mensaje, T03-H1); las salidas son idénticas. Las pruebas usan un proyecto temporal, no la partición test real. | INFORMATIVO | Cerrado |
| T03-H6 | Integrar la medición en `evaluate` exigió cambiar `src/thehouse_ml/stages.py` (fuera de `models/`), como en T02; el «EJECUTA» lo pide en el punto 6. `cli.py` no cambió: `TrainedEvalError` hereda de `EvaluationError`. | INFORMATIVO | **Autorizado retroactivamente** por el responsable (2026-11-24) |
| T03-H7 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T03)

- **Seguridad:** sin secretos, sin red en las pruebas, sin servicios de pago (COST-001: CPU local); sin pickle; los modelos solo se cargan si su SHA-256 coincide con el manifiesto (constitución §14.3).
- **Datos y licencias:** solo datos sintéticos de `data/processed/` (fuera de git); los reportes en git traen agregados y métricas por estudiante con solo el `student_id` sintético (como `per_student_baseline`), sin datos personales. Sin dependencias nuevas.
- **Coherencia:** tabla anterior.

### Traspaso a T04

- **T03-H2:** Traspaso a T04 (aprobado por el responsable el 2026-11-24): al reentrenar por pliegue para la ablación, T04 registra las 4 métricas (NDCG@10, NDCG@5, Precision@5 y MAP) por pliegue del baseline, la regresión logística y LightGBM y completa la tabla por pliegue de `reports/trained_report.md`. Las condiciones de promoción de T03 solo usan NDCG@10 y no cambian. T04 no vuelve a medir en test.
- **T03-H4:** el empate práctico entre la LR y LightGBM en test va a la model card (T06).

**Resultado T03: PASS** — aceptación de R03 cumplida y verificada; M-ML-039 **APROBADO** (César Yael López Trujillo, 2026-11-24), con `stages.py` autorizado retroactivamente (T03-H6) y la tabla por pliegue de 4 métricas traspasada a T04 (T03-H2). T03 **completa** (ML-05 en 3/7). Sin commit, por instrucción del responsable.

## T04 · Ablación y segmentos

Lecturas previas (T04), en el orden de `AGENTS.md`: `constitution.md` (§10 y §11.3) → `AGENTS.md` → `memoria.md` (M-ML-034 a M-ML-039) → `spec.md` (R04, §6.6 grupos, §6.7 ablación, segmentos y regla de la hipótesis), `plan.md`, `task.md` y el traspaso T03-H2 de esta evidencia → `reports/baseline_metrics.json`, `reports/train_cv_metrics.json`, `reports/trained_metrics.json` → `src/thehouse_ml/models/train*.py`, `trained_*.py`, `baseline_params.json` (ids de los factores) y `src/thehouse_ml/evaluation/ranking.py`.

### Qué se hizo

- **`src/thehouse_ml/models/factor_groups.py`** (nuevo): los 6 grupos de features con los `id` de los factores del baseline y `pets_compatible` fuera de los grupos (M-ML-040, decisión 1); `check_groups` y `kept_columns`.
- **`src/thehouse_ml/models/ablation.py`** (nuevo): pliegues de `train_manifest.json` (`cv_from_manifest`, nunca test), hiperparámetros elegidos de `train_cv_metrics.json` (`selected_configs`), reajuste por pliegue (`cv_scores`), 4 métricas por pliegue (`summarize_folds`) con verificación contra el NDCG@10 de T02 (`check_against_train`), ablación de los 6 grupos (`ablate`), segmentos en test sobre las métricas por estudiante de T03 (`build_segments`, `segment_results`, `check_segments`) e hipótesis > $3,500 (`hypothesis`).
- **`src/thehouse_ml/models/ablation_report.py`** (nuevo): `reports/ablation_metrics.json`, `reports/ablation_report.md` y la tabla de 4 métricas por pliegue.
- **`src/thehouse_ml/models/trained_eval.py`** y **`trained_report.py`**: `run_trained_evaluate` ejecuta T04 en la misma corrida de `evaluate` y `trained_report.md` gana la tabla «Las 4 métricas por pliegue (T04, traspaso T03-H2)». `stages.py` no cambió. `trained_metrics.json` no cambió (mismo SHA-256).
- **Pruebas** (sin red): `tests/unit/test_ablation.py` (23) y `tests/integration/test_ablation_stage.py` (6); `tests/integration/test_trained_eval_stage.py` ajustada (2 pruebas: la etapa ahora puntúa también los pliegues e imprime 3 líneas).
- **Decisiones nuevas:** M-ML-040 (**APROBADO**, César Yael López Trujillo, 2026-11-24).

### Traspaso T03-H2: las 4 métricas por pliegue (DATOS SINTÉTICOS)

Mismos 5 pliegues de `models/train_manifest.json` (1,699 estudiantes, 165,849 pares de train + validación) e hiperparámetros elegidos en T02, sin búsqueda nueva. El NDCG@10 por pliegue del baseline, la LR y LightGBM **reproduce exactamente** el de `reports/train_cv_metrics.json` (la etapa falla si no). Test no participa.

| Métrica | Modelo | P0 | P1 | P2 | P3 | P4 | Media ± desv. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NDCG@10 | Baseline | 0.5396 | 0.5470 | 0.5453 | 0.5241 | 0.5345 | 0.5381 ± 0.0093 |
| NDCG@10 | Regresión logística | 0.8664 | 0.8570 | 0.8616 | 0.8626 | 0.8640 | 0.8623 ± 0.0035 |
| NDCG@10 | LightGBM | 0.8733 | 0.8597 | 0.8651 | 0.8661 | 0.8684 | 0.8665 ± 0.0049 |
| NDCG@5 | Baseline | 0.5399 | 0.5512 | 0.5583 | 0.5185 | 0.5282 | 0.5392 ± 0.0163 |
| NDCG@5 | Regresión logística | 0.8683 | 0.8659 | 0.8747 | 0.8674 | 0.8648 | 0.8682 ± 0.0039 |
| NDCG@5 | LightGBM | 0.8773 | 0.8675 | 0.8775 | 0.8735 | 0.8708 | 0.8733 ± 0.0043 |
| Precision@5 | Baseline | 0.5765 | 0.5935 | 0.5882 | 0.5617 | 0.5576 | 0.5755 ± 0.0158 |
| Precision@5 | Regresión logística | 0.8906 | 0.8888 | 0.8900 | 0.8897 | 0.8800 | 0.8878 ± 0.0044 |
| Precision@5 | LightGBM | 0.8959 | 0.8871 | 0.8994 | 0.8973 | 0.8865 | 0.8932 ± 0.0060 |
| MAP | Baseline | 0.5114 | 0.5161 | 0.5095 | 0.5012 | 0.5103 | 0.5097 ± 0.0054 |
| MAP | Regresión logística | 0.8342 | 0.8244 | 0.8275 | 0.8230 | 0.8278 | 0.8274 ± 0.0043 |
| MAP | LightGBM | 0.8408 | 0.8275 | 0.8309 | 0.8318 | 0.8333 | 0.8329 ± 0.0049 |

### Ablación por grupo (solo pliegues; DATOS SINTÉTICOS)

Δ NDCG@10 = sin el grupo − modelo completo, media ± desviación muestral entre los 5 pliegues.

| Grupo quitado (features) | LR: NDCG@10 sin el grupo | LR: Δ | LightGBM: NDCG@10 sin el grupo | LightGBM: Δ |
| --- | ---: | ---: | ---: | ---: |
| (modelo completo) | 0.8623 | — | 0.8665 | — |
| Presupuesto (`price_to_budget`, `price_below_min`) | 0.7783 | −0.0840 ± 0.0090 (5/5) | 0.7843 | −0.0822 ± 0.0076 (5/5) |
| Distancia (`distance_km`, `effective_distance_km`, `distance_ratio`) | 0.6078 | **−0.2545 ± 0.0065** (5/5) | 0.6152 | −0.2514 ± 0.0074 (5/5) |
| Servicios esenciales (`services_coverage`) | 0.8239 | −0.0384 ± 0.0036 (5/5) | 0.8281 | −0.0384 ± 0.0040 (5/5) |
| Tipo de lugar (`place_type_match`) | 0.7505 | −0.1118 ± 0.0036 (5/5) | 0.7519 | −0.1146 ± 0.0048 (5/5) |
| Tipo de inmueble (`property_type_match`) | 0.8486 | −0.0137 ± 0.0015 (5/5) | 0.8530 | −0.0136 ± 0.0025 (5/5) |
| Extras (`extra_services_ratio`) | 0.8429 | −0.0195 ± 0.0021 (5/5) | 0.8475 | −0.0191 ± 0.0021 (5/5) |

Los 6 grupos aportan en los dos modelos (todos empeoran en 5/5 pliegues). `pets_compatible` (constante) se queda en todos.

### Segmentos de presupuesto en test (medición única de T03; DATOS SINTÉTICOS)

NDCG@10 con IC 95 % por bootstrap pareado (1,000 remuestreos, `bootstrap:20261122:<segment_id>`). Campus y «sin `seed_sin_verificar`» en `reports/ablation_report.md`.

| Segmento | Estudiantes | Baseline | Regresión logística | Δ LR [IC 95 %] | LightGBM | Δ LightGBM [IC 95 %] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total | 300 | 0.527 [0.503, 0.550] | 0.862 [0.852, 0.873] | +0.335 [+0.312, +0.360] | 0.863 [0.853, 0.872] | +0.336 [+0.313, +0.360] |
| `budget.max` ≤ $2,000 | 73 | 0.568 [0.525, 0.608] | 0.857 [0.837, 0.875] | +0.289 [+0.243, +0.336] | 0.865 [0.847, 0.882] | +0.298 [+0.254, +0.343] |
| `budget.max` $2,001–$3,500 | 158 | 0.534 [0.503, 0.566] | 0.867 [0.854, 0.880] | +0.332 [+0.300, +0.364] | 0.864 [0.850, 0.877] | +0.330 [+0.298, +0.361] |
| `budget.max` > $3,500 | 69 | 0.468 [0.410, 0.518] | 0.859 [0.834, 0.880] | **+0.391 [+0.338, +0.446]** | 0.859 [0.835, 0.881] | +0.391 [+0.343, +0.446] |

Verificaciones en la etapa: el intervalo del baseline de cada segmento coincide con `baseline_metrics.json`; las medias de campus y de presupuesto se agregan a las del total; el total coincide cifra por cifra con `trained_metrics.json` (T03).

### Hipótesis del segmento > $3,500

Regla de M-ML-040 (decisión 6), fijada en el código y en las pruebas **antes** de calcular los segmentos con los datos reales:

1. Δ NDCG@10 de la LR en > $3,500 = **+0.391**, la mayor de los tres rangos (+0.289 y +0.332): **sí**.
2. IC 95 % [+0.338, +0.446] por encima de 0: **sí**.
3. Quitar el grupo presupuesto cambia el NDCG@10 en **−0.0840** de media (5/5 pliegues por debajo): **sí, lo empeora**.

**Veredicto: CONFIRMADA** (aceptado por el responsable al aprobar M-ML-040, con las salvedades T04-H2 y T04-H3 para la model card). La mejora en > $3,500 viene de un baseline más débil (0.468) con la LR estable en los tres rangos (0.857–0.867), coherente con que el baseline puntúe igual cualquier renta dentro del presupuesto. Salvedades: los IC de los rangos se solapan (T04-H2) y la condición 3 usa la ablación global, no por segmento (T04-H3).

### Reproducibilidad, tiempos y perfil `ci`

| Comando | Duración | Resultado |
| --- | ---: | --- |
| `uv run python -m thehouse_ml evaluate` (1.ª, perfil `full`) | **426 s (7 min 6 s)** | exit 0; nuevos `reports/ablation_metrics.json` y `reports/ablation_report.md`; cambia solo `reports/trained_report.md` (tabla de 4 métricas) |
| `uv run python -m thehouse_ml evaluate` (2.ª, perfil `full`) | **429 s (7 min 9 s)** | exit 0; `diff` de los SHA-256 de `models/*`, params (`src/thehouse_ml/models/*.json`, `synthetic/*.json`, `feature_schema.json`), `reports/*.json` y `reports/*.md` contra la 1.ª: **vacío** |
| `THEHOUSE_ML_TRAIN_PROFILE=ci uv run python -m thehouse_ml evaluate` (con `models/ci/` de T02) | **40 s** | exit 0; escribe solo en `models/ci/` (`ablation_metrics.json`, `ablation_report.md`, `trained_*`), `profile: ci`, «no oficial», 2 pliegues, ablación solo de la LR; `diff` de los archivos versionados: **vacío** |

| Archivo | SHA-256 (las 2 corridas) |
| --- | --- |
| `reports/ablation_metrics.json` | `f8c1a80b8eb5728f6f347f0d59caa6a9703545a427cd335df7e96cd1e9ef1d92` |
| `reports/ablation_report.md` | `732cf41a2fa235767095effa17507c50151c06e47f67fa5f490de80d583287dc` |
| `reports/trained_report.md` | `e8ffe126bbe7e8a1859b835a5f12a2cdd687634de70e2cf1069df2ca9596a06f` (antes `32e0b90f…`: solo la tabla nueva) |
| `reports/trained_metrics.json` | `288258c212d7550fab127008b098d365156a13db28b57199e60001c53ce2ce68` (sin cambio desde T03) |

Domina el tiempo el reajuste de LightGBM: 35 ajustes de ~11 s (5 pliegues × modelo completo + 6 grupos); los 35 de la LR tardan ~0.1 s cada uno.

### Trazabilidad R04 → prueba

| Criterio | Prueba o comando |
| --- | --- |
| 4 métricas por pliegue con la misma asignación y los mismos hiperparámetros (traspaso T03-H2) | `test_ablation_stage.py::test_fold_metrics_have_the_four_metrics_and_reproduce_t02`; `test_ablation.py::test_cv_from_manifest_uses_the_manifest_folds_and_never_test`, `::test_selected_configs_come_from_train_cv_metrics`, `::test_summarize_folds_averages_each_metric_by_fold_and_counts_undefined`; verificación en la etapa con los datos reales |
| La ablación no usa test | `test_ablation_stage.py::test_ablation_never_uses_test_and_each_group_removes_exactly_its_features` (espía de los ajustes: sin estudiantes de test); `test_ablation.py::test_cv_scores_fit_on_other_folds_and_score_each_held_out_pair_once`; `test_trained_eval_stage.py::test_only_test_pairs_are_scored` |
| Cada grupo quita exactamente sus features (agrupamiento documentado, `pets_compatible` incluido) | `test_ablation.py::test_groups_cover_feature_order_once_and_match_the_baseline_factors`, `::test_each_group_removes_exactly_its_features` (6), `::test_invalid_groups_are_rejected`; `test_ablation_stage.py::test_ablation_never_uses_test_and_each_group_removes_exactly_its_features` |
| Segmentos que reproducen el total de T03 al agregarse y el baseline de referencia | `test_ablation_stage.py::test_segments_reproduce_t03_and_the_baseline_reference`; `test_ablation.py::test_segments_follow_the_baseline_ids_and_aggregate_to_the_total`, `::test_segment_check_detects_partitions_that_do_not_cover_the_total`, `::test_empty_segments_have_no_result` |
| Regla de la hipótesis con casos a mano | `test_ablation.py::test_hypothesis_confirmed_when_the_three_conditions_hold`, `::test_hypothesis_refuted_when_a_point_condition_fails` (3), `::test_hypothesis_refuted_when_the_interval_is_below_zero`, `::test_hypothesis_inconclusive_when_the_interval_includes_zero`, `::test_hypothesis_needs_the_budget_ranges` |
| Reproducibilidad | `test_ablation_stage.py::test_running_twice_does_not_change_any_file`; dos corridas reales (tabla anterior) |
| Perfil `ci` sin tocar los reportes versionados | `test_ablation_stage.py::test_ci_profile_writes_only_in_models_ci_and_skips_lightgbm_ablation`; `test_ablation.py::test_ci_profile_only_ablates_the_logistic_regression`; corrida real (40 s) |
| Nunca lee los archivos ocultos (decisión 1) | `test_trained_eval_stage.py::test_evaluate_never_reads_the_hidden_label_files` (corre T03 + T04); `test_ablation_stage.py::test_t04_modules_never_name_the_hidden_label_files` |

### Comandos y resultados (T04)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `160 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 48 source files` (PASS) |
| `uv run pytest` de las afectadas (`test_ablation.py`, `test_ablation_stage.py`, `test_trained_eval.py`, `test_trained_eval_stage.py`, `test_stages.py`, `test_cli.py`, `test_installed_package.py`, `test_evaluate_stage.py`, `test_train_stage.py`, `test_repo_hygiene.py`, `test_ci_workflow.py`, `test_baseline_report.py`) con cobertura | **193 passed** en 382.7 s; cobertura: `ablation.py` 95 %, `ablation_report.py` 97 %, `factor_groups.py` 100 %, `trained_eval.py` 93 %, `trained_report.py` 99 % (95.3 % en conjunto) (PASS) |
| `uv run python -m thehouse_ml evaluate` × 2 y con `THEHOUSE_ML_TRAIN_PROFILE=ci` | 426 s y 429 s, SHA-256 idénticos; `ci` 40 s, versionados intactos (PASS) |
| Suite completa (`uv run pytest --cov=thehouse_ml`) | **No ejecutada** en T04 (se pidieron las afectadas); corresponde a T07 |

### Tabla de coherencia (T04)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §10.2–§10.3 | 4 métricas por pliegue con media ± desviación; partición por estudiante sin fuga; IC por bootstrap en los segmentos | PASS |
| Constitución §11.3 | Segmentos por universidad (10 campus), sin `seed_sin_verificar` y por rango de presupuesto, con diferencias pareadas | PASS |
| Constitución §8 y §20.3 | «DATOS SINTÉTICOS» y advertencia de ML-07 en el JSON, el reporte y el mensaje de la etapa | PASS |
| Constitución §9 | Un comando (`evaluate`), mismas salidas byte a byte en dos corridas | PASS |
| Spec §6.2 y §7 regla 2 | La ablación y las métricas por pliegue solo usan train + validación; los segmentos reutilizan la medición única de T03 sin ajustar nada | PASS |
| Spec §6.6 | Los 6 grupos son los factores del desglose (mismos `id` que `baseline-1.0.0`); `pets_compatible` fuera, como su término va a la constante | PASS |
| Spec §6.7 | Ablación de los 6 grupos para los dos modelos sin nueva búsqueda; segmentos con diferencias pareadas e IC; veredicto de la hipótesis con la regla registrada | PASS |
| Spec §6.1 y decisión 1 | Solo `TRAINING_INPUTS`; la prueba obligatoria cubre la etapa completa | PASS |
| Traspaso T03-H2 (M-ML-039) | 4 métricas por pliegue al reentrenar y tabla completada en `reports/trained_report.md` | PASS |
| AGENTS: rutas permitidas | `src/thehouse_ml/models/` (3 nuevos, 2 modificados), `tests/` (2 nuevos, 1 modificado), `reports/` (2 nuevos, 1 modificado), `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05); `stages.py`, `evaluation/` y `features/` sin cambios | PASS |
| AGENTS: archivos protegidos por el «EJECUTA» | `train_params.json`, modelos oficiales, `baseline_params.json`, `baseline_eval_params.json` y params de ML-03/ML-04 con los mismos SHA-256 | PASS |
| AGENTS: sin dependencias nuevas | `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La suite completa y el CI del PR no se ejecutaron (T07 / PR) | PASS |

### Hallazgos (T04)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T04-H1 | La etapa `evaluate` oficial pasa de ~15 s a ~7 min por los 35 ajustes de LightGBM de los pliegues. El CI usa el perfil `ci` (40 s: 2 pliegues y sin ablación de LightGBM, M-ML-040 decisión 8); el tiempo en el runner de ubuntu se confirma en el CI del PR. | ADVERTENCIA → mitigada | Abierto: se cierra cuando el CI del PR confirme el tiempo en ubuntu (responsable, 2026-11-24) |
| T04-H2 | La condición (1) de la hipótesis compara medias puntuales: el IC de > $3,500 ([+0.338, +0.446]) se solapa con el de $2,001–$3,500 ([+0.300, +0.364]). La regla aprobada no exige que las diferencias entre rangos sean significativas; el veredicto se reporta con esa salvedad. | INFORMATIVO | Salvedad aceptada; **la model card (T06) debe incluirla** |
| T04-H3 | La condición (3) usa la ablación global de los pliegues (−0.0840), no una ablación por rango de presupuesto, como dice la spec §6.7. | INFORMATIVO | Salvedad aceptada; **la model card (T06) debe incluirla** |
| T04-H4 | Distancia es el grupo más influyente (−0.25), seguido de tipo de lugar (−0.11) y presupuesto (−0.08); LR y LightGBM coinciden grupo por grupo. Con etiquetas sintéticas, esto refleja los pesos de la utilidad oculta (circularidad, ML-05-RK01), no preferencias reales. | INFORMATIVO | Para la model card (T06) |
| T04-H5 | Dos pruebas de T03 (`test_only_test_pairs_are_scored`, `test_evaluate_stage_runs_the_baseline_and_then_the_trained_models`) se ajustaron: la etapa ahora también puntúa pliegues (nunca test) e imprime una tercera línea. Su intención no cambió. | INFORMATIVO | Corregido |
| T04-H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T04)

- **Seguridad:** sin secretos, sin red en las pruebas, sin servicios de pago (COST-001: CPU local); sin pickle.
- **Datos y licencias:** solo datos sintéticos de `data/processed/` (fuera de git); los reportes en git solo traen agregados por pliegue, grupo y segmento, sin datos personales. Sin dependencias nuevas.
- **Coherencia:** tabla anterior.

**Resultado T04: PASS** — aceptación de R04 cumplida y verificada, traspaso T03-H2 resuelto; hipótesis > $3,500 **confirmada** con la regla registrada y las salvedades T04-H2 y T04-H3 para la model card (T06); M-ML-040 **APROBADO** (César Yael López Trujillo, 2026-11-24); T04-H1 abierto hasta el CI del PR. T04 **completa** (ML-05 en 4/7). Sin commit, por instrucción del responsable.

## T05 · Calibración 0-100 y desglose

Lecturas previas (T05), en el orden de `AGENTS.md`: `constitution.md` (§11 y §12) → `AGENTS.md` → `memoria.md` (M-ML-029 a M-ML-040; sobre todo M-ML-031 redondeo y M-ML-036 decisión 5) → `spec.md` completo (§6.5, §6.6 y R05), `plan.md`, `task.md` y esta evidencia → `src/thehouse_ml/models/` (`train.py`/`logreg.json`, `baseline_score.py`, `factor_groups.py`, `trained_eval.py`) y `reports/trained_metrics.json`.

### Qué se hizo

- **`src/thehouse_ml/models/trained_breakdown.py`** (nuevo, puro): `build_breakdown_model` (calibración afín por factor sobre el recorte de `logreg.json`, sin reentrenar), `score_features_logreg` y `score_pair_logreg` (salida con formato del baseline, `modelVersion` `logreg-1.0.0`; reutiliza `_check_vector`, `factor_points`, `match_score`, `ExcludedPair`, `check_hard_constraints` y `compute_pair_features`), `rank_logreg`, verificación en lote (`score_pairs`, `check_sums`, `check_order`) y monotonía (`coefficient_table`, `budget_monotonicity`, `distance_monotonicity`, `budget_ratio_threshold`, `observed_budget_ratio`).
- **`src/thehouse_ml/models/trained_breakdown_report.py`** (nuevo): `reports/trained_breakdown.md` y el bloque `breakdown` de `reports/trained_metrics.json`.
- **`trained_eval.py`**: dentro de `evaluate_trained`, T05 calibra la LR, verifica el desglose en los mismos pares de test que puntuó T03 (no vuelve a llamar al modelo en test) y evalúa la condición 3 de verdad; resultados `promoted`/`logreg_promoted` en lugar de `candidate_subject_to_t05`/`logreg_candidate_subject_to_t05`; la etapa escribe `reports/trained_breakdown.md` e imprime una cuarta línea. **`trained_report.py`**: condición 3 con su detalle, decisión y bloque `breakdown`. `stages.py`, `evaluation/` y `features/` sin cambios.
- **Pruebas** (sin red): `tests/unit/test_trained_breakdown.py` (nuevo, 102 casos, 81 de ellos los dorados); `tests/unit/test_trained_eval.py` (3 pruebas de decisión nuevas en lugar de la de «sujeta a T05»); `tests/integration/test_trained_eval_stage.py` (salidas, módulos sin archivos ocultos y 4 líneas de la etapa).
- **Decisión nueva:** M-ML-041 (**APROBADO**, César Yael López Trujillo, 2026-11-24).
- **Segunda pasada (tras la decisión del responsable sobre T05-H1):** `trained_breakdown.py` gana `ACCEPTED_DEVIATIONS` (solo `monotonicity_budget`, con su motivo y la referencia a M-ML-041); la condición 3 es el criterio de la spec §6.5 (el desglose suma el total) y una falla de monotonía la bloquea salvo que sea una desviación aceptada. La monotonía se sigue calculando y reportando; el JSON gana `sums_total`, `failed_checks` y `accepted_deviations`, y la decisión de promoción nombra la desviación. 5 pruebas nuevas (4 unitarias de la regla y 1 de la decisión).

### Coeficientes (modelo `models/logreg.json`, SHA-256 `341ea3ec…0b27fa` verificado contra `train_manifest.json`; DATOS SINTÉTICOS)

| Feature | Factor | coef (estandarizado) | Logit por unidad | Signo coherente |
| --- | --- | ---: | ---: | --- |
| `distance_km` | distancia | −3.349741 | −0.531370 | sí |
| `effective_distance_km` | distancia | −1.869938 | −0.505759 | sí |
| `price_to_budget` | presupuesto | −0.996049 | −4.277202 | sí |
| `services_coverage` | servicios | +0.565531 | +3.049062 | sí |
| `place_type_match` | tipo de lugar | +1.006235 | +2.198060 | sí |
| `property_type_match` | tipo de inmueble | +0.359535 | +0.781024 | sí |
| `pets_compatible` | término fijo | 0 | 0 | — |
| `distance_ratio` | distancia | −0.100585 | −0.029395 | sí |
| `price_below_min` | presupuesto | −0.012314 | −0.056130 | **no** (por debajo del mínimo, más renta suma) |
| `extra_services_ratio` | extras | +0.439171 | +2.042204 | sí |

Intercepto −4.133224. `pets_compatible` es constante (1) en los candidatos: escala 1, coeficiente 0; va al término fijo con `b0`.

### Calibración y `maxPoints`

| Factor | min_k (logit) | max_k (logit) | maxPoints (sin redondear) | maxPoints | Peso del baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Presupuesto | −1.9443 | +2.3890 | 9.5698 | **9.6** | 30 |
| Distancia | −28.7731 | +4.1045 | 72.6075 | **72.6** | 25 |
| Servicios esenciales | −2.2348 | +0.8142 | 6.7336 | **6.7** | 20 |
| Tipo de lugar | −0.6570 | +1.5410 | 4.8542 | **4.9** | 10 |
| Tipo de inmueble | −0.5429 | +0.2381 | 1.7248 | **1.7** | 10 |
| Extras | −0.9863 | +1.0559 | 4.5100 | **4.5** | 5 |
| **Total** | | | 100.0000 | **100.0** | 100 |

S = 45.281261 unidades de logit. La distancia domina porque su `training_range` llega a 30 km (y 60 en `distance_ratio`) y los candidatos están cerca (T05-H2).

### Verificación en los 30,755 pares de test (300 estudiantes; misma medición de T03)

| Comprobación | Resultado |
| --- | --- |
| máx. \|Σ sin redondear − `matchScore`\| | 0.499973 ≤ 0.5; 0 fuera |
| máx. \|Σ reportada − Σ sin redondear\| | 0.212896 ≤ 0.3; 0 fuera |
| Factores fuera de [0, `maxPoints`] / totales fuera de [0, 100] | 0 / 0 |
| máx. \|logit − afín(raw_total)\| | 2.3 × 10⁻¹⁴ |
| Orden por `raw_total` = orden del logit | 300/300 estudiantes |
| Métricas con el puntaje 0-100 | idénticas a las de T03 |

Distribución de `matchScore` en test: mín. 8, p25 74, mediana 81, p75 85, p95 90, máx. 97, media 77.6 (detalle y 3 ejemplos con top 5 en `reports/trained_breakdown.md`).

### Monotonía por factor

| Factor | Rejilla | Pasos en que sube | Resultado |
| --- | --- | ---: | --- |
| Distancia (al aumentar la distancia) | 4 transportes × 8 `maxDistanceKm` × 0–36 km cada 0.25 (32 curvas, 4,608 pasos) | 0 | **monótono** |
| Presupuesto (al subir la renta) | 100 combinaciones `budget.min` ≤ `budget.max` del dominio NV-11, renta 0 → `budget.max` (10,204 pasos) | 46 | **no monótono**: sube si renta < `budget.min` y `budget.max / budget.min` > **76.20**; máx. +0.1238 puntos (`budget.min` 1, `budget.max` 50,000, renta 0 → 1) |

En los datos sintéticos, 73,815 pares tienen renta < `budget.min` y el mayor `budget.max / budget.min` implícito es **2.83**: ningún par observado cae en la zona. Aun así la rejilla sobre los `training_range` falla.

**Desviación aceptada (transparencia).** En la primera pasada, con la regla del «EJECUTA» (paso 6: suma **y** monotonía), la condición 3 no se cumplía y la decisión registrada fue `baseline_stays` (`trained_metrics.json` `8348939d…`). **Después de ver ese resultado**, el responsable eligió la opción 1: la condición 3 aprobada en la spec §6.5 es «desglose aditivo que suma el total» y se cumple; la monotonía del presupuesto fue una protección adicional del «EJECUTA». La no monotonía de `price_below_min` queda como desviación conocida y **no se corrige** en `logreg-1.0.0`. Trabajo futuro: `logreg-1.1.0` sin `price_below_min` o con coeficiente ≥ 0, evaluado con datos nuevos en ML-07 (test ya se usó, spec §6.2).

### Decisión de promoción (condición 3)

| Condición | Regresión logística |
| --- | --- |
| 1. NDCG@10 > baseline en 5/5 pliegues (T03) | sí |
| 2. IC 95 % de Δ NDCG@10 > 0 en test (T03) | sí (+0.335 [+0.312, +0.360]) |
| 3. Desglose aditivo que suma el total (spec §6.5) | **sí** (suma: sí; monotonía de distancia: sí; monotonía del presupuesto: no, desviación aceptada M-ML-041) |

**Decisión registrada en `reports/trained_metrics.json` y `reports/trained_report.md`: `logreg_promoted`** — la regresión logística queda **PROMOVIDA** como modelo de ML-05 (`logreg-1.0.0`); `baseline-1.0.0` queda como respaldo; LightGBM sigue como referencia (no se promueve en v1).

### Reproducibilidad y perfil `ci`

Primera pasada (regla del «EJECUTA», antes de la decisión sobre T05-H1): dos corridas `full` (900 s, en paralelo con pruebas, y 677 s) con SHA-256 idénticos y decisión `baseline_stays` (`trained_metrics.json` `8348939d…`, `trained_report.md` `fd0584e0…`, `trained_breakdown.md` `54800385…`); `ci` 49 s sin tocar versionados.

Segunda pasada (oficial, con M-ML-041 APROBADO y la desviación aceptada):

| Comando | Duración | Resultado |
| --- | ---: | --- |
| `uv run python -m thehouse_ml evaluate` (1.ª, `full`) | 396 s | exit 0; 4 líneas; decisión `logreg_promoted` |
| `uv run python -m thehouse_ml evaluate` (2.ª, `full`) | 437 s | exit 0; `diff` de los SHA-256 de `reports/*.json` y `reports/*.md` contra la 1.ª: **vacío** |
| `THEHOUSE_ML_TRAIN_PROFILE=ci uv run python -m thehouse_ml evaluate` | 34 s | exit 0; escribe solo en `models/ci/`; `diff` de los reportes versionados: **vacío** |

| Archivo | SHA-256 (las 2 corridas de la segunda pasada) |
| --- | --- |
| `reports/trained_breakdown.md` | `f2f2be0544c33f159d0a128ca9d73f0e198b2ed95d7a95a97dccd44f0aed743e` (nuevo) |
| `reports/trained_metrics.json` | `44eb1cd038bf07e91f9f0a7c2c2ebdc43bef55c453015f0227488c78412fa25b` (antes de T05 `288258c2…`) |
| `reports/trained_report.md` | `456a53f9dcb7b01c9bb79f96245ebac74ed2f72a3bd545a86fe3f3e8ff51310f` (antes de T05 `e8ffe126…`) |
| `reports/ablation_metrics.json`, `reports/ablation_report.md` | sin cambio (`f8c1a80b…`, `732cf41a…`) |
| `reports/baseline_metrics.json`, `reports/baseline_report.md` | sin cambio (`df36b2f1…`, `3b554e9b…`) |

Archivos protegidos con el mismo SHA-256 antes y después: `models/logreg.json` (`341ea3ec…`), `models/lightgbm_lambdarank.txt` (`5eb8d22d…`), `train_params.json` (`7b836eaa…`), `feature_schema.json` (`67af2e48…`), `baseline_params.json` (`51b663e8…`), `baseline_eval_params.json` (`c29ee164…`).

### Trazabilidad R05 → prueba

| Criterio | Prueba o comando |
| --- | --- |
| Recorte al `training_range` antes de puntuar; fuera del rango = entrada recortada | `test_trained_breakdown.py::test_inputs_beyond_the_training_range_score_as_the_clipped_input`, `::test_breakdown_properties_hold_for_random_vectors[beyond-training-range]` |
| Calibración afín, `maxPoints` proporcionales y Σ = 100; `pets_compatible` al término fijo | `::test_calibration_has_the_six_baseline_factors_with_max_points_summing_100`, `::test_max_points_are_proportional_to_the_contribution_range_of_each_factor`, `::test_pets_compatible_goes_to_the_fixed_term_not_to_a_factor`, `::test_only_the_standard_transform_and_the_schema_order_are_accepted` |
| Propiedades: total en [0, 100], factor en [0, `maxPoints`], tolerancias y redondeo de M-ML-029/031 | `::test_breakdown_properties_hold_for_random_vectors` (800 vectores), `::test_extremes_give_0_and_100`, `::test_batch_scores_match_the_single_pair_scores`; verificación en los 30,755 pares de test |
| Orden idéntico al logit; métricas sin cambios | `::test_ranking_by_raw_total_is_the_logit_ranking_inside_the_range`, `::test_check_sums_and_check_order_detect_a_logit_that_is_not_the_calibrated_one`; 300/300 estudiantes en la etapa |
| Pares excluidos sin puntaje y perfil incompleto como en ML-04 | `::test_excluded_pairs_and_incomplete_profiles_behave_as_in_ml04` (81 casos dorados) |
| Formato (lista `features`, `maxPoints` variables) | `::test_output_has_the_baseline_format_with_features_lists_and_variable_max_points` |
| Monotonía por factor | `::test_budget_factor_is_monotone_when_price_below_min_has_the_expected_sign`, `::test_collinear_budget_features_are_reported_with_the_exact_ratio_threshold`, `::test_distance_factor_with_a_wrong_sign_is_detected`, `::test_coefficient_table_flags_signs_against_the_natural_variable`, `::test_observed_budget_ratio_reads_the_implicit_max_over_min_of_the_pairs`, `::test_check_monotonicity_reports_observed_pairs_outside_the_violation_zone` |
| Decisión con la condición 3 | `test_trained_eval.py::test_logreg_meeting_the_three_conditions_is_promoted`, `::test_logreg_failing_the_breakdown_keeps_the_baseline_and_names_the_failed_checks`, `::test_without_a_verified_breakdown_the_logreg_is_not_promoted` |
| Pipeline solo con transformadores con convertidor en skl2onnx (lista verificada) | `::test_pipeline_uses_only_estimators_listed_with_a_skl2onnx_converter` (`StandardScaler`→`SklearnScaler`, `KBinsDiscretizer`→`SklearnKBinsDiscretizer`, `LogisticRegression`→`SklearnLinearClassifier`; recorte fuera del pipeline = nodo `Clip` en ML-06) |
| Reproducible, perfil `ci` sin tocar versionados, sin archivos ocultos | `test_trained_eval_stage.py::test_running_twice_does_not_change_any_file`, `::test_ci_profile_does_not_modify_the_versioned_reports`, `::test_evaluate_never_reads_the_hidden_label_files` (incluye los 2 módulos nuevos); corridas reales |

### Comandos y resultados (T05)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `163 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy` | `Success: no issues found in 50 source files` (PASS) |
| `uv run pytest tests/unit/test_trained_breakdown.py tests/unit/test_trained_eval.py` | 1.ª pasada 124 passed; 2.ª pasada **129 passed** (PASS) |
| `uv run pytest tests/integration/test_trained_eval_stage.py tests/integration/test_ablation_stage.py tests/unit/test_stages.py tests/unit/test_cli.py tests/integration/test_installed_package.py tests/test_repo_hygiene.py` (2.ª pasada) | **88 passed** en 78.2 s (PASS) |
| Cobertura (1.ª pasada: `test_trained_eval_stage.py` + unitarias de T05) | 136 passed; `trained_breakdown.py` 99 %, `trained_breakdown_report.py` 97 %, `trained_eval.py` 94 %, `trained_report.py` 97 % (96.6 % en conjunto) (PASS; no se repitió con la 2.ª pasada) |
| `evaluate` × 2 (`full`) y `ci`, en cada pasada | SHA-256 idénticos; versionados intactos con `ci` (PASS) |
| Suite completa | **No ejecutada** (se pidieron las afectadas); corresponde a T07 |

### Tabla de coherencia (T05)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §12 | Desglose por los 6 factores que suma el total en todos los pares de test | PASS |
| Constitución §11.1 | Pares excluidos por HC01–HC03 sin puntaje; perfil incompleto como en ML-04 | PASS |
| Constitución §8 y §20.3 | «DATOS SINTÉTICOS» en el reporte, el JSON y el mensaje | PASS |
| Constitución §9 / §13.1 | Dos corridas idénticas; sin pickle (JSON) | PASS |
| Spec §6.2 y §7 regla 2 | Sin reentrenar ni volver a medir en test; sin ajustar nada con sus resultados | PASS |
| Spec §6.6 | Recorte, calibración afín, `pets_compatible` al término fijo, redondeo de M-ML-029/031, formato con lista `features` | PASS |
| Spec §6.5 condición 3 | El desglose suma el total en los 30,755 pares de test; monotonía calculada y reportada; la del presupuesto falla y es desviación aceptada (M-ML-041), decidida **después** de ver la verificación (registrado) | PASS WITH WARNINGS |
| M-ML-031 | `factor_points` y `match_score` reutilizados, suma de izquierda a derecha | PASS |
| AGENTS: rutas | `src/thehouse_ml/models/` (2 nuevos, 2 modificados), `tests/` (1 nuevo, 2 modificados), `reports/` (1 nuevo, 2 modificados), `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05) | PASS |
| AGENTS: protegidos y dependencias | Mismos SHA-256; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: un control no ejecutado no es PASS | Suite completa y CI del PR no ejecutados | PASS |

### Hallazgos (T05)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T05-H1 | Monotonía del factor presupuesto: el coeficiente de `price_below_min` es negativo; por debajo de `budget.min` el factor sube con la renta cuando `budget.max / budget.min` > 76.20 (máx. +0.124 puntos sobre 100). No ocurre en los datos sintéticos (máx. 2.83), pero sí en el dominio de NV-11. No se corrigió. Con la regla del «EJECUTA» (paso 6) la 1.ª pasada dejó el baseline; **después de ver ese resultado**, el responsable aceptó la desviación (opción 1): la condición 3 de la spec es «suma el total» y la monotonía fue una protección adicional. Codificada en `ACCEPTED_DEVIATIONS`; corrección (sin `price_below_min` o coeficiente ≥ 0) como trabajo futuro en `logreg-1.1.0`, evaluada con datos nuevos en ML-07 porque test ya se usó. | ADVERTENCIA (desviación aceptada) | Aceptado — M-ML-041 APROBADO; la model card (T06) y NV-23 deben mencionarla |
| T05-H2 | `maxPoints` reflejan el rango de contribución de cada factor en todo el `training_range`, **no su importancia** (la medida de importancia es la ablación de T04): distancia 72.6 contra 25 del baseline; presupuesto 9.6 y servicios 6.7. Los cuartos cercanos tienden a puntajes altos (mediana 81; la distancia da en promedio el 87 % de su máximo). | ADVERTENCIA | Abierto para T06: la model card lo explica y se avisa a `TheHouse-GenAI` (no presentar `maxPoints` como pesos de importancia) |
| T05-H3 | CI de la rama `feat/ml-05-breakdown`: el job `quality` se cancelaba por timeout; el paso «Pruebas y cobertura» tardaba 14m50s con `timeout-minutes: 15`. **Resuelto** con autorización del responsable (2026-11-24, solo para esto): `.github/workflows/ci.yml` sube `timeout-minutes` del job `quality` de 15 a 30 (repositorio público, costo $0, COST-001), sin cambiar nombres de jobs, pasos ni acciones; `tests/test_ci_workflow.py::test_quality_timeout_is_30_minutes` exige 30 en `quality` (y se quitó la aserción de 15 de `test_quality_runs_train_with_the_ci_profile_in_the_cli_step`). **Nota para T07:** optimizar las pruebas lentas (fixtures de sesión y no regenerar los datos completos en cada prueba) para volver a un margen holgado. | ADVERTENCIA → mitigada | Mitigado; optimización pendiente en T07; se confirma en el CI del PR |
| T05-H4 | `min_k`/`max_k` usan la caja de los `training_range`, que incluye combinaciones no alcanzables a la vez (p. ej. `price_to_budget` 1 con `price_below_min` 1): el presupuesto nunca llega exactamente a 0. | INFORMATIVO | Documentado en M-ML-041 |
| T05-H5 | La prueba de T03 de la decisión «sujeta a T05» se reemplazó por 4 de la decisión final (promovida, con desviación nombrada, condición 3 fallida y sin verificar) y dos de integración se ajustaron (cuarta línea de la etapa, nuevo reporte). Su intención no cambió. | INFORMATIVO | Corregido |
| T05-H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T05)

- **Seguridad:** sin secretos, sin red en las pruebas, sin servicios de pago (COST-001); sin pickle.
- **Datos y licencias:** solo datos sintéticos de `data/processed/` (fuera de git); el reporte trae ids sintéticos y agregados, sin datos personales. Sin dependencias nuevas.
- **Coherencia:** tabla anterior.

**Resultado T05: PASS WITH WARNINGS** — aceptación de R05 cumplida y verificada: recorte al `training_range`, calibración afín 0-100 y desglose por los 6 factores que suma el total en los 30,755 pares de test (tolerancias de M-ML-029/031), orden idéntico al del logit, propiedades, formato y pipeline exportable; la regresión logística queda **PROMOVIDA** (`logreg-1.0.0`, `baseline-1.0.0` de respaldo). Advertencias: T05-H1 (no monotonía del presupuesto, desviación aceptada después de ver la verificación; corrección en `logreg-1.1.0`/ML-07) y T05-H2 (`maxPoints` ≠ importancia; para T06 y `TheHouse-GenAI`). M-ML-041 **APROBADO** (César Yael López Trujillo, 2026-11-24). T05 **completa** (ML-05 en 5/7). Sin commit, por instrucción del responsable.

## T06 · Model card

Lecturas previas: `constitution.md` (§8, §10, §11, §12) → `AGENTS.md` → `memoria.md` (M-ML-029 a M-ML-041) → `spec.md` (R06, §6.5–§6.7), `plan.md` y `task.md` de ML-05 → `reports/` (`baseline_metrics.json`, `train_cv_metrics.json`, `trained_metrics.json`, `ablation_metrics.json`, `trained_report.md`, `ablation_report.md`, `trained_breakdown.md`, `data_card.md`) → `src/thehouse_ml/models/` y `src/thehouse_ml/data_card.py` (patrón de generación).

### Qué se hizo

- **Nuevo** `src/thehouse_ml/models/model_card.py`: `collect_inputs` lee los 4 JSON de `reports/`, los pesos del baseline (`default_config()`) y las versiones de `uv.lock`; `check_inputs` exige la corrida oficial sintética (`synthetic`, perfil `full`, `official`, mismo SHA-256 de los modelos en los dos JSON); `render_model_card` (puro) escribe las 10 secciones; `run_model_card` es la única escritura. Ninguna métrica escrita a mano (M-ML-042).
- **Modificado** `src/thehouse_ml/models/trained_eval.py`: `run_trained_evaluate` regenera la ficha al final de `evaluate` solo con el perfil `full` (5.ª línea del mensaje). El perfil `ci` no la escribe.
- **Generado** `reports/model_card.md` llamando a `run_model_card` directamente (sin `train` ni `evaluate`; ningún JSON de `reports/` cambió).
- **Pruebas:** nuevo `tests/unit/test_model_card.py` (28); `tests/integration/test_trained_eval_stage.py` ajustada a la 5.ª línea del mensaje de `evaluate` (omisión en el proyecto temporal, que no tiene `uv.lock`).

### Trazabilidad R06 → prueba (`tests/unit/test_model_card.py`)

| Aceptación | Prueba |
| --- | --- |
| 10 secciones en orden | `test_has_the_ten_sections_in_order` |
| «DATOS SINTÉTICOS» en título, aviso y secciones de datos y métricas | `test_title_and_banner_say_synthetic_data` |
| Menciona T05-H1, T05-H2, T05-H3, T04-H2, T04-H3 y `TheHouse-GenAI` | `test_mentions_the_required_findings` (6 casos) |
| Cifras = JSON (resumen, uso, datos, rejilla y SHA-256, pliegues y test, ablación, segmentos, hipótesis, calibración, `maxPoints` y pesos, limitaciones, versiones) | `test_summary_…`, `test_intended_use_…`, `test_data_section_…`, `test_training_section_…`, `test_fold_and_test_metrics_…`, `test_ablation_segments_and_hypothesis_…`, `test_breakdown_…`, `test_limitations_…`, `test_future_work_and_traceability`, `test_library_versions_come_from_uv_lock` |
| Sin cifras escritas a mano | `test_no_figure_is_hardcoded_a_changed_json_value_changes_the_card` |
| La ficha versionada es la generada | `test_the_versioned_model_card_is_the_generated_one` |
| Reproducible (dos generaciones, mismo SHA-256, igual al versionado) | `test_two_generations_have_the_same_sha256` |
| Solo la corrida oficial sintética; SHA-256 coherentes | `test_only_the_official_synthetic_run_is_accepted` (4), `test_model_hashes_must_agree_between_reports` |
| Sin rutas absolutas ni datos personales; omisión sin entradas | `test_output_is_deterministic_and_has_no_absolute_paths_or_personal_data`, `test_without_inputs_it_is_skipped_and_writes_nothing` |

### Comandos y resultados (T06)

| Comando | Resultado |
| --- | --- |
| `run_model_card(ProjectPaths.resolve(None))` × 2 (vía `uv run python -c`) | Mismo SHA-256 `e4873dc1203cf895276667c766dd68c59017d7d9ce3769b3c84649a459a0ecb7` (PASS); tras la aprobación (M-ML-042 pasa a APROBADO en la sección 10), regenerada dos veces: `5a24b161a9a52ad781d08a1de7a591c48fa2e40da84afbff7b747d299f45dc09` (PASS) |
| `uv run ruff format --check .` | `167 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy` | `Success: no issues found in 51 source files` (PASS); además `mypy tests/unit/test_model_card.py` sin errores |
| `uv run pytest tests/unit/test_model_card.py` | **28 passed** (PASS) |
| `uv run pytest tests/test_repo_hygiene.py tests/unit/test_stages.py tests/unit/test_cli.py tests/unit/test_model_card.py` | **96 passed** (PASS) |
| `uv run pytest tests/integration/test_trained_eval_stage.py tests/integration/test_ablation_stage.py` | **18 passed** en 90.8 s (PASS) |
| `git diff` de `reports/*.json` y params de `src/thehouse_ml/models/` | Sin cambios (PASS) |
| Suite completa, cobertura y `train`/`evaluate`/`prepare` | **No ejecutados** por instrucción del responsable (los corre el CI; cobertura en T07) |

### Tabla de coherencia (T06)

| Referencia | Comprobación | Resultado |
| --- | --- | --- |
| Constitución §8 y §20.3 | «DATOS SINTÉTICOS» en el título y advertencia de circularidad (validan el pipeline, no el desempeño real; ML-07) | PASS |
| Constitución §10 | Comparación contra el baseline por pliegue y en test con IC y diferencia pareada | PASS |
| Constitución §11 | Sin atributos sensibles (solo `livingWith` declarado); HC01–HC03 sin puntaje; segmentos por campus, presupuesto y sin `seed_sin_verificar` | PASS |
| Constitución §12 | Desglose explicado y nota para `TheHouse-GenAI` (`maxPoints` no es importancia) | PASS |
| Spec R06 | Uso previsto, datos y etiquetas, modelos e hiperparámetros, métricas, ablación, segmentos, desglose de ejemplo (coeficientes y puntos por unidad; ejemplos completos enlazados, T06-H1 traspasado a T07), limitaciones, decisión y trabajo futuro | PASS |
| M-ML-040 / M-ML-041 | Salvedades T04-H2/H3, desviación T05-H1, T05-H2 y T05-H3 incluidas | PASS |
| AGENTS: rutas | `reports/model_card.md`, `src/thehouse_ml/models/` (1 nuevo, 1 modificado), `tests/` (1 nuevo, 1 modificado), `memoria.md`, `specs/ml-05-trained-model/`, `specs/README.md` (fila de ML-05) | PASS |
| AGENTS: dependencias, modelos, métricas y params | Sin dependencias nuevas; modelos, JSON de `reports/` y params sin cambios | PASS |
| AGENTS: un control no ejecutado no es PASS | Suite completa, cobertura y CI no ejecutados; declarados arriba | PASS |

### Hallazgos (T06)

| ID | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T06-H1 | La distribución de `matchScore`, los puntos por factor en test (mediana 81; distancia al 87 % de su máximo) y los ejemplos explicados solo están en `reports/trained_breakdown.md`, no en un JSON. La ficha los enlaza (§4 y §6) sin copiarlos y da como desglose de ejemplo los coeficientes con «puntos por unidad». Llevarlos a `trained_metrics.json` exigiría volver a correr `evaluate`. | ADVERTENCIA → traspasada | **Traspasado a T07** (responsable, 2026-11-24): agregar a `trained_metrics.json` la distribución de `matchScore` y los ejemplos explicados, y que la model card los tome de ahí |
| T06-H2 | La fecha de promoción, los PR #30–#35 y la lista de decisiones son constantes del módulo (no están en ningún JSON); la fila de T06 dice «PR pendiente» hasta que exista. | INFORMATIVO | **Aceptado** (responsable, 2026-11-24); actualizar la fila de T06 al abrir el PR |
| T06-H3 | No se calculó un IC pareado LightGBM − regresión logística (la spec compara cada modelo contra el baseline); la ficha describe el casi empate con las medias y los IC de cada modelo y lo declara. | INFORMATIVO | Documentado en la ficha |
| T06-H4 | La ficha incluye la nota para `TheHouse-GenAI` y la desviación T05-H1 para NV-23, pero el aviso a esos equipos es del responsable. | INFORMATIVO | Pendiente del responsable |
| T06-H5 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T06)

- **Seguridad:** sin secretos, sin red, sin servicios de pago (COST-001); sin pickle (la prueba de higiene pasa).
- **Datos y licencias:** la ficha solo tiene agregados e ids sintéticos (`S1078`); sin datos personales ni rutas absolutas. Sin dependencias nuevas.
- **Coherencia:** tabla anterior.

**Resultado T06: PASS** — aceptación de R06 cumplida y verificada: `reports/model_card.md` generada desde los JSON de `reports/` con las 10 secciones, «DATOS SINTÉTICOS», cifras iguales a los JSON y reproducible (mismo SHA-256 en dos generaciones). T06-H1 traspasado a T07 (distribución de `matchScore` y ejemplos en `trained_metrics.json`, leídos por la model card); T06-H2 aceptado. M-ML-042 **APROBADO** (César Yael López Trujillo, 2026-11-24). T06 **completa** (ML-05 en 6/7). Sin commit, por instrucción del responsable.

## T07 · Pruebas y evidencia (cierre de ML-05)

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-05 1.0.0, task T07 (ML-05-R07), en la rama feat/ml-05-cierre. Rutas permitidas: src/thehouse_ml/, tests/, reports/, memoria.md, specs/ml-05-trained-model/, specs/README.md y specs/ARQUITECTURA.md (solo lo de ML-05). Sin dependencias nuevas. No cambies los modelos entrenados, train_params.json ni ningún params» — César Yael López Trujillo, 2026-11-24 |
| Rama | `feat/ml-05-cierre` |
| Commit base | `44a89ee` (merge del PR #36, T06; igual a `origin/main`) |
| Commit (T07) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-24 |
| Decisión | M-ML-043 (APROBADO — César Yael López Trujillo, 2026-11-24) |
| Modelos y parámetros | `models/logreg.json`, `models/lightgbm_lambdarank.txt`, `models/train_manifest.json`, `train_params.json`, `feature_schema.json` 2.0.0, `baseline_params.json`, `baseline_eval_params.json`, `labels_params.json` y los params de ML-03 **sin cambios** (SHA-256 iguales antes y después) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` → `AGENTS.md` → `memoria.md` (M-ML-035 a M-ML-042) → `spec.md`, `plan.md`, `task.md` y esta evidencia (sobre todo los traspasos a T07: T06-H1, T06-H2 y T05-H3) → `specs/ml-04-baseline-model/evidence.md` (modelo del cierre).

### Qué se hizo

1. **T06-H1 — distribución y ejemplos en `trained_metrics.json`.** `src/thehouse_ml/models/trained_breakdown_report.py` calcula una sola vez `distribution_document` (pares, mínimo, máximo, media, percentiles p5–p95 con interpolación lineal, histograma por decenas con las mismas clases que el baseline —`HISTOGRAM_BINS`— y, por factor, media, p25, mediana, p75 y proporción del máximo) y `examples_document` (los ejemplos de la regla fija de M-ML-041 decisión 8 con su top 5 —`room_id`, `matchScore`, puntos por factor, relevancia sintética y `matchScore` del baseline— y la salida JSON del primer lugar). Los dos van al bloque `breakdown` de `reports/trained_metrics.json` (`distribution` y `examples`) y `reports/trained_breakdown.md` §4 y §6 se renderizan **desde esos mismos diccionarios**. `src/thehouse_ml/models/model_card.py` los lee del JSON (sección 7: tablas de distribución y de puntos por factor, los 3 ejemplos con su top 5 y la salida JSON del primer lugar como «desglose de ejemplo»; la explicación de T05-H2 cita la mediana y la proporción de la distancia del JSON) y `check_inputs` exige los dos bloques.
2. **Una sola corrida de `evaluate`** (perfil `full`) para regenerar los reportes, con SHA-256 de modelos, params, datos y reportes antes y después (ver «`evaluate` y SHA-256»).
3. **T05-H3 — pruebas lentas.** Se midieron las 15 más lentas con `--durations=15` y se optimizaron las 10 primeras sin quitar ninguna aserción (ver «Pruebas lentas»). `.github/workflows/ci.yml` y su `timeout-minutes: 30` **sin cambios**.
4. **T06-H2:** la fila de T06 de la model card lleva el PR #36 (`docs/ml-05-model-card`); se agrega la fila de T07 (pendiente de su PR) y M-ML-043 a la lista de decisiones. La limitación y el trabajo futuro de T05-H3 dicen ahora que las pruebas se optimizaron en T07.
5. **Suite completa una vez con cobertura** (`--cov-report=term-missing`).
6. **Documentos:** `spec.md`, `plan.md` (decía todavía «⛔ NO EJECUTAR HASTA «EJECUTA»», T07-H4) y `task.md` en ✅ COMPLETA (7/7); `specs/README.md` y `specs/ARQUITECTURA.md` (§3 salidas de `evaluate`, §4 módulos de `models/`, §6 fila de ML-05); M-ML-043 en `memoria.md` con la herencia para ML-06.

### `evaluate` y SHA-256

| Comando | Duración | Resultado |
| --- | ---: | --- |
| `uv run python -m thehouse_ml evaluate` (perfil `full`, **una vez**) | 352 s | exit 0; 5 líneas: baseline NDCG@10 0.527 [0.503, 0.550]; dif. NDCG@10 vs baseline logreg +0.335 [+0.312, +0.360], lightgbm +0.336 [+0.313, +0.360]; hipótesis > $3,500 confirmada; desglose de `logreg-1.0.0` en 30,755 pares, suma el total: sí; decisión `logreg_promoted`; model card regenerada |

SHA-256 de 52 archivos (`models/*`, `models/ci/*`, params de `models/` y `synthetic/`, `feature_schema.json`, `reports/*`, `data/processed/*.parquet` y los casos dorados) antes y después: **solo cambian 2**.

| Archivo | Antes (T06) | Después (T07) |
| --- | --- | --- |
| `reports/trained_metrics.json` | `44eb1cd038bf07e91f9f0a7c2c2ebdc43bef55c453015f0227488c78412fa25b` | `50c1a0c69fbe9aa662b79c5428ed9acc9d07b0708b9e794ee0668d91574bd54a` — solo gana `breakdown.distribution` y `breakdown.examples`: quitando esos dos bloques, el documento es **idéntico** al de T06 (comparación de los objetos JSON) |
| `reports/model_card.md` | `5a24b161a9a52ad781d08a1de7a591c48fa2e40da84afbff7b747d299f45dc09` | `85a60e302e11d90e15fcf95ad9a5772b3f2d1a7bd02e3caee218062d3ce5738a` (regenerada después con `run_model_card` tras un ajuste de redacción; dos generaciones con el mismo SHA-256) |
| `reports/trained_breakdown.md` | `f2f2be0544c33f159d0a128ca9d73f0e198b2ed95d7a95a97dccd44f0aed743e` | **igual** (el Markdown renderizado desde los bloques nuevos es byte a byte el de T05) |
| `models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` | **igual** |
| `models/lightgbm_lambdarank.txt` | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` | **igual** |
| `models/train_manifest.json` | `040c82b428ac9bc507f81bdafd3d46e40ac7ce4899600d6ac8bf7453a148009a` | **igual** |
| `src/thehouse_ml/models/train_params.json` | `7b836eaad4f4377d03dcf8972ee0695ce7b36d14b73007ba19e3f7980b005b8e` | **igual** |
| `reports/trained_report.md`, `ablation_*`, `baseline_*`, `training_report.md`, `train_cv_metrics.json`, demás reportes, params y Parquet | — | **iguales** |

Distribución en test (DATOS SINTÉTICOS, 30,755 pares): mínimo 8, p5 50, p10 61, p25 74, **p50 81**, p75 85, p90 88, p95 90, máximo 97, media 77.6; la distancia da en promedio el 87 % de su máximo (presupuesto 45 %, servicios 73 %, tipo de lugar 31 %, tipo de inmueble 73 %, extras 48 %). Ejemplos: `S0031` (≤ $2,000), `S0003` ($2,001–$3,500) y `S0044` (> $3,500), los tres de UTTT; los mismos de `trained_breakdown.md`.

### Pruebas lentas (T05-H3)

Las 10 más lentas de `uv run pytest --cov=thehouse_ml --cov-report=term-missing --durations=15` antes de T07 y qué se hizo con cada una:

| # | Prueba (fase) | Antes | Cambio | Después |
| ---: | --- | ---: | --- | ---: |
| 1 | `test_prepare_stage.py::test_cli_prepare_returns_zero` (call) | 84.45 s | Fixture de módulo `prepared_by_cli`: **una** corrida real de `prepare` por la CLI para las 3 pruebas de `prepare` completo | 91.55 s (setup del fixture compartido; call < 0.01 s) |
| 2 | `test_prepare_stage.py::test_run_stage_dispatches_prepare` (call) | 79.28 s | Reutiliza `prepared_by_cli` (espía del `run_stage` real que llama la CLI) | < 0.01 s |
| 3 | `test_prepare_stage.py::test_prepare_generates_the_data_card_with_all_sections` (call) | 77.16 s | Copia del proyecto de `prepared_by_cli` y `run_data_card` sobre la copia | 0.11 s |
| 4 | `test_synthetic_pairs.py::test_no_student_appears_in_two_splits` (setup: fixture `pair_set`, los 10 campus) | 46.45 s | **Sin cambio:** es la construcción única del conjunto completo (196,605 pares) que verifican las pruebas del módulo; reducirla quitaría lo que se prueba | 46.55 s |
| 5 | `test_synthetic_pairs.py::test_schema_2_0_0_keeps_the_same_pairs_splits_and_first_seven_features` (call) | 45.70 s | Comparación de esquemas en 2 campus (`few_pair_set`) + igualdad del subconjunto con el conjunto completo restringido | 9.11 s (+ 9.01 s del fixture `few_pair_set`, compartido con la n.º 10) ¹ |
| 6 | `test_trained_eval_stage.py::test_only_the_test_students_of_per_student_baseline_are_evaluated` (setup) | 36.58 s | Copia de `two_campus_trained_template` en lugar de `pairs` + `labels` + `train` + `evaluate` | 12.65 s |
| 7 | `test_synthetic_pairs.py::test_candidates_are_the_compatible_rooms_of_the_campus_up_to_the_maximum` (call) | 34.94 s | Recuento independiente compartido (`allowed_by_student`) con la prueba del 90 % (que tardaba 25.10 s y hacía el mismo recuento) | 0.01 s (+ 35.88 s del recuento compartido, una vez para las dos) |
| 8 | `test_ablation_stage.py::test_fold_metrics_have_the_four_metrics_and_reproduce_t02` (setup) | 30.39 s | Copia de `two_campus_trained_template` | 35.81 s: es el primer módulo que pide las plantillas y paga su construcción **una vez por sesión** (antes cada uno de los 6 módulos construía la suya) |
| 9 | `test_labels_stage.py::test_running_twice_does_not_change_any_file` (call) | 30.23 s | Una 2.ª corrida de `pairs` + `labels` comparada con el estado de la 1.ª (fixture `first_labels`), en lugar de dos corridas nuevas | 14.03 s |
| 10 | `test_synthetic_pairs.py::test_same_inputs_give_the_same_pairs_byte_for_byte` (call) | 28.62 s | La 1.ª construcción es `few_pair_set` (compartida); quedan la de orden inverso y la de otra semilla | 19.25 s |

Tiempos «antes» y «después» de las dos corridas completas con cobertura; ¹ las fases call de las pruebas 3, 5 y 7 y el fixture `few_pair_set` no salen entre las 15 más lentas de la corrida «después»: su tiempo es de una corrida auxiliar de `test_synthetic_pairs.py` y `test_prepare_stage.py` con cobertura (46 passed en 203 s). Las 10 sumaban 493.8 s antes y ~272 s después (incluidos los fixtures compartidos).

Qué verifican sigue igual:

- **`test_prepare_stage.py`:** el fixture de módulo `prepared_by_cli` corre **la CLI real** (`main(["--root", …, "prepare"])`) una vez; un espía alrededor del `run_stage` real registra la etapa y su mensaje. `test_cli_prepare_returns_zero` comprueba el código 0, `national_baselines.parquet` y además que la CLI imprime lo que devolvió la etapa; `test_run_stage_dispatches_prepare` comprueba que se despachó `prepare` y las 4 subetapas del mensaje; la de la data card trabaja sobre una **copia** del proyecto preparado. Antes cada una corría `prepare` completo.
- **Plantillas de sesión en `tests/conftest.py`** (`two_campus_pairs_template` → `two_campus_labels_template` → `two_campus_trained_template`; `copy_project`; `small_train_params`): el proyecto de 2 campus (UTTT y UCAN: 400 estudiantes, 600 cuartos) se construía igual en 6 módulos (`test_labels_stage`, `test_evaluate_stage`, `test_training_data_stage`, `test_train_stage`, `test_ablation_stage`, `test_trained_eval_stage`); ahora se construye una vez por sesión y **cada módulo trabaja sobre su propia copia**, así que el aislamiento entre módulos no cambia. La rejilla pequeña de `train` es la misma que tenían los módulos.
- **`test_labels_stage.py`:** `labels` se corre una vez por módulo (`first_labels`) para las pruebas que solo leen sus salidas; `test_running_twice_does_not_change_any_file` compara el estado tras la 1.ª corrida con el de una 2.ª corrida de `pairs` + `labels` (antes hacía dos corridas adicionales).
- **`test_synthetic_pairs.py`:** el recuento independiente con `filter_rooms` (T03) de los 2,000 estudiantes se hace una vez (`allowed_by_student`) y lo usan las dos pruebas que lo calculaban por separado; la prueba del esquema 2.0.0 compara esquemas en el subconjunto de 2 campus (`few_pair_set`) en lugar de reconstruir los 10 y **además** comprueba que ese subconjunto es exactamente el conjunto completo restringido a esos campus (particiones y candidatos se sortean por campus); `test_same_inputs_give_the_same_pairs_byte_for_byte` reutiliza `few_pair_set` como 1.ª construcción.

| Suite completa (`uv run pytest --cov=thehouse_ml --cov-report=term-missing --durations=15`) | Pruebas | Tiempo de pytest | Tiempo de pared |
| --- | ---: | ---: | ---: |
| Antes (base `44a89ee`, sin cambios de T07 en las pruebas) | 2,025 passed | **895.65 s (14 min 56 s)** | 897 s |
| Después (T07) | 2,031 passed (+6 pruebas nuevas de T06-H1/T06-H2) | **582.98 s (9 min 43 s)** | 584 s |
| Diferencia | | **−312.7 s (−34.9 %)** | −313 s |

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `167 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy` | `Success: no issues found in 51 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing --durations=15` (antes de T07) | **2025 passed in 895.65s (0:14:55)**, código 0; cobertura total reportada 94.24 %, **no válida** para `model_card.py` y `trained_breakdown_report.py` porque esos archivos se editaron mientras corría (T07-H6); vale como medición de tiempo |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing --durations=15` (después; **la** corrida de cobertura de T07) | **2031 passed in 582.98s (0:09:42)**, código 0; `Required test coverage of 80.0% reached. Total coverage: 96.25%` (PASS) |
| `uv run coverage report --include=<código de ML-05>` | No se repitió: la corrida auxiliar de tiempos sobrescribió `.coverage`. El agregado del código de ML-05 se calculó con las filas de `term-missing` de la corrida «después»: **96.64 %** (2,514 sentencias, 582 ramas) (PASS) |
| `uv run python -m thehouse_ml evaluate` (una vez) | exit 0, 352 s; ver «`evaluate` y SHA-256» (PASS) |
| `run_model_card(ProjectPaths.resolve(None))` × 2 | mismo SHA-256 `85a60e30…5738a` (PASS) |
| `git log --first-parent --format="%h %p \| %s" origin/main` y `git rev-list --first-parent --no-merges origin/main` | ML-05 entró solo por los PR #30–#36; el único commit de `main` que no es merge es el inicial `20b2c19` (PASS) |

No se ejecutaron `pip-audit` ni gitleaks en local: corren en el CI (`dependency-audit`, `secrets-gitleaks`); T07 no cambió dependencias (`pyproject.toml` y `uv.lock` sin cambios). No se reportan como PASS local (T07-H2). Ningún nombre nuevo de variable se llama `key`, `token`, `secret` ni `password` (los identificadores nuevos son `condition_id`, `factor_id`, `block`, `first_place`, etc.), para no repetir el falso positivo de gitleaks de T06.

### Cobertura por módulo de ML-05

De la corrida «después» (`--cov-report=term-missing`):

| Módulo | Sentencias | Sin cubrir | Ramas | Ramas parciales | Cobertura | Líneas sin cubrir |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `models/training_data.py` (T01) | 128 | 0 | 36 | 0 | 100 % | — |
| `models/train_config.py` (T02) | 212 | 3 | 72 | 2 | 98 % | 206-207, 209 |
| `models/train.py` (T02) | 368 | 9 | 66 | 8 | 96 % | 131, 200, 206, 281, 311, 315, 401, 403, 732 |
| `models/train_report.py` (T02) | 78 | 10 | 16 | 1 | **86 %** (mínimo) | 315-324 |
| `models/trained_eval.py` (T03–T06) | 344 | 13 | 86 | 13 | 94 % | 101, 109, 133, 199, 230, 243, 316, 376, 576, 578, 650, 692, 694 |
| `models/trained_report.py` (T03) | 86 | 1 | 26 | 2 | 97 % | 147, 250->261 |
| `models/factor_groups.py` (T04) | 24 | 0 | 4 | 0 | 100 % | — |
| `models/ablation.py` (T04) | 288 | 8 | 64 | 8 | 95 % | 132, 135, 138, 217, 242, 354, 463, 489 |
| `models/ablation_report.py` (T04) | 87 | 2 | 32 | 2 | 97 % | 64, 261 |
| `models/trained_breakdown.py` (T05) | 419 | 3 | 64 | 3 | 99 % | 412, 832, 863 |
| `models/trained_breakdown_report.py` (T05, T07) | 161 | 1 | 36 | 5 | 97 % | 74->73, 407->424, 429->453, 442->453, 543 |
| `models/model_card.py` (T06, T07) | 319 | 5 | 80 | 5 | 97 % | 127, 142, 186, 238, 721 |
| **Total del código de ML-05** | **2,514** | **55** | **582** | **49** | **96.64 %** | — |
| `stages.py` (modificado en T02–T03) | 46 | 0 | 14 | 0 | 100 % | — |
| **Total de `src/thehouse_ml`** | **7,826** | **209** | **2,106** | **159** | **96.25 %** | — |

Las líneas sin cubrir son validaciones defensivas y mensajes de error (p. ej., `train_report.py` 315-324: la sección del reporte para una rejilla sin configuración elegida). Todo módulo ≥ 80 %.

### Trazabilidad R07 → prueba

| Criterio de R07 | Prueba / evidencia |
| --- | --- |
| Pruebas unitarias e integración de lo implementado | `tests/unit/test_ml_dependencies.py`, `test_training_data.py`, `test_train_config.py`, `test_train.py`, `test_trained_eval.py`, `test_ablation.py`, `test_trained_breakdown.py`, `test_model_card.py`; `tests/integration/test_training_data_stage.py`, `test_train_stage.py`, `test_trained_eval_stage.py`, `test_ablation_stage.py`; `tests/test_repo_hygiene.py` (sin pickle); trazabilidad por requisito en las secciones T01–T06 |
| T06-H1 (nuevo en T07) | `test_trained_eval_stage.py::test_distribution_and_examples_are_in_the_json_and_match_the_markdown`; `test_model_card.py::test_distribution_and_examples_are_read_from_trained_metrics`, `::test_a_changed_distribution_or_example_changes_the_card`, `::test_without_distribution_or_examples_the_card_is_an_error` (2), `::test_without_examples_the_card_says_so`, `::test_the_versioned_model_card_is_the_generated_one` |
| T06-H2 | `test_model_card.py::test_future_work_and_traceability` (PR #30–#36, fila de T06 con #36, fila de T07 y M-ML-036 a M-ML-043) |
| Cobertura ≥ 80 % del código nuevo | «Cobertura por módulo de ML-05» |
| ruff, mypy y pip-audit | «Comandos y resultados» (pip-audit en el CI) |
| `evidence.md` con commit, comandos, resultados, hallazgos y coherencia | Esta sección, las tablas de coherencia y el «Resumen final de ML-05» |

### Tabla de coherencia — constitución

| § | Regla | Evidencia en el repo | Resultado |
| --- | --- | --- | --- |
| §2.4 / §13.2 | el contrato con el backend es `feature_schema.json` | Las entradas del modelo son las 10 columnas de `feature_order` del esquema 2.0.0 en ese orden (M-ML-037 decisión 5; `test_training_data.py`); `logreg.json` guarda `feature_order` y `feature_schema_version`; el esquema no cambió en ML-05 | PASS |
| §3 / SDD-001 | spec → plan → task → aprobación → «EJECUTA» → evidencia | `specs/ml-05-trained-model/{spec,plan,task,evidence}.md`; un «EJECUTA» por task en la cabecera; M-ML-036 a M-ML-043 | PASS |
| §4 / SDD-002 | specs independientes | `specs/README.md`: solo cambia la fila de ML-05; ML-06 y ML-07 siguen en borrador | PASS |
| §5 | stack de la tabla; dependencias solo con decisión M-ML | scikit-learn 1.9.1 y LightGBM 4.7.0 (tabla de stack) con M-ML-037; T02–T07 sin dependencias nuevas; `skl2onnx`/`onnxruntime` quedan para ML-06 | PASS |
| §6 / AGENTS | `data/raw`, `data/processed` y `models` fuera de git | `.gitignore`; `test_repo_hygiene.py`; `git status` no lista Parquet ni modelos | PASS |
| §7.4 | sin datos personales | Solo ids sintéticos (`S…`, `R…`) y agregados en reportes; `per_student_test` solo con `student_id`; los ejemplos nuevos de `trained_metrics.json` traen ids sintéticos, campus y `budget.max` | PASS |
| §8.1–§8.2 | sintéticos con semilla fija; las etiquetas de una utilidad se declaran | Semilla 20261124 (M-ML-038); «DATOS SINTÉTICOS» en `trained_metrics.json` (`synthetic`, `banner`, `warning`), `trained_report.md`, `ablation_*`, `trained_breakdown.md`, `training_report.md` y el título de `model_card.md`; advertencia de circularidad (ML-05-RK01) | PASS |
| §8.3 / §20.3 | desempeño real solo con señales reales | Ningún documento de ML-05 afirma desempeño real; ML-07 como reemplazo en cada reporte y en la model card | PASS |
| §9 | reproducibilidad | Dos corridas de `train` con SHA-256 idénticos (T02); `evaluate` repetido con SHA-256 idénticos (T03–T05); en T07 la única corrida de `evaluate` deja modelos, params, datos y demás reportes con el mismo SHA-256 y el JSON idéntico salvo los dos bloques nuevos; dos generaciones de la model card iguales; `test_running_twice_does_not_change_any_file` de `train`, `evaluate` y ablación | PASS |
| §10.1 | todo modelo entrenado se compara contra el modelo base; sin mejora no se promueve | Criterio de la spec §6.5 con 3 condiciones; regresión logística: 5/5 pliegues, Δ NDCG@10 pareada +0.335 [+0.312, +0.360], desglose que suma el total → **promovida**; LightGBM: referencia (no aditivo) | PASS |
| §10.2 | NDCG@10, Precision@5 y MAP por estudiante; partición por estudiante con validación cruzada | `evaluation/ranking.py` reutilizado; 5 pliegues `StratifiedGroupKFold` por estudiante y campus sobre train + validación (1,699 estudiantes), test (300) una sola vez | PASS |
| §10.3 | intervalos o variación | Media ± desviación entre pliegues (4 métricas); bootstrap pareado por estudiante 1,000/95 %/20261122 en test y por segmento | PASS |
| §11.1 | `livingWith` y presupuesto: restricciones duras | Solo se puntúan candidatos (pares que pasan HC01–HC03); `ExcludedPair` sin puntaje (`test_excluded_pairs_and_incomplete_profiles_behave_as_in_ml04`) | PASS |
| §11.2 | sin atributos sensibles como señal | Entradas = 10 features del esquema (sin género, edad ni origen); `test_training_data.py` | PASS |
| §11.3 | análisis por segmentos | 15 segmentos (total, 10 campus, sin `seed_sin_verificar`, 3 rangos de presupuesto) para baseline, LR y LightGBM con diferencias pareadas; en la model card | PASS |
| §12 | desglose por factor que suma el total | `logreg-1.0.0`: 6 factores, máx. \|Σ − `matchScore`\| 0.49997 ≤ 0.5 en los 30,755 pares de test; nota para `TheHouse-GenAI` en la model card | PASS (con la desviación aceptada T05-H1 sobre la monotonía, que no es parte de §12) |
| §13.1 | sin pickle | `logreg.json` (JSON) y `lightgbm_lambdarank.txt` (texto nativo); `test_no_module_of_the_package_uses_pickle` (AST) | PASS |
| §13.3 | versión semántica del modelo | `modelVersion` `logreg-1.0.0` en la salida, `trained_metrics.json` y la model card; `baseline-1.0.0` de respaldo | PASS |
| §13.4–§13.5 | paridad con casos dorados; 200 cuartos en < 20 ms | Herencia para ML-06 (M-ML-043): 81 casos del baseline más los de `logreg` | NO APLICA (ML-06) |
| §14 | sin secretos; dependencias fijas; artefactos con SHA-256 verificados | Sin secretos ni red en las pruebas; modelos cargados solo si su SHA-256 coincide con el manifiesto (T03); pip-audit y gitleaks en el CI (no ejecutados en local, T07-H2) | PASS (en el repo) · pip-audit y gitleaks: CI |
| §15 / COST-001 | costo $0 | CPU local (`train` ~9–13 min, `evaluate` ~6–7 min, suite ~10 min) y GitHub Actions del repositorio público; sin servicios de pago ni GPU | PASS |
| §16 | pruebas unitarias e integración; cobertura ≥ 80 % | 2,031 pruebas sin red (686 al cerrar ML-03, 1,678 al cerrar ML-04); 96.25 % total y 96.64 % en el código de ML-05 (mínimo por módulo 86 %) | PASS |
| §17 | CI con ruff, mypy, pytest+cobertura, pip-audit, gitleaks y pipeline | `ci.yml` con los 4 jobs; en ML-05 solo cambió con autorización: `env` del paso de la CLI (perfil `ci`, M-ML-038) y `timeout-minutes` de `quality` 15 → 30 (T05-H3); T07 no lo cambia; `tests/test_ci_workflow.py` en verde | PASS (local) · CI del PR de T07: pendiente |
| §18 | versión semántica | Spec 1.0.0 sin bump al cerrar; esquema 2.0.0 sin cambios | PASS |
| §19 | ramas con PR; sin commit directo a `main` | `git rev-list --first-parent --no-merges origin/main` → solo `20b2c19`; ML-05 entró por los PR #30–#36. La aprobación en GitHub no es auditable desde el clon local (sin `gh`) | PASS con observación |
| §20–§21 | leer antes; «EJECUTA»; sin métricas sintéticas como reales; sin scraping, pickle ni costos | «Lecturas previas» por task; autorizaciones por task; `constitution.md` y `AGENTS.md` sin modificar | PASS |

### Tabla de coherencia — AGENTS.md

| Regla | Evidencia en el repo | Resultado |
| --- | --- | --- |
| Leer constitution → AGENTS → memoria → spec/plan/task | «Lecturas previas» de cada sección | PASS |
| Responsable César (M-ML-011); cada task con su «EJECUTA» | Autorizaciones T01–T07 firmadas por César Yael López Trujillo | PASS |
| Aprobar spec / implementar tasks con «Apruebo» y «EJECUTA» | M-ML-036 (spec) y M-ML-037 a M-ML-043 (tasks) | PASS |
| Descargar datos de terceros | ML-05 no descargó datos (solo el wheel de LightGBM fijado en `uv.lock`, T01) | NO APLICA |
| Agregar dependencias solo con decisión M-ML | scikit-learn y LightGBM con M-ML-037; T07 sin dependencias | PASS |
| Publicar un artefacto (release) | ML-05 no publica artefactos (ML-06) | NO APLICA |
| Cambios en GitHub con permiso explícito | `ci.yml` solo en T02 (perfil `ci`) y T05 (timeout), con autorización expresa; T07 no toca GitHub | PASS |
| Merge solo con aprobación de otra persona | PR #30–#36 mezclados en GitHub; T07 sin commit ni PR; el agente no mezcla | PASS |
| Un plan aprobado no habilita tareas fuera de él | T07 solo hace los 7 puntos del «EJECUTA» | PASS |
| Flujo: implementar → probar → coherencia → seguridad y licencias | Cada sección T01–T07 con comandos, tabla de coherencia y revisión | PASS |
| Un control no ejecutado no se reporta como PASS | pip-audit, gitleaks, CI del PR de T07 y §13.4–§13.5 marcados como no ejecutados o NO APLICA | PASS |
| Cerrar una task solo con aceptación y evidencia reproducible | Comandos, resultados y SHA-256 por sección | PASS |
| Datos: nunca subir `raw`/`processed`/`models`; sin datos personales | `.gitignore`, `test_repo_hygiene.py`; reportes con ids sintéticos | PASS |
| Todo modelo se compara con el modelo base por reglas | Pliegues y test contra `baseline-1.0.0` (pareado) | PASS |
| Métricas sintéticas etiquetadas en todo reporte | «DATOS SINTÉTICOS» en los 7 reportes de ML-05 y en la evidencia | PASS |
| Sin pickle; solo ONNX; cada artefacto con `feature_schema.json`, `model_card.md`, `metrics.json` y SHA-256 | Sin pickle (higiene); `model_card.md` y métricas listas; el artefacto ONNX es de ML-06 | PASS · artefacto: NO APLICA (ML-06) |
| Rutas protegidas: `constitution.md` y `AGENTS.md` sin editar; sin secretos, scraping, pago ni commit directo a `main` | `git status` | PASS |
| Rutas permitidas de T07 | `git status`: `src/thehouse_ml/models/` (2 archivos), `tests/` (`conftest.py` y 9 módulos: 7 de integración y 2 unitarios), `reports/` (2), `memoria.md`, `specs/ml-05-trained-model/` (4), `specs/README.md` (fila de ML-05) y `specs/ARQUITECTURA.md` (solo ML-05); sin cambios en modelos, `train_params.json`, params, `pyproject.toml`, `uv.lock` ni `ci.yml` | PASS |
| Resultado de revisión PASS / PASS WITH WARNINGS / BLOCKED | Resultado por task y global | PASS |
| Convenciones: documentos en español, código en inglés, decisiones M-ML | Documentos en español; código, pruebas y claves JSON en inglés; M-ML-036 a M-ML-043 | PASS |

### Tabla de coherencia — decisiones M-ML-036 a M-ML-042

| Decisión | Qué exige | Archivo o prueba que lo cumple | Resultado |
| --- | --- | --- | --- |
| M-ML-036 · 1 | Única fuente `labels.parquet`; nunca los archivos ocultos (prueba obligatoria) | `test_train_never_reads_the_hidden_label_files`, `test_evaluate_never_reads_the_hidden_label_files` (T03–T05), `test_training_data_stage.py`; nombres ausentes de los módulos de ML-05 | PASS |
| M-ML-036 · 2 | 5 pliegues por estudiante y campus; test una vez | `train_manifest.json` (asignación); `train_cv_metrics.json`; T03 midió test una vez y T04–T07 reutilizan esa medición | PASS |
| M-ML-036 · 3 | LR pointwise exportable; LightGBM LambdaRank monotónico | `test_pipeline_uses_only_estimators_listed_with_a_skl2onnx_converter`; 0 violaciones monotónicas del LightGBM final | PASS |
| M-ML-036 · 4 | Criterio de 3 condiciones | `promotion` de `trained_metrics.json`: LR 5/5, IC > 0, desglose → `logreg_promoted` | PASS |
| M-ML-036 · 5 | Recorte al `training_range`, calibración afín, `breakdown` con lista `features` | `trained_breakdown.py`; `test_inputs_beyond_the_training_range_score_as_the_clipped_input`, `test_output_has_the_baseline_format_with_features_lists_and_variable_max_points` | PASS |
| M-ML-036 · 6 | scikit-learn y LightGBM con pip-audit limpio | `uv.lock`; CI del PR #31 | PASS |
| M-ML-037 | Versiones, override de mypy, `training_data.py`, `students.parquet` (2 columnas), validación del esquema, `join_labels`, prueba de humo, protección sin pickle | `test_ml_dependencies.py`, `test_training_data.py`, `test_no_module_of_the_package_uses_pickle` (21 prohibidos, 10 permitidos) | PASS |
| M-ML-038 | Semilla 20261124, `StratifiedGroupKFold` por estudiante, rejilla 8 + 8, selección, `num_threads` 1, JSON + texto nativo, perfil `ci` | `test_train.py`, `test_train_config.py`, `test_train_stage.py` (`test_two_runs_produce_identical_models_and_reports`, perfil `ci`) | PASS |
| M-ML-039 | Verificaciones de SHA-256, referencia reproducida, bootstrap pareado, condiciones 1 y 3, salidas | `test_trained_eval.py`, `test_trained_eval_stage.py` (`test_a_model_that_does_not_match_its_sha256_is_an_error`, `test_inputs_different_from_the_training_ones_are_an_error`) | PASS |
| M-ML-040 | Grupos de features, 4 métricas por pliegue, ablación, segmentos, regla de la hipótesis, perfil `ci` reducido | `test_ablation.py`, `test_ablation_stage.py`; hipótesis confirmada con salvedades T04-H2/H3 en la model card | PASS |
| M-ML-041 | Calibración, `maxPoints`, `pets_compatible` al término fijo, formato, `logreg-1.0.0`, monotonía con la desviación aceptada, ejemplos con regla fija | `test_trained_breakdown.py`; `ACCEPTED_DEVIATIONS` = `monotonicity_budget`; ejemplos de la regla fija ahora también en `trained_metrics.json` | PASS WITH WARNINGS (T05-H1 aceptada) |
| M-ML-042 | Model card generada desde los JSON, solo perfil `full`, nota para `TheHouse-GenAI` | `test_model_card.py` (33); **T06-H1 resuelto en T07**: la distribución y los ejemplos se leen de `trained_metrics.json` (la decisión 3 de M-ML-042, «enlazados, no copiados», queda superada por el traspaso aprobado) | PASS |

### Hallazgos (T07)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T07-H1 | `evaluate` se corrió una vez para regenerar `trained_metrics.json`; como la medición de test no cambia (mismos modelos, datos y código de cálculo), no es una segunda medición en el sentido de la spec §6.2: las métricas son idénticas a las de T03–T05 (JSON idéntico salvo los bloques nuevos) | INFORMATIVO | Cerrado |
| T07-H2 | `pip-audit` y gitleaks no se ejecutaron en local; corren en el CI. T07 no cambió dependencias. | INFORMATIVO | Pendiente del CI del PR; no se reporta como PASS |
| T07-H3 | La fila de T07 en la model card dice «pendiente» hasta que exista su PR (mismo caso que T06-H2). | INFORMATIVO | Actualizar al abrir el PR de T07 (o en ML-06) |
| T07-H4 | `plan.md` de ML-05 seguía con «⛔ NO EJECUTAR HASTA «EJECUTA»» en el estado de ejecución durante T01–T06 (como T07-H5 de ML-04). | INFORMATIVO | **RESUELTO** (✅ COMPLETA) |
| T07-H5 | Las plantillas de sesión de `tests/conftest.py` son de solo lectura por convención (cada módulo copia la suya); una prueba futura que escriba en la plantilla en lugar de en su copia afectaría a los demás módulos. | INFORMATIVO | Documentado en el docstring de las plantillas |
| T07-H6 | La cobertura de la corrida «antes» (94.24 %) no es válida para `model_card.py` (63 %) y `trained_breakdown_report.py` (65 %): esos archivos se editaron mientras la corrida estaba en marcha y `coverage` asocia las líneas ejecutadas al contenido del archivo en el momento del reporte (sus sentencias, 319 y 161, ya son las del código nuevo). Los módulos ya estaban importados, así que el **tiempo** de esa corrida sí vale. La cobertura oficial de T07 es la de la corrida «después» (96.25 %). | INFORMATIVO | Documentado; no afecta el resultado |
| T07-H7 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T07 |

### Revisión y resultado (T07)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas; modelos, params y datos con el mismo SHA-256; `ci.yml` sin cambios.
- **Coherencia:** tablas anteriores.

**Resultado T07: PASS** — se cumple la aceptación de R07: 2,031 pruebas en verde, cobertura 96.25 % total y 96.64 % en el código de ML-05 (mínimo 86 %), ruff y mypy sin errores; T06-H1 y T06-H2 resueltos; T05-H3 resuelto (suite de 895.65 s a 582.98 s, −34.9 %, sin cambiar el timeout del CI); una sola corrida de `evaluate` con modelos, params, datos y métricas existentes sin cambios (SHA-256). pip-audit y gitleaks quedan para el CI del PR. M-ML-043 **APROBADO** (César Yael López Trujillo, 2026-11-24). T07 **completa** (ML-05 en 7/7). Sin commit, por instrucción del responsable.

## Resumen final de ML-05 (cierre T07, 2026-11-24)

### Pull requests mezclados en `main` y sus commits

Verificado con `git log --first-parent --format="%h %p | %s" origin/main` y `git log <merge>^1..<merge>^2`. El único commit de `main` que no es merge es el inicial `20b2c19` (`git rev-list --first-parent --no-merges origin/main` → 1). ML-05 entró a `main` solo por PR:

| PR | Rama | Contenido | Commit de merge en `main` | Commit de la task |
| --- | --- | --- | --- | --- |
| #30 | `docs/ml-05-spec` | Spec 1.0.0 aprobada (M-ML-036) | `2bcb71e` | `8b04da7` |
| #31 | `feat/ml-05-deps-data` | T01 · dependencias y cargador (M-ML-037) | `386fd1d` | `15cf520` |
| #32 | `feat/ml-05-train` | T02 · validación cruzada y entrenamiento (M-ML-038) | `971beaf` | `98c0f88` |
| #33 | `feat/ml-05-eval` | T03 · medición única en test y decisión (M-ML-039) | `03a7b41` | `f406d07` |
| #34 | `feat/ml-05-ablation` | T04 · ablación, segmentos y 4 métricas por pliegue (M-ML-040) | `8dd9e6e` | `8ca8cc7` |
| #35 | `feat/ml-05-breakdown` | T05 · puntaje 0-100, desglose y promoción (M-ML-041) | `9c49499` | `233ef4d` y `9f084d6` (timeout de `quality`, T05-H3) |
| #36 | `docs/ml-05-model-card` | T06 · model card automática (M-ML-042) | `44a89ee` | `2019c40` |

T07 se entrega en `feat/ml-05-cierre` (base `44a89ee`) **sin commit ni PR**, por instrucción explícita del responsable.

### Resultados de ML-05 (DATOS SINTÉTICOS)

| | Baseline | Regresión logística (`logreg-1.0.0`) | LightGBM (referencia) |
| --- | ---: | ---: | ---: |
| NDCG@10 en pliegues (media ± desv.) | 0.5381 ± 0.0093 | 0.8623 ± 0.0035 | 0.8665 ± 0.0049 |
| NDCG@10 en test (300 estudiantes) | 0.527 [0.503, 0.550] | 0.862 [0.852, 0.873] | 0.863 [0.853, 0.872] |
| Δ NDCG@10 pareada contra el baseline | — | **+0.335 [+0.312, +0.360]** | +0.336 [+0.313, +0.360] |
| Pliegues por encima del baseline | — | 5/5 | 5/5 |
| Desglose aditivo que suma el total | sí | **sí** (30,755 pares) | no (no aditivo) |
| Decisión | respaldo | **PROMOVIDO** | referencia, no se promueve en v1 |

Ablación de la LR (Δ NDCG@10): distancia −0.2545, tipo de lugar −0.1118, presupuesto −0.0840, servicios −0.0384, extras −0.0195, tipo de inmueble −0.0137. Segmento > $3,500: Δ +0.391 [+0.338, +0.446], hipótesis confirmada con salvedades T04-H2/H3. `maxPoints` 9.6/72.6/6.7/4.9/1.7/4.5; mediana de `matchScore` en test 81. Son métricas sobre etiquetas sintéticas: validan el pipeline, no el desempeño real (ML-07).

### Cobertura

| Ámbito | Cobertura |
| --- | ---: |
| Total de `src/thehouse_ml` | **96.25 %** (mínimo exigido 80 %) |
| Código de ML-05 (12 módulos de `models/`) | **96.64 %** |
| Módulo de ML-05 con menor cobertura | `models/train_report.py` **86 %** |
| `stages.py` (modificado por ML-05) | **100 %** |
| Pruebas | **2,031** (1,678 al cerrar ML-04) |
| Tiempo de la suite con cobertura | 895.65 s antes de T07 → **582.98 s** después |

### Estado final de los hallazgos

| Origen | # | Hallazgo | Estado final |
| --- | --- | --- | --- |
| T01 | H1 | `libgomp.so.1` en el runner de Linux | RESUELTO (CI del PR #31) |
| T01 | H2–H5, H7 | `students.parquet` como 3.ª entrada, versión del esquema, `cloudpickle` transitiva, override de mypy, protección sin pickle | APROBADOS (M-ML-037) |
| T01 | H6 | Sin commit ni PR | CERRADO — PR #31 mezclado |
| T02 | H1 | Timeout del CI con la corrida completa | RESUELTO (perfil `ci`, M-ML-038) |
| T02 | H2 | Recorte fuera del pipeline (nodo `Clip` en ML-06) | Verificado en T05 (lista de transformadores); **conversión real en ML-06** (herencia de M-ML-043) |
| T02 | H3, H4, H5, H7, H9, H11 | Empate aparente de C, `quantile_bins`, `min_data_in_leaf` fijo, SHA-256 del manifiesto, apagado, modelos `ci` = oficiales | CERRADOS / documentados |
| T02 | H6 | Pruebas de `train: pendiente` | Corregido |
| T02 | H8 | Commit no registrado en `models/` | **Para ML-06** (lo registra el artefacto) |
| T02 | H10 | Sin commit ni PR | CERRADO — PR #32 mezclado |
| T03 | H1 | Consola cp1252 | Corregido |
| T03 | H2 | 4 métricas por pliegue | RESUELTO en T04 |
| T03 | H3–H6 | Perfil `ci` no oficial, empate LR–LightGBM, «test una vez», `stages.py` | CERRADOS / autorizado (H6) |
| T03 | H7 | Sin commit ni PR | CERRADO — PR #33 mezclado |
| T04 | H1 | `evaluate` oficial de ~7 min; tiempo del CI en ubuntu | Mitigado (perfil `ci`); los PR #34–#36 se mezclaron con los checks obligatorios del ruleset (§17), lo que indica que el job `quality` terminó; no verificado de forma independiente (sin `gh`) |
| T04 | H2, H3 | Salvedades de la hipótesis > $3,500 | ACEPTADAS; incluidas en la model card |
| T04 | H4, H5 | Distancia el grupo más influyente; pruebas ajustadas | INFORMATIVOS |
| T04 | H6 | Sin commit ni PR | CERRADO — PR #34 mezclado |
| T05 | H1 | No monotonía del presupuesto (`budget.max/budget.min` > 76.2) | **ACEPTADO como desviación (M-ML-041)**; en la model card; **pendiente de comunicar a NV-23**; corrección en `logreg-1.1.0` con datos de ML-07 |
| T05 | H2 | `maxPoints` ≠ importancia | RESUELTO en la model card (T06) con cifras del JSON (T07); **pendiente de comunicar a `TheHouse-GenAI`** |
| T05 | H3 | Timeout del job `quality` | Mitigado en T05 (30 min); **RESUELTO en T07** (pruebas lentas optimizadas: suite de 895.65 s a 582.98 s); se confirma en el CI del PR de T07 |
| T05 | H4, H5 | Caja de los `training_range`; pruebas de T03 reemplazadas | INFORMATIVOS |
| T05 | H6 | Sin commit ni PR | CERRADO — PR #35 mezclado |
| T06 | H1 | Distribución y ejemplos solo en `trained_breakdown.md` | **RESUELTO en T07** (`breakdown.distribution` y `breakdown.examples` en `trained_metrics.json`, leídos por la model card) |
| T06 | H2 | Fila de T06 «pendiente» | **RESUELTO en T07** (PR #36) |
| T06 | H3 | Sin IC pareado LightGBM − LR | INFORMATIVO (documentado en la ficha) |
| T06 | H4 | Aviso a `TheHouse-GenAI` y a NV-23 | **Pendiente del responsable** |
| T06 | H5 | Sin commit ni PR | CERRADO — PR #36 mezclado |
| T07 | H1–H7 | Ver «Hallazgos (T07)» | H1, H4 cerrados/resueltos; H2 pendiente del CI; H3 al abrir el PR; H5 documentado; H7 al mezclar el PR |
| Spec | RK01 | Circularidad de las etiquetas sintéticas | Mitigado (validación cruzada, test una vez, «DATOS SINTÉTICOS»); **ABIERTO hasta ML-07** |
| Spec | RK02 | Ningún modelo promovido | No ocurrió: `logreg-1.0.0` promovido |
| Spec | RK03 | Paridad del modelo en ONNX | Mitigado (pipeline con convertidores en skl2onnx); **pendiente en ML-06** |
| Backend | NV-23 | Esquema 2.0.0, round6 con `AwayFromZero`, recorte al `training_range`, `breakdown` con `maxPoints` variables y lista `features`, desviación T05-H1 | **ABIERTO — pendiente del backend** (aviso a cargo del responsable) |
| Backend | NV-12 | Contrato del cuarto (RK03 de ML-03) y `Property.AreaM2` | **ABIERTO — pendiente del backend** (heredado de M-ML-028/M-ML-035) |
| GenAI | `breakdown` | Nota de la sección 7 de la model card (`maxPoints` no es importancia; `value` sin recortar; sin `probability`/`logit`), formatos del par excluido y del perfil incompleto | **ABIERTO — coordinación del responsable con `TheHouse-GenAI`** |

### Pendiente y decisiones que esperan aprobación

- **Pendientes del backend:** NV-23 (esquema 2.0.0, round6/`AwayFromZero`, recorte, formato de `breakdown`, T05-H1); NV-12 (RK03) y `Property.AreaM2`.
- **Pendiente con `TheHouse-GenAI`:** la nota de la sección 7 de la model card y los formatos de `breakdown`, del par excluido y del perfil incompleto.
- **Pendiente del CI:** pip-audit, gitleaks, el tiempo del job `quality` y la corrida del pipeline en el PR de T07.
- **Decisiones:** M-ML-043 registrada como APROBADO según el «EJECUTA» de T07; ML-06 sigue en borrador y requiere su propia aprobación y «EJECUTA» (con la decisión M-ML que agregue `skl2onnx` y `onnxruntime`).

### Resultado global

**PASS WITH WARNINGS.** ML-05 1.0.0 queda **✅ COMPLETA (7/7)**: los 7 requisitos tienen evidencia (R01–R06 en sus secciones y R07 en «T07»), 2,031 pruebas en verde, cobertura de 96.25 % total y 96.64 % en el código de ML-05, ruff y mypy sin errores, `train` y `evaluate` reproducibles byte a byte y coherencia completa con la constitución, `AGENTS.md` y M-ML-036 a M-ML-042. **`logreg-1.0.0` queda promovido** (Δ NDCG@10 pareada +0.335 [+0.312, +0.360], 5/5 pliegues, desglose que suma el total) con `baseline-1.0.0` de respaldo, y la herencia para ML-06 está en M-ML-043. T07 queda en **PASS**; el global sigue con advertencias por: la **desviación aceptada T05-H1** (no monotonía del presupuesto con `budget.max/budget.min` > 76.2, corrección en `logreg-1.1.0`/ML-07); **pendientes externos** (NV-23: esquema 2.0.0, round6, recorte, formato de `breakdown` y T05-H1; NV-12/RK03 y `AreaM2`; nota de `maxPoints` y formatos con `TheHouse-GenAI`); **RK01** hasta ML-07; y pip-audit, gitleaks y el tiempo del CI a verificar en el PR de T07. Todas las métricas son **sintéticas**. Sin dependencias nuevas, sin descargas, sin secretos y sin commit, según lo autorizado.
