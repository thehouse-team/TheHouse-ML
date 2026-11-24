# ML-05 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-05 1.0.0 |
| Tareas ejecutadas | T01 · Dependencias y carga de datos (R01); T02 · Validación cruzada y entrenamiento reproducible (R02); T03 · Evaluación contra el baseline y decisión de promoción (R03); T04 · Ablación y segmentos (R04); T05 · Calibración 0-100 y desglose (R05). Cada task con sección propia abajo |
| Estado | **🟡 EN EJECUCIÓN — T01 a T05 completas (5/7)** |
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
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | T01: 2026-11-23 · T02: 2026-11-23 y 2026-11-24 (retomada tras un apagado del equipo) · T03: 2026-11-24 · T04: 2026-11-24 · T05: 2026-11-24 |
| Decisiones | M-ML-036 (APROBADO, decisiones 1 y 6); M-ML-037 (**APROBADO**, César Yael López Trujillo, 2026-11-23: versiones, override de mypy, ubicación, tercera entrada `students.parquet`, validación del esquema, reutilización de `join_labels`, formato de salida, prueba de humo y protección sin pickle); M-ML-038 (**APROBADO**, César Yael López Trujillo, 2026-11-24: semilla, pliegues, rejilla, selección, `num_threads`, persistencia, salidas y perfil `ci`); M-ML-039 (**APROBADO**, César Yael López Trujillo, 2026-11-24: ubicación y etapa, verificaciones, referencia reproducida, bootstrap pareado, condiciones 1 y 3, tabla por pliegue y salidas; `stages.py` autorizado retroactivamente; T03-H2 traspasado a T04); M-ML-040 (**APROBADO**, César Yael López Trujillo, 2026-11-24: grupos de features, pliegues y modelos de la validación cruzada, 4 métricas por pliegue, ablación, segmentos, regla de la hipótesis > $3,500, integración y perfil `ci`); M-ML-041 (**APROBADO**, César Yael López Trujillo, 2026-11-24: calibración afín, `maxPoints`, `pets_compatible` al término fijo, formato de `breakdown`, `modelVersion` `logreg-1.0.0`, regla de monotonía con la desviación aceptada T05-H1; `logreg-1.0.0` promovida) |

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
