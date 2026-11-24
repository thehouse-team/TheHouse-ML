# ML-05 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-05 1.0.0 |
| Tareas ejecutadas | T01 · Dependencias y carga de datos (R01). Cada task con sección propia abajo |
| Estado | **🟡 EN EJECUCIÓN — T01 completa (1/7)** |
| Autorización (T01) | «EJECUTA ML-05 1.0.0, task T01 (ML-05-R01), en la rama feat/ml-05-deps-data. Rutas permitidas: pyproject.toml, uv.lock, src/thehouse_ml/ (un módulo de carga de datos de entrenamiento, por ejemplo en models/ o evaluation/ según ARQUITECTURA.md), tests/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). No modifiques feature_schema.json ni ningún params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-23 |
| Rama (T01) | `feat/ml-05-deps-data` (commit base `2bcb71e`, merge del PR #30) |
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-23 |
| Decisiones | M-ML-036 (APROBADO, decisiones 1 y 6); M-ML-037 (**APROBADO**, César Yael López Trujillo, 2026-11-23: versiones, override de mypy, ubicación, tercera entrada `students.parquet`, validación del esquema, reutilización de `join_labels`, formato de salida, prueba de humo y protección sin pickle) |

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
| Linux (CI `ubuntu-latest`) | `test_ml_dependencies.py` corre en el job `quality` | Pendiente del PR |

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
| T01-H1 | `lib_lightgbm.so` del wheel de Linux necesita `libgomp.so.1` del sistema. `ubuntu-latest` incluye GCC (y con él `libgomp1`), pero no se ha ejecutado en el runner. **Cambio propuesto al CI** (no aplicado; `ci.yml` fuera de las rutas): en el job `quality`, antes de «Entorno reproducible (uv.lock)», un paso `run: sudo apt-get update && sudo apt-get install -y --no-install-recommends libgomp1`. El job `windows-latest` no lo necesita (verificado localmente) y `dependency-audit` no importa LightGBM. | ADVERTENCIA (no verificado en Linux) | Abierto: se confirma con `test_ml_dependencies.py` en el CI del PR; si falla, se aplica el cambio con autorización |
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
