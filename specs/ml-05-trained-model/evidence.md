# ML-05 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-05 1.0.0 |
| Tareas ejecutadas | T01 · Dependencias y carga de datos (R01); T02 · Validación cruzada y entrenamiento reproducible (R02). Cada task con sección propia abajo |
| Estado | **🟡 EN EJECUCIÓN — T01 y T02 completas (2/7)** |
| Autorización (T01) | «EJECUTA ML-05 1.0.0, task T01 (ML-05-R01), en la rama feat/ml-05-deps-data. Rutas permitidas: pyproject.toml, uv.lock, src/thehouse_ml/ (un módulo de carga de datos de entrenamiento, por ejemplo en models/ o evaluation/ según ARQUITECTURA.md), tests/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). No modifiques feature_schema.json ni ningún params de ML-03/ML-04» — César Yael López Trujillo, 2026-11-23 |
| Rama (T01) | `feat/ml-05-deps-data` (commit base `2bcb71e`, merge del PR #30); mezclada en `main` con el PR #31 (`386fd1d`) |
| Autorización (T02) | «EJECUTA ML-05 1.0.0, task T02 (ML-05-R02), en la rama feat/ml-05-train. Rutas permitidas: src/thehouse_ml/ (models/ para el entrenamiento; evaluation/ y features/ solo para reutilizar), tests/, reports/, memoria.md, specs/ml-05-trained-model/ y specs/README.md (solo la fila de ML-05). Sin dependencias nuevas. No modifiques feature_schema.json, baseline_params.json ni los params de ML-03/ML-04. Los modelos entrenados van a data/processed/ o models/ (fuera de git), nunca con pickle» — César Yael López Trujillo, 2026-11-23 |
| Rama (T02) | `feat/ml-05-train` (commit base `386fd1d`, merge del PR #31) |
| Aprobación (T02) | «Apruebo M-ML-038 (César Yael López Trujillo, 2026-11-24) con sus 9 decisiones y la elección de C=0.01 por la regla registrada (T02-H3)»; en el mismo mensaje, por T02-H1: «Autorizo .github/workflows/ci.yml solo para esto: agrega un perfil de entrenamiento "ci" […] que use 1 combinación por modelo (las elegidas) y 2 pliegues, escriba sus salidas en una ubicación que no sobrescriba reports/training_report.md ni reports/train_cv_metrics.json versionados, y deje la corrida completa como la oficial. No cambies nombres de jobs ni timeouts» |
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | T01: 2026-11-23 · T02: 2026-11-23 y 2026-11-24 (retomada tras un apagado del equipo) |
| Decisiones | M-ML-036 (APROBADO, decisiones 1 y 6); M-ML-037 (**APROBADO**, César Yael López Trujillo, 2026-11-23: versiones, override de mypy, ubicación, tercera entrada `students.parquet`, validación del esquema, reutilización de `join_labels`, formato de salida, prueba de humo y protección sin pickle); M-ML-038 (**APROBADO**, César Yael López Trujillo, 2026-11-24: semilla, pliegues, rejilla, selección, `num_threads`, persistencia, salidas y perfil `ci`) |

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
