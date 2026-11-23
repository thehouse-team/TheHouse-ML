# ML-04 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-04 1.0.0 |
| Tareas ejecutadas | T02 · Esquema 2.0.0 y features nuevas (R02) y, después, T01 · Configuración versionada (R01); en una segunda ejecución, T03 · Puntaje y desglose (R03); en una tercera, T04 · Simulador de etiquetas sintéticas (R04); en una cuarta, T05 · Casos dorados (R05); en una quinta, T06 · Reporte del modelo base (R06); en una sexta, T07 · Pruebas y evidencia (R07, cierre). Cada una con sección propia abajo |
| Estado | **✅ ML-04 COMPLETA (7/7)** — cierre M-ML-035, 2026-11-23; ver «T07 · Pruebas y evidencia» y «Resumen final de ML-04» (al final) |
| Autorización (T02 y T01) | «EJECUTA ML-04 1.0.0, tasks T02 y después T01 (en ese orden, porque T01 valida contra las features de T02), en la rama feat/ml-04-config-schema. Rutas permitidas: src/thehouse_ml/ (features/ y el módulo del modelo base donde indique specs/ARQUITECTURA.md), tests/, reports/ (solo lo que regenere prepare), memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). data/processed/ se regenera pero sigue fuera de git. Sin dependencias nuevas. No modifiques students_params.json, rooms_params.json ni pairs_params.json» — César Yael López Trujillo, 2026-11-20 |
| Rama (T02 y T01) | `feat/ml-04-config-schema` (commit base `fac23bc`; mezclada en el PR #24, `643f94e`) |
| Autorización (T03) | «EJECUTA ML-04 1.0.0, task T03 (ML-04-R03), en la rama feat/ml-04-score. Rutas permitidas: src/thehouse_ml/models/, src/thehouse_ml/features/ (solo para reutilizar, no cambiar el esquema), tests/, memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). Sin dependencias nuevas (nada de hypothesis: las pruebas de propiedades usan rejillas o random.Random con semilla). No modifiques feature_schema.json (2.0.0) ni baseline_params.json (baseline-1.0.0)» — César Yael López Trujillo, 2026-11-20 |
| Rama (T03) | `feat/ml-04-score` (commit base `643f94e`, merge del PR #24) |
| Commit | Sin commit, por instrucción explícita del responsable (las dos ejecuciones) |
| Fecha de ejecución | 2026-11-20 |
| Decisiones | M-ML-029 (APROBADO); M-ML-030 (**APROBADO**, César Yael López Trujillo, 2026-11-20: lo que la spec no fijaba y la resolución de H1 y H2); M-ML-031 (**APROBADO**, César Yael López Trujillo, 2026-11-20: lo que T03 decidió y la corrección del redondeo por factor de T03-H1); M-ML-032 (**APROBADO**, César Yael López Trujillo, 2026-11-20: parámetros de las etiquetas sintéticas y desviaciones T04-H1 a T04-H3); M-ML-033 (**APROBADO**, César Yael López Trujillo, 2026-11-23: formato y casos de T05, con la decisión 7 del perfil incompleto); M-ML-034 (**APROBADO**, César Yael López Trujillo, 2026-11-23: valores SUPUESTO de la evaluación y dos desviaciones de T06) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§10–§13 y §18) → `AGENTS.md` → `memoria.md` (M-ML-022 a M-ML-029) → `specs/ARQUITECTURA.md` → `spec.md` (sobre todo §6 y §10), `plan.md` y `task.md` de ML-04 → `src/thehouse_ml/features/` y `specs/ml-03-feature-schema/evidence.md` (formato).

## T02 · Esquema 2.0.0 y features nuevas

### Qué se hizo

1. **`src/thehouse_ml/features/feature_schema.json` 1.1.0 → 2.0.0** (versión mayor, M-ML-029):
   - `metadata.schema_version` 2.0.0, `spec` «ML-04 1.0.0», `decision` y `generated_by` con 2.0.0.
   - `metadata.changelog` con la entrada 2.0.0 (M-ML-029): motivo «cambia las entradas del ONNX y lo que calcula NV-23».
   - `metadata.versioning`: un cambio en NV-11 **y** agregar, quitar o reordenar features de `feature_order` son versión mayor; ampliar un rango sin cambiar fórmulas ni orden es menor.
   - Tres features nuevas **al final** de `pair_features` y `feature_order` (posiciones 8–10). Las 7 anteriores no cambian: SHA-256 de su definición canónica idéntico antes y después (`938939bb…0318`).

   | Orden | Feature | Fórmula | `range` | `training_range` |
   | --- | --- | --- | --- | --- |
   | 8 | `distance_ratio` | `distance_km / maxDistanceKm` resuelto (nulo → 30); distancia real | `{"min": 0, "max": null}` | [0, 60] (30 km / 0.5 km; observado 59.797) |
   | 9 | `price_below_min` | `(budget.min − price_mxn) / budget.min` si `budget.min` > 0 y `price_mxn < budget.min`; si no, 0 | 0–1 | — |
   | 10 | `extra_services_ratio` | servicios del cuarto no pedidos / (11 − pedidos); 0 si pidió los 11 | 0–1 | — |

2. **`src/thehouse_ml/features/pairs.py`** — `compute_pair_features` calcula las 10 en el orden del esquema con una sola implementación: reutiliza `resolve_student` (faltantes de M-ML-022) y la misma `distance` de `distance_km`; el tamaño del catálogo (11) se lee del esquema (`catalogs.nv11.essentialServices`).
3. **`tests/fixtures/pair_feature_cases.json` 2.0.0** — los 35 casos de ML-03 conservan entradas y valores de las 7 features (comprobado: 0 cambios) y agregan los 3 valores nuevos; **13 casos nuevos** (48 en total). Valores calculados con un recuento independiente que no importa `thehouse_ml` (aritmética exacta con `fractions.Fraction` y la regla del meridiano 6371 km × Δφ de ML-03; script en el scratchpad de la sesión, fuera del repo). `hard_constraint_cases.json` solo sube `schema_version` a 2.0.0.
4. **`pairs.parquet` regenerado** con `prepare` (ver «Reproducibilidad»).
5. **Pruebas** — `tests/unit/test_feature_schema_v2.py` (nuevo) y ajustes en `test_pair_features.py`, `test_feature_schema_nv11.py`, `test_feature_schema_validation.py`, `test_synthetic_pairs.py` (prueba nueva de invariancia de pares) y `tests/integration/test_pairs_stage.py` (rangos de las 3 columnas). Nota en `tests/fixtures/README.md`.

### Casos nuevos de la tabla

| Escenario pedido | Caso (`id`) | Valor nuevo esperado |
| --- | --- | --- |
| `maxDistanceKm` nulo | `distance_ratio_null_max_distance` | 20.015087 / 30 = 0.667170 |
| Distancia = `maxDistanceKm` | `distance_ratio_equal_max_distance` | 1.0 (10 km / 10) |
| Quiebre de la curva (extra) | `distance_ratio_half` | 0.5 (10 km / 20) |
| `budget.min` nulo | `price_below_min_null_budget_min` | 0 (renta 500) |
| Renta igual a `budget.min` | `price_equal_budget_min` | 0 |
| Renta menor | `price_below_budget_min` | 500 / 2000 = 0.25 |
| Renta mucho menor | `price_far_below_budget_min` | 1800 / 2000 = 0.9 |
| Renta 0 (extra) | `price_zero_with_budget_min` | 1.0 |
| Sin servicios pedidos | `extras_no_services_requested` | 4/11 = 0.363636 |
| Pidió los 11 | `extras_all_eleven_requested` | 0 (cobertura 5/11) |
| Cuarto sin servicios extra | `extras_room_without_extra_services` | 0 |
| Cuarto con todos (extra) | `extras_room_with_every_service`, `extras_partial` | 9/9 = 1.0; 2/8 = 0.25 |

Además, en los 35 casos anteriores: `null_budget_min_and_max_distance` y `all_optional_null` cubren `maxDistanceKm` nulo (0 / 30 y 10.007543 / 30); `transport_*` dan `distance_ratio` 1.000754 (justo encima de 1); `distance_over_30_km` da 8.006 (40.03 km / 5).

### Reproducibilidad y comparación con 1.1.0

| Comprobación | Resultado |
| --- | --- |
| `pairs.parquet` 1.1.0 (antes de T02) | `8e6d4690…d11d` (igual al cierre de ML-03); copia en el scratchpad para comparar |
| Filas | 196,605 antes y después |
| `student_id`, `room_id`, `university_code`, `seed_verified`, `split` | **Idénticos** (mismo orden; `DataFrame.equals`) |
| Las 7 columnas anteriores y `synthetic` | **Idénticas** (igualdad exacta) |
| Columnas nuevas | `distance_ratio`, `price_below_min`, `extra_services_ratio` (float64), entre `pets_compatible` y `synthetic` |
| Particiones | train 1,400 · validation 300 · test 300 estudiantes (135,993 · 29,857 · 30,755 pares), sin cambios |
| Rangos observados | `distance_ratio` 0.000025–59.797 (26.3 % > 1); `price_below_min` 0–0.931 (62.5 % en 0); `extra_services_ratio` 0–1 |
| `students.parquet` / `rooms.parquet` | SHA-256 sin cambios (`d4ad048a…73c1`, `e2ab7e8e…8dca`) |
| `prepare` corrida 1 | `pairs.parquet` `1f5c4d16…1cae` |
| `prepare` corrida 2 | Código 0; `pairs.parquet` `1f5c4d16…1cae` (**idéntico** a la corrida 1); los 7 parquet y todos los `reports/*.md` con el mismo SHA-256 entre corridas |
| Reportes regenerados por `prepare` | `pairs_summary.md` (versión 2.0.0, columnas y tabla de §7 con las 3 features), `students_synthetic.md` y `rooms_catalog.md` (solo la línea «Esquema de features 2.0.0») |

La invariancia también queda como prueba reproducible sin `data/processed/`: `test_synthetic_pairs.py::test_schema_2_0_0_keeps_the_same_pairs_splits_and_first_seven_features` construye los pares con un esquema reducido a las 7 features de 1.1.0 y con el 2.0.0 y exige los mismos pares, particiones y valores.

### Trazabilidad R02 → prueba

| Criterio | Prueba |
| --- | --- |
| El esquema carga en 2.0.0 con changelog y versioning | `test_feature_schema_v2.py::test_the_schema_loads_as_major_version_2_0_0`, `::test_versioning_says_that_adding_removing_or_reordering_features_is_major`, `test_feature_schema_nv11.py::test_the_packaged_schema_loads_validates_and_exposes_its_version` |
| Nuevas al final; las 7 anteriores no cambian de posición ni de definición | `::test_the_new_features_are_appended_at_the_end`, `::test_the_seven_previous_definitions_did_not_change` (SHA-256), `test_feature_schema_nv11.py::test_feature_order_has_no_duplicates_and_covers_every_pair_feature` |
| Las 7 anteriores no cambian de valor | `::test_the_first_seven_values_equal_those_of_schema_1_1_0` (44 casos), `test_synthetic_pairs.py::test_schema_2_0_0_keeps_the_same_pairs_splits_and_first_seven_features` |
| Rangos de las 3 nuevas y `training_range` | `::test_ranges_of_the_new_features`, `::test_the_training_range_of_distance_ratio_is_the_bound_of_the_synthetic_catalog`, `test_pair_features.py::test_only_unbounded_features_can_leave_their_training_range`, `::test_every_computed_feature_is_valid_against_its_schema_range` |
| Fórmulas | `::test_distance_ratio_uses_the_real_distance_and_the_resolved_max_distance` (20), `::test_price_below_min_is_in_range_zero_at_or_above_the_minimum_and_non_increasing` (6), `::test_extra_services_ratio_over_subsets_of_the_catalog`, `::test_a_null_essential_services_counts_as_nothing_requested` |
| Casos de la tabla | `test_pair_features.py::test_table_case[*]` (48), `::test_the_table_covers_every_requested_scenario`, `test_feature_schema_v2.py::test_the_table_has_the_35_previous_cases_and_the_new_ones` |
| Reproducibilidad de `prepare` | `tests/integration/test_pairs_stage.py::test_running_twice_does_not_change_any_file` y dos corridas reales (tabla anterior) |

## T01 · Configuración versionada

### Qué se hizo

1. **`src/thehouse_ml/models/baseline_params.json`** (`baseline-1.0.0`, ubicación de `specs/ARQUITECTURA.md` §4 y del plan), empaquetado con `thehouse_ml.models`:
   - `metadata`: `model_version` `baseline-1.0.0`, `spec` ML-04 1.0.0, `feature_schema_version` **2.0.0**, regla de compatibilidad, pesos fijados antes de las etiquetas (spec §7 reglas 2 y 3) y sin atributos sensibles (constitución §11.2).
   - `curve_semantics`: lineal por tramos, extremos constantes.
   - `factors` (orden de la tabla §6.2 = orden del `breakdown`):

     | `id` | Peso | Feature | Puntos de la curva |
     | --- | --- | --- | --- |
     | `budget` | 30 | `price_below_min` | (0, 1) → (1, 0) |
     | `distance` | 25 | `distance_ratio` | (0, 1) → (0.5, 1) → (1, 0); 0 después |
     | `essential_services` | 20 | `services_coverage` | identidad (0, 0) → (1, 1) |
     | `place_type` | 10 | `place_type_match` | identidad |
     | `property_type` | 10 | `property_type_match` | identidad («modalidad», constitución §12) |
     | `extras` | 5 | `extra_services_ratio` | identidad |

   - `non_scoring_features` con motivo: `distance_km`, `effective_distance_km`, `price_to_budget` y `pets_compatible`.
   - `rounding`: 1 decimal por factor (`half_up`); total `floor(round6(Σ) + 0.5)` en [0, 100]; ranking por la suma sin redondear (desc) con desempate `room_id` (asc); tolerancias 0.5 y 0.3; nota de .NET con `MidpointRounding.AwayFromZero` (`ToEven` no compatible).
2. **`src/thehouse_ml/models/baseline_config.py`** — cargador tipado con la biblioteca estándar (`BaselineConfig`, `Factor`, `Curve`, `NonScoringFeature`, `RoundingRules`; `load_config`, `parse_config`, `default_config`, `schema_is_compatible`). Acumula **todos** los problemas y lanza `BaselineConfigError` («baseline_params.json inválido: …»). Valida: pesos positivos que suman 100; cada feature existe en `feature_schema.json`; cada feature del esquema puntúa en un solo factor o tiene motivo; curvas con ≥ 2 puntos, `x` estrictamente creciente dentro del `range` de la feature e `y` en [0, 1]; esquema de la misma versión mayor y no anterior; modos y campos de redondeo soportados. `Curve.evaluate` define la semántica declarativa (la usará T03).
3. **Pruebas** — `tests/unit/test_baseline_config.py` (76).

### Trazabilidad R01 → prueba

| Criterio | Prueba |
| --- | --- |
| Archivo `baseline-1.0.0` con factores, pesos, features y curvas de la §6.2 | `test_baseline_config.py::test_the_packaged_config_is_baseline_1_0_0_for_schema_2_0_0`, `::test_factors_weights_and_features_follow_the_spec_table`, `::test_curves_are_the_declared_piecewise_linear_points`, `::test_curve_values_match_the_formulas_of_the_spec` (15) |
| Ejemplo de la §6.4 (77.976190) con estos pesos y curvas | `::test_the_example_of_spec_6_4_adds_up_to_77_976190` |
| Features que no puntúan, con motivo | `::test_every_schema_feature_scores_or_has_a_reason` |
| Redondeo de M-ML-029 y nota de .NET | `::test_rounding_rules_of_m_ml_029` |
| Pesos que no suman 100 | `::test_broken_configs_are_rejected_with_a_clear_message` (8 casos de pesos) |
| Feature inexistente | ídem (8 casos de features y cobertura) |
| Curva inválida | ídem (11 casos de curvas) |
| Versión de esquema distinta | `::test_a_schema_with_another_major_version_is_rejected` (1.1.0 y 3.0.0), `::test_a_later_minor_schema_is_accepted`, `::test_schema_compatibility_is_same_major_and_not_older` (8), y 3 casos de metadatos |
| Todos los problemas a la vez | `::test_every_problem_is_reported_at_once` |
| Sin atributos sensibles | `::test_no_factor_uses_a_sensitive_attribute` |

«La salida y el reporte muestran la versión» (aceptación de T01): la configuración expone `model_version` y `feature_schema_version`; la salida (`modelVersion`, `featureSchemaVersion`, spec §6.4) es T03 y el reporte, T06.

## Comandos y resultados (T02 y T01)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | Primera corrida: `117 files already formatted`, sin `src/thehouse_ml/models/` (H1). **Después de resolver H1:** `120 files already formatted`; ya incluye los 3 archivos de `models/` (PASS) |
| `uv run ruff check .` | Después de resolver H1: `All checks passed!` sobre todo el repositorio (PASS) |
| `uv run mypy src` | `Success: no issues found in 33 source files` (PASS; mypy sí incluye `models/`) |
| `uv run pytest --cov=thehouse_ml` | **868 passed** en 1090 s; cobertura total **95.89 %** ≥ 80 %; `features/pairs.py` 100 % (183 sentencias, 66 ramas), `features/loader.py` 100 %, `models/baseline_config.py` **100 %** (254 sentencias, 88 ramas), `models/__init__.py` 100 % (PASS) |
| `uv run python -m thehouse_ml prepare` (dos veces) | Ver «Reproducibilidad y comparación con 1.1.0» |
| `uv run python -m thehouse_ml prepare` (3.ª, tras H2) | Código 0; `pairs.parquet` `1f5c4d16…1cae` (sin cambios); `reports/pairs_summary.md` §2 dice ahora «Las 10 features» |
| `uv run pytest` sobre las pruebas afectadas tras H1 y H2 | **500 passed** (71 s): `tests/test_repo_hygiene.py`, `tests/unit/test_baseline_config.py`, `test_feature_schema_v2.py`, `test_pair_features.py`, `test_feature_schema_nv11.py`, `test_feature_schema_validation.py`, `test_hard_constraints.py`, `test_synthetic_pairs.py` y `tests/integration/test_pairs_stage.py`. La suite completa (868 passed) se corrió antes de H1 y H2; los cambios posteriores solo tocan `.gitignore`, `test_repo_hygiene.py`, una línea de texto de `pairs_stage.py` y documentos (PASS) |

## Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitución §2.4 / §13.2: el contrato es `feature_schema.json` | Fórmulas, rangos y catálogo de servicios se leen del esquema; la configuración se valida contra él | PASS |
| Constitución §18: cambio de esquema = versión mayor | 1.1.0 → 2.0.0 con changelog y `versioning` | PASS |
| Constitución §10.1 / §12: modelo base y desglose por factor | 6 factores con pesos que suman 100; «modalidad» = `property_type` | PASS (configuración; el puntaje es T03) |
| Constitución §11: restricciones duras y sin atributos sensibles | `pets_compatible` y `price_to_budget` no puntúan (HC03/HC01); ninguna feature depende de género, edad u origen | PASS |
| Constitución §8: datos sintéticos etiquetados | `synthetic: true` en casos y `pairs.parquet`; reportes con «DATOS SINTÉTICOS» | PASS |
| Constitución §5 / AGENTS: sin dependencias nuevas | Solo biblioteca estándar; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: `data/processed/` fuera de git | Regenerado y sigue ignorado (`git status --ignored`) | PASS |
| AGENTS: rutas permitidas | `git status` lista `src/thehouse_ml/features/`, `tests/`, `reports/` (regenerados por `prepare`), `memoria.md`, `specs/ml-04-baseline-model/` y `specs/README.md`; `students_params.json`, `rooms_params.json` y `pairs_params.json` sin cambios. Con la autorización de M-ML-030: `.gitignore`, `tests/test_repo_hygiene.py` y una línea de `src/thehouse_ml/synthetic/pairs_stage.py`. `src/thehouse_ml/models/` aparece como archivo nuevo (ya no ignorado) | PASS |
| M-ML-022: reglas para faltantes | `maxDistanceKm` nulo → 30, `budget.min` nulo → 0, `essentialServices` nulo → vacío (casos y pruebas) | PASS |
| M-ML-029: decisiones 1–3 | Esquema 2.0.0, pesos/curvas y redondeo tal como la §6.2–6.3 | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La paridad .NET y el puntaje no se ejecutan aquí (T03, T05, ML-06) | No ejecutado (fuera de alcance) |

## Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| H1 | La línea `models/` de `.gitignore` (sin anclar) ignora también `src/thehouse_ml/models/` (`git check-ignore -v` → `.gitignore:4:models/`). `baseline_params.json` y `baseline_config.py` existen y pasan sus pruebas, pero **no pueden entrar a git** y `ruff … .` no los revisa. `.gitignore` está fuera de las rutas autorizadas. Propuesta: `models/` → `/models/` y ajustar `tests/test_repo_hygiene.py::test_gitignore_excludes_data_and_models`. | ADVERTENCIA | **RESUELTO (M-ML-030)**: `.gitignore` usa `/models/`; `test_repo_hygiene.py` comprueba con `git check-ignore` que `models/*.onnx`, `data/raw/` y `data/processed/` siguen ignorados y que `src/thehouse_ml/models/` no; cualquier `.onnx` versionado es prohibido |
| H2 | `reports/pairs_summary.md` §2 dice «Las 7 features se calculan…» (texto fijo en `src/thehouse_ml/synthetic/pairs_stage.py`, fuera de las rutas); §1 y §7 sí muestran las 10. | INFORMATIVO | **RESUELTO (M-ML-030)**: el texto toma `len(schema.feature_order)` (10); reporte regenerado con `prepare` |
| H3 | `training_range` de `distance_ratio` = [0, 60] (cota) en lugar del máximo observado exacto (59.797). | INFORMATIVO | APROBADO (M-ML-030) |
| H4 | `distance_km` se agregó a `non_scoring_features` (la spec §6.2 nombraba 3; el «EJECUTA» pide 4). | INFORMATIVO | APROBADO (M-ML-030) |
| H5 | El backend debe actualizar su spec por el esquema 2.0.0 (NV-23, constitución §2.4). | INFORMATIVO (dependencia externa, no afecta la aceptación de T01 ni T02) | Acción del responsable |
| H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

## Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Casos y pares son sintéticos; coordenadas de campus de `data/universities.csv` (ODbL, M-ML-016/M-ML-017).
- **Verificación de cambios menores (pedida por el responsable):** `git diff` de `reports/students_synthetic.md` y `reports/rooms_catalog.md` = 1 línea cada uno («Esquema de features 1.1.0» → «2.0.0»); `tests/fixtures/hard_constraint_cases.json` = 2 líneas (`spec` y `schema_version`). Ningún valor ni caso cambió.
- **Resultado:** **PASS** (T01 y T02) — se cumple la aceptación de R02 y R01; ruff, mypy y pytest pasan; `prepare` es reproducible; H1 y H2 están resueltos y M-ML-030 está APROBADO. Queda fuera del resultado el aviso al backend (H5, acción del responsable).

---

## T03 · Puntaje y desglose

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§11 y §12) → `AGENTS.md` → `memoria.md` (M-ML-029 y M-ML-030) → `spec.md` (sobre todo §6.2–§6.4 y §10), `plan.md`, `task.md` y esta evidencia → `src/thehouse_ml/models/` y `src/thehouse_ml/features/`.

### Qué se hizo

1. **`src/thehouse_ml/models/baseline_score.py`** (nuevo), exportado por `thehouse_ml.models`. Puro: sin I/O, red ni aleatoriedad. Pesos, curvas, decimales y versión salen de `baseline_params.json`; el orden de las features, de `feature_schema.json`.

   ```python
   def score_features(features: Sequence[float], config: BaselineConfig | None = None,
                      schema: FeatureSchema | None = None) -> ScoreResult
   def score_pair(student: StudentProfile, room: Room, campuses: Mapping[str, Campus],
                  config: BaselineConfig | None = None,
                  schema: FeatureSchema | None = None) -> ScoreResult | ExcludedPair
   def rank(candidates: Iterable[ScoredRoom]) -> list[ScoredRoom]
   def match_score(raw_total: float, config: BaselineConfig | None = None) -> int
   def factor_points(raw_points: float, config: BaselineConfig | None = None) -> float
   def round_half_up(value: float, decimals: int) -> float

   # dataclasses congeladas
   FactorScore(factor, feature, value, raw_points, points, max_points)
   ScoreResult(model_version, feature_schema_version, match_score, raw_total, breakdown)
       # + reported_sum, to_dict(include_raw=False)
   ExcludedPair(model_version, feature_schema_version, violated)  # + to_dict()
   ScoredRoom(room_id, score)
   class ScoreInputError(ValueError)
   ```

   - `score_features` recibe las 10 features en el orden de `feature_order` (lo que recibirá el ONNX y calculará NV-23), valida el vector (longitud, finitos, rango del esquema, enteros en las `integer`) y calcula por factor `peso × curva(feature)`; `raw_total` es la suma en el orden de la configuración; los puntos del desglose son `floor(round6(puntos) × 10 + 0.5) / 10` (`factor_points`) y `matchScore = floor(round6(raw_total) + 0.5)`.
   - `score_pair` = `check_hard_constraints` → si se excluye, `ExcludedPair` con las restricciones violadas y **sin puntaje** (constitución §11.1; no calcula features) → si pasa, `compute_pair_features` → `feature_vector` → `score_features`. Una sola implementación de cada paso.
   - `rank` ordena por `raw_total` descendente y `room_id` ascendente (M-ML-029).
   - Redondeo «mitad hacia arriba» con el algoritmo de .NET `MidpointRounding.AwayFromZero` (M-ML-031 decisión 2).
2. **`src/thehouse_ml/models/__init__.py`** — exporta la API nueva.
3. **Corrección del redondeo por factor (T03-H1, M-ML-031 decisión 10, autorizada por el responsable):** `baseline-1.0.0` no se ha publicado, así que se corrige **sin cambiar la versión**:
   - `baseline_params.json`: `rounding.factor_points` agrega `pre_round_decimals: 6` y `formula: "floor(round6(puntos sin redondear) * 10 + 0.5) / 10"` y corrige su descripción; la nota `dotnet` pide round6 antes de redondear, con `double` o con `decimal`, y `MidpointRounding.AwayFromZero`; `metadata.changelog` (nuevo) registra la versión inicial (M-ML-029/M-ML-030) y esta corrección (M-ML-031, T03-H1). Pesos, curvas, factores y el resto del redondeo no cambian.
   - `baseline_config.py`: `RoundingRules` agrega `factor_pre_round_decimals` y `factor_formula`, leídos y validados como los demás campos.
   - `tests/unit/test_baseline_config.py`: `test_rounding_rules_of_m_ml_029` comprueba los campos nuevos, la nota de .NET y el changelog; 2 casos rotos nuevos (sin `pre_round_decimals`, sin `formula`).
4. **`tests/unit/test_baseline_score.py`** (nuevo, 168 pruebas) — sin red; propiedades con rejillas y `random.Random("score:20261120:<salt>")`; sin dependencias nuevas.
5. `feature_schema.json` (2.0.0) **sin cambios**; `src/thehouse_ml/features/` solo se reutiliza.

### Formato de salida (JSON)

Contrato (spec §6.4): `matchScore` con el nombre y tipo de `ExplainMatchRequest` de `TheHouse-GenAI`; `breakdown` es el campo nuevo propuesto. Ejemplo de la §6.4 calculado por `score_features` (`ScoreResult.to_dict()`):

```json
{
  "modelVersion": "baseline-1.0.0",
  "featureSchemaVersion": "2.0.0",
  "matchScore": 78,
  "breakdown": [
    {"factor": "budget", "points": 25.5, "maxPoints": 30, "feature": "price_below_min", "value": 0.15},
    {"factor": "distance", "points": 17.0, "maxPoints": 25, "feature": "distance_ratio", "value": 0.66},
    {"factor": "essential_services", "points": 13.3, "maxPoints": 20, "feature": "services_coverage", "value": 0.6666666666666666},
    {"factor": "place_type", "points": 10.0, "maxPoints": 10, "feature": "place_type_match", "value": 1},
    {"factor": "property_type", "points": 10.0, "maxPoints": 10, "feature": "property_type_match", "value": 1},
    {"factor": "extras", "points": 2.1, "maxPoints": 5, "feature": "extra_services_ratio", "value": 0.42857142857142855}
  ]
}
```

| Campo | Tipo | Regla |
| --- | --- | --- |
| `modelVersion` | texto | `metadata.model_version` de `baseline_params.json` |
| `featureSchemaVersion` | texto | versión del esquema cargado |
| `matchScore` | entero 0–100 | `floor(round6(Σ puntos sin redondear) + 0.5)`, mitad hacia arriba |
| `breakdown[]` | lista | un elemento por factor, en el orden de la configuración (= tabla §6.2) |
| `breakdown[].factor` | texto | `id` del factor |
| `breakdown[].points` | número, 1 decimal | `floor(round6(puntos) × 10 + 0.5) / 10`, mitad hacia arriba |
| `breakdown[].maxPoints` | entero | peso del factor (suman 100) |
| `breakdown[].feature` | texto | feature del esquema que usa el factor |
| `breakdown[].value` | número | valor de la feature **sin redondear** (entero en las features `integer`); los 6 decimales del ejemplo de la §6.4 son de presentación |

- **Valores crudos** (`to_dict(include_raw=True)`, para reportes y casos dorados; **no** forman parte del contrato con `TheHouse-GenAI`): agrega `rawTotal` (77.97619047619047) después de `matchScore` y `rawPoints` en cada factor (25.5, 17.0, 13.333333333333332, 10.0, 10.0, 2.142857142857143).
- **Par excluido** (`ExcludedPair.to_dict()`), sin `matchScore` ni `breakdown`:

  ```json
  {"modelVersion": "baseline-1.0.0", "featureSchemaVersion": "2.0.0", "excluded": true, "violatedConstraints": ["HC01_budget"]}
  ```

- Coherencia del ejemplo: suma sin redondear 77.976190 → `matchScore` 78 (diferencia 0.024 ≤ 0.5); suma reportada 77.9 (diferencia 0.076 ≤ 0.3).

### Trazabilidad R03 → prueba

Todas en `tests/unit/test_baseline_score.py`.

| Criterio | Prueba |
| --- | --- |
| Ejemplo de la §6.4: total 78 y puntos 25.5 · 17.0 · 13.3 · 10.0 · 10.0 · 2.1 | `test_the_example_of_spec_6_4_scores_78`, `test_the_json_output_is_the_format_of_spec_6_4` |
| Formato JSON serializable, `matchScore` entero, claves de la §6.4 | `test_the_json_output_is_the_format_of_spec_6_4`, `test_include_raw_adds_the_unrounded_values` |
| Pesos y curvas leídos de la configuración (nada escrito a mano) | `test_weights_and_curves_are_read_from_the_config` (configuración alterada: 66.926190 → 67, calculado a mano), `test_the_breakdown_follows_the_config_order_and_weights` |
| Monotonía en precio (`price_below_min` sube → el puntaje no sube) | `test_price_monotonicity[a,b]` (rejilla 0–1 con paso 0.01 × 200 bases aleatorias con semilla, cada una) |
| Monotonía en distancia (`distance_ratio` sube → el puntaje no sube) | `test_distance_monotonicity[a,b]` (rejilla 0–3 con paso 0.01 más 10, 59.797, 60 y 1000 × 200 bases), `test_distance_curve_at_its_breakpoints` |
| Total entre 0 y 100 | `test_total_is_between_0_and_100` (2,000 vectores con semilla), `test_the_total_is_between_0_and_100_on_the_corners_of_the_grid` (256 esquinas) |
| Features perfectas → 100; todo en 0 → 0 | `test_perfect_features_score_100`, `test_the_worst_features_score_0`, `test_a_vector_of_zeros_only_keeps_budget_and_distance` (vector literal de ceros = 55; M-ML-031 decisión 9) |
| Coherencia: diferencia entre la suma reportada y `raw_total` ≤ 0.3 (≤ 0.05 por factor) y entre `matchScore` y `raw_total` ≤ 0.5 | `test_the_breakdown_is_coherent_with_the_total` (2,000 vectores; tolerancias leídas de la configuración) |
| Redondeo en los bordes, calculado a mano | `test_match_score_rounding_edges` (17: 77.5 → 78, 77.4999995 → 78, 77.5000004 → 78, 77.4999994 → 77, 77.49999949 → 77, 2.5 → 3, 99.5 → 100…), `test_round_half_up` (13), `test_a_total_of_exactly_77_5_from_features_scores_78`, `test_factor_points_ending_in_5_round_up_from_features`, `test_factor_points_are_pre_rounded_to_6_decimals` (3, a mano: 13.35 → 13.4, 24.35 → 24.4, 2.35 → 2.4; en binary64 quedan en 13.349999999999998, 24.349999999999998 y 2.3499999999999996 y sin round6 bajarían), `test_factor_points_rounding_edges` (6) |
| Pares excluidos sin puntaje | `test_an_excluded_pair_has_no_score` (HC01, HC02, HC03 y las tres), `test_an_excluded_pair_does_not_need_campus_coordinates`, `test_hard_constraint_cases` (27 casos de ML-03) |
| Una sola implementación de cada paso | `test_a_pair_that_passes_is_scored_with_its_features` (`score_pair` = `score_features(feature_vector(compute_pair_features(…)))`), `test_pair_feature_cases` (48 casos de `pair_feature_cases.json`) |
| `pets_compatible` y las demás features que no puntúan no afectan el resultado | `test_non_scoring_features_do_not_change_the_result[distance_km, effective_distance_km, price_to_budget, pets_compatible]`, `test_pets_compatible_does_not_change_the_example` |
| Desempate por `room_id` | `test_rank_breaks_ties_by_room_id_ascending`, `test_rank_orders_by_unrounded_total_descending` (99.85 y 99.7 dan 100 y se ordenan por la suma sin redondear), `test_rank_is_stable_against_input_order` |
| Entradas inválidas y esquema incompatible | `test_invalid_vectors_are_rejected` (11), `test_every_problem_of_a_vector_is_reported_at_once`, `test_an_incompatible_schema_is_rejected`, `test_a_config_feature_missing_from_the_schema_order_is_rejected` |
| Determinismo | `test_the_result_is_deterministic` |

### Comandos y resultados (T03)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `122 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 34 source files` (PASS) |
| `uv run pytest tests/unit/test_baseline_score.py tests/unit/test_baseline_config.py --cov=thehouse_ml.models` (tras la corrección de T03-H1) | **246 passed** (168 + 78); `models/` **100 %** (PASS) |
| `uv run pytest --cov=thehouse_ml` (suite completa, con la corrección de T03-H1) | **1046 passed** en 945.7 s (15 min 46 s), código 0; cobertura total **96.01 %** ≥ 80 %; código nuevo o cambiado: `models/baseline_score.py` **100 %** (125 sentencias, 34 ramas), `models/baseline_config.py` **100 %** (256 sentencias, 88 ramas), `models/__init__.py` 100 % (PASS) |

### Tabla de coherencia (T03)

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitución §11.1: los pares que violan restricciones duras no se recomiendan | `score_pair` devuelve `ExcludedPair` sin `matchScore`; `ScoredRoom` solo acepta `ScoreResult`, así que un excluido no entra al ranking | PASS |
| Constitución §11.2: sin atributos sensibles | Solo las 6 features de la configuración puntúan; ninguna depende de género, edad u origen; las que no puntúan no cambian el resultado (prueba) | PASS |
| Constitución §12: desglose por factor que suma el total antes de redondear | `raw_total` = suma de `raw_points`; tolerancias de la §6.3 probadas en 2,000 vectores | PASS |
| Spec §6.2: pesos y curvas | Leídos de `baseline_params.json` (sin cambios en pesos ni curvas); con otra configuración el resultado cambia como se calcula a mano | PASS |
| Spec §6.3: redondeo y ranking | `floor(round6(Σ) + 0.5)`; por factor `floor(round6(puntos) × 10 + 0.5) / 10` (1 decimal, mitad hacia arriba, con la corrección de T03-H1); ranking por suma sin redondear y `room_id` | PASS |
| Spec §6.4: formato y `TheHouse-GenAI` | `matchScore` entero 0–100 y `breakdown` con las claves del ejemplo; total 78 en el ejemplo | PASS |
| Constitución §2.4 / M-ML-030: el contrato es el esquema | Vector en el orden de `feature_order`; configuración compatible con el esquema (misma mayor) | PASS |
| AGENTS / constitución §5: sin dependencias nuevas | Solo biblioteca estándar; `pyproject.toml` y `uv.lock` sin cambios; sin `hypothesis` | PASS |
| AGENTS: rutas permitidas | `git status`: `src/thehouse_ml/models/` (`__init__.py`, `baseline_score.py`, `baseline_config.py` y `baseline_params.json`; este último con la autorización expresa de M-ML-031 para corregir el redondeo por factor sin cambiar la versión), `tests/unit/test_baseline_score.py` y `test_baseline_config.py`, `memoria.md`, `specs/ml-04-baseline-model/` y la fila de ML-04 de `specs/README.md`; `feature_schema.json` y `src/thehouse_ml/features/` sin cambios | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La paridad con .NET no se ejecuta aquí (casos dorados en T05; backend en NV-23) | No ejecutado (fuera de alcance) |

### Hallazgos (T03)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T03-H1 | `baseline_params.json` redondeaba cada factor a 1 decimal **sin** pre-redondeo a 6 decimales: con ruido de binary64, un factor que a mano termina en .x5 bajaba (`price_below_min` 0.555 → 13.349999999999998 → 13.3; a mano, 13.4). | ADVERTENCIA | **RESUELTO (M-ML-031, APROBADO)**: por factor `floor(round6(puntos) × 10 + 0.5) / 10`; `baseline_params.json` corregido sin cambiar la versión (no publicada) y con changelog; 3 casos a mano y 6 bordes con prueba |
| T03-H2 | «Todo en 0 → 0» se interpreta como todas las curvas en 0; el vector literal de ceros da 55 porque `price_below_min` = 0 y `distance_ratio` = 0 son los mejores valores. | INFORMATIVO | APROBADO (M-ML-031 decisión 9) |
| T03-H3 | El formato del par excluido (`excluded`, `violatedConstraints`) y `breakdown` son propuestas que el responsable coordina con `TheHouse-GenAI` y el backend. | INFORMATIVO (dependencia externa) | Acción del responsable |
| T03-H4 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Nota para el backend (NV-23, paridad)

Para obtener los mismos `matchScore` y `points` que Python:

1. Evaluar las curvas y sumar los puntos de los factores **en el orden de `baseline_params.json`**, de izquierda a derecha.
2. **Aplicar round6 antes de redondear**, tanto en el total como en cada factor, **con `double` o con `decimal`**:
   - total: `Math.Round(Math.Round(sum, 6, MidpointRounding.AwayFromZero), MidpointRounding.AwayFromZero)`;
   - cada factor: `Math.Round(Math.Round(points, 6, MidpointRounding.AwayFromZero), 1, MidpointRounding.AwayFromZero)`.
3. Usar siempre `MidpointRounding.AwayFromZero`; el `Math.Round` por omisión (`ToEven`, mitad al par) **no** es compatible (2.5 → 2 en lugar de 3).
4. Con `decimal`, 30 × (1 − 0.555) da 13.35 exacto; con `double`, 13.349999999999998. Con round6 los dos dan 13.4; sin round6, `double` daría 13.3. Casos de comprobación: 13.35 → 13.4, 24.35 → 24.4, 2.35 → 2.4, total 77.4999995 → 78 y 77.4999994 → 77. T05 entregará ≥ 50 casos dorados con estos bordes.

### Revisión y resultado (T03)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas; las pruebas usan vectores generados con semilla y los casos sintéticos de ML-03.
- **Resultado:** **PASS** (T03) — se cumple la aceptación de R03: `matchScore` 0–100 con `breakdown` en el formato de la §6.4 (ejemplo = 78), pares excluidos sin puntaje, redondeo de la §6.3 con round6 también por factor (T03-H1 resuelto), propiedades de monotonía, rango y coherencia probadas; ruff, mypy y la suite completa pasan; M-ML-031 APROBADO. Quedan fuera del resultado la coordinación de `breakdown` y del formato del par excluido con `TheHouse-GenAI` y el backend (T03-H3, acción del responsable) y la paridad .NET (T05 / NV-23).

## T04 · Simulador de etiquetas sintéticas

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-04 1.0.0, task T04 (ML-04-R04), en la rama feat/ml-04-labels. Rutas permitidas: src/thehouse_ml/ (synthetic/ para el simulador; models/ y features/ solo para reutilizar), tests/, reports/, memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). data/processed/ se regenera fuera de git. Sin dependencias nuevas. No modifiques feature_schema.json, baseline_params.json ni los params de students, rooms o pairs» — César Yael López Trujillo, 2026-11-20 |
| Rama | `feat/ml-04-labels` (commit base `85d1e4b`, merge del PR #25) |
| Commit | Sin commit, por instrucción explícita del responsable |
| Decisión | M-ML-032 (**APROBADO**, César Yael López Trujillo, 2026-11-20, con los valores propuestos y las desviaciones T04-H1, T04-H2 y T04-H3) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§8, §10 y §11) → `AGENTS.md` → `memoria.md` (M-ML-025 a M-ML-031) → `spec.md` (sobre todo §6.5, §6.6, §10 decisión 4 y ML-04-RK02), `plan.md`, `task.md` y esta evidencia → `src/thehouse_ml/synthetic/` y `src/thehouse_ml/models/`.

### Qué se hizo

1. **`src/thehouse_ml/synthetic/labels_params.json` 1.0.0** (nuevo, en git; `metadata.synthetic = true`, `decision = "M-ML-032 (APROBADO)"`; se pasó de PROPUESTO a APROBADO sin cambiar ningún valor). Cada parámetro con `value` y `basis` «SUPUESTO documentado»:

   | Parámetro | Valor | Nota |
   | --- | --- | --- |
   | `seed` | 20261121 | distinta de 20261119 · 20261905 · 20261120 |
   | `utility.base_weights` | 20 · 35 · 20 · 10 · 5 · 10 | baseline 30 · 25 · 20 · 10 · 10 · 5 |
   | `utility.curves` | presupuesto `1 − price_to_budget`; distancia `exp(−effective_distance_km / 3)`; el resto identidad | baseline: `1 − price_below_min` y `distance_ratio` por tramos |
   | `weights.multiplier_sigma` | 0.3 | `exp(0.3 · z)` por factor, renormalizado a 100 |
   | `noise.sigma` | 5.0 | puntos, por par |
   | `relevance.bands` | 3: 0.10 · 2: 0.10 · 1: 0.10 | cortes acumulados ⌈0.1 n⌉, ⌈0.2 n⌉, ⌈0.3 n⌉ |
   | `check.spearman_range` | [0.3, 0.9] | `prepare` falla fuera del rango |

2. **`src/thehouse_ml/synthetic/labels.py`** (nuevo): `load_params`/`parse_params` (validación con la biblioteca estándar: factores y orden, pesos > 0 que suman 100, formas `identity`/`one_minus`/`exp_decay`, features del esquema, bandas en (0, 1) que no suman más de 1, rango de Spearman, base de cada parámetro y `metadata.synthetic`), `simulate_labels(pairs, params) -> LabelSet`, `hidden_weights`, `utility`, `grade_cutoffs` y `relevance_by_position`. Puro (sin I/O, red ni reloj); entrada `LabelInput(student_id, room_id, features)`; **no** importa `thehouse_ml.models`; independiente del orden de entrada; una etiqueta por par, en orden de `student_id` y `room_id`; rechaza pares duplicados y features faltantes, no finitas o negativas.
3. **`src/thehouse_ml/synthetic/labels_stage.py`** (nuevo): `run_labels(paths)` lee `pairs.parquet`, escribe `labels.parquet`, `labels_hidden_utility.parquet`, `labels_hidden_weights.parquet` (los tres fuera de git) y `reports/labels_synthetic.md`; calcula Spearman (Pearson de rangos promedio, sin scipy) entre `score_features(...).raw_total` (T03) y la utilidad oculta; `LabelsError` si sale de [0.3, 0.9] (después de escribir el reporte) o si faltan columnas. Sin `pairs.parquet`, se omite con mensaje.
4. **`stages.py`** (`prepare` corre `run_labels` después de `run_pairs` y antes de la data card), **`cli.py`** (código 1 con `LabelsError`, `LabelParamsError` y `LabelInputError`) y **`synthetic/__init__.py`** (exporta la API).
5. **Pruebas:** `tests/unit/test_synthetic_labels.py` (nuevo, 290 pruebas), `tests/integration/test_labels_stage.py` (nuevo, 10) y `tests/unit/test_stages.py` (orden de las etapas con `run_labels`).
6. `feature_schema.json`, `baseline_params.json`, `students_params.json`, `rooms_params.json` y `pairs_params.json` **sin cambios**; `models/` y `features/` solo se reutilizan.

### Resultado con los datos de ML-03

Relevancia (de `reports/labels_synthetic.md`):

| Grupo | Estudiantes | Pares | 3 | 2 | 1 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 1,400 | 135,993 | 14,097 (10.4 %) | 13,543 (10.0 %) | 13,653 (10.0 %) | 94,700 (69.6 %) |
| `validation` | 300 | 29,857 | 3,092 (10.4 %) | 2,972 (10.0 %) | 3,002 (10.1 %) | 20,791 (69.6 %) |
| `test` | 300 | 30,755 | 3,173 (10.3 %) | 3,073 (10.0 %) | 3,088 (10.0 %) | 21,421 (69.7 %) |
| **Total** | 2,000 | 196,605 | 20,362 (10.4 %) | 19,588 (10.0 %) | 19,743 (10.0 %) | 136,912 (69.6 %) |

Por campus, las proporciones de 3, 2 y 1 quedan entre 9.9 % y 10.4 % y la de 0 entre 69.6 % y 69.7 % en los 10 campus. La relevancia 3 supera el 10 % porque ⌈0.1 n⌉ redondea hacia arriba.

| Control de no circularidad (ML-04-RK02) | Valor |
| --- | ---: |
| Spearman baseline (suma sin redondear) – utilidad oculta, todos los pares | **0.513** |
| Rango documentado | [0.3, 0.9] — cumple |
| Mediana por estudiante (1,999 con series no constantes) | 0.547 |
| p25 – p75 por estudiante | 0.377 – 0.685 |

Estudiantes con pocos candidatos: 1 con un candidato (`S1078`, relevancia 3), 0 con dos y 23 con 3–9.

### Trazabilidad R04 → prueba

| Criterio | Prueba |
| --- | --- |
| Semilla fija y distinta de estudiantes, cuartos y pares | `test_seed_is_fixed_and_distinct_from_students_rooms_and_pairs` |
| `labels_params.json` versionado, sintético, con base por parámetro y validado | `test_packaged_params_load_and_declare_synthetic`, `test_invalid_params_are_rejected` (15), `test_non_object_params_are_rejected` |
| Pesos ocultos distintos del baseline y renormalizados a 100 | `test_hidden_base_weights_differ_from_the_baseline_and_sum_to_100`, `test_hidden_weights_are_renormalized_per_student`, `test_hidden_weights_use_the_lognormal_multipliers`, `test_hidden_weights_are_per_student_and_sum_to_100` (integración) |
| Forma de la utilidad distinta (presupuesto y distancia) | `test_utility_shape_differs_from_the_baseline_in_budget_and_distance`, `test_curves_are_bounded_in_zero_one`, `test_without_noise_or_variation_utility_is_the_weighted_sum` (a mano) |
| Relevancia 0–3 por posición con la regla de pocos candidatos | `test_relevance_for_few_candidates` (n = 1…6 y 10), `test_relevance_counts_follow_the_cumulative_ceiling_cuts` (n = 1…250), `test_two_hundred_candidates_split_20_20_20_140`, `test_relevance_per_student_follows_utility_and_the_cuts`, `test_ties_in_utility_are_broken_by_room_id`, `test_relevance_proportions_per_student_follow_the_cuts` (integración) |
| Misma semilla → mismas etiquetas; otra semilla las cambia | `test_same_seed_gives_the_same_labels_and_another_seed_changes_them` |
| Una etiqueta por par, ninguna fuera de los candidatos | `test_exactly_one_label_per_input_pair_in_order`, `test_run_labels_writes_one_synthetic_label_per_candidate_pair` (integración, `merge` exterior con `pairs.parquet`) |
| Correlación baseline–utilidad en [0.3, 0.9] | `test_report_is_marked_synthetic_and_the_correlation_is_in_range`, `test_correlation_out_of_range_is_an_error_and_the_report_says_so`, `test_spearman_known_values` |
| Sin campos personales ni de género | `test_simulator_has_no_personal_or_gender_fields`, `test_outputs_have_no_personal_or_gender_columns`, `test_simulator_does_not_import_the_baseline` |
| Reporte «DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES» con advertencia de ML-07 | `test_report_is_marked_synthetic_and_the_correlation_is_in_range` |
| Reproducible byte por byte (dos corridas) | `test_running_twice_does_not_change_any_file` (SHA-256 de pares, etiquetas, archivos ocultos y reporte) y doble `prepare` abajo |
| Omisión y errores | `test_without_pairs_the_stage_is_skipped`, `test_missing_feature_columns_are_an_error`, `test_cli_reports_label_errors_with_exit_code_one`, `test_duplicate_pairs_and_bad_features_are_rejected` |

### Comandos y resultados (T04)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `127 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 36 source files` (PASS) |
| `uv run pytest tests/unit/test_synthetic_labels.py tests/integration/test_labels_stage.py tests/unit/test_stages.py tests/unit/test_cli.py` | 290 + 27 passed (PASS) |
| `uv run pytest --cov=thehouse_ml` (suite completa) | **1346 passed** en 1388.4 s (23 min 8 s), código 0; cobertura total **95.96 %** ≥ 80 %; código nuevo: `synthetic/labels.py` **94 %** (255 sentencias), `synthetic/labels_stage.py` **97 %** (171 sentencias), `stages.py` y `cli.py` 100 % (PASS) |
| `uv run python -m thehouse_ml prepare` (dos veces) | código 0 las dos; `labels: 196605 etiquetas sintéticas de 2000 estudiantes (Spearman con el baseline 0.513)`; SHA-256 de todos los `data/processed/*.parquet` y `reports/*.md` **idénticos** entre corridas; ningún otro archivo versionado cambia (PASS) |

SHA-256 tras las dos corridas: `labels.parquet` `b720f056…cdd54`, `labels_hidden_utility.parquet` `ddf71238…6acff`, `labels_hidden_weights.parquet` `086a8fd2…03525`, `reports/labels_synthetic.md` `9bf89886…5b9ac`.

### Tabla de coherencia (T04)

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitución §8.2: las etiquetas de una utilidad se declaran y no son desempeño real | Título «DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES», advertencia de utilidad + ruido, ML-04-RK02 y ML-07; `synthetic = true` en los tres Parquet y en `labels_params.json` | PASS |
| Constitución §9: semilla fija y reproducible | Semilla 20261121 por estudiante; doble `prepare` con el mismo SHA-256 | PASS |
| Constitución §11.2 y spec §8: sin atributos sensibles | El simulador solo recibe ids y 6 features del par; sin columnas personales ni de género (prueba) | PASS |
| Spec §6.5 / ML-04-RK02: pesos ocultos distintos, variación por estudiante y ruido | 4 de 6 pesos distintos, 2 curvas distintas, lognormal 0.3, ruido 5; Spearman 0.513 en [0.3, 0.9] | PASS |
| Spec §7 regla 3: el baseline no se ajusta con las etiquetas | `baseline_params.json` sin cambios; el simulador no importa `thehouse_ml.models` (prueba) | PASS |
| AGENTS: datos fuera de git | `labels*.parquet` en `data/processed/` (ignorado); `git status` sin ellos | PASS |
| AGENTS / constitución §5: sin dependencias nuevas | Biblioteca estándar y pandas; Spearman sin scipy; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: rutas permitidas | `src/thehouse_ml/synthetic/` (nuevos `labels.py`, `labels_stage.py`, `labels_params.json`; `__init__.py`), `stages.py`, `cli.py`, `tests/`, `reports/labels_synthetic.md`, `memoria.md`, `specs/ml-04-baseline-model/evidence.md`; params protegidos sin cambios | PASS |
| Spec §6.5: valores SUPUESTO fijados con su M-ML | M-ML-032 **APROBADO** (César Yael López Trujillo, 2026-11-20) | PASS |

### Hallazgos (T04)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T04-H1 | La spec §6.5 dice «Σ peso oculto × `f` de la §6.2»; por el «EJECUTA», presupuesto y distancia usan otras features y curvas (`1 − price_to_budget`, `exp(−effective_distance_km / 3)`). | ADVERTENCIA (desviación del texto de la spec) | **APROBADO** (M-ML-032) |
| T04-H2 | Bandas 10/10/10 % en lugar de la propuesta 10/20/30 % de la spec §6.5, por el «EJECUTA». Con ⌈·⌉ acumulado, 2 o 3 candidatos solo dan un 3 y el resto 0. | ADVERTENCIA (desviación de una propuesta SUPUESTO) | **APROBADO** (M-ML-032) |
| T04-H3 | La spec ponía `utility` en `labels.parquet`; va aparte en `labels_hidden_utility.parquet` (y los pesos en `labels_hidden_weights.parquet`), por el «EJECUTA». `labels.parquet` agrega `labels_params_version`, que pide la spec. | INFORMATIVO | **APROBADO** (M-ML-032) |
| T04-H4 | `price_to_budget` no tiene tope en el esquema (`[0, ∞)`); las curvas se acotan a [0, 1] con `min(x, 1)`. En los candidatos `price_to_budget` ≤ 1 (HC01), así que el recorte no actúa. | INFORMATIVO | Documentado en `labels_params.json` |
| T04-H5 | `utility` es binary64 con `math.exp` y `random.gauss`: los bytes de `labels_hidden_utility.parquet` podrían diferir en el último ulp entre plataformas (libm); `labels.parquet` solo depende del orden. La prueba de reproducibilidad es en la misma plataforma. | INFORMATIVO | Aceptado |
| T04-H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T04)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas; solo datos sintéticos de ML-03.
- **Resultado:** **PASS** (T04) — se cumple la aceptación de R04 y todos los controles pasan; M-ML-032 **APROBADO** por César Yael López Trujillo el 2026-11-20 con los valores propuestos y las desviaciones T04-H1, T04-H2 y T04-H3. `labels_params.json` pasó a `metadata.decision = "M-ML-032 (APROBADO)"` sin cambiar ningún valor y `reports/labels_synthetic.md` se regeneró con `prepare` (ver «Cierre tras la aprobación»). T04 completa; ML-04 en 4/7.

### Nota para ML-05

- `data/processed/labels_hidden_utility.parquet` (utilidad oculta por par) y `data/processed/labels_hidden_weights.parquet` (pesos ocultos por estudiante) **NO deben usarse como entrada del modelo ni para entrenarlo**, ni para elegir features o hiperparámetros: son la «respuesta» del simulador y usarlos sería fuga de la etiqueta. ML-05 entrena **solo** con `labels.parquet` (`relevance`, partición `train`) unido a `pairs.parquet` por `student_id` y `room_id`.
- **Una prueba de ML-05 deberá verificarlo**: por ejemplo, que el código de entrenamiento y evaluación no lea esos dos archivos (sus nombres no aparecen en el módulo de entrenamiento) y que las columnas de entrada del modelo sean exactamente las de `feature_order` del esquema, sin `utility` ni `weight_*`.
- Las métricas que ML-05 calcule con `labels.parquet` son sobre **etiquetas sintéticas** y se reportan como tales (constitución §8.2); se reemplazan con señales reales en ML-07.

### Cierre tras la aprobación (T04)

| Comando o cambio | Resultado |
| --- | --- |
| `labels_params.json`: `metadata.decision` `M-ML-032 (PROPUESTO)` → `M-ML-032 (APROBADO)` | Único cambio; ningún valor ni `basis` cambia |
| `uv run python -m thehouse_ml prepare` | código 0; `labels: 196605 etiquetas sintéticas de 2000 estudiantes (Spearman con el baseline 0.513)`; el reporte dice «decisión M-ML-032 (APROBADO)» (PASS) |
| SHA-256 después de la aprobación | `labels.parquet` `b720f056…cdd54`, `labels_hidden_utility.parquet` `ddf71238…6acff` y `labels_hidden_weights.parquet` `086a8fd2…03525`, **iguales** a los de antes (los valores no cambiaron); `reports/labels_synthetic.md` `b1260eb1…511a6`, que solo cambia en el texto de la decisión (PASS) |
| `uv run ruff format --check .` / `uv run ruff check .` / `uv run mypy src` | `127 files already formatted` / `All checks passed!` / `Success: no issues found in 36 source files` (PASS) |
| `uv run pytest tests/unit/test_synthetic_labels.py tests/integration/test_labels_stage.py tests/unit/test_stages.py tests/unit/test_cli.py tests/test_repo_hygiene.py` (pruebas afectadas) | **335 passed** (PASS); la suite completa (1346 passed, 95.96 %) corrió antes del cambio de `metadata.decision`, que ninguna prueba lee; el CI la vuelve a correr |

## T05 · Casos dorados

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-04 1.0.0, task T05 (ML-04-R05), en la rama feat/ml-04-golden. Rutas permitidas: tests/ (incluido tests/fixtures/), src/thehouse_ml/models/ (solo si hace falta un exportador; no cambies baseline_params.json ni la lógica de puntaje), memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). Sin dependencias nuevas. No modifiques feature_schema.json» — César Yael López Trujillo, 2026-11-23 |
| Rama | `feat/ml-04-golden` (commit base `b0bbdc6`, merge del PR #26) |
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-23 |
| Decisión | M-ML-033 (**APROBADO**, César Yael López Trujillo, 2026-11-23, con sus 6 decisiones y una séptima: el perfil incompleto como tercer tipo de resultado) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§12 y §13.4) → `AGENTS.md` → `memoria.md` (M-ML-022 a M-ML-024 y M-ML-029 a M-ML-032) → `spec.md` (§6.3, §6.4, R05 y riesgo RK03), `plan.md`, `task.md` y esta evidencia (nota para el backend de T03) → `src/thehouse_ml/models/` y `src/thehouse_ml/features/pairs.py` → `tests/fixtures/pair_feature_cases.json` y `hard_constraint_cases.json` (formato y regla del meridiano).

### Qué se hizo

1. **`tests/golden_recount.py`** (nuevo): recuento independiente. No importa `thehouse_ml` (una prueba lo verifica con `ast`). Escribe aparte las 10 fórmulas de las features, las reglas para faltantes (M-ML-022), los factores de transporte y las tres restricciones duras (M-ML-024); lee como datos los pesos, curvas y decimales de `baseline_params.json` y el orden, el catálogo de servicios y los ids de las restricciones de `feature_schema.json`. Aritmética exacta con `fractions.Fraction`: curvas lineales por tramos con extremos constantes, suma exacta, round6 y mitad hacia arriba sobre el valor exacto. La distancia es la única parte en binary64 (haversine con `atan2`; el paquete usa `asin`) y se convierte a `Fraction`.
2. **`tests/build_baseline_golden_cases.py`** (nuevo): define las entradas de los 81 casos y escribe `tests/fixtures/baseline_golden_cases.json` solo con el recuento (`uv run python tests/build_baseline_golden_cases.py`). Rechaza un caso si un valor que depende de la distancia queda a menos de 1e-9 de un punto medio de round6 (M-ML-033 decisión 3).
3. **`tests/fixtures/baseline_golden_cases.json`** (nuevo, 81 casos, 257,076 bytes, SHA-256 `a92852c7…85ae3f2`): encabezado y casos con el formato de abajo.
4. **`tests/unit/test_baseline_golden_cases.py`** (nuevo, 258 pruebas): archivo idéntico a la salida del recuento, paridad con el paquete por caso, encabezado contra `baseline_params.json` y `feature_schema.json`, campos completos y cobertura de los casos límite.
5. **`tests/fixtures/README.md`**: entrada de los casos dorados.
6. `src/thehouse_ml/` **sin cambios** (no hizo falta exportador), igual que `feature_schema.json` y `baseline_params.json`.

### Formato del archivo para el backend (NV-23)

`tests/fixtures/baseline_golden_cases.json` (UTF-8). En **ML-06 se empaquetará como `golden_cases.json` junto al ONNX** (constitución §13.4), con la misma estructura, para la paridad Python ↔ onnxruntime y la prueba .NET del backend.

**Encabezado (`metadata`):**

| Clave | Contenido |
| --- | --- |
| `model_version` / `feature_schema_version` | `baseline-1.0.0` / `2.0.0`; la prueba falla si dejan de coincidir con `baseline_params.json` y `feature_schema.json` |
| `spec`, `decision`, `generated_at`, `generated_by`, `synthetic` | ML-04 1.0.0; M-ML-033 (APROBADO) sobre M-ML-029 a M-ML-031; 2026-11-23; el generador; `true` |
| `how_computed` | Recuento independiente con `Fraction`, sin el código bajo prueba |
| `dotnet` | Nota de paridad: sumar en el orden de `breakdown`; **round6 antes de redondear, con `double` o con `decimal`**; `matchScore = Math.Round(Math.Round(sum, 6, MidpointRounding.AwayFromZero), MidpointRounding.AwayFromZero)`; `points = Math.Round(Math.Round(points, 6, MidpointRounding.AwayFromZero), 1, MidpointRounding.AwayFromZero)`; `ToEven` no es compatible |
| `tolerance` | `features`, `value` y `rawTotal` 1e-6 (absoluta); `points` y `matchScore` 0 (exactos) |
| `format` | Descripción de cada bloque del caso |
| `campuses` | `lat`, `lon` y `final_status` de los 10 campus (`data/universities.csv`, `final_lat`/`final_lon`); la prueba los compara con el CSV |
| `feature_order`, `factors`, `hard_constraints`, `required_fields`, `case_count` | Orden de las 10 features, los 6 factores (`factor`, `feature`, `maxPoints`), los ids HC01–HC03, los campos del quick setup (`budget.max`, `livingWith`, `placeType`) y 81 |

**Cada caso (`cases[]`):**

| Clave | Contenido |
| --- | --- |
| `id`, `category`, `description` | Identificador único, grupo de cobertura y explicación con el cálculo a mano |
| `input.student` | Perfil de NV-11 con los nombres de `feature_schema.json` (`budget.min`, `budget.max`, `maxDistanceKm`, `essentialServices`, `livingWith`, `placeType`, `pets`, `transport`, `propertyTypes`, `university_code`); los 10 siempre presentes y `null` donde el campo es nulo (el backend aplica las reglas para faltantes del esquema) |
| `input.room` | `price_mxn`, `lat`, `lon`, `services`, `placeType`, `propertyType`, `pets_allowed`, `livingWith`, `bedrooms`, `bathrooms`, `area_m2`, `university_code`, `synthetic` (ninguno nulo) |
| `tolerance` | `{"features": 1e-6, "value": 1e-6, "points": 0, "matchScore": 0}` |
| `expected.features` | Las 10 features en el orden de `feature_order` (también en los pares excluidos, como referencia); `null` si el perfil está incompleto |
| `expected.result` | Si pasa: `modelVersion`, `featureSchemaVersion`, `matchScore` (entero) y `breakdown` (6 entradas en el orden de los factores con `factor`, `points` a 1 decimal, `maxPoints`, `feature` y `value` sin redondear). Si no pasa: `modelVersion`, `featureSchemaVersion`, `excluded: true` y `violatedConstraints` en el orden del esquema, sin `matchScore` ni `breakdown`. Si el perfil está incompleto: ver abajo |
| `expected.rawTotal` | Suma sin redondear (solo pares que pasan); informativa para depurar el redondeo, no es parte del contrato con `TheHouse-GenAI` |

**Tres tipos de resultado** (el backend los distingue en este orden):

| Orden | Condición | `expected.result` | `expected.features` |
| --- | --- | --- | --- |
| 1 | Algún campo del quick setup es nulo (`budget.max`, `livingWith`, `placeType`) | `{"modelVersion", "featureSchemaVersion", "incomplete": true, "missingFields": [...]}`; `missingFields` con **todos** los nulos, con los nombres de NV-11, en el orden `budget.max`, `livingWith`, `placeType` (`metadata.required_fields`). Sin `matchScore`, `breakdown` ni `violatedConstraints`: las restricciones no se evalúan | `null` |
| 2 | Perfil completo y el par viola HC01–HC03 | `{"modelVersion", "featureSchemaVersion", "excluded": true, "violatedConstraints": [...]}` en el orden del esquema | Las 10 (referencia) |
| 3 | El par pasa | `{"modelVersion", "featureSchemaVersion", "matchScore", "breakdown"}` (spec §6.4) | Las 10 |

El formato del perfil incompleto no estaba en la spec; lo fija M-ML-033 (decisión 7). En Python equivale a `IncompleteProfileError` con `.missing` (M-ML-023), que lanzan `score_pair` y `compute_pair_features`; el backend puede devolverlo como respuesta o como error de validación, pero con los mismos campos.

### Cobertura de casos

| Categoría | Casos | Qué cubre |
| --- | ---: | --- |
| `perfect` | 3 | 100 con todo en 1; con todos los opcionales nulos; renta = `budget.max` y `distance_ratio` 0.5 |
| `worst` | 2 | 0 con todas las curvas en 0 (`distance_ratio` 1.5 y exactamente 1) |
| `budget` | 7 | `price_below_min` 0 (renta = `budget.min`), 0.2, 0.5 y 1 (renta 0); `budget.min` nulo; `budget.min` 0; `budget.max` 0 con renta 0 |
| `distance` | 8 | `distance_ratio` 0.5, 0.6 (`maxDistanceKm` decimal), 2/3 y 0.5 con `maxDistanceKm` nulo, 0.75, 1, 1.2 y 59.8 (`maxDistanceKm` 0.5) |
| `services` | 6 | Cobertura 0, 1/3, 1/2 y 1; `essentialServices` vacío; los 11 pedidos (5/11, extras sin denominador) |
| `place_property` | 6 | Tipo de lugar 0, tipo de inmueble 0, los dos en 0, `propertyTypes` nulo, vacío y con varios tipos |
| `extras` | 4 | `extra_services_ratio` 0, 2/9, 3/8 y 1 (con 10 pedidos) |
| `factor_rounding` | 7 | 13.35 → 13.4 (presupuesto), 24.35 → 24.4 (distancia), 2.35 → 2.4 (presupuesto y distancia), 28.65 → 28.7, 1.25 → 1.3 y 3.75 → 3.8 (extras) |
| `total_rounding` | 5 | 86.5 → 87 exacto; 68.5 → 69 con ruido de binary64 (68.49999999999999); 82.49999955 → 83; 82.4999994 → 82; 82.5000004 → 83 |
| `spec_example` | 1 | Ejemplo de la §6.4 con entradas de la app → 78 |
| `transport` | 5 | `walk`, `bike`, `public`, `car` y nulo → `public` |
| `nulls` | 3 | `pets` nulo con cuarto sin mascotas; todos los opcionales nulos con cuarto parcial; `essentialServices` nulo con cuarto sin servicios |
| `exclusion` | 9 | HC01, HC02 (women/men y men/any), HC03, HC01 + HC03, las tres; y 3 controles que pasan (any/women, mascota permitida, women/women con renta = `budget.max`) |
| `campus` | 10 | Un caso por campus con el cuarto desplazado en latitud y longitud (haversine general); TESJI, UMB-VDC y UCAN marcados `seed_sin_verificar` |
| `incomplete` | 5 | Falta `budget.max`; falta `livingWith`; falta `placeType`; faltan los tres; falta `livingWith` con mascota y renta 9000 (gana `incomplete` sobre `excluded`) |
| **Total** | **81** | 70 con puntaje, 6 excluidos y 5 incompletos; `matchScore` de 0 a 100 |

Además de su categoría, los 10 campus aparecen en los casos y los 6 campos opcionales de NV-11 son nulos en al menos un caso (prueba `test_inputs_cover_transports_nulls_campuses_and_exclusions`).

### Trazabilidad R05 → prueba

| Criterio | Prueba |
| --- | --- |
| Al menos 50 casos, ids únicos | `test_file_has_at_least_50_cases_with_unique_ids` |
| Entradas, features, desglose y total en cada caso | `test_every_case_has_all_fields` (81) |
| Calculados con un recuento independiente, no con la función de T03 | `test_fixture_is_exactly_what_the_independent_recount_produces`, `test_recount_does_not_import_the_package` (2), `test_recount_rules_written_apart_agree_with_the_schema`, `test_recount_rounding_matches_hand_values` |
| El paquete devuelve exactamente lo esperado | `test_package_score_matches` (81: `matchScore`, `points`, `maxPoints`, `factor` y `feature` exactos; `value` y `rawTotal` con 1e-6; excluidos con `to_dict()` idéntico; incompletos con `IncompleteProfileError.missing` igual a `missingFields`), `test_package_features_match` (81, 1e-6; los incompletos lanzan el mismo error) |
| Perfil incompleto: cada campo, los tres y precedencia sobre las restricciones | `test_incomplete_profiles_cover_each_required_field_and_all_three`; `required_fields` = campos `reject` del esquema = `REQUIRED_FIELDS` del recuento (`test_header_versions_match_the_config_and_the_schema`) |
| Versiones y contrato iguales a `baseline_params.json` y `feature_schema.json` | `test_header_versions_match_the_config_and_the_schema`, `test_header_campuses_match_universities_csv`, `test_header_documents_generation_decision_and_dotnet_rounding` |
| Casos límite de la §6.2–6.3 y del «EJECUTA» | `test_coverage_by_category`, `test_perfect_and_worst_scores`, `test_factor_borders_are_covered`, `test_rounding_borders_are_covered`, `test_inputs_cover_transports_nulls_campuses_and_exclusions` |
| La prueba detecta un esperado alterado | Mutación manual: cambiar `points` 13.4 → 13.3 en `budget_13_35` hace fallar 3 pruebas (archivo ≠ recuento, paridad del caso y bordes de redondeo); restaurado, 242 passed |

### Comandos y resultados (T05)

| Comando | Resultado |
| --- | --- |
| `uv run python tests/build_baseline_golden_cases.py` | `tests\fixtures\baseline_golden_cases.json: 81 casos` (PASS) |
| `uv run ruff format --check .` | `130 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 36 source files` (PASS) |
| `uv run pytest tests/unit/test_baseline_golden_cases.py` | **258 passed** (PASS) |
| `uv run pytest tests/unit/test_baseline_golden_cases.py tests/unit/test_baseline_score.py tests/unit/test_baseline_config.py tests/unit/test_pair_features.py tests/unit/test_hard_constraints.py tests/test_repo_hygiene.py` (pruebas afectadas) | **717 passed** en 7.6 s (PASS) |
| Suite completa con cobertura | **No ejecutada aquí** (la corre el CI, por indicación del responsable); no se reporta como PASS. No hay código nuevo en `src/`, así que la cobertura del paquete no cambia |

### Recuento frente al paquete

**Sin discrepancias:** los 81 casos coinciden sin ajustar ningún esperado. Valores sin redondear en los casos más delicados:

| Caso | Recuento | Paquete (binary64) | Sin round6 | Con round6 |
| --- | --- | --- | ---: | ---: |
| `total_x5_binary64_noise` | 68.5 (exacto) | 68.49999999999999 | 68 | **69** |
| `budget_13_35` (factor) | 13.35 (exacto) | 13.349999999999998 | 13.3 | **13.4** |
| `transport_car` | 87.49999999999967 (distancia binary64 del recuento) | 87.50000000000075 | 87 o 88 | **88** |
| `total_x4999995_up` | 82.4999995500004 | 82.4999995500012 | 82 | **83** |

En `transport_car` el recuento y el paquete calculan la distancia con haversine distintos y quedan en lados opuestos de 87.5; con round6 los dos dan 88. Es la misma situación que tendrá .NET: sin round6 el resultado dependería de la implementación de haversine.

### Tabla de coherencia (T05)

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitución §13.4: casos dorados con tolerancia documentada, entregados al backend | 81 casos (las tres salidas posibles) con tolerancias en el encabezado y por caso; formato y nota de .NET arriba; empaquetado en ML-06 | PASS |
| Constitución §12: el desglose suma el total antes de redondear | `rawTotal` = Σ de los puntos sin redondear; `matchScore` y `points` con la §6.3 | PASS |
| Constitución §11.1: un par excluido no recibe puntaje | 6 casos con `excluded`, sin `matchScore` ni `breakdown`; `to_dict()` idéntico | PASS |
| Spec §6.3 / M-ML-031: round6 y mitad hacia arriba en total y factores | Bordes 13.35, 24.35, 2.35, 28.65, 1.25, 3.75, 86.5, 68.5 (ruido), x.49999955, x.4999994 y x.5000004 | PASS |
| Spec R05: recuento independiente, no la función de T03 | El recuento y el generador no importan `thehouse_ml` (prueba con `ast`); archivo = salida del recuento | PASS |
| Constitución §8 / spec §8: solo datos sintéticos, sin datos personales | Entradas inventadas; `synthetic: true` en el encabezado y en cada cuarto | PASS |
| AGENTS / constitución §5: sin dependencias nuevas | Biblioteca estándar (`fractions`, `json`, `csv`, `math`); `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: rutas permitidas | `git status`: `tests/` (3 archivos `.py` nuevos, `fixtures/baseline_golden_cases.json` y `fixtures/README.md`), `memoria.md`, `specs/ml-04-baseline-model/` y la fila de ML-04 de `specs/README.md`; `src/`, `feature_schema.json` y `baseline_params.json` sin cambios | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | Suite completa y paridad en .NET no ejecutadas aquí (CI y NV-23) | No ejecutado (fuera de alcance) |

### Hallazgos (T05)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T05-H1 | Recuento y paquete coinciden en los 81 casos; ningún esperado se ajustó. | INFORMATIVO | Cerrado |
| T05-H2 | Un total con exactamente 7 decimales (x.4999995) no se puede construir con entradas válidas de la app; se usa x.49999955 con margen ≥ 1e-9 a los puntos medios de round6, más x.4999994 y x.5000004. | ADVERTENCIA (desviación del texto del «EJECUTA») | **APROBADO** (M-ML-033 decisión 3) |
| T05-H3 | El vector del ejemplo de la §6.4 (cobertura 2/3 y extras 3/7) no se puede obtener de entradas de la app; el caso usa extras 3/8 y conserva `matchScore` 78. | ADVERTENCIA (el ejemplo de la spec es solo un vector) | **APROBADO** (M-ML-033 decisión 4) |
| T05-H4 | Extras no puede valer 2.35 con entradas de la app (denominador ≤ 11); el borde 2.35 → 2.4 se cubre en presupuesto y distancia, y extras cubre 1.25 → 1.3 y 3.75 → 3.8. | INFORMATIVO | **APROBADO** (M-ML-033 decisión 5) |
| T05-H5 | En `transport_car` el recuento y el paquete quedan en lados opuestos de 87.5 por el haversine; round6 los iguala. Refuerza la nota para .NET: round6 es obligatorio también con `decimal`. | INFORMATIVO (para NV-23) | Documentado en el encabezado y arriba |
| T05-H6 | Las coordenadas de TESJI, UMB-VDC y UCAN son `seed_sin_verificar`; si cambian en `data/universities.csv`, `test_header_campuses_match_universities_csv` falla y se regeneran los casos con el generador. | INFORMATIVO | Aceptado |
| T05-H7 | El perfil incompleto (quick setup) no estaba en los casos y su formato no está en la spec. | ADVERTENCIA | **RESUELTO** (M-ML-033 decisión 7, APROBADO): tercer tipo de resultado `incomplete` con `missingFields`, 5 casos y formato documentado arriba para el backend |
| T05-H8 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T05)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas; entradas sintéticas y coordenadas de campus de `data/universities.csv` (ya en git).
- **Resultado:** **PASS** (T05) — se cumple la aceptación de R05: 81 casos (≥ 50) con entradas, 10 features, desglose y total, exclusión o perfil incompleto, calculados con un recuento independiente y con los casos límite de la §6.2–6.3; el paquete coincide en todos; ruff, mypy y las pruebas afectadas pasan. M-ML-033 **APROBADO** por César Yael López Trujillo el 2026-11-23 con las desviaciones T05-H2 a T05-H4 y la decisión 7 (T05-H7 resuelto); el encabezado del archivo dice `M-ML-033 (APROBADO)`. T05 completa; ML-04 en 5/7.

## T06 · Reporte del modelo base

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-04 1.0.0, task T06 (ML-04-R06), en la rama feat/ml-04-report. Rutas permitidas: src/thehouse_ml/ (models/ para el reporte y un módulo de métricas de ranking reutilizable por ML-05; synthetic/ y features/ solo para reutilizar), tests/, reports/, memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). Sin dependencias nuevas. No modifiques feature_schema.json, baseline_params.json ni los params de students, rooms, pairs o labels» — César Yael López Trujillo, 2026-11-23 |
| Rama | `feat/ml-04-report` (commit base `683d834`, merge del PR #27) |
| Commit | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-23 |
| Decisión | M-ML-034 (**APROBADO**, César Yael López Trujillo, 2026-11-23, con los valores SUPUESTO y las desviaciones T06-H2 y T06-H3) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§10, §11.3 y §12) → `AGENTS.md` → `memoria.md` (M-ML-027 a M-ML-033, nota para ML-05 de M-ML-032) → `spec.md` (R06, §6.6 y decisión 5), `plan.md`, `task.md` → `specs/ARQUITECTURA.md` (§3 etapas y §4 estructura) → `src/thehouse_ml/models/` y `src/thehouse_ml/synthetic/`.

### Qué se hizo

1. **`src/thehouse_ml/evaluation/ranking.py`** (nuevo, puro, reutilizable por ML-05): `dcg_at_k`, `ndcg_at_k`, `precision_at_k`, `average_precision`, `student_metrics`, `percentile` y `bootstrap_means` (bootstrap pareado por estudiante con intervalo percentil). Definiciones y tratamiento del estudiante sin relevantes en el docstring y en M-ML-034.
2. **`src/thehouse_ml/models/baseline_eval_params.json`** (nuevo, 1.0.0; `M-ML-034 (APROBADO)` tras la aprobación, sin cambiar ningún valor): partición, mínimo de candidatos, umbral de relevancia, bootstrap, semilla del orden al azar, rangos de presupuesto y regla del ejemplo, cada uno con `value` y `basis`.
3. **`src/thehouse_ml/models/baseline_report.py`** (nuevo): etapa `evaluate`. Une `pairs.parquet` con `labels.parquet` por (`student_id`, `room_id`) (uno a uno, misma partición, `synthetic = true`), puntúa con `score_features` y ordena con `rank` (T03), calcula baseline, azar e ideal por estudiante, segmentos y bootstrap, y escribe `reports/baseline_report.md` y `reports/baseline_metrics.json`.
4. **`src/thehouse_ml/stages.py`** y **`cli.py`**: `evaluate` llama a `run_evaluate`; `EvaluationError` sale con código 1. `train` y `export` siguen pendientes.
5. **Pruebas nuevas:** `tests/unit/test_ranking_metrics.py` (ejemplos a mano), `tests/unit/test_baseline_report.py` (parámetros, exclusiones, órdenes, explicaciones) y `tests/integration/test_evaluate_stage.py` (etapa completa con 2 campus). `tests/unit/test_cli.py`: `evaluate` deja de ser etapa vacía y se prueba su omisión sin entradas.
6. **Sin cambios** en `feature_schema.json`, `baseline_params.json` ni los params de estudiantes, cuartos, pares o etiquetas; `data/processed/` no se regeneró (la etapa solo lee).

### Resultado con los datos de ML-03 y T04 (DATOS SINTÉTICOS)

Partición test: 300 estudiantes, **300 evaluados**, 0 con menos de 2 candidatos, 0 sin cuarto relevante. Media e IC 95 % (bootstrap por estudiante, 1,000 remuestreos, semilla 20261122):

| Segmento | Estudiantes | NDCG@10 | NDCG@5 | Precision@5 | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total | 300 | 0.527 [0.503, 0.550] | 0.535 [0.507, 0.560] | 0.586 [0.557, 0.615] | 0.503 [0.485, 0.522] |
| Sin `seed_sin_verificar` | 210 | 0.511 [0.480, 0.539] | 0.515 [0.480, 0.548] | 0.576 [0.536, 0.613] | 0.494 [0.469, 0.519] |
| `budget.max` ≤ $2,000 | 73 | 0.568 [0.525, 0.608] | 0.583 [0.536, 0.631] | 0.666 [0.608, 0.723] | 0.561 [0.523, 0.601] |
| `budget.max` $2,001–$3,500 | 158 | 0.534 [0.503, 0.566] | 0.535 [0.496, 0.571] | 0.570 [0.533, 0.606] | 0.494 [0.469, 0.519] |
| `budget.max` > $3,500 | 69 | 0.468 [0.410, 0.518] | 0.485 [0.419, 0.545] | 0.539 [0.469, 0.606] | 0.463 [0.421, 0.501] |

Por campus (30 estudiantes cada uno) en `reports/baseline_report.md` §4. Contra el azar (total): NDCG@10 azar 0.197 [0.182, 0.214], diferencia **+0.330 [+0.302, +0.355]** (41 % de la brecha azar → ideal); Precision@5 azar 0.204, techo 0.983; MAP azar 0.251. El intervalo de la diferencia baseline − azar queda por encima de 0 en todos los segmentos y métricas.

### Trazabilidad R06 → prueba

| Criterio | Prueba |
| --- | --- |
| NDCG@10/@5 con `2^rel − 1` y `log2`, Precision@5 y MAP con relevante ≥ 2, a mano | `tests/unit/test_ranking_metrics.py` (DCG, NDCG, corte en k, Precision con denominador fijo, AP = 13/18, AP con la lista completa) |
| Estudiante sin cuarto relevante (tratamiento documentado) | `test_student_without_any_relevant_room_has_undefined_ap_and_is_excluded`, `test_students_with_one_candidate_or_no_relevant_room_are_excluded_and_counted` |
| Solo test y estudiantes con ≥ 2 candidatos; cuántos quedan fuera | `test_only_test_students_with_two_or_more_candidates_are_evaluated`, `test_students_with_one_candidate_or_no_relevant_room_are_excluded_and_counted` |
| Baseline = `score_features` + `rank`; unión por (`student_id`, `room_id`) | `test_baseline_metrics_match_an_independent_recomputation`, `test_join_labels_requires_one_label_per_pair_with_the_same_split`, `test_labels_that_do_not_match_the_pairs_are_an_error` |
| No lee `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet` | `test_evaluate_never_reads_the_hidden_label_files` (espía de `pd.read_parquet`, archivos borrados, salida idéntica) |
| Bootstrap por estudiante, 1,000, 95 %, semilla fija; pareado | `test_bootstrap_*`, `test_percentile_interpolates_linearly`, `test_packaged_params_hold_the_supuestos_of_m_ml_034`, `test_every_segment_has_intervals_and_the_baseline_beats_random` |
| Segmentos: total, campus, sin `seed_sin_verificar`, presupuesto | `test_every_segment_has_intervals_and_the_baseline_beats_random`, `test_budget_ranges_are_inclusive_and_cover_every_value` |
| Azar (piso) e ideal (techo) | `test_random_order_is_seeded_per_student_and_independent_of_input_order`, `test_ideal_order_sorts_by_relevance_then_room_id`; ideal NDCG = MAP = 1 en cada segmento |
| Reporte: «DATOS SINTÉTICOS», versiones, distribución, ejemplo, tablas | `test_report_and_metrics_are_marked_synthetic_with_versions`, `test_report_has_distribution_example_and_tables`, `test_explain_factor_describes_each_factor` |
| Sin datos personales | `test_outputs_have_no_personal_or_sensitive_fields` |
| Dos corridas sin cambios | `test_running_twice_does_not_change_any_file` y doble corrida real (abajo) |

### Comandos y resultados (T06)

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `137 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 39 source files` (PASS) |
| `uv run pytest tests/unit/test_ranking_metrics.py tests/unit/test_baseline_report.py tests/unit/test_cli.py tests/unit/test_stages.py tests/integration/test_evaluate_stage.py tests/integration/test_installed_package.py` (afectadas) | **93 passed** (PASS; corregido: la evidencia previa decía 104 por error) |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing` (suite completa) | **1678 passed** en 10 min 39 s; cobertura total **96 %**; `evaluation/ranking.py` **100 %**, `models/baseline_report.py` **97 %**, `stages.py` y `cli.py` 100 % (PASS) |
| `uv run python -m thehouse_ml evaluate` ×2 | ~7 s cada una; SHA-256 iguales en las dos corridas: `baseline_report.md` `216e6dc7…227c6b37b`, `baseline_metrics.json` `2ba0242f…3bda7eb5` (PASS). Tras la aprobación (solo cambia el texto de la decisión): `3b554e9b…bbe924f30` y `df36b2f1…49d05561`, iguales en dos corridas (PASS) |

### Tabla de coherencia (T06)

| Norma | Verificación | Resultado |
| --- | --- | --- |
| Constitución §8.2 y §20.3 | Título y aviso «DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS» en el reporte; `synthetic: true` y advertencia en el JSON | PASS |
| Constitución §10.1–10.3 | NDCG@10, Precision@5 y MAP por estudiante en test, sin fuga (partición de ML-03), con intervalos; JSON como referencia de ML-05 | PASS |
| Constitución §11.2–11.3 | Segmentos por campus y presupuesto; ningún factor ni segmento usa género, edad u origen | PASS |
| Constitución §12 | Ejemplo con desglose por factor y salida §6.4; nota de la suma antes de redondear | PASS |
| Constitución §9 | Semillas fijas documentadas; doble corrida con el mismo SHA-256 | PASS |
| Spec §7 regla 3 | El baseline no se ajusta con las etiquetas (solo se mide) | PASS |
| M-ML-032 | Solo `labels.parquet`; los archivos ocultos no se leen (prueba) | PASS |
| `specs/ARQUITECTURA.md` §3–4 | Etapa `evaluate` (ML-04/05) y métricas en `evaluation/` | PASS |
| AGENTS: sin dependencias nuevas | Biblioteca estándar y pandas | PASS |
| AGENTS: rutas permitidas | `src/thehouse_ml/` (`evaluation/`, `models/`, `stages.py`, `cli.py`), `tests/`, `reports/`, `memoria.md`, `specs/ml-04-baseline-model/` | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | Todos los controles de arriba se ejecutaron | PASS |

### Hallazgos (T06)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T06-H1 | Umbral de relevancia, bootstrap y rangos de presupuesto son SUPUESTO (spec §6.6). | BLOQUEANTE para cerrar T06 | **RESUELTO** (M-ML-034 APROBADO) |
| T06-H2 | La spec §6.6.4 propone `random.Random("bootstrap:<seed>")`; se usa una semilla por segmento (`bootstrap:<seed>:<segment_id>`) para que los segmentos no dependan entre sí. | ADVERTENCIA (desviación menor) | **APROBADO** (M-ML-034 decisión 4) |
| T06-H3 | La etapa es `evaluate` (ARQUITECTURA §3) y las métricas van en `evaluation/` (ARQUITECTURA §4), además de `models/` que nombra el «EJECUTA»; ambas dentro de `src/thehouse_ml/`. | INFORMATIVO | **APROBADO** (M-ML-034 decisión 1) |
| T06-H4 | Los segmentos de un campus tienen 30 estudiantes: intervalos anchos que no permiten comparar campus; el reporte lo dice. | INFORMATIVO | Aceptado |
| T06-H5 | ARQUITECTURA §3 nombra `reports/metrics.json` para `evaluate`; T06 escribe `reports/baseline_metrics.json` (spec §6.6 y «EJECUTA»). ARQUITECTURA no está en las rutas permitidas. | ADVERTENCIA (coherencia documental) | Se resuelve en T07 (indicación del responsable) |
| T06-H6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR |

### Revisión y resultado (T06)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos ni datos personales (solo ids sintéticos), sin dependencias nuevas.
- **Resultado (antes de la aprobación):** **PASS WITH WARNINGS** — los controles técnicos pasaban y se cumplía la aceptación de R06; T06 quedó abierta hasta la aprobación de M-ML-034 (T06-H1). Resultado final en «Cierre tras la aprobación (T06)». Al aprobarse: `baseline_eval_params.json` pasa a `M-ML-034 (APROBADO)`, se regenera con `evaluate` y se actualizan `task.md`, `spec.md` §14 y `specs/README.md` (6/7).

### Nota para ML-05

- **Referencia:** `reports/baseline_metrics.json` (mismas etiquetas, estudiantes de test y definiciones de `thehouse_ml.evaluation`); `per_student_baseline` permite comparaciones pareadas por estudiante. Un modelo entrenado solo se promueve si supera esta referencia (constitución §10.1).
- **Segmento más débil: `budget.max` > $3,500** (69 estudiantes): NDCG@10 **0.468 [0.410, 0.518]**, frente a 0.568 en ≤ $2,000 y 0.534 en $2,001–$3,500. Motivo (señalado por el responsable): el baseline puntúa igual cualquier renta dentro del presupuesto (factor `budget` = `1 − price_below_min`, plano entre `budget.min` y `budget.max`), mientras que la utilidad oculta prefiere lo más barato dentro del presupuesto (`1 − price_to_budget`, M-ML-032). Con un presupuesto amplio, más candidatos empatan en ese factor y el baseline no distingue lo que las etiquetas sí premian. **Es el patrón principal que el modelo entrenado debería aprender** (`price_to_budget` sigue en el esquema como feature).
- Es una explicación por el diseño del simulador; T06 no hizo un análisis de ablación. ML-05 puede confirmarla comparando su NDCG@10 por rango de presupuesto con esta tabla: la mejora debería ser mayor en > $3,500.
- Todo esto es sobre **datos sintéticos**: lo que ML-05 aprenda aquí es la función del simulador (ML-04-RK02), no preferencias reales.

### Cierre tras la aprobación (T06)

- M-ML-034 **APROBADO** por César Yael López Trujillo el 2026-11-23: valores SUPUESTO (relevante si `relevance ≥ 2`; bootstrap 1,000 con intervalo percentil del 95 % y semilla 20261122; rangos ≤ $2,000, $2,001–$3,500 y > $3,500) y las desviaciones T06-H2 (semilla por segmento) y T06-H3 (métricas en `evaluation/`). T06-H5 se resuelve en T07.
- `baseline_eval_params.json` pasa a `M-ML-034 (APROBADO)` sin cambiar valores; reportes regenerados con `evaluate` (mismas cifras; dos corridas con el mismo SHA-256).
- **Resultado final (T06): PASS.** T06 completa; ML-04 en 6/7.

## T07 · Pruebas y evidencia (cierre de ML-04)

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-04 1.0.0, task T07 (ML-04-R07), en la rama feat/ml-04-cierre. Rutas permitidas: src/thehouse_ml/, tests/, reports/, memoria.md, specs/ml-04-baseline-model/, specs/README.md y specs/ARQUITECTURA.md (solo lo de ML-04 y T06-H5). Sin dependencias nuevas» — César Yael López Trujillo, 2026-11-23 |
| Rama | `feat/ml-04-cierre` |
| Commit base | `fc69f1d` (merge del PR #28, T06; igual a `origin/main`) |
| Commit (T07) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-23 |
| Decisión | M-ML-035 (APROBADO — César Yael López Trujillo, 2026-11-23) |
| Esquema y parámetros | `feature_schema.json` 2.0.0, `baseline_params.json` (`baseline-1.0.0`), `labels_params.json` 1.0.0, `baseline_eval_params.json` 1.0.0 y los params de estudiantes, cuartos y pares **sin cambios** |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` → `AGENTS.md` → `CONTRIBUTING.md` → `memoria.md` (M-ML-028 a M-ML-034) → `spec.md`, `plan.md`, `task.md` y esta evidencia → `specs/ml-03-feature-schema/evidence.md` (modelo del cierre).

### Qué se hizo

1. **Suite completa con cobertura** (`--cov-report=term-missing`): cobertura total y del código nuevo de ML-04 por encima del 80 % en todos los módulos (el menor, `synthetic/labels.py`, 94 %). No hizo falta agregar pruebas; `src/` y `tests/` **sin cambios**.
2. **Reproducibilidad:** `prepare` y `evaluate` dos veces seguidas, con SHA-256 de todos los archivos versionados (y nuevos no ignorados) más los 10 Parquet de `data/processed/` antes, entre y después de las corridas (ver «Reproducibilidad comprobada»).
3. **T06-H5 resuelto en `specs/ARQUITECTURA.md`:** §3 (diagrama y tabla de etapas): `prepare` = ML-02 · ML-03 · ML-04 (10 features y etiquetas sintéticas); `train` = ML-05 (el modelo base por reglas no se entrena: su configuración está versionada); `evaluate` escribe en ML-04 `reports/baseline_report.md` y `reports/baseline_metrics.json` (antes decía `reports/metrics.json` y `reports/model_card.md`, que quedan para ML-05). §4: módulos reales de `models/`, `evaluation/ranking.py` y `synthetic/labels*`. §6: fila de ML-04 con lo que entrega (esquema 2.0.0, `baseline-1.0.0`, etiquetas sintéticas, 81 casos dorados y métricas de referencia).
4. **Documentos:** `spec.md` (encabezado, §14 y §15), `plan.md` y `task.md` en ✅ COMPLETA (7/7); `specs/README.md` con ML-04 ✅ COMPLETA 7/7; M-ML-035 en `memoria.md` con la herencia para ML-05. `plan.md` decía todavía «⛔ NO EJECUTAR HASTA «EJECUTA»» en el estado de ejecución (T07-H5).

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `137 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 39 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing` | **1678 passed in 684.12s (0:11:24)**, código 0; `Required test coverage of 80.0% reached. Total coverage: 96.10%` (PASS) |
| `uv run coverage report --include="*/thehouse_ml/models/*,*/thehouse_ml/evaluation/*,*/thehouse_ml/synthetic/labels*,*/thehouse_ml/features/*"` | Código nuevo o modificado de ML-04: **98.29 %** (1,984 sentencias, 640 ramas) (PASS) |
| `uv run python -m thehouse_ml prepare` (1.ª y 2.ª corrida) | Código 0 las dos; ~36 s cada una; misma salida de consola: 2,000 estudiantes, 3,000 cuartos, 196,605 pares, `labels: 196605 etiquetas sintéticas de 2000 estudiantes (Spearman con el baseline 0.513)` (PASS) |
| `uv run python -m thehouse_ml evaluate` (1.ª y 2.ª corrida, cada una después de su `prepare`) | Código 0 las dos; ~7.5 s; misma salida: `evaluate: baseline-1.0.0 en test con 300 estudiantes; NDCG@10 0.527 [0.503, 0.550] (DATOS SINTÉTICOS)` (PASS) |
| `git log --first-parent --format="%h %p \| %s" origin/main` y `git rev-list --first-parent --no-merges origin/main` | ML-04 entró solo por los PR #23–#28; el único commit de `main` que no es merge es el inicial `20b2c19` (PASS) |

No se ejecutaron `pip-audit` ni gitleaks en local (corren en el CI: jobs `dependency-audit` y `secrets-gitleaks`); ML-04 no cambió dependencias. No se reportan como PASS local (T07-H2).

### Cobertura por módulo de ML-04

| Módulo | Sentencias | Sin cubrir | Ramas | Ramas parciales | Cobertura | Líneas sin cubrir |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `evaluation/__init__.py` | 3 | 0 | 0 | 0 | 100 % | — |
| `evaluation/ranking.py` | 104 | 0 | 36 | 0 | 100 % | — |
| `models/__init__.py` | 4 | 0 | 0 | 0 | 100 % | — |
| `models/baseline_config.py` | 256 | 0 | 88 | 0 | 100 % | — |
| `models/baseline_score.py` | 125 | 0 | 34 | 0 | 100 % | — |
| `models/baseline_report.py` | 438 | 9 | 146 | 11 | 97 % | 137->143, 228, 442, 473->478, 565, 577-578, 671, 733, 887->896, 968, 977 (validaciones y casos defensivos) |
| `synthetic/labels.py` | 255 | 11 | 82 | 8 | **94 %** (mínimo) | 163, 171, 187, 193, 210, 236, 242-243, 267, 270-271 (validación defensiva de parámetros) |
| `synthetic/labels_stage.py` | 171 | 3 | 34 | 3 | 97 % | 199, 209, 380 |
| `features/pairs.py` (modificado en T02) | 183 | 0 | 66 | 0 | 100 % | — |
| `features/` restantes (`__init__`, `campuses`, `constraints`, `loader`) | 445 | 0 | 154 | 0 | 100 % | — |
| `stages.py` y `cli.py` (modificados en T04, T06 y T07-H1) | 86 | 0 | 14 | 0 | 100 % | — |
| **Total `models/` + `evaluation/` + `synthetic/labels*` + `features/`** | **1,984** | **23** | **640** | **22** | **98.29 %** | — |
| **Total de `src/thehouse_ml`** | **5,304** | **154** | **1,520** | **110** | **96.10 %** | — |

### Reproducibilidad comprobada

Comando: `sh hashes.sh <salida>` (script en el scratchpad de la sesión, fuera del repo): SHA-256 de `git ls-files -co --exclude-standard` más `data/processed/*.parquet` (**182 archivos**). Se tomó antes de la 1.ª corrida (h0), después de `prepare` + `evaluate` (h1) y después de la 2.ª `prepare` + `evaluate` (h2), con `git status --short` en cada punto.

| Comprobación | Resultado |
| --- | --- |
| h1 = h2 (2.ª corrida seguida de `prepare` y `evaluate`) | **idéntico** (`cmp` sin diferencias) |
| h0 = h2 sin `memoria.md` | **idéntico**: la única diferencia entre h0 y h1 es `memoria.md`, que el agente editó (M-ML-035) mientras corría la 1.ª corrida; ninguna etapa la escribe (T07-H4) |
| `git status --short` entre h1 y h2 | **idéntico**; las etapas no modifican ningún archivo versionado |
| Salida de consola de las dos corridas | **idéntica** |
| SHA-256 de la lista h2 completa | `7ce77c6620a1a105a3487ef9c7a06951f55a7d19202e3c603bd0f319b15a3189` |

| Archivo | SHA-256 (iguales en h0, h1 y h2) |
| --- | --- |
| `data/processed/pairs.parquet` | `1f5c4d167c2e45c571681857e6b6b34d2889f9cf5d3578729b6557f0b0431cae` (el mismo `1f5c4d16…1cae` de T02) |
| `data/processed/labels.parquet` | `b720f05602fb19cd1d68b8ee9a38d3a593b4cca6eb4985ae3a6a99984d9cdd54` (el mismo de T04) |
| `data/processed/labels_hidden_utility.parquet` | `ddf71238ed994ce8a009b287d3da8cd5cb91fb0abf95ba157d76cd3bb526acff` |
| `data/processed/labels_hidden_weights.parquet` | `086a8fd2ff3ed8463ec89d7d1e69e518734ad49a316831bdc47e35413b603525` |
| `data/processed/students.parquet` | `d4ad048ac31392bf98d33437d4bf91ba07839ffeb83d1ebec9842fb49ef373c1` (igual desde ML-03) |
| `data/processed/rooms.parquet` | `e2ab7e8edaac5cbe55f2ce4971ad50841ce521143d23ce42505e89be3d918dca` (igual desde ML-03) |
| `reports/baseline_metrics.json` | `df36b2f19c205cb56c3290afe8d15c7ef4e4f48088a7b8a676a98f3149d05561` (el mismo del cierre de T06) |
| `reports/baseline_report.md` | `3b554e9bafa67c7d1fb8464f1704c60d897472cf2720745752b0befbbe924f30` (el mismo del cierre de T06) |
| `reports/labels_synthetic.md` | `b1260eb1874fba068bd1a2da188fe366ae8898fb43f676096010cbfc8b2511a6` (el mismo del cierre de T04) |
| `reports/pairs_summary.md` | `07105fcfcf8b77a402a7167874eb37448a640a4cff8f1c27e45bf5103a3b57de` |
| `src/thehouse_ml/features/feature_schema.json` | `67af2e487e2b34c6bc5943e0eca4c209e9812a165f30698246c2f2090ffaa8bb` |
| `src/thehouse_ml/models/baseline_params.json` | `51b663e840eb0b5d12592802093050e28091bf91279db6aece7c237dd5737111` |
| `src/thehouse_ml/models/baseline_eval_params.json` | `c29ee16499a413d3a3716ffd21d8aaf67ed398b503bc7807d8c5156faf8b844e` |
| `src/thehouse_ml/synthetic/labels_params.json` | `077e0ea1d79ed3d33f60deb3a474d1b3df16b4932894a3a4837465a09035744f` |
| `tests/fixtures/baseline_golden_cases.json` | `a92852c7dd289d1040e4f8e5d0b134d23eab8f3a7798c04439b787f9f85ae3f2` (el mismo `a92852c7…85ae3f2` de T05) |

### Trazabilidad R07 → prueba

| Criterio de R07 | Prueba / evidencia |
| --- | --- |
| Pruebas unitarias e integración de lo implementado | `tests/unit/test_feature_schema_v2.py`, `test_pair_features.py`, `test_baseline_config.py`, `test_baseline_score.py`, `test_synthetic_labels.py`, `test_baseline_golden_cases.py`, `test_ranking_metrics.py`, `test_baseline_report.py`, `test_stages.py`, `test_cli.py`; `tests/integration/test_pairs_stage.py`, `test_labels_stage.py`, `test_evaluate_stage.py`; trazabilidad por requisito en las secciones T01–T06 |
| Cobertura ≥ 80 % del código nuevo | «Cobertura por módulo de ML-04»: 98.29 % en el código nuevo o modificado, mínimo por módulo 94 % |
| ruff y mypy sin errores | «Comandos y resultados» |
| `evidence.md` con commit, comandos, resultados, hallazgos y coherencia | Esta sección, las tablas de coherencia de abajo y el «Resumen final de ML-04» |
| Reproducibilidad (constitución §9) | «Reproducibilidad comprobada»; `test_running_twice_does_not_change_any_file` de `pairs`, `labels` y `evaluate` |

### Tabla de coherencia — constitución

| § | Regla | Evidencia en el repo | Resultado |
| --- | --- | --- | --- |
| §2.4 / §13.2 | el contrato con el backend es `feature_schema.json` | Esquema 2.0.0 (10 features) en `src/thehouse_ml/features/feature_schema.json`; `baseline_params.json` se valida contra él (`test_the_packaged_config_is_baseline_1_0_0_for_schema_2_0_0`); los casos dorados copian orden y versión (`test_header_versions_match_the_config_and_the_schema`) | PASS |
| §3 / SDD-001 | spec → plan → task → aprobación → «EJECUTA» → evidencia | `specs/ml-04-baseline-model/{spec,plan,task,evidence}.md`; un «EJECUTA» por task en la cabecera de cada sección; M-ML-029 a M-ML-035 | PASS |
| §4 / SDD-002 | specs independientes | `specs/README.md`: solo ML-04 cambia de estado; ML-05 a ML-07 siguen en borrador | PASS |
| §5 | stack de la tabla; dependencias solo con decisión M-ML | `pyproject.toml` y `uv.lock` sin cambios en toda ML-04 (biblioteca estándar, pandas y pyarrow; Spearman y bootstrap sin scipy; sin `hypothesis`) | PASS |
| §6 / AGENTS | `data/raw`, `data/processed` y `models` fuera de git | `.gitignore` con `/models/` (M-ML-030 H1); `tests/test_repo_hygiene.py` (`test_gitignore_excludes_data_and_models`, `test_artifacts_and_data_are_ignored_by_git`, `test_no_data_or_model_files_are_tracked`); `git status` no lista ningún Parquet | PASS |
| §7.4 | sin datos personales | Ids sintéticos; `test_outputs_have_no_personal_or_sensitive_fields` (evaluate), `test_simulator_has_no_personal_or_gender_fields` (labels); casos dorados con entradas inventadas | PASS |
| §8.1 | sintéticos con distribuciones documentadas y semilla fija | `labels_params.json` (cada parámetro con `value` y `basis`), semilla 20261121 distinta de las demás (`test_seed_is_fixed_and_distinct_from_students_rooms_and_pairs`) | PASS |
| §8.2 | etiquetas de una utilidad: se declara; no es desempeño real | «DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES» en `reports/labels_synthetic.md` (`test_report_is_marked_synthetic_and_the_correlation_is_in_range`); «DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS», `synthetic: true` y `warning` en `reports/baseline_report.md` y `baseline_metrics.json` (`test_report_and_metrics_are_marked_synthetic_with_versions`); `synthetic = true` en `labels.parquet` | PASS |
| §8.3 | desempeño real solo con señales reales | Ningún documento de ML-04 afirma desempeño real; ML-07 queda como reemplazo en cada reporte | PASS |
| §9 | reproducibilidad | Semillas fijas (etiquetas 20261121, bootstrap 20261122, azar 20261123); doble `prepare` + `evaluate` con SHA-256 idénticos (arriba) | PASS |
| §10.1 | todo modelo entrenado se compara contra el modelo base | ML-04 **es** el modelo base: fija la referencia (`reports/baseline_metrics.json`, `reference_for: ML-05`) y se compara contra azar e ideal; la comparación de un modelo entrenado es de ML-05 | PASS (referencia) · comparación: NO APLICA (ML-05) |
| §10.2 | NDCG@10, Precision@5 y MAP por estudiante; partición por estudiante sin fuga | `evaluation/ranking.py` (`tests/unit/test_ranking_metrics.py`); partición test de ML-03 (300 estudiantes, sin fuga); `test_only_test_students_with_two_or_more_candidates_are_evaluated`. La validación cruzada es de ML-05 (el modelo base no se entrena) | PASS |
| §10.3 | intervalos o variación | Bootstrap por estudiante, 1,000, IC 95 %, pareado (`test_packaged_params_hold_the_supuestos_of_m_ml_034`, `test_every_segment_has_intervals_and_the_baseline_beats_random`) | PASS |
| §11.1 | `livingWith` y presupuesto: restricciones duras | `score_pair` devuelve `ExcludedPair` sin puntaje (`test_an_excluded_pair_has_no_score`); 6 casos dorados excluidos; candidatos de `pairs.parquet` solo entre los que pasan HC01–HC03 | PASS |
| §11.2 | sin género ni atributos sensibles como señal | `test_no_factor_uses_a_sensitive_attribute`, `test_non_scoring_features_do_not_change_the_result`, `test_simulator_has_no_personal_or_gender_fields`, `test_outputs_have_no_personal_or_sensitive_fields` | PASS |
| §11.3 | análisis por segmentos (universidad, presupuesto) | 15 segmentos en `reports/baseline_metrics.json`: total, 10 campus, sin `seed_sin_verificar` y 3 rangos de presupuesto (`test_every_segment_has_intervals_and_the_baseline_beats_random`, `test_budget_ranges_are_inclusive_and_cover_every_value`) | PASS |
| §12 | desglose por factor que suma el total | 6 factores (presupuesto, distancia, servicios, tipo de lugar, modalidad = tipo de inmueble, extras); `raw_total` = Σ puntos sin redondear; tolerancias 0.5/0.3 (`test_the_breakdown_is_coherent_with_the_total`, `test_the_json_output_is_the_format_of_spec_6_4`) | PASS |
| §13.1 | sin pickle | `tests/test_repo_hygiene.py` prohíbe `.pkl`, `.pickle`, `.joblib` y `.onnx` versionados (`test_onnx_files_are_flagged_wherever_they_are_tracked`); ML-04 solo escribe JSON, Markdown y Parquet | PASS |
| §13.3 | versión semántica del modelo | `modelVersion` `baseline-1.0.0` en la configuración, la salida, el reporte y los casos dorados | PASS |
| §13.4 | casos dorados con tolerancia documentada | 81 casos con recuento independiente (`test_fixture_is_exactly_what_the_independent_recount_produces`, `test_recount_does_not_import_the_package`, `test_package_score_matches`); tolerancias y nota .NET en el encabezado (`test_header_documents_generation_decision_and_dotnet_rounding`) | Casos: PASS · paridad ONNX y .NET: NO APLICA (ML-06 / NV-23) |
| §13.5 | 200 cuartos en < 20 ms | Presupuesto del artefacto ONNX | NO APLICA (ML-06) |
| §14 | sin secretos; dependencias fijas | Sin secretos ni red; `uv.lock` sin cambios; gitleaks y pip-audit en el CI (no ejecutados en local, T07-H2) | PASS (en el repo) · pip-audit y gitleaks: no ejecutados en local |
| §15 / COST-001 | costo $0 | CPU local (`prepare` ~36 s, `evaluate` ~7.5 s, suite ~11 min) y GitHub Actions del repositorio público (M-ML-007); sin servicios de pago | PASS |
| §16 | pruebas unitarias e integración; cobertura ≥ 80 % | 1,678 pruebas sin red; 96.10 % total y 98.29 % en el código de ML-04 | PASS |
| §17 | CI con ruff, mypy, pytest+cobertura, pip-audit, gitleaks y pipeline | `.github/workflows/ci.yml` sin cambios en ML-04: `quality` (ruff, mypy, pytest con cobertura y las cinco etapas, que ahora incluyen `evaluate`), `tests-windows`, `dependency-audit` y `secrets-gitleaks`; `tests/test_ci_workflow.py` (3) en verde | PASS (local) · CI del PR de T07: pendiente |
| §18 | versión semántica; cambio de esquema = mayor | `feature_schema.json` 1.1.0 → **2.0.0** (`test_the_schema_loads_as_major_version_2_0_0`, `test_the_new_features_are_appended_at_the_end`); spec 1.0.0 sin bump al cerrar | PASS |
| §19 | ramas con PR y aprobación ajena; sin commit directo a `main` | `git rev-list --first-parent --no-merges origin/main` → solo `20b2c19`; ML-04 entró por los PR #23–#28. La aprobación en GitHub no es auditable desde el clon local (sin `gh`) | PASS con observación |
| §20–§21 | leer antes; «EJECUTA»; sin métricas sintéticas como reales; sin scraping, pickle ni costos | «Lecturas previas»; autorizaciones por task; `constitution.md` y `AGENTS.md` sin modificar (`git status`) | PASS |

### Tabla de coherencia — AGENTS.md

| Regla | Evidencia en el repo | Resultado |
| --- | --- | --- |
| Leer constitution → AGENTS → memoria → spec/plan/task | «Lecturas previas» de cada sección | PASS |
| Estado vigente: responsable César (M-ML-011); cada task con su «EJECUTA» | Autorizaciones T01–T07 firmadas por César Yael López Trujillo | PASS |
| Aprobar spec / implementar tasks con «Apruebo» y «EJECUTA» | M-ML-029 (spec) y M-ML-030 a M-ML-035 (tasks) | PASS |
| Descargar datos de terceros | ML-04 no descargó nada | NO APLICA |
| Agregar dependencias solo con decisión M-ML | Sin dependencias nuevas en ML-04 | PASS |
| Publicar un artefacto (release) | ML-04 no publica artefactos (ML-06) | NO APLICA |
| Cambios en GitHub con permiso explícito | Ningún cambio de repositorio, ruleset, secretos ni workflow en ML-04 | PASS |
| Merge solo con aprobación de otra persona | PR #23–#28 mezclados en GitHub; T07 sin commit ni PR; el agente no mezcla | PASS |
| Un plan aprobado no habilita tareas fuera de él | Solo T07; el texto de `train` (T07-H1) se cambió solo después de la autorización expresa del responsable y solo en las 3 rutas autorizadas | PASS |
| Flujo: implementar → probar → coherencia → seguridad y licencias | Cada sección T01–T07 tiene comandos, tabla de coherencia y revisión de seguridad | PASS |
| Un control no ejecutado no se reporta como PASS | pip-audit, gitleaks, CI del PR, paridad .NET/ONNX y §13.5 marcados como no ejecutados o NO APLICA | PASS |
| Cerrar una task solo con aceptación y evidencia reproducible | Comandos y resultados por sección; doble corrida con SHA-256 | PASS |
| Datos: nunca subir `raw`/`processed`/`models`; sin datos personales | `.gitignore`, `test_repo_hygiene.py`, pruebas de columnas | PASS |
| Todo modelo se compara con el modelo base por reglas | ML-04 es el modelo base y deja la referencia | NO APLICA (ML-05) |
| Métricas sintéticas etiquetadas en todo reporte | `labels_synthetic.md`, `baseline_report.md`, `baseline_metrics.json`, evidencia y M-ML-034/035 | PASS |
| Sin pickle; solo ONNX; cada artefacto con `feature_schema.json`, `model_card.md`, `metrics.json` y SHA-256 | Sin pickle (higiene); sin artefacto todavía | PASS · artefacto: NO APLICA (ML-06) |
| Rutas protegidas: `constitution.md` y `AGENTS.md` sin editar; sin secretos, scraping, pago ni commit directo a `main` | `git status`; sin red en las pruebas | PASS |
| Rutas permitidas de T07 | `git status`: `memoria.md`, `specs/ml-04-baseline-model/` (`spec.md`, `plan.md`, `task.md`, `evidence.md`), `specs/README.md` (fila de ML-04), `specs/ARQUITECTURA.md` (§3, §4, fila de ML-04 y nota en la de ML-03 de §6) y, por T07-H1, `src/thehouse_ml/stages.py`, `tests/unit/test_stages.py` y `tests/integration/test_installed_package.py` (una línea cada uno); `reports/` sin cambios | PASS |
| Resultado de revisión PASS / PASS WITH WARNINGS / BLOCKED | Resultado por task y global | PASS |
| Convenciones: documentos en español, código en inglés, decisiones M-ML | Documentos de ML-04 en español; código y pruebas en inglés; M-ML-029 a M-ML-035 | PASS |

### Tabla de coherencia — decisiones M-ML-029 a M-ML-034

| Decisión | Qué exige | Archivo o prueba que lo cumple | Resultado |
| --- | --- | --- | --- |
| M-ML-029 · 1 | Esquema 2.0.0 mayor con 3 features al final | `feature_schema.json` 2.0.0; `test_feature_schema_v2.py` (`test_the_new_features_are_appended_at_the_end`, `test_the_seven_previous_definitions_did_not_change`) | PASS |
| M-ML-029 · 2 | Pesos 30/25/20/10/10/5 y curvas; excluidos sin puntaje; sin atributos sensibles | `baseline_params.json`; `test_factors_weights_and_features_follow_the_spec_table`, `test_an_excluded_pair_has_no_score`, `test_no_factor_uses_a_sensitive_attribute` | PASS |
| M-ML-029 · 3 | Redondeo mitad hacia arriba, tolerancias, ranking, `matchScore` + `breakdown` | `baseline_score.py`; `test_match_score_rounding_edges`, `test_the_breakdown_is_coherent_with_the_total`, `test_rank_breaks_ties_by_room_id_ascending`, `test_the_json_output_is_the_format_of_spec_6_4` | PASS |
| M-ML-029 · 4 | Etiquetas sintéticas fuera de git y declaradas no reales | `labels_stage.py`; `test_run_labels_writes_one_synthetic_label_per_candidate_pair`, `test_report_is_marked_synthetic_and_the_correlation_is_in_range` | PASS |
| M-ML-029 · 5 | Métricas en test, ≥ 2 candidatos, bootstrap, segmentos, referencia de ML-05 | `baseline_report.py`; `test_only_test_students_with_two_or_more_candidates_are_evaluated`, `test_every_segment_has_intervals_and_the_baseline_beats_random` | PASS |
| M-ML-030 | `training_range` [0, 60]; ubicación y compatibilidad de la configuración; curvas por tramos; `distance_km` no puntúa; `.gitignore` `/models/` (H1); texto de 10 features (H2) | `test_the_training_range_of_distance_ratio_is_the_bound_of_the_synthetic_catalog`, `test_schema_compatibility_is_same_major_and_not_older`, `test_curves_are_the_declared_piecewise_linear_points`, `test_every_schema_feature_scores_or_has_a_reason`, `test_the_models_package_of_the_code_is_not_ignored`; `reports/pairs_summary.md` §2 | PASS |
| M-ML-031 | API de `baseline_score`, `AwayFromZero`, suma izquierda→derecha, validación, formato JSON, par excluido, round6 por factor | `test_baseline_score.py` (168): `test_factor_points_are_pre_rounded_to_6_decimals`, `test_invalid_vectors_are_rejected`, `test_a_vector_of_zeros_only_keeps_budget_and_distance`, `test_include_raw_adds_the_unrounded_values` | PASS |
| M-ML-032 | `labels_params.json` 1.0.0: semilla, pesos ocultos distintos, curvas distintas, lognormal 0.3, ruido 5, bandas 10/10/10, Spearman en [0.3, 0.9]; utilidad aparte | `test_hidden_base_weights_differ_from_the_baseline_and_sum_to_100`, `test_utility_shape_differs_from_the_baseline_in_budget_and_distance`, `test_relevance_for_few_candidates`, `test_correlation_out_of_range_is_an_error_and_the_report_says_so`, `test_simulator_does_not_import_the_baseline`; Spearman 0.513 en esta corrida | PASS |
| M-ML-033 | 81 casos dorados con recuento independiente; margen 1e-9; ejemplo §6.4 con extras 3/8; perfil incompleto | `tests/fixtures/baseline_golden_cases.json`; `test_fixture_is_exactly_what_the_independent_recount_produces`, `test_recount_does_not_import_the_package`, `test_package_score_matches` (81), `test_incomplete_profiles_cover_each_required_field_and_all_three` | PASS |
| M-ML-034 | Relevante ≥ 2; bootstrap 1,000/95 %/20261122 por segmento; rangos de presupuesto; métricas en `evaluation/`; azar e ideal; `per_student_baseline`; sin archivos ocultos | `baseline_eval_params.json`; `test_packaged_params_hold_the_supuestos_of_m_ml_034`, `test_budget_ranges_are_inclusive_and_cover_every_value`, `test_random_order_is_seeded_per_student_and_independent_of_input_order`, `test_evaluate_never_reads_the_hidden_label_files`; `reports/baseline_metrics.json` con 300 filas en `per_student_baseline` | PASS |
| T06-H5 (M-ML-034) | ARQUITECTURA coherente con la salida de `evaluate` | `specs/ARQUITECTURA.md` §3, §4 y §6 (arriba) | **RESUELTO** en T07 |

### Hallazgos (T07)

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| T07-H1 | `src/thehouse_ml/stages.py` declaraba `Stage("train", "Entrena el modelo base y los modelos candidatos.", "ML-04")` y la CLI respondía `train: pendiente de ML-04`, aunque ML-04 queda completa y el modelo base no se entrena; `specs/ARQUITECTURA.md` §3 ya dice `train` = ML-05. Las pruebas `tests/unit/test_stages.py` y `tests/integration/test_installed_package.py` fijan ese texto. Propuesta: `spec = "ML-05"` y descripción «Entrena los modelos candidatos (el modelo base por reglas no se entrena)», con las dos pruebas ajustadas. No se cambió porque no está entre los puntos del «EJECUTA» de T07. | ADVERTENCIA (coherencia código ↔ documentos) | **RESUELTO** (autorización del responsable, 2026-11-23): `stages.py` usa `"ML-05"` y la CLI responde `train: pendiente de ML-05`; `tests/unit/test_stages.py` y `tests/integration/test_installed_package.py` ajustadas (ver «Cierre tras T07-H1») |
| T07-H2 | `pip-audit` y gitleaks no se ejecutaron en local; corren en el CI (`dependency-audit`, `secrets-gitleaks`). ML-04 no cambió dependencias. | INFORMATIVO | Pendiente del CI del PR; no se reporta como PASS |
| T07-H3 | El menor módulo de ML-04 es `synthetic/labels.py` (94 %): líneas de validación defensiva de parámetros. | INFORMATIVO | Aceptado (≥ 80 %) |
| T07-H4 | `memoria.md` cambió entre h0 y h1 porque el agente escribió M-ML-035 durante la 1.ª corrida; ninguna etapa la escribe. La reproducibilidad se demuestra con h1 = h2 y con h0 = h2 sin `memoria.md`. | INFORMATIVO | Documentado |
| T07-H5 | `plan.md` seguía con «⛔ NO EJECUTAR HASTA «EJECUTA»» en el estado de ejecución durante T01–T06. | INFORMATIVO | **RESUELTO** (✅ COMPLETA) |
| T07-H6 | La fila de ML-03 en `specs/ARQUITECTURA.md` §6 menciona el esquema 1.1.0 (lo que entregó ML-03). | INFORMATIVO | **RESUELTO** (autorización del responsable): la fila conserva su historia y agrega que ML-04 subió el esquema a 2.0.0 (M-ML-029) |
| T07-H7 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T07 |

### Revisión y resultado (T07)

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas; `reports/` sin cambios; en `src/` y `tests/` solo el texto de `train` (T07-H1).
- **Resultado (antes de T07-H1):** PASS WITH WARNINGS — advertencia T07-H1, propuesta al responsable.

### Cierre tras T07-H1

| Comando o cambio | Resultado |
| --- | --- |
| `src/thehouse_ml/stages.py`: `Stage("train", …, "ML-04")` → `"ML-05"` | Único cambio en `src/`; coherente con `specs/ARQUITECTURA.md` §3 |
| `tests/unit/test_stages.py` y `tests/integration/test_installed_package.py`: `train: pendiente de ML-04` → `ML-05` | Una línea cada una |
| `specs/ARQUITECTURA.md` §6, fila de ML-03 | Conserva lo que entregó ML-03 (1.1.0) y agrega que ML-04 subió el esquema a 2.0.0 (M-ML-029/M-ML-030) |
| `uv run ruff format --check .` / `uv run ruff check .` / `uv run mypy src` | `137 files already formatted` / `All checks passed!` / `Success: no issues found in 39 source files` (PASS) |
| `uv run pytest tests/unit/test_stages.py tests/unit/test_cli.py tests/integration/test_installed_package.py tests/test_repo_hygiene.py` (pruebas afectadas) | **37 passed** en 2.91 s (PASS). La suite completa (1,678 passed, 96.10 %) corrió antes de este cambio de una cadena; `stages.py` conserva su cobertura (mismas sentencias y ramas). El CI la vuelve a correr |
| `uv run python -m thehouse_ml train` | `train: pendiente de ML-05`, código 0 (PASS) |

- **Resultado final (T07): PASS** — se cumple la aceptación de R07 y T07-H1 está resuelto.

## Resumen final de ML-04 (cierre T07, 2026-11-23)

### Pull requests mezclados en `main` y sus commits

Verificado con `git log --first-parent --format="%h %p | %s" origin/main` y `git log <merge>^1..<merge>^2`. El único commit de `main` que no es merge es el inicial `20b2c19` (`git rev-list --first-parent --no-merges origin/main` → 1). ML-04 entró a `main` solo por PR:

| PR | Rama | Contenido | Commit de merge en `main` | Commit de la task |
| --- | --- | --- | --- | --- |
| #23 | `docs/ml-04-spec` | Spec 1.0.0 aprobada (M-ML-029) | `fac23bc` | `0040af7` |
| #24 | `feat/ml-04-config-schema` | T02 · esquema 2.0.0 y T01 · `baseline-1.0.0` (M-ML-030) | `643f94e` | `1136a88` |
| #25 | `feat/ml-04-score` | T03 · puntaje 0–100 con desglose (M-ML-031) | `85d1e4b` | `004c52d` |
| #26 | `feat/ml-04-labels` | T04 · etiquetas sintéticas (M-ML-032) | `b0bbdc6` | `c9bde9c` |
| #27 | `feat/ml-04-golden` | T05 · 81 casos dorados (M-ML-033) | `683d834` | `1f04014` |
| #28 | `feat/ml-04-report` | T06 · reporte y métricas de ranking (M-ML-034) | `fc69f1d` | `0285328` |

T07 se entrega en `feat/ml-04-cierre` (base `fc69f1d`) **sin commit ni PR**, por instrucción explícita del responsable.

### Comandos ejecutados y resultado

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | 137 files already formatted — PASS |
| `uv run ruff check .` | All checks passed! — PASS |
| `uv run mypy src` | Success: no issues found in 39 source files — PASS |
| `uv run pytest --cov=thehouse_ml --cov-report=term-missing` | 1678 passed in 684.12s; cobertura 96.10 % — PASS |
| `uv run coverage report --include=<código de ML-04>` | 98.29 % — PASS |
| `uv run python -m thehouse_ml prepare` y `evaluate` (dos veces seguidas) | Código 0 las cuatro; SHA-256 de 182 archivos idénticos entre corridas (incluidos los 10 Parquet y los dos reportes de `evaluate`) — PASS |
| `git log --first-parent origin/main` | Solo merges de PR desde `20b2c19` — PASS |
| `pip-audit`, gitleaks | No ejecutados en local (CI) — no se reportan como PASS |

### Cobertura

| Ámbito | Cobertura |
| --- | ---: |
| Total de `src/thehouse_ml` | **96.10 %** (mínimo exigido 80 %) |
| Código nuevo o modificado de ML-04 (`models/`, `evaluation/`, `synthetic/labels*`, `features/`) | **98.29 %** |
| Módulo de ML-04 con menor cobertura | `synthetic/labels.py` **94 %** |
| `stages.py` y `cli.py` (modificados por ML-04) | **100 %** |
| Pruebas | **1,678** (686 al cerrar ML-03) |

### Resultados de ML-04 (DATOS SINTÉTICOS)

Esquema `feature_schema.json` 2.0.0 · modelo `baseline-1.0.0` · 196,605 pares con etiqueta sintética (Spearman baseline–utilidad 0.513) · 81 casos dorados · en test (300 estudiantes, 0 fuera): **NDCG@10 0.527 [0.503, 0.550]**, NDCG@5 0.535 [0.507, 0.560], Precision@5 0.586 [0.557, 0.615], MAP 0.503 [0.485, 0.522]; azar NDCG@10 0.197; segmento más débil `budget.max` > $3,500 (0.468 [0.410, 0.518]). Son la referencia de ML-05, no desempeño real.

### Estado final de los hallazgos

| Origen | # | Hallazgo | Estado final |
| --- | --- | --- | --- |
| T02/T01 | H1 | `.gitignore` ignoraba `src/thehouse_ml/models/` | RESUELTO (M-ML-030) |
| T02/T01 | H2 | «Las 7 features» en `pairs_summary.md` | RESUELTO (M-ML-030) |
| T02/T01 | H3, H4 | `training_range` [0, 60]; `distance_km` no puntúa | APROBADOS (M-ML-030) |
| T02/T01 | H5 | El backend debe actualizar su spec por el esquema 2.0.0 | **ABIERTO — pendiente del backend (NV-23)**, ver abajo |
| T02/T01 | H6 | Sin commit ni PR | CERRADO — PR #24 mezclado |
| T03 | H1 | Redondeo por factor sin round6 | RESUELTO (M-ML-031 decisión 10) |
| T03 | H2 | Vector de ceros = 55 | APROBADO (M-ML-031 decisión 9) |
| T03 | H3 | `breakdown` y formato del par excluido por coordinar con `TheHouse-GenAI` y el backend | **ABIERTO — acción del responsable** |
| T03 | H4 | Sin commit ni PR | CERRADO — PR #25 mezclado |
| T04 | H1, H2, H3 | Curvas, bandas y utilidad aparte distintas del texto de la spec | APROBADOS (M-ML-032) |
| T04 | H4, H5 | Recorte de `price_to_budget`; ulp entre plataformas en la utilidad oculta | INFORMATIVOS — documentado / aceptado |
| T04 | H6 | Sin commit ni PR | CERRADO — PR #26 mezclado |
| T05 | H1 | Recuento y paquete coinciden en 81 casos | CERRADO |
| T05 | H2, H3, H4 | x.49999955, ejemplo con extras 3/8, borde 2.35 en presupuesto y distancia | APROBADOS (M-ML-033) |
| T05 | H5 | Haversine a lados opuestos de 87.5; round6 obligatorio en .NET | Documentado — **para NV-23** |
| T05 | H6 | Coordenadas `seed_sin_verificar` | Aceptado (la prueba avisa si cambian) |
| T05 | H7 | Perfil incompleto sin formato en la spec | RESUELTO (M-ML-033 decisión 7) |
| T05 | H8 | Sin commit ni PR | CERRADO — PR #27 mezclado |
| T06 | H1 | Valores SUPUESTO de la evaluación | RESUELTO (M-ML-034) |
| T06 | H2, H3 | Semilla por segmento; métricas en `evaluation/` | APROBADOS (M-ML-034) |
| T06 | H4 | Intervalos anchos por campus (30 estudiantes) | Aceptado |
| T06 | H5 | ARQUITECTURA nombraba `reports/metrics.json` | **RESUELTO en T07** (`specs/ARQUITECTURA.md` §3, §4 y §6) |
| T06 | H6 | Sin commit ni PR | CERRADO — PR #28 mezclado |
| T07 | H1 | Texto de la etapa `train` (`pendiente de ML-04`) | **RESUELTO** — `pendiente de ML-05` (autorización del responsable) |
| T07 | H2 | pip-audit y gitleaks no ejecutados en local | Pendiente del CI del PR |
| T07 | H3–H6 | Cobertura mínima 94 %; `memoria.md` entre h0 y h1; `plan.md` desactualizado; fila de ML-03 en ARQUITECTURA | INFORMATIVOS — H3 y H4 aceptados; H5 y H6 resueltos |
| T07 | H7 | Sin commit ni PR | **Se cierra al mezclar el PR de T07** |
| Spec | RK02 | Circularidad de las etiquetas sintéticas | Mitigado (pesos y curvas ocultos distintos, ruido, Spearman 0.513); **ABIERTO hasta ML-07** |
| Spec | RK03 | Paridad Python/backend | Mitigado con 81 casos dorados y nota .NET; **paridad pendiente en NV-23 / ML-06** |
| Backend | NV-23 | Esquema **2.0.0** (3 features nuevas: `distance_ratio`, `price_below_min`, `extra_services_ratio`) y redondeo **round6** con `MidpointRounding.AwayFromZero` en total y factores | **ABIERTO — pendiente del backend** (aviso a cargo del responsable) |
| Backend | NV-12 | Contrato del cuarto (RK03 de ML-03: coordenadas, `placeType`, `propertyType`, mascotas, `livingWith`, servicios) y `Property.AreaM2` | **ABIERTO — pendiente del backend** (heredado de M-ML-028) |
| GenAI | `breakdown` | Campo nuevo propuesto en `ExplainMatchRequest`, y formatos del par excluido y del perfil incompleto | **ABIERTO — coordinación del responsable con `TheHouse-GenAI`** |

### Pendiente y decisiones que esperan aprobación

- **Pendientes del backend:** NV-23 con el esquema 2.0.0 y round6; NV-12 (RK03) y `Property.AreaM2`.
- **Pendiente con `TheHouse-GenAI`:** `breakdown` y los formatos del par excluido y del perfil incompleto.
- **Pendiente del CI:** pip-audit, gitleaks y la corrida del pipeline en el PR de T07.
- **Decisiones:** M-ML-035 registrada como APROBADO según el «EJECUTA» de T07; ML-05 sigue en borrador y requiere su propia aprobación y «EJECUTA».

### Resultado global

**PASS WITH WARNINGS.** ML-04 1.0.0 queda **✅ COMPLETA (7/7)**: los 7 requisitos tienen evidencia (R01–R06 en sus secciones y R07 en «T07»), 1,678 pruebas en verde, cobertura de 96.10 % total y 98.29 % en el código de ML-04, ruff y mypy sin errores, `prepare` y `evaluate` reproducibles byte por byte y coherencia completa con la constitución, `AGENTS.md` y M-ML-029 a M-ML-034. T07 queda en **PASS** (T07-H1 resuelto); el global sigue con advertencias por lo que no depende de este repositorio: **pendientes externos** (NV-23 con esquema 2.0.0 y round6, NV-12/RK03 y `AreaM2` en el backend; `breakdown` con `TheHouse-GenAI`); **RK02** hasta ML-07; y pip-audit/gitleaks a verificar en el CI del PR. Todas las métricas son **sintéticas**. Sin dependencias nuevas, sin descargas, sin secretos y sin commit, según lo autorizado.
