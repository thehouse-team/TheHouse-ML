# ML-04 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-04 1.0.0 |
| Tareas ejecutadas | T02 · Esquema 2.0.0 y features nuevas (R02) y, después, T01 · Configuración versionada (R01); en una segunda ejecución, T03 · Puntaje y desglose (R03). Cada una con sección propia abajo |
| Estado | **🟡 EN EJECUCIÓN — 3/7** (T01, T02 y T03 completas; T04–T07 abiertas) |
| Autorización (T02 y T01) | «EJECUTA ML-04 1.0.0, tasks T02 y después T01 (en ese orden, porque T01 valida contra las features de T02), en la rama feat/ml-04-config-schema. Rutas permitidas: src/thehouse_ml/ (features/ y el módulo del modelo base donde indique specs/ARQUITECTURA.md), tests/, reports/ (solo lo que regenere prepare), memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). data/processed/ se regenera pero sigue fuera de git. Sin dependencias nuevas. No modifiques students_params.json, rooms_params.json ni pairs_params.json» — César Yael López Trujillo, 2026-11-20 |
| Rama (T02 y T01) | `feat/ml-04-config-schema` (commit base `fac23bc`; mezclada en el PR #24, `643f94e`) |
| Autorización (T03) | «EJECUTA ML-04 1.0.0, task T03 (ML-04-R03), en la rama feat/ml-04-score. Rutas permitidas: src/thehouse_ml/models/, src/thehouse_ml/features/ (solo para reutilizar, no cambiar el esquema), tests/, memoria.md, specs/ml-04-baseline-model/ y specs/README.md (solo la fila de ML-04). Sin dependencias nuevas (nada de hypothesis: las pruebas de propiedades usan rejillas o random.Random con semilla). No modifiques feature_schema.json (2.0.0) ni baseline_params.json (baseline-1.0.0)» — César Yael López Trujillo, 2026-11-20 |
| Rama (T03) | `feat/ml-04-score` (commit base `643f94e`, merge del PR #24) |
| Commit | Sin commit, por instrucción explícita del responsable (las dos ejecuciones) |
| Fecha de ejecución | 2026-11-20 |
| Decisiones | M-ML-029 (APROBADO); M-ML-030 (**APROBADO**, César Yael López Trujillo, 2026-11-20: lo que la spec no fijaba y la resolución de H1 y H2); M-ML-031 (**APROBADO**, César Yael López Trujillo, 2026-11-20: lo que T03 decidió y la corrección del redondeo por factor de T03-H1) |

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
