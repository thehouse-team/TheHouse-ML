# ML-03 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-03 1.0.0 |
| Tareas ejecutadas | T01 · Esquema de features (R01); T02 · Cálculo de features de referencia (R02) y T03 · Restricciones duras (R03), cada una con sección propia abajo |
| Autorización T01 | «EJECUTA ML-03 1.0.0, task T01 (ML-03-R01), en la rama feat/ml-03-schema. Rutas permitidas: src/thehouse_ml/, tests/, data/seed/, memoria.md y specs/ml-03-feature-schema/. Sin dependencias nuevas (nada de jsonschema ni pydantic: valida con la biblioteca estándar)» — César Yael López Trujillo, 2026-11-19 |
| Autorización T01 (ampliación) | Respuestas del responsable a las advertencias 1–5 y autorización para mover el esquema a `src/thehouse_ml/features/` (ubicación de `specs/ARQUITECTURA.md`), actualizar `specs/ARQUITECTURA.md` (ML-03 con 7 tasks y catálogo sintético de cuartos) y `specs/README.md` (ML-03 en ejecución, 1/7); aprueba `backend_status = "ml_only"` y el radio de 6371 km; sin commit — César Yael López Trujillo, 2026-11-19 |
| Rama (T01) | `feat/ml-03-schema` |
| Commit base | `ea3b615` (merge del PR #15, spec ML-03 1.0.0 aprobada) |
| Commit (T01) | Sin commit, por instrucción explícita del responsable en el «EJECUTA» |
| Fecha de ejecución T01 | 2026-11-19 |
| Decisión | M-ML-022 (APROBADO) |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` → `AGENTS.md` → `memoria.md` (M-ML-016 a M-ML-021) → `specs/ARQUITECTURA.md` → `spec.md` (sobre todo §6, §7 y §10), `plan.md` y `task.md` de ML-03.

## T01 · Esquema de features

### Qué se hizo

1. **Snapshot de NV-11** — `data/seed/nv11_catalogs.json` (en git) con origen «TheHouse-Backend, `TheHouse.Application/Common/PreferenceCatalogs.cs` y `Contracts/Profile/ProfileContracts.cs`, commit `b7d7595` (verificado en `ea10a4c`, 2026-11-19)». **Verificado por César Yael López Trujillo con el código del backend en el commit `ea10a4c`** (`source.verified_by`); tipos del backend registrados en `backend_types`: `BudgetMin`/`BudgetMax` `int?` (`BudgetResponse(int? Min, int? Max, string Currency)`), `MaxDistanceKm` `decimal?` y `Pets` `bool?`. Contenido: moneda `MXN`; presupuesto 0–50,000 con `budget.min <= budget.max`; `maxDistanceKm` 0.5–30 (decimal); catálogos `essentialServices` (11), `livingWith` (3), `placeType` (4), `transport` (4) y `propertyTypes` (3); `pets` booleano; 9 campos nulables y los 3 del quick setup (`budget.max`, `livingWith`, `placeType`).
2. **Esquema** — `src/thehouse_ml/features/feature_schema.json`, versión `1.0.0`:
   - `metadata`: `schema_version`, `spec` «ML-03 1.0.0», `nv11_source` (commit `b7d7595`, verificado en `ea10a4c`, ruta del snapshot), `generated_by`, `synthetic_training_data: true` y la regla de versionado.
   - `catalogs.nv11` (copia exacta del snapshot) y `catalogs.ml.university_code` (10 códigos de `data/universities.csv`); `universities.unverified_codes` = TESJI, UMB-VDC y UCAN (`seed_sin_verificar`, spec §7 regla 4).
   - `student`: los 9 campos de NV-11 más `university_code`, cada uno con tipo, rango o catálogo, `nullable`, `required_for_scoring` y regla para faltantes (`default`, `reject` o `accept_all`).
   - `room`: 13 campos con tipo, rango o catálogo y `backend_status` (`present`, `missing_in_property` o `ml_only`). Tipos de Property en `backend_type`: `Price` `decimal` (→ `price_mxn` se redondea a entero), `Bedrooms` y `Bathrooms` `int`, `AreaM2` `double`. Los rangos `bedrooms` 1–10, `bathrooms` 1–6 y `area_m2` 6–400 son SUPUESTOS documentados (`range_basis`).
   - `pair_features`: 7 definiciones con tipo, rango, dependencias y fórmula en texto (el cálculo es T02). Factores de transporte APROBADOS (`parameters_status`; spec §10; M-ML-022).
   - `hard_constraints`: `HC01_budget`, `HC02_living_with` y `HC03_pets`.
   - `feature_order`: lista fija de 7 features para el ONNX.
3. **Módulo** — `src/thehouse_ml/features/` (`__init__.py`, `loader.py`; paquete `thehouse_ml.features`): carga el JSON empaquetado con `importlib.resources` (o desde una ruta) a dataclasses congeladas (`FeatureSchema`, `FieldSpec`, `MissingRule`, `ValueRange`, `PairFeature`, `HardConstraint`, `Metadata`), valida estructura y semántica acumulando todos los problemas en `SchemaError` y expone la versión con `schema_version()`. Solo biblioteca estándar (`json`, `re`, `dataclasses`, `importlib.resources`).
4. **Pruebas sin red** — `tests/unit/test_feature_schema_nv11.py` (contrato con NV-11 y universidades) y `tests/unit/test_feature_schema_validation.py` (esquemas rotos): 82 pruebas nuevas.
5. **Documentos** — M-ML-022 en `memoria.md` (con la verificación humana, los tipos del backend y la aclaración de M-ML-021 sobre los factores de transporte); T01 marcada en `task.md`; `spec.md` §10 con la fila de factores de transporte APROBADO y §14 con 1 task completada; `specs/ARQUITECTURA.md` (ML-03 con 7 tasks, catálogo sintético de cuartos, total 42 tasks) y `specs/README.md` (ML-03 🟡 EN EJECUCIÓN, 1/7).

### Reglas para faltantes (M-ML-022)

| Campo | Nulable | Quick setup | Regla |
| --- | --- | --- | --- |
| `budget.min` | sí | no | → 0 |
| `budget.max` | sí | **sí** | `reject`: no se calcula el puntaje |
| `maxDistanceKm` | sí | no | → 30 |
| `essentialServices` | sí | no | → conjunto vacío |
| `livingWith` | sí | **sí** | `reject` |
| `placeType` | sí | **sí** | `reject` |
| `pets` | sí | no | → `false` |
| `transport` | sí | no | → `"public"` |
| `propertyTypes` | sí | no | nulo o vacío → acepta todos (`accept_all`) |
| `university_code` | no | — | obligatorio (de `data/universities.csv`) |

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `92 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 21 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **340 passed in 8.99s**; cobertura total **93.67 %** ≥ 80 %; `features/loader.py` **100 %** (309 sentencias, 120 ramas) y `features/__init__.py` **100 %** (PASS) |
| `uv build --wheel -o <scratchpad>` (fuera del repo) | La wheel incluye `thehouse_ml/features/feature_schema.json`, `__init__.py` y `loader.py` (PASS) |

### Trazabilidad R01 → prueba

| Criterio | Prueba |
| --- | --- |
| El esquema carga, valida y expone la versión | `test_feature_schema_nv11.py::test_the_packaged_schema_loads_validates_and_exposes_its_version` |
| Procedencia de NV-11 (commit) | `::test_the_schema_cites_the_snapshot_commit` |
| Catálogos **exactamente** los del snapshot (orden incluido) | `::test_catalogs_are_exactly_those_of_nv11`, `::test_student_and_room_fields_use_the_nv11_catalogs` |
| Rangos de NV-11 iguales | `::test_budget_and_distance_ranges_are_those_of_nv11`, `::test_booleans_of_nv11_are_booleans` |
| Nulables = los de NV-11 y todos con regla para faltantes | `::test_nullable_fields_are_those_of_nv11`, `::test_every_nullable_field_has_a_missing_rule`, `::test_missing_rules_of_m_ml_022` |
| Quick setup no nulo para el puntaje | `::test_quick_setup_fields_cannot_be_null_for_scoring` |
| Orden de features sin duplicados | `::test_feature_order_has_no_duplicates_and_covers_every_pair_feature` |
| Factores de transporte cubren el catálogo | `::test_transport_factors_cover_the_transport_catalog` |
| Restricciones duras con id | `::test_hard_constraints_have_ids_for_budget_living_with_and_pets` |
| `backend_status` del cuarto | `::test_room_backend_status_matches_property` |
| Tipos del backend y verificación humana | `::test_student_types_match_the_backend_types_of_the_snapshot`, `::test_the_snapshot_records_the_human_verification`, `::test_room_types_from_property_and_documented_assumptions` |
| `university_code` = `data/universities.csv` | `::test_university_codes_match_the_universities_csv` |
| La validación rechaza esquemas rotos | `test_feature_schema_validation.py::test_broken_schemas_are_rejected` (45 casos), `::test_every_problem_is_reported_at_once`, `::test_invalid_json_is_reported` |
| Sin red | Las pruebas solo leen archivos versionados del repo |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §2.4 / §13.2: el contrato es `feature_schema.json` con nombres, tipos, rangos y catálogos | `feature_schema.json`; pruebas de contrato | PASS |
| Constitution §3 / SDD-001: spec → «EJECUTA» → evidencia | Autorización en la cabecera; M-ML-022 | PASS |
| Constitution §5 / AGENTS: sin dependencias nuevas | `pyproject.toml` y `uv.lock` sin cambios (`git status`) | PASS |
| Constitution §6: nada de `data/raw`, `data/processed` ni `models` en git | Solo archivos nuevos en `data/seed/`, `src/` y `tests/`; `tests/test_repo_hygiene.py` en verde | PASS |
| Constitution §7: sin datos personales | El esquema no tiene ningún campo de identidad; el snapshot solo trae catálogos y rangos | PASS |
| Constitution §8: datos sintéticos etiquetados | `metadata.synthetic_training_data: true` (validado) y `room.synthetic` | PASS |
| Constitution §11: `livingWith` y presupuesto como restricciones duras; sin inferir género | `HC01_budget`, `HC02_living_with`, `HC03_pets`; la descripción de `livingWith` lo declara | PASS (definición; la exclusión se implementa en T03) |
| Constitution §18 / spec §7 regla 1: cambio en NV-11 → versión mayor | Pruebas de catálogos, rangos y nulables contra el snapshot | PASS |
| Spec §7 regla 4: universidades sin verificar marcadas | `universities.unverified_codes` comparado con `final_status` | PASS |
| AGENTS: no editar `constitution.md` ni `AGENTS.md`; rutas permitidas | `git status` solo lista rutas autorizadas | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | T02–T07 siguen abiertas; la paridad y las restricciones en ejecución no se reportan | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | El snapshot de NV-11 se escribió con los valores del «EJECUTA», sin copia local del backend. | ADVERTENCIA | **RESUELTO** — verificado por César Yael López Trujillo con el código del backend en `ea10a4c`; registrado en `nv11_catalogs.json` (`verified_by`) y en el esquema |
| 2 | Tipo del presupuesto fijado como `integer`. | ADVERTENCIA | **RESUELTO** — correcto: `BudgetMin`/`BudgetMax` son `int?`; `MaxDistanceKm` `decimal?`, `Pets` `bool?` (`backend_types`, con prueba) |
| 3 | Rangos del cuarto que no vienen de NV-11. | ADVERTENCIA | **RESUELTO como SUPUESTO documentado** — `Bedrooms`/`Bathrooms` `int`, `AreaM2` `double`, `Price` `decimal`; rangos 1–10, 1–6 y 6–400 marcados `range_basis: SUPUESTO`; `price_mxn` se redondea a entero desde `Price` |
| 4 | Factores de transporte PROPUESTOS en §6 y ausentes de §10. | ADVERTENCIA | **RESUELTO** — APROBADOS; fila en `spec.md` §10 y aclaración de M-ML-021 en M-ML-022 |
| 5 | `synthetic` con `backend_status = "ml_only"`. | INFORMATIVO | **APROBADO** por el responsable |
| 6 | `distance_km` y `effective_distance_km` con rango 0–30 (dominio del catálogo sintético); el tratamiento fuera de rango lo define T02. | INFORMATIVO | Pasa a T02 |
| 7 | Ubicación del módulo distinta de `specs/ARQUITECTURA.md` y conteo de 6 tasks. | INFORMATIVO | **RESUELTO** — módulo movido a `src/thehouse_ml/features/`; ARQUITECTURA con 7 tasks y catálogo sintético de cuartos |
| 8 | `specs/README.md` desactualizado para ML-03. | INFORMATIVO | **RESUELTO** — 🟡 EN EJECUCIÓN, 1/7 |
| 9 | Avisar al backend (NV-12) de los campos `missing_in_property` del cuarto (spec RK03). | ADVERTENCIA | **RESUELTO (2026-11-19)** — aviso enviado por el responsable; ver «Acción pendiente resuelta» en T03. RK03 sigue abierto hasta que el backend implemente NV-12 |
| 10 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T01 |
| 11 | Al cambiar ML-03 de 6 a 7 tasks, el total de `specs/ARQUITECTURA.md` §6 pasa de 41 a 42 tasks. | INFORMATIVO | Ajustado por consistencia |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. El snapshot es un dato propio del proyecto TheHouse.
- **Resultado:** **PASS WITH WARNINGS** — la aceptación de R01 se cumple (`feature_schema.json` con nombres, tipos, rangos, catálogos y versión, validado contra el snapshot de NV-11 verificado por el responsable) y los cuatro controles pasan. Única advertencia abierta: el aviso al backend (NV-12, hallazgo 9), acción pendiente del responsable; no bloquea T01 pero sí el uso del modelo en producción.

## T02 · Cálculo de features de referencia

| Campo | Valor |
| --- | --- |
| Autorización T02 | «EJECUTA ML-03 1.0.0, task T02 (ML-03-R02), en la rama feat/ml-03-pair-features. Rutas permitidas: src/thehouse_ml/features/, tests/, memoria.md y specs/ml-03-feature-schema/. Sin dependencias nuevas» (el mismo mensaje pide actualizar `specs/README.md`) — César Yael López Trujillo, 2026-11-19 |
| Rama | `feat/ml-03-pair-features` |
| Commit base | `5376da9` (merge del PR #16, T01) |
| Commit (T02) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-19 |
| Decisión | M-ML-023 (APROBADO — César Yael López Trujillo, 2026-11-19, con las decisiones sobre los hallazgos 1, 2 y 4) |
| Esquema | `feature_schema.json` 1.0.0 → **1.1.0** (ampliación compatible; ver «Cambio de esquema 1.1.0») |

Lecturas previas: `constitution.md` → `AGENTS.md` → `memoria.md` (M-ML-021 y M-ML-022) → `specs/ARQUITECTURA.md` → `spec.md`, `plan.md`, `task.md` y `evidence.md` de ML-03 → `src/thehouse_ml/features/feature_schema.json`.

### Qué se hizo

1. **`src/thehouse_ml/features/pairs.py`** — implementación de referencia, pura (sin I/O, sin red, sin aleatoriedad):
   - Dataclasses tipadas del esquema: `StudentProfile` (NV-11, todo nulable salvo `university_code`), `ResolvedStudent` (sin nulos) y `Room` (13 campos de `room`). `STUDENT_ATTRIBUTES` y `ROOM_ATTRIBUTES` mapean los nombres del esquema a atributos; una prueba exige que cubran exactamente los campos del esquema.
   - `resolve_student` aplica las reglas para faltantes **leídas del esquema** (`default`, `reject`, `accept_all`); si falta algún dato del quick setup lanza `IncompleteProfileError` con todos los faltantes. Después valida tipo, rango y catálogo de cada campo y `budget.min <= budget.max` (`PairInputError`). `validate_room` hace lo mismo con el cuarto.
   - `compute_pair_features(student, room, campuses, schema=None) -> dict[str, float]` devuelve las 7 features en el orden de `schema.feature_order`; si el esquema nombra una feature sin implementación, falla. `feature_vector` da el vector para el modelo.
   - `distance_km` = `universities.haversine_m(...) / 1000` (misma función de ML-02, radio 6371 km); `effective_distance_km` con los factores de `schema_transport_factors`; `price_to_budget`, `services_coverage`, `place_type_match`, `property_type_match` y `pets_compatible` según las fórmulas del esquema. Las features `integer` valen 0 o 1.
   - Sin recorte de distancia: los valores mayores a 30 km o a `maxDistanceKm` se devuelven tal cual.
   - `student_from_record` / `room_from_record` construyen las dataclasses desde registros con los nombres del esquema (clave ausente en el estudiante = nulo).
2. **`src/thehouse_ml/features/campuses.py`** — `Campus` y `load_campuses(path)`: lee `final_lat`/`final_lon` de `data/universities.csv` y marca `seed_verified = false` en las `seed_sin_verificar`. Es la única parte con I/O; la función de features recibe los campus ya cargados.
3. **`tests/fixtures/pair_feature_cases.json`** — 35 casos por tabla (esquema 1.1.0) (entrada + salida esperada, tolerancia 1e-6), entradas **sintéticas**. Las salidas se calcularon **a mano**, sin el código bajo prueba: los cuartos con distancia > 0 están sobre el meridiano del campus, donde haversine = 6371 km × Δφ (0.09° → 10.007543 km; 0.36° → 40.030174 km). El generador quedó en el scratchpad de la sesión, fuera del repo.
4. **Pruebas** — `tests/unit/test_pair_features.py` (87) y `tests/unit/test_campuses.py` (9), más 7 casos nuevos en `test_feature_schema_validation.py`.

En total, 136 pruebas nuevas:

- `tests/unit/test_pair_features.py`: 120 (incluye los 33 añadidos para el hallazgo 9).
- `tests/unit/test_campuses.py`: 9.
- `tests/unit/test_feature_schema_validation.py`: 7.
5. **Documentos** — M-ML-023 (APROBADO) en `memoria.md`; T02 marcada en `task.md`; `spec.md` §14 con 2 tasks completadas; `specs/README.md` (ML-03 2/7); nota del fixture en `tests/fixtures/README.md`.

### Comprobación de las fórmulas contra el esquema

| Feature | Esquema | Pedido en el «EJECUTA» | Diferencia |
| --- | --- | --- | --- |
| `distance_km` | haversine campus–cuarto, radio 6371 km | haversine con `haversine_m` de ML-02 | Ninguna |
| `effective_distance_km` | `distance_km × factor`; walk 1.0, bike 0.5, public 0.4, car 0.25; nulo → public | Igual | Ninguna |
| `price_to_budget` | 1.0.0: `price_mxn / budget.max`; `budget.max = 0` → 0. **1.1.0:** `price_mxn / max(budget.max, 1)` | `price_mxn / budget.max`; definir el caso 0 | En 1.0.0 el esquema ya definía el caso 0 (hallazgo 2); resuelto en 1.1.0 con una sola fórmula |
| `services_coverage` | cubiertos / pedidos; vacío o nulo → 1 | Igual | Ninguna |
| `place_type_match` | 1 si coincide | Igual | Ninguna |
| `property_type_match` | 1 si nulo, vacío o contiene | Igual | Ninguna |
| `pets_compatible` | 0 solo si mascota y no admite; nulo → false | Igual | Ninguna |

### Casos por tabla

| Escenario pedido | Casos (`id`) |
| --- | --- |
| Presupuesto exacto | `budget_exact` (1.0) |
| Distancia 0 | `distance_zero`, `distance_zero_other_campus` |
| Sin servicios pedidos | `no_services_requested`, `null_essential_services` |
| Cuarto sin servicios | `room_without_services` (0.0); además `partial_services` (2/3) |
| Mascotas, 4 combinaciones | `pets_student_{false,true}_room_{false,true}` |
| `propertyTypes` vacío | `property_types_empty`; además `property_types_contains`, `property_types_mismatch`, `null_property_types` |
| Cada transporte | `transport_walk`, `transport_bike`, `transport_public`, `transport_car`, `null_transport` |
| Campos nulos con su regla | `null_budget_min_and_max_distance`, `null_essential_services`, `null_pets`, `null_property_types`, `null_transport`, `all_optional_null` |
| Distancia > 30 km | `distance_over_30_km` (40.03 km con `maxDistanceKm` 5), `distance_over_30_km_car` |
| Quick setup incompleto | `incomplete_budget_max`, `incomplete_livingWith`, `incomplete_placeType`, `incomplete_all_quick_setup` |
| `budget.max = 0` (1.1.0) | `budget_max_zero_price_zero` (0 / 1 = 0), `budget_max_zero_price_positive` (3000 / 1 = 3000), `budget_max_one` (3000 / 1 = 3000) |
| Rangos SUPUESTOS del cuarto (1.1.0) | `room_assumed_ranges_exceeded` (12 recámaras, 8 baños, 450 m²: se acepta) |
| Otros | `place_type_mismatch`, `price_over_budget` (1.25, sin recorte) |

### Cambio de esquema 1.1.0 (M-ML-023)

Ampliación compatible (versión menor): todo valor válido en 1.0.0 sigue siéndolo y ningún catálogo, rango ni nulable de NV-11 cambia. Registrado en `metadata.changelog`.

| Elemento | 1.0.0 | 1.1.0 |
| --- | --- | --- |
| `distance_km.range` | `{"min": 0, "max": 30}` | `{"min": 0, "max": null}` (sin límite superior) + `"training_range": [0, 30]` |
| `effective_distance_km.range` | `{"min": 0, "max": 30}` | `{"min": 0, "max": null}` + `"training_range": [0, 30]` |
| `price_to_budget.formula` | `price_mxn / budget.max`; `budget.max = 0` → 0 | `price_mxn / max(budget.max, 1)`, sin casos especiales |
| `price_to_budget.range` | `{"min": 0, "max": 1}` | `{"min": 0, "max": null}` + `"training_range": [0, 1]` (pares que pasan HC01; hallazgo 9) |
| `bedrooms`, `bathrooms`, `area_m2` (`range_basis`) | SUPUESTO documentado | SUPUESTO para el catálogo sintético; en la entrada solo tipo y no negativo |
| `metadata` | — | `schema_version` 1.1.0, `decision` y `changelog` |

En el cargador (`loader.py`): `ValueRange.max` admite `None` (solo en `pair_features`; un campo del estudiante o del cuarto con `max: null` se rechaza), `PairFeature.training_range` (par `[min, max]` cerrado y dentro de `range`) y `FieldSpec.range_basis` / `range_is_assumption`. En `pairs.py`: la fórmula única de `price_to_budget` y la validación de los campos con rango SUPUESTO (tipo y no negativo).

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `96 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 23 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **476 passed**; cobertura total **94.32 %** ≥ 80 %; `features/pairs.py` **100 %** (173 sentencias, 62 ramas), `features/loader.py` **100 %** (334 sentencias, 132 ramas), `features/campuses.py` **100 %** (45 sentencias, 14 ramas), `features/__init__.py` 100 % (PASS) |

Corrida final, después de aplicar las decisiones de M-ML-023 (esquema 1.1.0), incluida la del hallazgo 9.

### Trazabilidad R02 → prueba

| Criterio | Prueba |
| --- | --- |
| Casos por tabla (35) con tolerancia 1e-6 | `test_pair_features.py::test_table_case[*]` |
| Una distancia > 30 km es válida contra el esquema 1.1.0 (y queda fuera de `training_range`) | `::test_a_distance_beyond_30_km_is_valid_against_the_schema` |
| `price_to_budget` = `price_mxn / max(budget.max, 1)` | `::test_price_to_budget_is_one_formula_with_max_budget_one` |
| `price_to_budget` = 3000.0 es válido contra el esquema y queda fuera de `training_range` [0, 1] | `::test_a_price_to_budget_of_3000_is_valid_against_the_schema` |
| Todo valor calculado de la tabla cumple el `range` de su feature (31 casos) | `::test_every_computed_feature_is_valid_against_its_schema_range[*]` |
| Solo las features sin límite superior llevan `training_range`; las de rango cerrado no lo pueden rebasar | `::test_only_unbounded_features_can_leave_their_training_range` |
| Los rangos SUPUESTOS del cuarto no lo rechazan | `::test_assumed_room_ranges_do_not_reject_the_room` |
| `max: null` y `training_range` en el cargador | `test_feature_schema_validation.py::test_broken_schemas_are_rejected` (7 casos nuevos) |
| La tabla cubre todos los escenarios pedidos | `::test_the_table_covers_every_requested_scenario` |
| Coordenadas de la tabla = `data/universities.csv`; metadatos = esquema | `::test_the_table_uses_the_campus_coordinates_of_universities_csv`, `::test_the_table_metadata_matches_the_schema` |
| Orden exacto del esquema | `::test_features_come_back_in_the_exact_order_of_the_schema`, `::test_the_order_is_taken_from_the_schema_not_from_the_code`, `::test_a_schema_feature_without_reference_implementation_fails` |
| Distancias > 30 km o > `maxDistanceKm` sin recorte | `::test_distances_beyond_30_km_or_max_distance_are_returned_unclipped` |
| Reglas para faltantes leídas del esquema | `::test_missing_rules_come_from_the_schema`, `::test_dataclasses_cover_every_field_of_the_schema` |
| Validación de entradas | `::test_invalid_student_values_are_rejected` (16), `::test_invalid_room_values_are_rejected` (12), `::test_records_with_unknown_or_missing_keys_are_rejected` y otras |
| Pureza (determinista, sin mutar, sin I/O ni aleatoriedad) | `::test_the_function_is_deterministic_and_does_not_mutate_inputs`, `::test_the_pairs_module_has_no_io_or_randomness` |
| Una sola implementación de haversine | `::test_distance_reuses_the_single_haversine_of_ml02` |
| Campus desde `data/universities.csv` | `test_campuses.py` (9) |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §2.4 / §13: el contrato es `feature_schema.json` | Reglas, factores y orden se leen del esquema; la fórmula de `price_to_budget` es la misma en esquema y código | PASS |
| Constitution §18: versión semántica | 1.0.0 → 1.1.0 (menor): ampliación compatible aprobada en M-ML-023; NV-11 sin cambios (pruebas contra el snapshot en verde) | PASS |
| Constitution §5 / AGENTS: sin dependencias nuevas | Solo biblioteca estándar; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| Constitution §8: datos sintéticos etiquetados | `metadata.synthetic: true` en `pair_feature_cases.json` | PASS |
| Constitution §11: restricciones duras | No se implementan aquí (T03); la distancia no se trata como restricción dura | PASS (fuera de alcance de T02) |
| Constitution §13.4: casos dorados para el backend | La tabla queda lista para ML-06; la paridad ONNX no se ejecuta en T02 | No ejecutado (ML-06) |
| Spec §7 regla 2: una sola implementación de referencia | `pairs.py`; `haversine_m` de ML-02 reutilizada | PASS |
| AGENTS: rutas permitidas | `git status` solo lista `src/thehouse_ml/features/`, `tests/`, `memoria.md`, `specs/ml-03-feature-schema/` y `specs/README.md` (pedido en el «EJECUTA») | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | T03–T07 abiertas; la paridad ONNX no se reporta | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | El esquema declara `range` 0–30 para `distance_km` y `effective_distance_km`, pero la función devuelve distancias reales mayores (p. ej. 40.03 km en `distance_over_30_km`). El esquema **no se cambió**. Opciones: ampliar el rango (cambio de esquema; según la regla de versionado, posible versión mayor) o documentar que 0–30 es el dominio del catálogo sintético y no un límite de la entrada. | ADVERTENCIA | **RESUELTO (M-ML-023)** — `range` con mínimo 0 y sin límite superior + `training_range` [0, 30]; esquema 1.1.0; prueba de que > 30 km es válido |
| 2 | `budget.max = 0`: el esquema ya fija `price_to_budget = 0`; no hay división entre cero. Con renta positiva el valor 0 parece «precio ideal», aunque solo es seguro porque HC01 excluye ese par (T03). Alternativa: devolver un valor ≥ 1 para que ese par nunca parezca barato. | ADVERTENCIA | **RESUELTO (M-ML-023)** — `price_mxn / max(budget.max, 1)` en esquema y código; casos recalculados a mano (0 → 0; 3000 → 3000) |
| 3 | `price_to_budget` puede pasar de 1 (`price_over_budget` = 1.25): se devuelve sin recorte; el esquema ya dice que el rango 0–1 solo se cumple en los pares que pasan HC01. | INFORMATIVO | **RESUELTO con el hallazgo 9** — `range` sin límite superior y `training_range` [0, 1] |
| 4 | La función valida contra los rangos del cuarto que son SUPUESTOS (`bedrooms` 1–10, `bathrooms` 1–6, `area_m2` 6–400): un cuarto real fuera de esos rangos se rechaza aunque esos campos no entren en ninguna feature. | ADVERTENCIA | **RESUELTO (M-ML-023)** — solo tipo y no negativo; los rangos quedan como SUPUESTO del catálogo sintético |
| 5 | `budget.min`, `maxDistanceKm` y `livingWith` no entran en ninguna de las 7 features (son de HC/ML-04); `livingWith` igual es obligatorio por la regla `reject` del esquema. | INFORMATIVO | Coherente con el esquema |
| 6 | Las universidades sin verificar (TESJI, UMB-VDC, UCAN) dan distancias orientativas; `Campus.seed_verified` lo conserva, pero la salida de features no lo marca. | INFORMATIVO | Para T06 (métricas con y sin ellas) |
| 7 | Aviso al backend (NV-12) de los campos `missing_in_property` (hallazgo 9 de T01). | ADVERTENCIA | **RESUELTO (2026-11-19)** — ver «Acción pendiente resuelta» en T03 |
| 8 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T02 |
| 9 | Con la fórmula 1.1.0, `price_to_budget` puede valer miles (p. ej. 3000 con presupuesto 0), mientras el esquema mantiene `range` 0–1 para esta feature (válido solo en pares que pasan HC01). No se cambió: queda igual que el hallazgo 3. Si se quiere que todo valor calculado cumpla su `range`, la opción análoga a la de la distancia sería `max: null` con `training_range` [0, 1]. | INFORMATIVO | **RESUELTO (M-ML-023)** — dentro de 1.1.0 (sin publicar): `range` `{"min": 0, "max": null}` y `training_range` [0, 1]; prueba de que 3000.0 es válido y queda fuera de `training_range` |
| 10 | Revisión del resto de las features: `services_coverage` (0–1) es una proporción de un subconjunto, así que no puede salir de [0, 1]; `place_type_match`, `property_type_match` y `pets_compatible` solo valen 0 o 1. Ninguna otra feature puede rebasar su `range`. Los mínimos en 0 se cumplen porque las distancias, la renta y `max(budget.max, 1)` no son negativos (entradas validadas). | INFORMATIVO | Verificado con pruebas: todo valor de la tabla cumple su `range` |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Los casos por tabla son sintéticos; las coordenadas de campus vienen de `data/universities.csv` (ODbL, registrado en M-ML-016/M-ML-017).
- **Resultado:** **PASS WITH WARNINGS** — la aceptación de R02 se cumple (función pura con casos límite: presupuesto exacto, distancia 0, sin servicios, y el resto de la tabla) y los cuatro controles pasan. Los hallazgos 1, 2, 3, 4 y 9 se resolvieron con M-ML-023 (esquema 1.1.0). Única advertencia abierta: el aviso al backend (NV-12, hallazgo 7), acción pendiente del responsable.

## T03 · Restricciones duras

| Campo | Valor |
| --- | --- |
| Autorización T03 | «EJECUTA ML-03 1.0.0, task T03 (ML-03-R03), en la rama feat/ml-03-constraints. Rutas permitidas: src/thehouse_ml/features/, tests/, memoria.md, specs/ml-03-feature-schema/ y specs/README.md (solo la fila de ML-03). Sin dependencias nuevas. No modifiques feature_schema.json (1.1.0); si hace falta un cambio, repórtamelo como hallazgo» — César Yael López Trujillo, 2026-11-19 |
| Rama | `feat/ml-03-constraints` |
| Commit base | `ed927b3` (merge del PR #17, T02) |
| Commit (T03) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-19 |
| Decisión | M-ML-024 (APROBADO — César Yael López Trujillo, 2026-11-19) |
| Esquema | `feature_schema.json` **1.1.0, sin cambios** |

Lecturas previas: `constitution.md` (§11) → `AGENTS.md` → `memoria.md` (M-ML-021 a M-ML-023) → `spec.md`, `plan.md`, `task.md` y `evidence.md` de ML-03 → `src/thehouse_ml/features/` (`feature_schema.json`, `loader.py` y `pairs.py`).

### Qué se hizo

1. **`src/thehouse_ml/features/constraints.py`** — implementación de referencia, pura (sin I/O, sin red, sin aleatoriedad):
   - `check_hard_constraints(student: StudentProfile, room: Room, schema: FeatureSchema | None = None) -> ConstraintResult`. Primero `resolve_student` (la misma de `compute_pair_features`: reglas para faltantes y validación; quick setup incompleto → `IncompleteProfileError`) y `validate_room`; después evalúa **las restricciones de `hard_constraints` del esquema, en su orden**.
   - `ConstraintResult(allowed: bool, violated: tuple[str, ...])`: `violated` lista **todas** las restricciones violadas, en el orden del esquema.
   - `filter_rooms(student: StudentProfile, rooms: Iterable[Room], schema: FeatureSchema | None = None) -> FilterResult`, con `FilterResult(allowed: tuple[Room, ...], exclusions: Mapping[str, int], excluded: int)`: resuelve el perfil una vez, conserva el orden de los cuartos y cuenta exclusiones por id (todas las del esquema, con 0 si no excluyen; un cuarto con varias violaciones cuenta en cada una).
   - Cada id del esquema se liga a un predicado (`HC01_budget`, `HC02_living_with`, `HC03_pets`); si el esquema trae un id sin implementación, ambas funciones lanzan `PairInputError`. `implemented_constraints()` expone los ids implementados.
   - La distancia **no** es restricción dura (M-ML-023): no se evalúa aquí.
2. **`tests/fixtures/hard_constraint_cases.json`** — 27 casos por tabla (esquema 1.1.0), entradas **sintéticas** y salidas escritas a mano (el generador quedó en el scratchpad de la sesión, fuera del repo, con las salidas como literales). Nota en `tests/fixtures/README.md`.
3. **`tests/unit/test_hard_constraints.py`** — 49 pruebas nuevas (27 de la tabla y 22 de reglas, filtro, coherencia con T02, privacidad y pureza).
4. **`src/thehouse_ml/features/__init__.py`** — exporta `ConstraintResult`, `FilterResult`, `check_hard_constraints`, `filter_rooms` e `implemented_constraints`.
5. **Documentos** — M-ML-024 (APROBADO) en `memoria.md`; T03 marcada en `task.md`; `spec.md` §14 con 3 tasks completadas; fila de ML-03 en `specs/README.md` (3/7).

### Restricciones del esquema contra la implementación

| Id | `excludes_when` del esquema | Implementación | Diferencia |
| --- | --- | --- | --- |
| `HC01_budget` | `room.price_mxn > student.budget.max` | `room.price_mxn > student.budget_max` (renta igual al presupuesto pasa) | Ninguna |
| `HC02_living_with` | `student.livingWith != "any" and room.livingWith != student.livingWith` | Igual | Ninguna |
| `HC03_pets` | `student.pets = true and room.pets_allowed = false` | `student.pets and not room.pets_allowed` (pets nulo → `false`) | Ninguna |

### `livingWith`: 9 combinaciones (spec §7 regla 3)

| Estudiante ↓ / Cuarto → | `women` | `men` | `any` |
| --- | --- | --- | --- |
| `any` | pasa | pasa | pasa |
| `women` | pasa | excluye (HC02) | excluye (HC02) |
| `men` | excluye (HC02) | pasa | excluye (HC02) |

### Casos por tabla

| Escenario pedido | Casos (`id`) |
| --- | --- |
| Renta < = > presupuesto | `budget_rent_below` (3999), `budget_rent_equal` (4000, pasa), `budget_rent_above` (4001, HC01); además `budget_max_zero_rent_zero` (pasa) y `budget_max_zero_rent_one` (HC01) |
| 9 combinaciones de `livingWith` | `living_with_student_{any,women,men}_room_{women,men,any}` |
| 4 combinaciones de mascotas | `pets_student_{false,true}_room_{false,true}` |
| `pets` nulo (regla para faltantes → `false`) | `pets_null_room_false` (pasa) |
| Varias violadas a la vez | `violates_budget_and_pets`, `violates_living_with_and_pets`, `violates_all_three` |
| Distancia no es restricción dura | `distance_over_max_and_30_km_not_excluded` (40.03 km con `maxDistanceKm` 5: pasa) |
| Quick setup incompleto | `incomplete_budget_max`, `incomplete_livingWith`, `incomplete_placeType`, `incomplete_all_quick_setup` (`IncompleteProfileError`) |

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `98 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 24 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **525 passed**; cobertura total **94.43 %** ≥ 80 %; `features/constraints.py` **100 %** (54 sentencias, 8 ramas); `pairs.py`, `loader.py`, `campuses.py` y `__init__.py` siguen en 100 % (PASS) |

### Trazabilidad R03 → prueba

| Criterio | Prueba (`tests/unit/test_hard_constraints.py`) |
| --- | --- |
| Casos por tabla (27) | `::test_table_case[*]` |
| La tabla cubre todos los escenarios pedidos; metadatos = esquema | `::test_the_table_covers_every_requested_scenario`, `::test_the_table_metadata_matches_the_schema` |
| HC01: renta igual pasa, un peso más excluye | `::test_rent_equal_to_budget_passes_and_one_peso_more_is_excluded` |
| HC02: matriz 3 × 3 de la spec | `::test_living_with_matrix_of_the_spec` |
| HC03 con `pets` nulo según la regla del esquema | `::test_null_pets_follows_the_missing_rule_of_the_schema` |
| Todas las violadas, en el orden del esquema | `::test_every_violated_constraint_is_reported_in_schema_order` |
| Restricciones leídas del esquema, no del código | `::test_constraints_are_read_from_the_schema`, `::test_every_schema_constraint_has_an_implementation`, `::test_the_implemented_rules_are_the_ones_written_in_the_schema` |
| Restricción sin implementación → falla | `::test_a_schema_constraint_without_implementation_fails` |
| Una sola implementación de las reglas para faltantes (T02) | `::test_missing_rules_are_the_single_implementation_of_t02`, `::test_invalid_inputs_are_rejected_like_in_t02` |
| Filtro: permitidos y exclusiones por id | `::test_filter_returns_allowed_rooms_in_order_and_exclusions_per_constraint`, `::test_filter_agrees_with_the_pair_function`, `::test_filter_with_no_rooms_reports_zero_for_every_constraint`, `::test_filter_with_incomplete_profile_raises_the_t02_error` |
| La distancia no es restricción dura | `::test_distance_is_not_a_hard_constraint` |
| Coherencia con T02 (casos que pasan: `price_to_budget ≤ 1` y `pets_compatible = 1`) | `::test_pairs_that_pass_the_constraints_have_price_to_budget_at_most_one_and_compatible_pets` (≥ 20 casos de `pair_feature_cases.json`), `::test_pair_cases_excluded_by_t02_signals_violate_the_matching_constraint` |
| Privacidad: sin campo de género; HC02 solo usa `livingWith` | `::test_no_gender_field_exists_in_the_profile_or_is_used_by_the_constraints` |
| Pureza | `::test_the_constraints_module_has_no_io_or_randomness`, `::test_the_function_is_deterministic_and_does_not_mutate_inputs` |

### Acción pendiente resuelta

| Acción | Estado |
| --- | --- |
| Aviso a TheHouse-Backend sobre los campos que faltan en Property (lat/lon, placeType, propertyType, pets_allowed, livingWith y servicios del catálogo de NV-11) y sobre las correcciones de `reports/universities_check.md` | **RESUELTA** — enviado por César Yael López Trujillo por WhatsApp el 2026-11-19 a las 12:33 y recibido a las 12:35. NV-12 queda como responsabilidad del backend (riesgo RK03 sigue abierto hasta que el backend lo implemente). Cierra el hallazgo 9 de T01 y el 7 de T02 |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §11.1: `livingWith` y presupuesto como restricciones duras | `HC01_budget` y `HC02_living_with` excluyen el cuarto; `HC03_pets` también | PASS |
| Constitution §11.2: sin género como señal; solo en `livingWith` que elige el estudiante | Sin campo de género en `StudentProfile`, `ResolvedStudent` ni el esquema; HC02 solo compara `livingWith` (prueba) | PASS |
| Constitution §2.4 / §13: el contrato es `feature_schema.json` | Las restricciones y su orden se leen del esquema; el texto `excludes_when` está fijado por prueba | PASS |
| Constitution §13.4: casos dorados para el backend | `hard_constraint_cases.json` listo para ML-06; la paridad no se ejecuta en T03 | No ejecutado (ML-06) |
| Constitution §5 / AGENTS: sin dependencias nuevas | Solo biblioteca estándar; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| Constitution §8: datos sintéticos etiquetados | `metadata.synthetic: true` en `hard_constraint_cases.json` | PASS |
| Spec §7 regla 2: una sola implementación | `resolve_student` y `validate_room` de T02 reutilizadas (prueba de identidad) | PASS |
| M-ML-023: la distancia no es restricción dura | Caso `distance_over_max_and_30_km_not_excluded` y prueba del filtro | PASS |
| `feature_schema.json` sin cambios | `git status` no lo lista | PASS |
| AGENTS: rutas permitidas | `git status` solo lista `src/thehouse_ml/features/`, `tests/`, `memoria.md`, `specs/ml-03-feature-schema/` y la fila de ML-03 en `specs/README.md` | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | T04–T07 abiertas; la paridad ONNX y la prueba de ≥ 90 % (T06) no se reportan | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | No hace falta cambiar `feature_schema.json`: las tres restricciones del esquema coinciden con lo pedido en el «EJECUTA». | INFORMATIVO | Esquema 1.1.0 sin cambios |
| 2 | El esquema define cada restricción con texto (`excludes_when`), no con una expresión ejecutable; el código liga cada id a su predicado. Un cambio del texto con el mismo id no se detectaría solo. | ADVERTENCIA | **Mitigado** — `test_the_implemented_rules_are_the_ones_written_in_the_schema` fija el texto de las tres; si cambia, la prueba falla y obliga a revisar el código |
| 3 | Por la regla aprobada (spec §7 regla 3 y §10), un estudiante `women` o `men` excluye los cuartos `any`: solo ve cuartos con su misma política. Esto reduce los cuartos compatibles de esos estudiantes y afecta el umbral de ≥ 90 % de T06 según las distribuciones de `livingWith` de T04 y T05. | INFORMATIVO | Para T04–T06: ver «Nota para T05» |
| 4 | La implicación con T02 va en un solo sentido: si el par pasa HC01, `price_to_budget ≤ 1`; pero `price_to_budget = 1` no implica que pase (presupuesto 0 y renta 1 → 1 / max(0, 1) = 1, y HC01 lo excluye; caso `budget_max_zero_rent_one`). No afecta el `training_range` [0, 1] de 1.1.0. | INFORMATIVO | Documentado; sin cambio |
| 5 | Decisiones de implementación nuevas (API, `FilterResult`, error para ids sin implementación, conteo de exclusiones con varias violaciones). | ADVERTENCIA | **RESUELTO** — M-ML-024 APROBADO por César Yael López Trujillo (2026-11-19) |
| 6 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T03 |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Los casos por tabla son sintéticos; las coordenadas de UTTT vienen de `data/universities.csv` (ODbL, M-ML-016/M-ML-017).
- **Resultado:** **PASS** — la aceptación de R03 se cumple (los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla), los cuatro controles pasan y M-ML-024 quedó APROBADO (hallazgo 5 resuelto). El hallazgo 2 queda mitigado con prueba. El aviso al backend quedó resuelto; RK03 sigue abierto hasta que el backend implemente NV-12 (riesgo de la spec, no bloquea T03).

### Nota para T05

La distribución de `livingWith` de los cuartos debe incluir suficientes cuartos `women` y `men` para que se cumpla el umbral de ≥ 90 % de T06, porque los estudiantes `women` y `men` no ven cuartos `any` (HC02). Origen: hallazgo 3 de T03; M-ML-024.
