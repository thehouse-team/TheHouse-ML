# ML-03 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-03 1.0.0 |
| Tareas ejecutadas | T01 · Esquema de features (R01); T02 · Cálculo de features de referencia (R02); T03 · Restricciones duras (R03); T04 · Estudiantes sintéticos (R04); T05 · Catálogo sintético de cuartos (R05) y T06 · Conjunto de pares (R06), cada una con sección propia abajo |
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

## T04 · Estudiantes sintéticos

| Campo | Valor |
| --- | --- |
| Autorización T04 | «EJECUTA ML-03 1.0.0, task T04 (ML-03-R04), en la rama feat/ml-03-students. Rutas permitidas: src/thehouse_ml/ (incluido src/thehouse_ml/synthetic/), tests/, reports/, memoria.md, specs/ml-03-feature-schema/ y specs/README.md (solo la fila de ML-03). Sin dependencias nuevas (usa random.Random de la biblioteca estándar o numpy solo si ya está instalado como dependencia de pandas; no agregues paquetes). No modifiques feature_schema.json (1.1.0)» — César Yael López Trujillo, 2026-11-19 |
| Rama | `feat/ml-03-students` |
| Commit base | `f776f00` (merge del PR #18, T03) |
| Commit (T04) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-19 |
| Decisión | M-ML-025 (APROBADO — César Yael López Trujillo — 2026-11-19, con los valores propuestos sin cambios) |
| Esquema | `feature_schema.json` **1.1.0, sin cambios** |

Lecturas previas: `constitution.md` (§8 y §11) → `AGENTS.md` → `memoria.md` (M-ML-014, M-ML-017, M-ML-020 a M-ML-024) → `spec.md` (§6 y §10), `plan.md`, `task.md` y `evidence.md` de ML-03 (nota para T05) → `src/thehouse_ml/features/`.

### Qué se hizo

1. **`src/thehouse_ml/synthetic/students_params.json`** (en git, empaquetado) — todos los parámetros del generador; cada uno es `{value, basis}` con `basis` = «calibrado: <fuente>» o «SUPUESTO documentado: …». Metadatos `synthetic: true` y `decision: "M-ML-025 (APROBADO)"`. Resumen en la tabla de abajo.
2. **`src/thehouse_ml/synthetic/students.py`** — generador **puro** (sin I/O, red ni reloj): `load_params`/`parse_params` validan los parámetros contra el esquema (catálogos, rangos de NV-11, probabilidades que suman 1, `n_students = per_campus × 10`); `generate_students(params, rent_p50, campuses, schema)` usa un `random.Random("<seed>:<code>")` por campus y un orden fijo de extracciones (las de nulos siempre se hacen, se usen o no), valida cada perfil con `resolve_student` de T02 y devuelve `SyntheticStudent(student_id, seed_verified, profile: StudentProfile)`. `student_record` produce el registro con los nombres del esquema. Sin numpy: solo `random.Random`, `math` y `statistics`.
3. **`src/thehouse_ml/synthetic/students_stage.py`** — etapa `run_students` de `prepare`: lee la renta p50 de `regional_calibration.parquet` (indicador `renta`) por el municipio de cada campus y los campus de `data/universities.csv`; escribe `data/processed/students.parquet` (fuera de git; columnas `student_id`, `university_code`, `seed_verified`, los 9 campos de NV-11 con los nombres del esquema y `synthetic = true`; tipos fijos, nulos con `Int64`/`Float64`/`boolean`) y `reports/students_synthetic.md` (título «DATOS SINTÉTICOS», sin fechas ni rutas absolutas). Sin la calibración regional la etapa se omite con un mensaje; si falta la renta de un municipio o un campus, lanza `StudentsError`.
4. **`src/thehouse_ml/stages.py` y `cli.py`** — `prepare` corre `run_students` después del DENUE y antes de la data card; la CLI devuelve 1 con `StudentParamsError` o `StudentsError`.
5. **Pruebas** — `tests/unit/test_synthetic_students.py` (38) y `tests/integration/test_students_stage.py` (7); `tests/unit/test_stages.py` actualizado con el orden nuevo de `prepare`.
6. **Documentos** — M-ML-025 (APROBADO) en `memoria.md`; T04 marcada en `task.md`; `spec.md` §14 con 4 tasks; fila de ML-03 en `specs/README.md` (4/7).

### Parámetros (`students_params.json` 1.0.0)

| Parámetro | Valor | Base |
| --- | --- | --- |
| `seed` | 20261119 (un generador por campus: `"<seed>:<code>"`) | SUPUESTO |
| `n_students` | 2,000 | SUPUESTO (aprobado en M-ML-021) |
| `student_id.prefix` / `.width` | `S` / 4 → `S0001`…`S2000` | SUPUESTO |
| `universities.per_campus` | 200 (partes iguales; sin matrícula pública verificada) | SUPUESTO |
| `universities.municipio_clave` | campus → municipio (ciudad del seed = municipio de la calibración) | calibrado: `universities_seed.csv` + `regional_calibration.parquet` |
| `budget.max.median_factor` | **1.3** (mediana = renta p50 regional × 1.3) | SUPUESTO; la renta p50 es calibrado: `regional_calibration.parquet` (ENVI 2020, jul-2024, respaldo estatal M-ML-015) |
| `budget.max.log_sigma` | 0.35 (lognormal; p25 ≈ 0.79×, p75 ≈ 1.27× la mediana) | SUPUESTO |
| `budget.max.round_to` | 100 | SUPUESTO |
| `budget.max.clip` | [0, 50,000] | calibrado: NV-11 |
| `budget.max.calibration_tolerance` | ±10 % | SUPUESTO |
| `budget.min.null_probability` | 0.30 | SUPUESTO |
| `budget.min.fraction` | U(0.4, 0.8) × budget.max, hacia abajo a múltiplos de 100 (≤ budget.max) | SUPUESTO |
| `transport.probabilities` | walk 0.20 · bike 0.05 · public 0.60 · car 0.15 | SUPUESTO |
| `transport.null_probability` | 0.10 | SUPUESTO |
| `maxDistanceKm.by_transport` | walk {0.5, 1, 2, 3} · bike {2, 3, 5, 8} · public {3, 5, 8, 10, 15, 20} · car {5, 10, 15, 20, 30}, con pesos | SUPUESTO |
| `maxDistanceKm.null_probability` | 0.15 | SUPUESTO |
| `essentialServices.probabilities` | wifi 0.85 · water 0.90 · electricity 0.90 · gas 0.60 · hot_water 0.65 · laundry 0.35 · kitchen 0.50 · furnished 0.40 · parking 0.15 · security 0.45 · cleaning 0.15 | SUPUESTO |
| `essentialServices.null_probability` | 0.10 | SUPUESTO |
| `livingWith.probabilities` | women 0.30 · men 0.25 · any 0.45 (nunca nulo) | SUPUESTO |
| `placeType.probabilities` | room 0.45 · room_private_bath 0.25 · studio 0.15 · shared 0.15 (nunca nulo) | SUPUESTO |
| `propertyTypes.inclusion_probabilities` | house 0.6 · apartment 0.6 · student_residence 0.4 (si no sale ninguno, uno al azar) | SUPUESTO |
| `propertyTypes.null_probability` | 0.25 | SUPUESTO |
| `pets.true_probability` | 0.12 | SUPUESTO |
| `pets.null_probability` | 0.20 | SUPUESTO |

### Calibración por campus (`reports/students_synthetic.md` §2)

| Campus | Municipio | Nivel | Renta p50 regional | Objetivo (× 1.3) | Mediana `budget.max` | Desviación |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| UTTT | Tula de Allende | estado | $1,872 | $2,434 | $2,500 | +2.7 % |
| UAEH-ESTE | Tepeji del Río de Ocampo | estado | $1,872 | $2,434 | $2,400 | −1.4 % |
| UAEH-ESTL | Tlahuelilpan | estado | $1,872 | $2,434 | $2,400 | −1.4 % |
| UAEH-ESAT | Atotonilco de Tula | estado | $1,872 | $2,434 | $2,400 | −1.4 % |
| ITSOEH | Mixquiahuala de Juárez | estado | $1,872 | $2,434 | $2,400 | −1.4 % |
| TESJI (sin verificar) | Jilotepec | estado | $2,496 | $3,245 | $3,300 | +1.7 % |
| UMB-JILO | Jilotepec | estado | $2,496 | $3,245 | $3,450 | +6.3 % |
| UMB-VDC (sin verificar) | Villa del Carbón | estado | $2,496 | $3,245 | $3,400 | +4.8 % |
| UCAN (sin verificar) | Tula de Allende | estado | $1,872 | $2,434 | $2,600 | +6.8 % |
| UNE-TULA | Tula de Allende | estado | $1,872 | $2,434 | $2,400 | −1.4 % |

Los 10 campus quedan dentro de ±10 %. Nulos observados frente a los parámetros: `budget.min` 28.4 % (30 %), `maxDistanceKm` 14.7 % (15 %), `essentialServices` 9.4 % (10 %), `pets` 20.2 % (20 %), `transport` 9.7 % (10 %), `propertyTypes` 24.0 % (25 %); quick setup 0 %.

### Hallazgo obligatorio: `livingWith`

- **Estudiantes limitados:** con la distribución propuesta (women 0.30 · men 0.25 · any 0.45), **1,102 de 2,000 (55.1 %)** quedan limitados a cuartos `women` (604, 30.2 %) o `men` (498, 24.9 %): por HC02 no ven cuartos `any`. Por campus va de 52.0 % (ITSOEH) a 61.5 % (UMB-JILO). Solo 5.3 % del total además lleva mascota.
- **Estimación para T05** (simulación de exploración en el scratchpad de la sesión, fuera del repo; **no** es evidencia de T06): 300 cuartos por campus con renta lognormal centrada en la renta p50 regional y la dispersión de la ENVI (p25–p75 de `regional_calibration.parquet`), `pets_allowed` 20 % o 30 %, 20 repeticiones, contra los 2,000 estudiantes generados:

  | `livingWith` de los cuartos (women · men · any) | Estudiantes con ≥ 1 cuarto compatible (media) | Peor campus (mínimo) |
  | --- | ---: | ---: |
  | 0.02 · 0.02 · 0.96 | 95.7–96.4 % | **45.5 %** |
  | 0.05 · 0.05 · 0.90 | 99.1–99.5 % | 95.0 % |
  | 0.10 · 0.10 · 0.80 | 99.8–99.9 % | 96.0–98.0 % |
  | 0.20 · 0.20 · 0.60 | 100.0 % | 99.0–99.5 % |
  | 0.30 · 0.25 · 0.45 | 100.0 % | 99.5 % |

  Lo que limita es el **número absoluto** de cuartos `women` y `men` por campus, no el presupuesto (la mediana de `budget.max` es 1.3 veces la renta p50): con 6 cuartos por valor el peor campus cae a 45.5 %.
- **Recomendación para T05:** `livingWith` de los cuartos con **women ≥ 0.20 y men ≥ 0.20** (propuesta: **women 0.25 · men 0.25 · any 0.50**) y una prueba de T05 que exija **al menos 40 cuartos `women` y 40 `men` por campus** (≈ 13 % de 300). Así el umbral de ≥ 90 % de T06 se cumple con margen también por campus, aunque la renta o `pets_allowed` de T05 queden más restrictivos que en la simulación. Es una recomendación; la distribución de cuartos se decide y aprueba en T05.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `104 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 27 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **570 passed**; cobertura total **94.40 %** ≥ 80 %; `synthetic/students_stage.py` **100 %**, `synthetic/students.py` **91 %** (las líneas sin cubrir son ramas de validación de parámetros mal formados), `synthetic/__init__.py`, `stages.py` y `cli.py` 100 % (PASS) |
| `uv run python -m thehouse_ml prepare` (dos veces) | Código 0 las dos veces; `students: 2000 estudiantes sintéticos -> data/processed/students.parquet; reporte reports/students_synthetic.md`. SHA-256 de todos los archivos versionados y no ignorados idéntico entre corridas; `students.parquet` idéntico byte por byte (PASS) |

### Trazabilidad R04 → prueba

| Criterio | Prueba |
| --- | --- |
| Cada parámetro con valor y base documentada; marcado sintético y M-ML-025 APROBADO | `test_synthetic_students.py::test_every_parameter_has_a_value_and_a_documented_basis`, `::test_parameters_are_marked_synthetic_and_approved_as_m_ml_025`, `::test_requested_values_are_in_the_params` |
| Campus → municipio coherente con el seed y la calibración | `::test_campus_municipality_matches_the_seed_city_and_the_regional_calibration` |
| Parámetros inválidos rechazados | `::test_invalid_parameters_are_rejected[*]` (9), `::test_missing_and_unknown_parameters_are_rejected`, `::test_load_params_reports_invalid_json` |
| Misma semilla → mismos datos (registros y Parquet byte por byte); otra semilla → distintos | `::test_same_seed_gives_the_same_students_byte_for_byte`, `::test_another_seed_gives_different_students` |
| 2,000 estudiantes, 200 por campus | `::test_two_thousand_students_two_hundred_per_campus` |
| Todos válidos con catálogos y rangos de NV-11 (`resolve_student` y snapshot `nv11_catalogs.json`) | `::test_every_student_is_valid_with_nv11_catalogs_and_ranges`, `::test_present_property_types_are_never_empty` |
| 3 campus sin verificar marcados | `::test_unverified_campuses_are_marked` |
| Calibración: mediana de `budget.max` por campus a ±10 % de renta p50 × 1.3 | `::test_budget_max_median_per_campus_is_within_the_documented_tolerance` (rentas p50 de `reports/regional_calibration.md`), `::test_calibration_follows_other_anchors[*]`, `::test_zero_rent_gives_zero_budgets`, `::test_missing_rent_or_campus_is_an_error` |
| Quick setup nunca nulo | `::test_quick_setup_is_never_null[*]` (3) |
| Nulos de opcionales parecidos a los parámetros (±3 puntos) | `::test_null_share_of_optional_fields_resembles_the_parameter[*]` (6) |
| Sin campo de género; ids sin datos personales | `::test_no_gender_field_anywhere`, `::test_ids_are_synthetic_consecutive_and_without_personal_data` |
| Parquet con columnas documentadas y `synthetic = true` | `::test_frame_has_the_documented_columns_and_synthetic_flag`, `test_students_stage.py::test_run_students_writes_the_parquet_and_the_report` |
| Reporte «DATOS SINTÉTICOS» con distribuciones y tabla de calibración | `test_students_stage.py::test_run_students_writes_the_parquet_and_the_report` |
| Correr dos veces no cambia archivos | `test_students_stage.py::test_running_twice_does_not_change_any_file` y la doble corrida de `prepare` de arriba |
| Errores y omisión de la etapa | `test_students_stage.py::test_without_regional_calibration_the_stage_is_skipped`, `::test_missing_rent_for_a_campus_municipality_is_an_error`, `::test_missing_campus_in_universities_csv_is_an_error`, `::test_cli_reports_students_errors_with_exit_code_one` |
| Orden de `prepare` | `test_stages.py::test_prepare_generates_the_data_card_after_the_other_outputs` |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §8.1: estudiantes desde el esquema de NV-11, distribuciones documentadas y semilla fija | `students_params.json` con base por parámetro; semilla fija; perfiles validados con `resolve_student` | PASS |
| Constitution §8.2–8.3 y AGENTS: sintéticos etiquetados | Columna `synthetic = true`; reporte titulado «DATOS SINTÉTICOS»; `metadata.synthetic: true` | PASS |
| Constitution §7.4: sin datos personales | Ids `S0001`…`S2000`; ninguna columna personal (prueba) | PASS |
| Constitution §11.2: sin género | Ningún campo de género en parámetros, perfil ni Parquet (prueba); `livingWith` es elección del estudiante | PASS |
| Constitution §9: reproducibilidad | Misma semilla → mismos bytes; `prepare` dos veces sin cambios en archivos versionados | PASS |
| Constitution §5 / AGENTS: sin dependencias nuevas | Solo biblioteca estándar (`random`, `math`, `statistics`) más pandas ya instalado; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: `data/processed/` fuera de git | `students.parquet` ignorado (`git status` no lo lista) | PASS |
| M-ML-014: pesos de julio de 2024 | Presupuesto anclado en la renta p50 deflactada a jul-2024 | PASS |
| M-ML-015: respaldo municipio → estado | Los 10 campus usan el nivel `estado` (Hidalgo y Edomex), visible en el reporte | PASS |
| M-ML-017 / spec §7 regla 4: 3 campus sin verificar marcados | `seed_verified = false` para TESJI, UMB-VDC y UCAN, leído de `data/universities.csv` | PASS |
| M-ML-021: 2,000 estudiantes; prueba de calibración en T04 | 2,000 (200 × 10); prueba de la mediana por campus | PASS |
| M-ML-022: reglas para faltantes | Nulos en los 6 opcionales; quick setup nunca nulo | PASS |
| M-ML-024 nota para T05 | Hallazgo `livingWith` con estimación y recomendación | PASS |
| `feature_schema.json` sin cambios | `git status` no lo lista | PASS |
| AGENTS: rutas permitidas | `git status` solo lista `src/thehouse_ml/`, `tests/`, `reports/`, `memoria.md`, `specs/ml-03-feature-schema/` y la fila de ML-03 en `specs/README.md` | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La prueba de ≥ 90 % es de T06 y no se reporta; la simulación de `livingWith` es solo una estimación | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | Todos los valores marcados SUPUESTO y el factor de calibración 1.3 son propuestas del agente. | ADVERTENCIA | **RESUELTO** — M-ML-025 APROBADO por César Yael López Trujillo — 2026-11-19, sin cambios en los valores |
| 2 | La renta p50 regional es **estatal** para los 10 campus (M-ML-015): los 6 campus de Hidalgo comparten objetivo ($2,434) y los 3 de Edomex también ($3,245); el presupuesto no distingue municipios. | ADVERTENCIA | Documentado; se resuelve solo con una fuente municipal de renta (fuera de ML-03) |
| 3 | La renta de la ENVI es de **vivienda completa**, no de cuarto; el factor 1.3 se aplica sobre esa renta. Si T05 calibra los cuartos con la misma renta p50, la relación presupuesto/renta es coherente entre ambos catálogos, pero los montos absolutos pueden diferir del mercado de cuartos para estudiantes. | ADVERTENCIA | Documentado; riesgo RK02 de la spec (validar con el equipo, ML-07). Para T05: ajuste de la renta por `placeType` (ver «Nota para T05») |
| 4 | La tolerancia de ±10 % supone objetivos de al menos ~$1,500: con un objetivo de $1,170 (renta $900) el redondeo a 100 pesa ±4 % y la mediana se desvió +11.1 %. Los objetivos reales son $2,434 y $3,245 (máxima desviación observada +6.8 %). | INFORMATIVO | Documentado en la prueba `test_calibration_follows_other_anchors` y en la «Nota para T05» |
| 5 | 55.1 % de los estudiantes queda limitado por `livingWith`; la recomendación para T05 es women ≥ 0.20 y men ≥ 0.20 (propuesta 0.25 · 0.25 · 0.50) con ≥ 40 cuartos de cada uno por campus. | ADVERTENCIA | Para T05 (ver «Nota para T05») |
| 6 | Las distribuciones de transporte, servicios, tipo de lugar, mascotas y nulos no tienen fuente pública por campus; son supuestos iguales para todos los campus (solo `maxDistanceKm` depende del transporte). | INFORMATIVO | Riesgo RK02; reemplazar con señales reales en ML-07 |
| 7 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T04 |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Insumos: `regional_calibration.parquet` (derivado de INEGI, términos de libre uso, ya registrado en ML-02) y `data/universities.csv` (ODbL, M-ML-016/M-ML-017); el reporte solo publica agregados sintéticos y la renta p50 que ya publica `reports/regional_calibration.md`.
- **Resultado:** **PASS** — la aceptación de R04 se cumple (generador reproducible que respeta rangos y catálogos, reporte de distribuciones y prueba de calibración por universidad), los cinco controles pasan y M-ML-025 quedó APROBADO por el responsable (hallazgo 1 resuelto). Los hallazgos 2, 3, 4 y 5 son limitaciones documentadas que no bloquean T04; los que afectan a T05 pasan a la nota de abajo.

### Nota para T05

Origen: hallazgos 3, 4 y 5 de T04 y aprobación de M-ML-025 (César Yael López Trujillo, 2026-11-19). Son indicaciones del responsable para la spec de T05; no se ejecutan en T04.

1. **`livingWith` de los cuartos:** distribución propuesta **women 0.25 · men 0.25 · any 0.50**, más una prueba de T05 que exija **al menos 40 cuartos `women` y 40 `men` por campus**. Con estudiantes women 0.30 · men 0.25 · any 0.45 (M-ML-025), el 55.1 % solo ve cuartos de su mismo valor (HC02), y el umbral de ≥ 90 % de T06 depende del número absoluto de esos cuartos por campus (ver «Hallazgo obligatorio: `livingWith`»).
2. **Renta por `placeType`:** la renta del INEGI (ENVI 2020, `regional_calibration.parquet`) es de **vivienda completa**, no de cuarto. La renta de cada cuarto se ajusta con **multiplicadores por `placeType` documentados** en `rooms_params.json`, en el orden `shared` < `room` < `room_private_bath` < `studio`, y se normaliza para que la **mediana general** de la renta de cada campus siga calibrada con la renta p50 regional (la prueba de T05 compara esa mediana con la calibrada). Así la relación con `budget.max` de los estudiantes (mediana = renta p50 × 1.3) se mantiene.
3. **Limitación de la tolerancia:** la tolerancia de ±10 % alrededor de la mediana supone objetivos de al menos **~$1,500**. Con objetivos menores, el redondeo a 100 pesos ya pesa ±4 % (un objetivo de $1,170 dio +11.1 %). Si T05 redondea rentas o calibra subgrupos (por ejemplo por `placeType`) con medianas menores a ~$1,500, debe documentar una tolerancia acorde o redondear con un paso menor.

## T05 · Catálogo sintético de cuartos

| Campo | Valor |
| --- | --- |
| Autorización T05 | «EJECUTA ML-03 1.0.0, task T05 (ML-03-R05), en la rama feat/ml-03-rooms. Rutas permitidas: src/thehouse_ml/ (incluido src/thehouse_ml/synthetic/), tests/, reports/, memoria.md, specs/ml-03-feature-schema/ y specs/README.md (solo la fila de ML-03). Sin dependencias nuevas. No modifiques feature_schema.json (1.1.0) ni students_params.json» — César Yael López Trujillo, 2026-11-19 |
| Rama | `feat/ml-03-rooms` |
| Commit base | `f7960b7` (merge del PR #19, T04) |
| Commit (T05) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-19 |
| Decisión | M-ML-026 (APROBADO — César Yael López Trujillo — 2026-11-19, con los valores propuestos sin cambios) |
| Esquema | `feature_schema.json` **1.1.0, sin cambios**; `students_params.json` **sin cambios** |

Lecturas previas: `constitution.md` (§8 y §11) → `AGENTS.md` → `memoria.md` (M-ML-014, M-ML-015, M-ML-017, M-ML-019 a M-ML-025) → `spec.md` (§6 y §10), `plan.md`, `task.md` y `evidence.md` de ML-03 (las dos «Nota para T05») → `src/thehouse_ml/features/` y `src/thehouse_ml/synthetic/students*`.

### Qué se hizo

1. **`src/thehouse_ml/synthetic/rooms_params.json`** (en git, empaquetado) — 28 parámetros del generador, cada uno `{value, basis}` con `basis` = «calibrado: <fuente>» o «SUPUESTO documentado: …». Metadatos `synthetic: true` y `decision: "M-ML-026 (APROBADO)"`. Resumen en la tabla de abajo.
2. **`src/thehouse_ml/synthetic/rooms.py`** — generador **puro** (sin I/O, red ni reloj): `load_params`/`parse_params` validan los parámetros contra el esquema (catálogos, rango de `price_mxn`, rangos SUPUESTOS de recámaras, baños y superficie, bandas contiguas hasta 30 km, probabilidades que suman 1, orden de multiplicadores `shared` < `room` < `room_private_bath` < `studio`, `n_rooms = per_campus × 10`). `RentShape` convierte el histograma en una distribución continua (uniforme por intervalo; `cdf`, `quantile`, `median`); `mixture_normalizer` resuelve por bisección el factor común de los multiplicadores para que la mediana de la mezcla sea la del histograma; `destination_point` coloca el cuarto sobre la esfera de 6,371 km (mismo radio de `haversine_m`). `generate_rooms(params, inputs, campuses, schema)` usa un `random.Random("rooms:<seed>:<code>")` por campus con un orden fijo de extracciones (primero el `placeType` de los 300 cuartos, luego el cuantil de renta estratificado dentro de cada `placeType`, luego el resto cuarto por cuarto), valida cada cuarto con `validate_room` de T02 y devuelve `SyntheticRoom(room_id, seed_verified, campus_distance_km, room: Room)`. Sin numpy.
3. **`src/thehouse_ml/synthetic/rooms_stage.py`** — etapa `run_rooms` de `prepare`: lee la renta p50 (con `rent_anchors` de T04) y las proporciones del Censo de `regional_calibration.parquet`, el histograma del estado de `rent_histogram.parquet` y los campus de `data/universities.csv`; escribe `data/processed/rooms.parquet` (fuera de git; columnas `room_id`, `university_code`, `seed_verified`, `campus_distance_km`, los 12 campos de `room` del esquema y `synthetic = true`; tipos fijos) y `reports/rooms_catalog.md` (título «DATOS SINTÉTICOS», sin fechas ni rutas absolutas). Sin insumos la etapa se omite con un mensaje; si falta la renta, un indicador del Censo, el histograma de una entidad o un campus, lanza `RoomsError`.
4. **`src/thehouse_ml/stages.py`, `cli.py` y `synthetic/__init__.py`** — `prepare` corre `run_rooms` después de `run_students` y antes de la data card; la CLI devuelve 1 con `RoomParamsError` o `RoomsError`; el paquete exporta el generador.
5. **Pruebas** — `tests/unit/test_synthetic_rooms.py` (62) y `tests/integration/test_rooms_stage.py` (8); `tests/unit/test_stages.py` con el orden nuevo de `prepare`. Las pruebas unitarias usan solo insumos versionados: renta p50 y proporciones del Censo de `reports/regional_calibration.md` e histogramas agrupados (≥ 30 viviendas por celda) de `reports/data_card.md` §7.
6. **Documentos** — M-ML-026 (APROBADO) en `memoria.md`; T05 marcada en `task.md`; `spec.md` §14 con 5 tasks; fila de ML-03 en `specs/README.md` (5/7).

### Parámetros (`rooms_params.json` 1.0.0)

| Parámetro | Valor | Base |
| --- | --- | --- |
| `seed` | 20261905 (generador por campus `"rooms:<seed>:<code>"`; estudiantes usan 20261119 y `"<seed>:<code>"`) | SUPUESTO |
| `n_rooms` / `universities.per_campus` | 3,000 / 300 por campus | SUPUESTO (aprobado en M-ML-021) |
| `room_id.prefix` / `.width` | `R` / 5 → `R00001`…`R03000` | SUPUESTO |
| `universities.municipio_clave` | igual a `students_params.json` (prueba) | calibrado: `universities_seed.csv` + `regional_calibration.parquet` |
| `location.max_distance_km` | 30 | calibrado: M-ML-021 y `training_range` de `distance_km` |
| `location.distance_bands_km` | 0–1 km 25 % · 1–2 km 20 % · 2–5 km 25 % · 5–10 km 15 % · 10–20 km 10 % · 20–30 km 5 % (uniforme dentro de la banda) | SUPUESTO |
| `location.bearing_degrees` / `.coordinate_decimals` | uniforme [0, 360) / 6 decimales | SUPUESTO |
| `rent.median_anchor` | renta p50 de `regional_calibration.parquet` (Hidalgo $1,872; Edomex $2,496) | calibrado: ENVI 2020, jul-2024, M-ML-015 |
| `rent.shape` | histograma ENVI del estado del campus (`rent_histogram.parquet`), escalado a la p50 | calibrado: ML-02 T08 |
| `rent.open_bin_upper` | 15,000 (tope del intervalo 10,000+) | SUPUESTO |
| `rent.place_type_multipliers` | shared 0.60 · room 0.90 · room_private_bath 1.15 · studio 1.50 (efectivos 0.622 · 0.933 · 1.192 · 1.555) | SUPUESTO (nota para T05) |
| `rent.round_to` / `rent.clip` | 50 / [0, 50,000] | SUPUESTO / calibrado: `price_mxn` del esquema |
| `rent.calibration_tolerance` | ±10 % | SUPUESTO (igual a M-ML-025) |
| `services.census_indicators` | water ← `agua_entubada` · electricity ← `electricidad` · gas ← `gas` · wifi ← `internet` | calibrado: Censo 2020 por municipio (M-ML-015) |
| `services.census_tolerance_se` | ±3 errores estándar binomiales (n = 300), mínimo ±2 puntos | SUPUESTO |
| `services.priors_by_place_type` | hot_water, laundry, kitchen, furnished, parking, security, cleaning por `placeType` (ver M-ML-026) | SUPUESTO |
| `livingWith.probabilities` / `.min_per_campus` | women 0.25 · men 0.25 · any 0.50 / 40 | SUPUESTO (nota para T05, M-ML-025) |
| `placeType.probabilities` | room 0.40 · room_private_bath 0.25 · studio 0.15 · shared 0.20 | SUPUESTO |
| `propertyType.by_place_type` | studio sobre todo `apartment` (0.65); cuartos sobre todo `house` (0.50–0.60) | SUPUESTO |
| `pets_allowed.by_property_type` | house 0.35 · apartment 0.25 · student_residence 0.10 | SUPUESTO |
| `bedrooms` / `bathrooms` / `area_m2` por `placeType` | recámaras 2–5 (estudio 1); baños 1–3, con baño propio 2–4, estudio 1, recortados a recámaras + 1; superficie del espacio ofrecido 9–45 m² | SUPUESTO (dentro de los rangos SUPUESTOS del esquema) |

### Renta por campus contra la calibración (`reports/rooms_catalog.md` §2)

| Campus | Municipio | Nivel | Renta p50 regional | p25 | p50 | p75 | Desviación p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| UTTT | Tula de Allende | estado | $1,872 | $1,200 | $1,925 | $3,100 | +2.8 % |
| UAEH-ESTE | Tepeji del Río de Ocampo | estado | $1,872 | $1,200 | $1,850 | $3,050 | −1.2 % |
| UAEH-ESTL | Tlahuelilpan | estado | $1,872 | $1,150 | $1,800 | $2,925 | −3.9 % |
| UAEH-ESAT | Atotonilco de Tula | estado | $1,872 | $1,150 | $1,850 | $2,962 | −1.2 % |
| ITSOEH | Mixquiahuala de Juárez | estado | $1,872 | $1,200 | $1,900 | $3,012 | +1.5 % |
| TESJI (sin verificar) | Jilotepec | estado | $2,496 | $1,650 | $2,500 | $4,012 | +0.2 % |
| UMB-JILO | Jilotepec | estado | $2,496 | $1,650 | $2,500 | $4,062 | +0.2 % |
| UMB-VDC (sin verificar) | Villa del Carbón | estado | $2,496 | $1,550 | $2,400 | $3,900 | −3.9 % |
| UCAN (sin verificar) | Tula de Allende | estado | $1,872 | $1,150 | $1,850 | $3,012 | −1.2 % |
| UNE-TULA | Tula de Allende | estado | $1,872 | $1,150 | $1,850 | $2,962 | −1.2 % |

Los 10 campus dentro de ±10 %. Referencia de la ENVI estatal (`regional_calibration.parquet`): p25–p75 $1,248–$2,995 en Hidalgo y $1,623–$3,495 en el Estado de México; la dispersión de los cuartos es algo mayor por los multiplicadores de `placeType`. Mediana por `placeType` (todo el catálogo): shared $1,250 < room $1,900 < room_private_bath $2,450 < studio $3,150; el orden se cumple en los 10 campus (§3 del reporte). Distancia: máximo 29.94 km, mediana 2.59 km, 44.5 % a ≤ 2 km. Censo: 40 de 40 proporciones (4 servicios × 10 campus) dentro de la tolerancia; la mayor diferencia es `wifi` en UTTT (42.0 % contra 46.9 %, tolerancia ±8.6). `livingWith`: de 62 a 85 cuartos `women` y de 62 a 84 `men` por campus (mínimo exigido 40).

### Hallazgo obligatorio: umbral de estudiantes con al menos un cuarto compatible

Estimación de exploración (script en el scratchpad de la sesión, fuera del repo; **no** es la prueba formal de T06): los 2,000 estudiantes de `students.parquet` (T04) contra los 3,000 cuartos de `rooms.parquet`, con `filter_rooms` (HC01 presupuesto, HC02 `livingWith`, HC03 mascotas; M-ML-024) y la distancia haversine del campus del estudiante al cuarto contra su `maxDistanceKm` resuelto (nulo → 30, M-ML-022).

| Campus | Estudiantes | HC (cuartos de su campus) | HC + `maxDistanceKm` (cuartos de su campus) | HC + `maxDistanceKm` (todo el catálogo) |
| --- | ---: | ---: | ---: | ---: |
| UTTT | 200 | 100.0 % | 100.0 % | 100.0 % |
| UAEH-ESTE | 200 | 100.0 % | 100.0 % | 100.0 % |
| UAEH-ESTL | 200 | 100.0 % | 100.0 % | 100.0 % |
| UAEH-ESAT | 200 | 100.0 % | 100.0 % | 100.0 % |
| ITSOEH | 200 | 100.0 % | 100.0 % | 100.0 % |
| TESJI (sin verificar) | 200 | 100.0 % | 100.0 % | 100.0 % |
| UMB-JILO | 200 | 100.0 % | 100.0 % | 100.0 % |
| UMB-VDC (sin verificar) | 200 | 100.0 % | 100.0 % | 100.0 % |
| UCAN (sin verificar) | 200 | 100.0 % | 99.5 % | 99.5 % |
| UNE-TULA | 200 | 100.0 % | 100.0 % | 100.0 % |
| **Total** | 2,000 | **100.0 %** | **99.95 %** | **99.95 %** |

- Ningún campus queda por debajo del 90 %: **no se propone ajuste**. El único estudiante sin cuarto con la distancia (UCAN) pide `men`, lleva mascota y `maxDistanceKm` = 0.5 km.
- **Para T06 (candidatos):** el número de cuartos compatibles por estudiante (HC, cuartos de su campus) es mínimo 1, p10 28 y mediana 62; para `women` y `men` el máximo es 84 y la mediana 46–49; con `maxDistanceKm` la mediana baja a 50. Solo **21.2 %** de los estudiantes tiene ≥ 200 cuartos compatibles, así que los **200 candidatos por estudiante** de M-ML-021 no pueden ser todos compatibles con 300 cuartos por campus: T06 deberá definir si los candidatos incluyen cuartos excluidos (como negativos) o de otros campus.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `109 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 29 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **640 passed**; cobertura total **95.19 %** ≥ 80 %; `synthetic/rooms.py` **99.19 %** (sin cubrir: la salida de respaldo de `quantile` y el error de distancia > 30 km, defensivos), `synthetic/rooms_stage.py` **99.18 %** (sin cubrir: un `placeType` sin cuartos), `synthetic/__init__.py`, `stages.py` y `cli.py` 100 % (PASS) |
| `uv run python -m thehouse_ml prepare` (dos veces) | Código 0 las dos veces; `rooms: 3000 cuartos sintéticos -> data/processed/rooms.parquet; reporte reports/rooms_catalog.md`. SHA-256 de los 138 archivos versionados y no ignorados más `rooms.parquet` y `students.parquet` idéntico entre corridas (`rooms.parquet` `e2ab7e8e…8dca`, `reports/rooms_catalog.md` `ce83040d…6631` tras la aprobación; `rooms.parquet` no cambió al aprobar) (PASS) |

### Trazabilidad R05 → prueba

| Criterio | Prueba |
| --- | --- |
| Cada parámetro con valor y base documentada; sintético; M-ML-026 APROBADO | `test_synthetic_rooms.py::test_every_parameter_has_a_value_and_a_documented_basis`, `::test_parameters_are_marked_synthetic_and_approved_as_m_ml_026`, `::test_requested_values_are_in_the_params`, `::test_distance_bands_put_most_rooms_near_with_a_tail_to_thirty_km`, `::test_studio_is_more_likely_to_have_a_kitchen_than_a_room` |
| Campus → municipio igual que en estudiantes | `::test_campus_municipality_matches_the_students_params` |
| Parámetros inválidos rechazados | `::test_invalid_parameters_are_rejected[*]` (34), `::test_missing_unknown_and_unexplained_parameters_are_rejected`, `::test_load_params_reports_invalid_json`, `::test_load_params_reads_an_explicit_path` |
| Renta: forma, normalización y ubicación | `::test_rent_shape_quantile_inverts_the_cdf`, `::test_rent_shape_rejects_invalid_histograms`, `::test_mixture_normalizer_keeps_the_overall_median`, `::test_destination_point_lands_at_the_requested_haversine_distance`, `::test_rent_is_rounded_and_clipped` |
| Misma semilla → mismos cuartos (registros y Parquet byte por byte); otra semilla → distintos | `::test_same_seed_gives_the_same_rooms_byte_for_byte`, `::test_another_seed_gives_different_rooms` |
| ~3,000 cuartos, 300 por campus, todos a ≤ 30 km | `::test_three_thousand_rooms_three_hundred_per_campus`, `::test_every_room_is_within_thirty_km_of_its_campus` |
| Validez contra el esquema y NV-11 (`validate_room`, snapshot `nv11_catalogs.json`, rangos SUPUESTOS) | `::test_every_room_is_valid_with_the_schema_and_nv11` |
| 3 campus sin verificar marcados | `::test_unverified_campuses_are_marked` |
| Mediana de la renta por campus a ±10 % de la p50 regional | `::test_rent_median_per_campus_is_within_ten_percent_of_the_regional_p50` |
| Orden de medianas por `placeType` (en cada campus y en total) | `::test_rent_medians_follow_the_place_type_order_in_every_campus` |
| water/electricity/gas/wifi cerca del Censo (±3 ee, mínimo 2 pts) | `::test_census_services_follow_the_municipal_census_within_the_documented_tolerance` |
| ≥ 40 cuartos `women` y ≥ 40 `men` por campus | `::test_each_campus_has_at_least_forty_women_and_forty_men_rooms`, `::test_categorical_shares_resemble_the_parameters` |
| Sin campos personales ni de género; ids sintéticos | `::test_no_personal_or_gender_field_anywhere`, `::test_frame_has_the_documented_columns_and_synthetic_flag` |
| Insumos faltantes | `::test_missing_inputs_are_errors`, `test_rooms_stage.py::test_missing_census_indicator_is_an_error`, `::test_missing_rent_is_an_error`, `::test_missing_state_histogram_is_an_error`, `::test_missing_campus_in_universities_csv_is_an_error`, `::test_cli_reports_rooms_errors_with_exit_code_one`, `::test_without_inputs_the_stage_is_skipped` |
| Parquet y reporte «DATOS SINTÉTICOS» con renta, distancia, servicios contra Censo, `placeType`, `propertyType`, mascotas y `livingWith` | `test_rooms_stage.py::test_run_rooms_writes_the_parquet_and_the_report` |
| Correr dos veces no cambia archivos | `test_rooms_stage.py::test_running_twice_does_not_change_any_file` y la doble corrida de `prepare` de arriba |
| Orden de `prepare` | `test_stages.py::test_prepare_generates_the_data_card_after_the_other_outputs` |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §8: sintéticos honestos, distribuciones documentadas y semilla fija | `rooms_params.json` con base por parámetro; semilla fija; `synthetic = true`; reporte «DATOS SINTÉTICOS» | PASS |
| Constitution §7.4: sin datos personales | Ids `R00001`…`R03000`; sin títulos, descripciones, nombres ni anfitriones (prueba de columnas) | PASS |
| Constitution §11.2: sin género | Ningún campo de género; `livingWith` es la política del cuarto (prueba) | PASS |
| Constitution §9: reproducibilidad | Misma semilla → mismos bytes; `prepare` dos veces sin cambios | PASS |
| Constitution §5 / AGENTS: sin dependencias nuevas | Solo biblioteca estándar (`random`, `math`, `statistics`, `itertools`) más pandas ya instalado; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: `data/processed/` fuera de git | `rooms.parquet` ignorado (`git status` no lo lista) | PASS |
| M-ML-014: pesos de julio de 2024 | Renta anclada en la p50 y el histograma deflactados a jul-2024 | PASS |
| M-ML-015: Censo por municipio y renta con respaldo estatal | 4 servicios con nivel `municipio`; renta con nivel `estado`, visibles en el reporte | PASS |
| M-ML-017 / spec §7 regla 4: 3 campus sin verificar | `seed_verified = false` para TESJI, UMB-VDC y UCAN; marcados en el reporte | PASS |
| M-ML-019: sin Inside Airbnb | Ningún dato de anuncios; catálogo sintético calibrado con INEGI | PASS |
| M-ML-021: ~3,000 cuartos, ≤ 30 km, renta y servicios calibrados | 3,000 (300 × 10), máximo 29.94 km; pruebas de mediana y Censo | PASS |
| M-ML-024 y M-ML-025 (notas para T05) | `livingWith` 0.25 · 0.25 · 0.50 con ≥ 40 por campus; multiplicadores por `placeType` normalizados; redondeo a 50 (objetivos ≥ $1,872) | PASS |
| `feature_schema.json` 1.1.0 y `students_params.json` sin cambios | `git status` no los lista | PASS |
| AGENTS: rutas permitidas | `git status` solo lista `src/thehouse_ml/`, `tests/`, `reports/`, `memoria.md`, `specs/ml-03-feature-schema/` y la fila de ML-03 en `specs/README.md` | PASS |
| AGENTS: un control no ejecutado no se reporta como PASS | La prueba de ≥ 90 % es de T06 y no se reporta como PASS; la tabla de arriba es una estimación | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | Todos los valores marcados SUPUESTO (ubicación, multiplicadores, priors de servicios, `placeType`, `propertyType`, mascotas, `livingWith`, recámaras, baños y superficie) son propuestas del agente. | ADVERTENCIA | **RESUELTO** — M-ML-026 APROBADO por César Yael López Trujillo — 2026-11-19, sin cambios en los valores |
| 2 | Para T06: con 300 cuartos por campus solo 21.2 % de los estudiantes tiene ≥ 200 cuartos compatibles (los `women`/`men`, como máximo 84); los 200 candidatos por estudiante de M-ML-021 no pueden ser todos compatibles. | ADVERTENCIA | Para T06 (ver «Nota para T06»: candidatos = cuartos del campus que pasan las restricciones duras, hasta 200) |
| 3 | `area_m2` se interpreta como superficie del espacio ofrecido (cuarto o estudio) y recámaras/baños como del inmueble; el backend no define la semántica de `Property.AreaM2`. | ADVERTENCIA | Documentado en `rooms_params.json`; acción pendiente del responsable (ver abajo) |
| 4 | La renta regional es estatal y de vivienda completa; las proporciones del Censo son de todas las viviendas del municipio, no solo de las rentadas. | INFORMATIVO | Documentado; riesgo RK02 (ML-07) |
| 5 | El generador usa los 21 intervalos del histograma estatal (algunos con menos de 30 viviendas en muestra, que la data card agrupa al publicar). `rooms.parquet` queda fuera de git y el reporte solo publica cuartiles de los cuartos sintéticos, no los intervalos. | INFORMATIVO | Sin acción; las pruebas en CI usan los histogramas agrupados ya publicados |
| 6 | Las distancias de TESJI, UMB-VDC y UCAN son orientativas (coordenadas sin verificar, M-ML-017). | INFORMATIVO | Marcadas en el Parquet y el reporte |
| 7 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T05 |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Insumos: `regional_calibration.parquet` y `rent_histogram.parquet` (derivados de INEGI, términos de libre uso, ya registrados en ML-02) y `data/universities.csv` (ODbL, M-ML-016/M-ML-017). El reporte solo publica agregados sintéticos y las proporciones y la renta p50 que ya publica `reports/regional_calibration.md`.
- **Resultado:** **PASS** — la aceptación de R05 se cumple (generador reproducible, 3,000 cuartos a ≤ 30 km, rangos y catálogos de NV-11, prueba de la mediana de renta contra la calibrada y reporte sintético de distribuciones), los cinco controles pasan y M-ML-026 quedó APROBADO por el responsable (hallazgo 1 resuelto). Los hallazgos 2 y 3 no bloquean T05: pasan a la «Nota para T06» y a la acción pendiente del responsable.

### Acción pendiente del responsable

Confirmar con el backend el significado de `Property.AreaM2` (superficie del espacio ofrecido —cuarto o estudio— o del inmueble completo), junto con NV-12. T05 usa la superficie del espacio ofrecido (`area_m2.by_place_type` en `rooms_params.json`); si el backend define la del inmueble, se cambia ese parámetro, se sube la versión de `rooms_params.json` y se vuelve a correr `prepare`. Origen: hallazgo 3 de T05.

### Nota para T06

Origen: hallazgo 2 de T05 y aprobación de M-ML-026 (César Yael López Trujillo, 2026-11-19). Es una indicación del responsable para T06; no se ejecuta en T05.

1. **Candidatos por estudiante:** como solo **21.2 %** de los estudiantes tiene 200 o más cuartos compatibles (los `women` y `men`, como máximo 84), se propone que los candidatos de cada estudiante sean **los cuartos de su campus que pasan las restricciones duras** (`filter_rooms`, HC01–HC03), **hasta un máximo de 200**; si hay más, se toma una muestra con semilla fija. No se incluyen cuartos que no pasan las restricciones duras. Se registrará en T06 como aclaración de M-ML-021 («hasta 200» candidatos por estudiante).
2. **Estimación del umbral:** la tabla de «Hallazgo obligatorio» de T05 salió de un script de exploración en el scratchpad de la sesión (fuera del repo, no versionado). No se copia al repositorio: T06 implementa la prueba formal de ≥ 90 % con `filter_rooms` sobre `students.parquet` y `rooms.parquet`, y esa prueba sustituye a la estimación.

## T06 · Conjunto de pares

| Campo | Valor |
| --- | --- |
| Autorización T06 | «EJECUTA ML-03 1.0.0, task T06 (ML-03-R06), en la rama feat/ml-03-pairs. Rutas permitidas: src/thehouse_ml/ (incluido src/thehouse_ml/synthetic/ y features/), tests/, reports/, memoria.md, specs/ml-03-feature-schema/ y specs/README.md (solo la fila de ML-03). Sin dependencias nuevas. No modifiques feature_schema.json, students_params.json ni rooms_params.json» — César Yael López Trujillo, 2026-11-19. El mensaje llegó truncado en el punto 6; el responsable lo completó el mismo día (punto 6: pruebas de fuga, proporciones 70/15/15 con tolerancia documentada, restricciones, máximo de 200, campus, features y reproducibilidad; punto 7: marcar T06 completa solo si todo pasa y actualizar `evidence.md`, `task.md`, `spec.md` §14 y `specs/README.md` con ML-03 6/7) y pidió no hacer commit |
| Rama | `feat/ml-03-pairs` |
| Commit base | `e2a0679` (merge del PR #20, T05) |
| Commit (T06) | Sin commit, por instrucción explícita del responsable |
| Fecha de ejecución | 2026-11-19 |
| Decisión | M-ML-027 (APROBADO — César Yael López Trujillo — 2026-11-19, con los valores propuestos sin cambios) |
| Esquema | `feature_schema.json` **1.1.0, sin cambios**; `students_params.json` y `rooms_params.json` **sin cambios** |

Lecturas previas: `constitution.md` (§8) → `AGENTS.md` → `memoria.md` (M-ML-021 a M-ML-026) → `spec.md`, `plan.md`, `task.md` y `evidence.md` de ML-03 (sobre todo la «Nota para T06») → `src/thehouse_ml/features/` y `src/thehouse_ml/synthetic/`.

### Qué se hizo

1. **`src/thehouse_ml/synthetic/pairs_params.json`** (en git, empaquetado) — 5 parámetros `{value, basis}`: `seed` 20261120, `candidates.max_per_student` 200, `split.fractions` 0.70 · 0.15 · 0.15, `split.stratify_by` `university_code` y `coverage.min_share` 0.90. Metadatos `synthetic: true` y `decision: "M-ML-027 (APROBADO)"` (antes `PROPUESTO`; ningún valor cambió al aprobar).
2. **`src/thehouse_ml/synthetic/pairs.py`** — módulo **puro** (sin I/O, red ni reloj): `load_params`/`parse_params` validan los parámetros; `split_sizes` (mayor residuo, al menos 1 por partición) y `assign_splits` (partición por estudiante, estratificada por campus, `random.Random("pairs:split:<seed>:<code>")`); `choose_candidates` (todos los permitidos hasta 200; si hay más, muestra uniforme sin reemplazo con `random.Random("pairs:candidates:<seed>:<student_id>")`, en orden de `room_id`); `build_pairs` usa `filter_rooms` (T03) con los cuartos del campus del estudiante y `compute_pair_features` (T02) para cada candidato; `coverage` calcula el umbral por campus y total, con o sin las `seed_sin_verificar`.
3. **`src/thehouse_ml/synthetic/pairs_stage.py`** — etapa `run_pairs` de `prepare`: lee `students.parquet` y `rooms.parquet` (conversión Parquet → `StudentProfile`/`Room` con `student_from_record`/`room_from_record`, nulos de NV-11 conservados), escribe `data/processed/pairs.parquet` (fuera de git; `student_id`, `room_id`, `university_code`, `seed_verified`, `split`, las 7 features en el orden del esquema —`integer` → int64, `number` → float64— y `synthetic = true`) y `reports/pairs_summary.md` («DATOS SINTÉTICOS», sin fechas ni rutas absolutas). Después comprueba el umbral y lanza `PairsError` si no se cumple (la CLI devuelve 1). Sin insumos, la etapa se omite con un mensaje.
4. **`stages.py`, `cli.py` y `synthetic/__init__.py`** — `prepare` corre `run_pairs` después de `run_rooms` y antes de la data card; la CLI devuelve 1 con `PairParamsError`, `PairSetError` o `PairsError`; el paquete exporta `build_pairs` y los parámetros.
5. **Pruebas** — `tests/unit/test_synthetic_pairs.py` (37, dos agregadas al completar el punto 6: proporciones 70/15/15 con tolerancia y comprobación exhaustiva de todos los pares) y `tests/integration/test_pairs_stage.py` (5); `tests/unit/test_stages.py` con el orden nuevo de `prepare`; `tests/conftest.py` con `versioned_synthetic_data()`, que genera los 2,000 estudiantes y los 3,000 cuartos solo con insumos versionados (renta p50 de `reports/regional_calibration.md` e histogramas agrupados de `reports/data_card.md` §7).
6. **Documentos** — M-ML-027 (APROBADO) en `memoria.md`; T06 marcada en `task.md`; `spec.md` §14 con 6 tasks; fila de ML-03 en `specs/README.md` (6/7); «Nota para ML-04/ML-05» al final de esta sección.

### Regla de candidatos y partición

- **Candidatos:** cuartos **del campus del estudiante** que pasan HC01–HC03, hasta 200. No entran cuartos excluidos ni de otros campus. Los cuartos más lejos que `maxDistanceKm` **sí** entran (M-ML-023): 51,794 pares (26.3 %).
- **Partición por estudiante**, no por par: cada `student_id` está en una sola partición. La partición se calcula antes que los candidatos y no depende del catálogo, así que un estudiante sin candidatos conserva su partición y no aporta pares.
- **Los cuartos se repiten entre particiones:** son el catálogo, como en una app real (todos los estudiantes ven los mismos cuartos). Con los datos de ML-02, **2,751 de 2,893** cuartos con algún par aparecen en más de una partición. Por eso la evaluación de ML-04/ML-05 mide la generalización a **estudiantes nuevos** sobre el mismo catálogo, no a cuartos nuevos.
- **Tolerancia documentada de las proporciones 70/15/15:** estudiantes — en cada campus, cada partición difiere de fracción por n en **menos de 1 estudiante** (mayor residuo; con 200 por campus es exacto, 140 · 30 · 30) y **±0.5 puntos** en el total; pares — **±3 puntos** por partición, porque el número de candidatos varía por estudiante. Con los datos de ML-02: estudiantes 70.0 / 15.0 / 15.0 % y pares 69.2 / 15.2 / 15.6 % (train / validation / test).

### Resultados con los datos de ML-02 (`reports/pairs_summary.md`)

| Campus | Verificado | Estudiantes train · val · test | Pares train · val · test | Candidatos mín. · mediana · máx. | Con muestra (> 200) | Con ≥ 1 candidato |
| --- | --- | --- | --- | --- | ---: | ---: |
| UTTT | sí | 140 · 30 · 30 | 12,107 · 3,261 · 2,784 | 5 · 58 · 200 | 35 | 100.0 % |
| UAEH-ESTE | sí | 140 · 30 · 30 | 13,470 · 2,857 · 3,377 | 6 · 58 · 200 | 43 | 100.0 % |
| UAEH-ESTL | sí | 140 · 30 · 30 | 14,810 · 3,399 · 2,718 | 8 · 70 · 200 | 43 | 100.0 % |
| UAEH-ESAT | sí | 140 · 30 · 30 | 13,741 · 2,858 · 2,791 | 7 · 61 · 200 | 40 | 100.0 % |
| ITSOEH | sí | 140 · 30 · 30 | 13,722 · 3,082 · 3,120 | 4 · 65 · 200 | 41 | 100.0 % |
| TESJI | **no** | 140 · 30 · 30 | 13,982 · 2,715 · 3,214 | 1 · 66 · 200 | 44 | 100.0 % |
| UMB-JILO | sí | 140 · 30 · 30 | 12,505 · 2,718 · 3,472 | 10 · 60 · 200 | 39 | 100.0 % |
| UMB-VDC | **no** | 140 · 30 · 30 | 14,061 · 2,985 · 3,297 | 8 · 61.5 · 200 | 50 | 100.0 % |
| UCAN | **no** | 140 · 30 · 30 | 14,255 · 2,988 · 3,690 | 8 · 71.5 · 200 | 44 | 100.0 % |
| UNE-TULA | sí | 140 · 30 · 30 | 13,340 · 2,994 · 2,292 | 5 · 55 · 200 | 41 | 100.0 % |
| **Total** | — | 1,400 · 300 · 300 | 135,993 · 29,857 · 30,755 | 1 · 62 · 200 | 420 | **100.0 %** |
| **Total sin `seed_sin_verificar`** | — | 980 · 210 · 210 | 93,695 · 21,169 · 20,554 | 4 · 61 · 200 | 282 | **100.0 %** |

Candidatos por estudiante (total): mínimo 1, p25 44, mediana 62, p75 182, máximo 200. Exclusiones sobre 600,000 pares evaluados (estudiantes por cuartos de su campus): 387,954 excluidos (64.7 %); `HC01_budget` 213,960 (35.7 %), `HC02_living_with` 248,792 (41.5 %) y `HC03_pets` 40,745 (6.8 %); sin las `seed_sin_verificar`, 274,200 de 420,000 (65.3 %). Estudiantes sin candidatos: 0.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run ruff format --check .` | `114 files already formatted` (PASS) |
| `uv run ruff check .` | `All checks passed!` (PASS) |
| `uv run mypy src` | `Success: no issues found in 31 source files` (PASS) |
| `uv run pytest --cov=thehouse_ml` | **682 passed** en 414 s; cobertura total **95.60 %** ≥ 80 %; `synthetic/pairs.py` **99 %** (sin cubrir: línea 328, estudiante con campus fuera del catálogo, defensiva), `synthetic/pairs_stage.py` **99 %** (sin cubrir: línea 73, NaN de tipo `float` en la conversión desde Parquet), `synthetic/__init__.py`, `stages.py` y `cli.py` 100 % (PASS) |
| `uv run python -m thehouse_ml prepare` (dos veces, tras la aprobación) | Código 0 las dos veces; `pairs: 196605 pares sintéticos de 2000 estudiantes -> data/processed/pairs.parquet; reporte reports/pairs_summary.md`. SHA-256 de los 147 archivos (versionados y no ignorados más `students.parquet`, `rooms.parquet` y `pairs.parquet`) idéntico entre corridas: `pairs.parquet` `8e6d4690…d11d` (el mismo que antes de aprobar: los datos no cambiaron), `reports/pairs_summary.md` `6eeb2f7a…4763` (solo cambia el texto de la decisión, ahora APROBADO) (PASS) |

### Trazabilidad R06 → prueba

| Criterio | Prueba |
| --- | --- |
| Parámetros con valor y base; sintéticos; M-ML-027; 200, 70/15/15, campus, 90 %; semilla propia | `test_synthetic_pairs.py::test_every_parameter_has_a_value_and_a_documented_basis`, `::test_parameters_are_marked_synthetic_and_approved_as_m_ml_027`, `::test_requested_values_are_in_the_params` |
| Parámetros inválidos rechazados | `::test_invalid_parameters_are_rejected[*]` (12), `::test_missing_unknown_and_unexplained_parameters_are_rejected`, `::test_load_params_reads_an_explicit_path_and_reports_invalid_json` |
| **Umbral formal ≥ 90 %** (en total, por campus y sin las `seed_sin_verificar`), con recuento independiente de `filter_rooms` | `::test_at_least_ninety_percent_of_students_have_a_room_that_passes_the_hard_constraints`, `::test_coverage_below_the_threshold_does_not_meet`; con los datos reales, la etapa (`test_pairs_stage.py::test_threshold_not_met_is_an_error_and_the_report_says_so`, `::test_cli_reports_pairs_errors_with_exit_code_one`) y la doble corrida de `prepare` |
| **Sin fuga:** ningún estudiante en dos particiones; particiones disjuntas; cada par con la partición de su estudiante | `::test_no_student_appears_in_two_splits`, `test_pairs_stage.py::test_run_pairs_writes_the_parquet_and_the_report` |
| Cada campus en las tres particiones (140 · 30 · 30); cuotas por mayor residuo | `::test_every_campus_appears_in_the_three_splits_with_140_30_30`, `::test_split_sizes_use_the_largest_remainder_and_fill_every_split` |
| Proporciones 70/15/15 dentro de la tolerancia documentada (estudiantes por campus < 1 estudiante y ±0.5 puntos en total; pares ±3 puntos) | `::test_split_proportions_are_70_15_15_within_the_documented_tolerance` |
| **Todos** los pares (no una muestra): ninguno viola una restricción dura, todos los candidatos son del campus del estudiante y ninguno tiene más de 200 | `::test_no_pair_violates_a_hard_constraint_every_candidate_is_from_its_campus_and_at_most_200` (además de la muestra con `check_hard_constraints` de `::test_every_pair_passes_the_hard_constraints_and_is_from_the_student_campus`) |
| Cuartos repetidos entre particiones (documentado) | `::test_rooms_may_repeat_across_splits_because_they_are_the_catalog` |
| Partición independiente del orden y de los cuartos; cambia con la semilla | `::test_split_does_not_depend_on_rooms_and_changes_with_the_seed` |
| Candidatos = cuartos del campus que pasan HC01–HC03, hasta 200 (muestra fija si hay más) | `::test_candidates_are_the_compatible_rooms_of_the_campus_up_to_the_maximum`, `::test_every_pair_passes_the_hard_constraints_and_is_from_the_student_campus`, `::test_choose_candidates_is_a_fixed_uniform_sample_in_room_order` |
| Cuartos más lejos que `maxDistanceKm` sí son candidatos | `::test_rooms_farther_than_max_distance_are_candidates` |
| Estudiantes con 0 candidatos: sin pares, contados, con exclusiones por restricción | `::test_student_without_candidates_generates_no_pairs_but_is_counted` |
| 7 features con `compute_pair_features`, en el orden del esquema | `::test_features_come_from_compute_pair_features_in_the_schema_order` |
| Columnas pedidas, `synthetic = true`, **sin etiquetas** | `::test_frame_has_the_ids_split_seven_features_and_no_labels` |
| Reproducibilidad (Parquet byte por byte); otra semilla → otros pares | `::test_same_inputs_give_the_same_pairs_byte_for_byte`, `test_pairs_stage.py::test_running_twice_does_not_change_any_file` y la doble corrida de `prepare` |
| Parquet → objetos sin pérdida (nulos de NV-11) | `::test_parquet_round_trip_keeps_students_and_rooms` |
| Reporte «DATOS SINTÉTICOS» con particiones, distribución de candidatos, exclusiones HC01–HC03 y cobertura, con y sin las `seed_sin_verificar` | `::test_report_is_synthetic_and_shows_numbers_with_and_without_unverified_campuses`, `test_pairs_stage.py::test_run_pairs_writes_the_parquet_and_the_report` |
| Entradas inválidas y etapa sin insumos | `::test_invalid_inputs_are_errors`, `test_pairs_stage.py::test_without_inputs_the_stage_is_skipped` |
| Orden de `prepare` | `test_stages.py::test_prepare_generates_the_data_card_after_the_other_outputs` |

### Tabla de coherencia

| Regla | Evidencia | Resultado |
| --- | --- | --- |
| Constitution §8: sintéticos honestos | Parquet con `synthetic = true`; reporte «DATOS SINTÉTICOS»; sin etiquetas ni métricas de desempeño | PASS |
| Constitution §7 y §11: sin datos personales ni género | Solo ids `S…`/`R…`, campus, partición y features; `livingWith` solo en HC02 | PASS |
| Constitution §9: reproducibilidad | Semilla fija con espacios de nombres; `prepare` dos veces sin cambios | PASS |
| Spec §7 regla 2 / M-ML-023: una sola implementación de las features | `compute_pair_features` de T02; prueba de igualdad y de orden | PASS |
| M-ML-024: restricciones con `filter_rooms` | Candidatos y exclusiones salen de `filter_rooms`; prueba con `check_hard_constraints` | PASS |
| M-ML-023: la distancia no es restricción dura | 26.3 % de los pares más lejos que `maxDistanceKm`; prueba dedicada | PASS |
| M-ML-021: umbral ≥ 90 % | 100 % en los 10 campus, en total y sin las `seed_sin_verificar`; la etapa falla si no se cumple | PASS |
| M-ML-017 / spec §7 regla 4: 3 campus sin verificar | `seed_verified` en el Parquet; totales con y sin ellos en todas las tablas | PASS |
| AGENTS: sin dependencias nuevas | Solo biblioteca estándar más pandas/pyarrow ya instalados; `pyproject.toml` y `uv.lock` sin cambios | PASS |
| AGENTS: `data/processed/` fuera de git | `pairs.parquet` ignorado (`git status` no lo lista) | PASS |
| `feature_schema.json`, `students_params.json` y `rooms_params.json` sin cambios | `git status` no los lista | PASS |
| AGENTS: rutas permitidas | `git status` solo lista `src/thehouse_ml/`, `tests/`, `reports/`, `memoria.md`, `specs/ml-03-feature-schema/` y la fila de ML-03 en `specs/README.md` | PASS |

### Hallazgos

| # | Hallazgo | Clasificación | Estado |
| --- | --- | --- | --- |
| 1 | El «EJECUTA» llegó truncado en el punto 6. | ADVERTENCIA | **RESUELTO** — el responsable confirmó la interpretación y completó los puntos 6 y 7; cada criterio del punto 6 tiene su prueba (tabla de trazabilidad) y se agregaron las dos que faltaban |
| 2 | Dentro de los candidatos, `pets_compatible` vale siempre 1 y `price_to_budget` ≤ 1, porque HC03 y HC01 ya filtran. | ADVERTENCIA | Para ML-04 («Nota para ML-04/ML-05», punto 1); el esquema no cambia en T06 |
| 3 | El conjunto no trae negativos por restricción: todos los candidatos son compatibles y ML-04 debe etiquetar la relevancia dentro de los compatibles. | INFORMATIVO | Por diseño (M-ML-027) |
| 4 | Los cuartos se repiten entre particiones (2,751 de 2,893): la evaluación mide la generalización a estudiantes nuevos, no a cuartos nuevos. | INFORMATIVO | Documentado aquí y en el reporte |
| 5 | 420 estudiantes (21 %) tienen más de 200 cuartos compatibles y reciben una muestra de 200. | INFORMATIVO | Por diseño (M-ML-021 aclarada en M-ML-027) |
| 6 | La prueba formal del umbral en CI usa el catálogo generado con los histogramas agrupados versionados (no idéntico a `rooms.parquet`); sobre los datos reales la comprueba la etapa `prepare`, que falla si no se cumple. Los dos dan 100 %. | INFORMATIVO | Sin acción |
| 7 | Las pruebas de pares tardan unos 35–45 s (generan los 2,000 estudiantes y 3,000 cuartos una vez por sesión). | INFORMATIVO | Sin acción |
| 8 | M-ML-027 estaba PROPUESTO (semilla 20261120, porcentajes 70/15/15, regla de candidatos). | ADVERTENCIA | **RESUELTO** — APROBADO por César Yael López Trujillo — 2026-11-19, sin cambios en los valores |
| 9 | Con los datos de ML-02, 1 estudiante (`S1078`, TESJI, campus sin verificar, partición `train`) tiene un solo candidato; 1,999 tienen 2 o más. Con un solo candidato no hay ranking que evaluar. | ADVERTENCIA | Para ML-05 («Nota para ML-04/ML-05», punto 2) |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. Insumos: `students.parquet` y `rooms.parquet` (sintéticos de T04 y T05) y `data/universities.csv` (ODbL, M-ML-016/M-ML-017). El reporte solo publica agregados de datos sintéticos.
- **Resultado:** **PASS** — la aceptación de R06 se cumple (pares candidatos por estudiante, partición por estudiante sin fuga para entrenamiento, validación y prueba, y prueba de ≥ 90 % con 100 % en todos los campus), cada criterio del punto 6 tiene su prueba en verde, los cinco controles pasan y M-ML-027 quedó APROBADO (hallazgos 1 y 8 resueltos). Los hallazgos 2 y 9 no bloquean T06: pasan a la nota de abajo.

### Nota para ML-04/ML-05

Origen: hallazgos 2 y 9 de T06 y aprobación de M-ML-027 (César Yael López Trujillo, 2026-11-19). Es una indicación del responsable para ML-04 y ML-05; no se ejecuta en T06.

1. **Features constantes dentro de los candidatos:** como los candidatos ya pasan HC03 y HC01, `pets_compatible` vale siempre **1** y `price_to_budget` es siempre **≤ 1** en `pairs.parquet` (`reports/pairs_summary.md` §7: mínimo, mediana y máximo de `pets_compatible` 1.000; máximo de `price_to_budget` 1.000). ML-04 decide si `pets_compatible` se conserva como feature del modelo (con este conjunto no aporta información) o queda solo como restricción; cualquier cambio de las features o de su orden es un cambio de `feature_schema.json` con su versión.
2. **Estudiantes con un solo candidato:** hay estudiantes con 1 candidato (con los datos de ML-02, 1: `S1078`, TESJI, `train`); con un solo cuarto no hay orden que medir. En ML-05 las métricas de ranking se calculan **solo con estudiantes de 2 o más candidatos**, y el reporte dice cuántos quedan fuera (por partición y por campus, con y sin las `seed_sin_verificar`).
