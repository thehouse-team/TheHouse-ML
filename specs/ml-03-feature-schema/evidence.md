# ML-03 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-03 1.0.0 |
| Tareas ejecutadas | T01 · Esquema de features (R01) |
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
| 9 | Avisar al backend (NV-12) de los campos `missing_in_property` del cuarto (spec RK03). | ADVERTENCIA | **ABIERTO — acción pendiente del responsable** (César Yael López Trujillo) |
| 10 | Sin commit ni PR, por instrucción del responsable. | INFORMATIVO | Se cierra al mezclar el PR de T01 |
| 11 | Al cambiar ML-03 de 6 a 7 tasks, el total de `specs/ARQUITECTURA.md` §6 pasa de 41 a 42 tasks. | INFORMATIVO | Ajustado por consistencia |

### Revisión y resultado

- **Seguridad y licencias:** sin descargas, sin red, sin secretos, sin datos personales y sin dependencias nuevas. El snapshot es un dato propio del proyecto TheHouse.
- **Resultado:** **PASS WITH WARNINGS** — la aceptación de R01 se cumple (`feature_schema.json` con nombres, tipos, rangos, catálogos y versión, validado contra el snapshot de NV-11 verificado por el responsable) y los cuatro controles pasan. Única advertencia abierta: el aviso al backend (NV-12, hallazgo 9), acción pendiente del responsable; no bloquea T01 pero sí el uso del modelo en producción.
