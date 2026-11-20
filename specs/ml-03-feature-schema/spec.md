# ML-03 — Esquema de features, estudiantes sintéticos y catálogo sintético de cuartos

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
| Documento | `spec.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-07 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-19 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0 |
| Tasks | `task.md` 1.0.0 |
| Precondición | ML-02 completa ✅ (M-ML-020); catálogos de NV-11 verificados en el commit `ea10a4c` del backend |
| Relación con el backend | El esquema de preferencias de NV-11 es la entrada del modelo; `feature_schema.json` es el contrato con NV-23 |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

El modelo necesita la misma información que la app guarda: las preferencias de NV-11 y los datos públicos del cuarto. Sin usuarios todavía, los perfiles de estudiante se generan de forma sintética y documentada. El catálogo de cuartos también se genera de forma sintética (~3,000 cuartos entre los 10 campuses del seed): sin datos reales de cuartos, el pareo sintético parte de un catálogo reproducible, calibrado con los datos de ML-02 y etiquetado como sintético (constitución §8).

## 2. Objetivo

Definir el esquema de features compartido con el backend y generar estudiantes sintéticos realistas y reproducibles, más el catálogo sintético de cuartos que da pares reales de entrenamiento y evaluación sin datos reales de cuartos.

## 3. Precondiciones

1. ML-02 completa ✅ (M-ML-020).
2. NV-11 verificada en el commit `ea10a4c` del backend (catálogos cerrados: presupuesto 0–50,000 MXN; `maxDistanceKm` 0.5–30; `placeType` y `propertyTypes` de NV-11).
3. Autorización «EJECUTA».

## 4. Alcance

1. `feature_schema.json` versionado: entradas del estudiante, del cuarto y features del par.
2. Restricciones duras definidas en el esquema.
3. Generador de estudiantes sintéticos con semilla.
4. Catálogo sintético de cuartos (heredado de M-ML-019/M-ML-020): ~3,000 cuartos entre los 10 campuses del seed a ≤30 km del campus; renta calibrada con `regional_calibration.parquet` y la forma de `rent_histogram.parquet`; servicios con las proporciones del Censo por municipio (agua, drenaje, electricidad, gas, internet→wifi); priors documentados para los que el Censo no mide (laundry, kitchen, furnished, parking, security, cleaning, hot_water); `placeType`, `propertyTypes`, mascotas y política de convivencia (`livingWith`) con distribuciones documentadas como supuestos; semilla fija; parámetros en `src/thehouse_ml/synthetic/rooms_params.json`; salida `data/processed/rooms.parquet` (sintético, etiquetado) y `reports/rooms_catalog.md`; todo etiquetado como sintético (constitución §8).
5. Pares estudiante–cuarto candidatos para entrenamiento y evaluación.

## 5. Fuera de alcance

- Etiquetas y modelos (ML-04, ML-05).
- Datos personales de cualquier tipo.
- El entorno DENUE no entra como feature del modelo en v1 (el backend no tiene esos datos); solo se documenta como riesgo (RK03).

## 6. Diseño

- Estudiante (de NV-11): `budget.min`, `budget.max` (0–50,000 MXN), `maxDistanceKm` (0.5–30), `essentialServices` (multi-hot de `wifi`, `water`, `electricity`, `gas`, `hot_water`, `laundry`, `kitchen`, `furnished`, `parking`, `security`, `cleaning`), `livingWith` (`women`, `men`, `any`), `placeType`, `pets`, `transport`, `propertyTypes`, universidad (coordenadas).
- Cuarto: renta mensual (MXN, jul-2024), coordenadas (haversine a ≤30 km del campus), servicios (mismo catálogo que `essentialServices`), `placeType` (NV-11: `room`, `room_private_bath`, `studio`, `shared`), `propertyType` (NV-11: `house`, `apartment`, `student_residence`), admite mascotas (`pets`), política de convivencia (`livingWith`: `women`, `men`, `any` — el cuarto declara quién vive ahí), recámaras, baños y área.
- Par: distancia en km (haversine), distancia efectiva según `transport` (factores PROPUESTOS: caminar 1.0, bici 0.5, transporte público 0.4, auto 0.25), razón precio/presupuesto, cobertura de servicios esenciales, coincidencia de tipo de lugar y de inmueble, compatibilidad de mascotas.
- Restricciones duras: renta mayor a `budget.max`, `livingWith` incompatible o mascota no admitida → el cuarto se excluye.
- Catálogo sintético de cuartos (R05): generación reproducible (semilla fija; parámetros en `src/thehouse_ml/synthetic/rooms_params.json`); renta con la forma de `rent_histogram.parquet` (intervalos de 500 pesos hasta 10,000 y `10,000+`) calibrada por campus: mediana versus `regional_calibration.parquet`; servicios por municipio del Censo (agua, drenaje, electricidad, gas, internet→wifi) y priors documentados (hot_water, laundry, kitchen, furnished, parking, security, cleaning); `placeType`, `propertyTypes`, mascotas y `livingWith` (el cuarto declara `women`, `men`, `any`) con distribuciones documentadas como supuestos; las 3 universidades sin verificar entran marcadas (`seed_verified = false`); salida `data/processed/rooms.parquet` (sintético) y `reports/rooms_catalog.md` (etiquetado como sintético).
- Estudiantes sintéticos: distribuciones por universidad documentadas (presupuesto, transporte, servicios) con semilla fija; presupuesto calibrado con la renta regional (`regional_calibration.parquet`): prueba en T04 de que la mediana de `budget.max` por universidad es coherente con la renta p50 regional; parámetros en `src/thehouse_ml/synthetic/students_params.json`.

## 7. Reglas

1. El esquema usa exactamente los catálogos de NV-11; un cambio en NV-11 obliga a una versión mayor del esquema (incluidos los del cuarto: `placeType`, `propertyTypes` y los servicios).
2. Las features se calculan con una sola implementación de referencia que también genera los casos dorados para el backend.
3. Restricción de convivencia (R03): el par es válido si `student.livingWith == "any"` o `room.livingWith == student.livingWith`; el cuarto declara quién vive ahí (`women`, `men`, `any`); señal: no se infiere el género del estudiante (constitución §11).
4. El catálogo de cuartos es sintético y etiquetado como tal; las 3 universidades sin verificar (TESJI, UMB-VDC, UCAN) entran marcadas (`seed_verified = false`) y sus distancias se reportan como orientativas.

## 8. Seguridad, datos y privacidad

- Los perfiles sintéticos no imitan personas reales.
- El catálogo de cuartos es sintético y etiquetado como tal (constitución §8); no se usan datos personales (constitución §7).
- El género solo interviene en la restricción `livingWith` que elige el estudiante (constitution §11).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-03-R01 | Esquema de features | `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba. | F1 | T01 |
| ML-03-R02 | Cálculo de features de referencia | Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios). | F2 | T02 |
| ML-03-R03 | Restricciones duras | Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla. | F2 | T03 |
| ML-03-R04 | Estudiantes sintéticos | Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones; presupuesto calibrado con la renta regional (`regional_calibration.parquet`): prueba de que la mediana de `budget.max` por universidad es coherente con la renta p50 regional. | F2 | T04 |
| ML-03-R05 | Catálogo sintético de cuartos | Generador reproducible (misma semilla → mismos cuartos): ~3,000 cuartos entre los 10 campuses del seed, todos a ≤30 km del campus; rangos y catálogos de NV-11; renta coherente con la calibración por municipio — prueba que compara la mediana generada contra la calibrada; reporte de distribuciones del catálogo (servicios, `placeType`, `propertyTypes`, mascotas, `livingWith`, distancia a campus) etiquetado como sintético. | F2 | T05 |
| ML-03-R06 | Conjunto de pares | Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga; prueba de que la mayoría de los estudiantes tiene al menos un cuarto que pasa las restricciones duras (≥ 90 % de estudiantes con al menos un cuarto compatible). | F3 | T06 |
| ML-03-R07 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T07 |

## 10. Decisiones

Se registran en `memoria.md` (M-ML-021, APROBADO).

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Número de estudiantes sintéticos y candidatos | 2,000 estudiantes y 200 candidatos por estudiante (~3,000 cuartos) | APROBADO |
| Número y distribución de cuartos | ~3,000 cuartos entre los 10 campuses del seed, ≤30 km del campus (~300 por campus) | APROBADO |
| Calibración de renta | Renta de los cuartos calibrada con `regional_calibration.parquet` y la forma del histograma de `rent_histogram.parquet`; mediana por campus contra la renta p50 regional | APROBADO |
| Servicios de los cuartos | Proporciones del Censo por municipio: agua (AGUA_ENTUBADA), drenaje (sin equivalente en NV-11), electricidad (Electricidad), gas (Gas), internet → wifi (INTERNET); priors documentados: laundry, kitchen, furnished, parking, security, cleaning, hot_water | APROBADO |
| Distribuciones de placeType, propertyTypes, mascotas y livingWith de los cuartos | Distribuciones documentadas como supuestos | APROBADO |
| livingWith (cuarto vs estudiante) | El cuarto declara quién vive ahí (`women`, `men`, `any`); estudiante `women` → solo cuartos `women`; `men` → solo cuartos `men`; `any` → todos; señal: no se infiere el género del estudiante (constitución §11) | APROBADO |
| campuses sin verificar | Las 3 `seed_sin_verificar` (TESJI, UMB-VDC, UCAN) entran marcadas (`seed_verified = false`); métricas reportadas con y sin ellas | APROBADO |
| Entorno DENUE | No entra como feature del modelo en v1; solo se documenta (riesgo RK03) | APROBADO |
| Semilla y parámetros del generador | Semilla fija; parámetros de estudiantes en `src/thehouse_ml/synthetic/students_params.json` y de cuartos en `src/thehouse_ml/synthetic/rooms_params.json` | APROBADO |
| Umbral de estudiantes con al menos un cuarto compatible | Se verifica en T06: la mayoría de los estudiantes tiene al menos un cuarto que pasa las restricciones duras; umbral exacto: ≥ 90 % | APROBADO |
| Factores de distancia efectiva por transporte | `walk` 1.0, `bike` 0.5, `public` 0.4, `car` 0.25 (`effective_distance_km = distance_km × factor`); omitidos por error en M-ML-021 y aclarados en M-ML-022 | APROBADO |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-03-RK01 | Esquema distinto al de la app | Modelo inutilizable | Prueba contra NV-11 y versión mayor ante cambios |
| ML-03-RK02 | Sintéticos poco realistas | Modelo que no generaliza | Validar distribuciones con el equipo y reemplazar con señales reales en ML-07 |
| ML-03-RK03 | Contrato del cuarto no implementado en el backend | Modelo no se puede usar en producción: Property no tiene coordenadas, `placeType`, `propertyTypes`, mascotas ni política de convivencia | `feature_schema.json` define esos campos y se avisa al backend para NV-12; sin ellos el modelo no se puede usar en producción |

## 13. Dependencias

- ML-02.
- NV-11 1.1.0 (backend).
- NV-12 — campos del cuarto que Property no tiene (coordenadas, `placeType`, `propertyTypes`, mascotas, política de convivencia): avisar sobre el contrato nuevo del cuarto.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 7 |
| Tasks | 7 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 6 (T01, T02, T03, T04, T05, T06) |
| Estado documental | **✅ APROBADA** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación

Aprobada por César Yael López Trujillo el 2026-11-19 (ML-03 1.0.0; decisión registrada como M-ML-021). Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
