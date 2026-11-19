# ML-02 — Datos: bases nacionales, calibración regional y licencias

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `spec.md` |
| Versión | 1.0.1 |
| Fecha | 2026-11-12 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-12 (M-ML-012): aprueba ML-02 1.0.1 |
| Estado de ejecución | **✅ COMPLETA — T01–T05 y T07–T09 completas (8 tasks); T06 DESCARTADA (M-ML-019)** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.1 |
| Tasks | `task.md` 1.0.1 |
| Precondición | ML-01 completa; fuentes de M-ML-003 aprobadas |
| Relación con el backend | Verifica las coordenadas de las 10 universidades del seed de NV-11 (3 están aproximadas al centro del municipio) y entrega las correcciones |

Spec aprobada (1.0.1, M-ML-012). No autoriza código ni descargas hasta recibir «EJECUTA» con tasks y rutas. T03 se ejecutó el 2026-11-17 bajo M-ML-014, T04 el 2026-11-17 bajo M-ML-015 y T07 el 2026-11-17 bajo M-ML-016 (antes que T05, porque T05 necesita coordenadas verificadas); la T05 se ejecutó el 2026-11-18 bajo M-ML-018. La T06 quedó **DESCARTADA** el 2026-11-18 por M-ML-019 (Inside Airbnb no se descarga ni se verifica); la T08 se ejecutó el 2026-11-18 en la rama `feat/ml-02-data-card`. La T09 (ML-02-R09) se ejecutó el 2026-11-19 en la rama `feat/ml-02-cierre`: pruebas, evidencia, revisión de coherencia y cierre de la spec. ML-02 queda **COMPLETA**: 8 tasks completas y T06 descartada.

## 1. Contexto

No existe un dataset público de estudiantes buscando cuarto en México, y las encuestas tienen pocas viviendas por municipio. Por eso los datos se trabajan en dos niveles: **nacional** (INEGI) para obtener bases y proporciones con muestra suficiente, y **regional** para calibrarlas a los 7 municipios de las universidades del seed (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón). INEGI describe hogares y viviendas, no cuartos. Inside Airbnb quedó **descartado** (M-ML-019), así que el catálogo de cuartos se generará **sintético en ML-03** con distribuciones documentadas, calibrado con INEGI y etiquetado como sintético; más adelante se contrastará con los cuartos reales de la app (ML-07).

## 2. Objetivo

Obtener con licencia verificada: (1) bases nacionales de renta, tenencia y servicios de vivienda; (2) su calibración a la región; (3) las bases para el catálogo **sintético** de cuartos de ML-03, sin datos personales y mapeado a los catálogos de NV-11 (M-ML-019); (4) el entorno de cada universidad (DENUE); y (5) las coordenadas verificadas de las 10 universidades.

## 3. Precondiciones

1. ML-01 completa.
2. Fuentes y licencias registradas como APROBADO en M-ML-003 (términos de libre uso de INEGI y OpenStreetMap). Inside Airbnb quedó DESCARTADO por M-ML-019 y no se descarga.
3. Variables confirmadas en los diccionarios oficiales antes de escribir código; las del Censo 2020 ya se confirmaron (§10).
4. Autorización «EJECUTA».

## 4. Alcance

1. Registro de fuentes: URL, edición, fecha, licencia, atribución, `dictionary_url`, `variables_status` y variables usadas de cada fuente, en `data/sources.toml` (en git).
2. `download`: descarga reproducible de cada fuente con manifiesto SHA-256 en `data/manifests/`.
3. Bases nacionales (INEGI): Censo de Población y Vivienda 2020, muestra del cuestionario ampliado (`Viviendas_CA`: `TENENCIA`, servicios, `FACTOR` y `MUN`), que **no** pregunta el monto de renta; Encuesta Nacional de Vivienda (ENVI) 2020 como fuente del monto y motivo de renta; ENIGH 2022 solo como validación del gasto en vivienda de inquilinos.
4. Calibración regional: estimaciones de los 7 municipios (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón) con respaldo jerárquico municipio → estado → nacional cuando la muestra no alcanza.
5. Entorno de las universidades: conteos del DENUE por categoría SCIAN (abarrotes, lavanderías, farmacias, papelerías, transporte, comida) en radios de 1, 2 y 5 km de cada campus.
6. Catálogo de cuartos: **DESCARTADA la T06** (M-ML-019). No se usa Inside Airbnb. ML-03 generará el catálogo de cuartos de forma sintética, calibrado con ENVI 2020/ENIGH 2022 (renta), Censo 2020 (servicios) y DENUE (entorno), sin datos personales, mapeado a NV-11 y etiquetado como sintético (constitution §8).
7. Universidades: verificación contra OpenStreetMap de las coordenadas del seed de NV-11, con prioridad en las 3 aproximadas (UAEH-ESTL, UMB-VDC, UCAN).
8. Data card con fuentes, licencias, atribuciones, ponderación, transformaciones, tamaños de muestra y limitaciones.

## 5. Fuera de alcance

- Scraping de portales inmobiliarios (M-ML-004).
- Datos de usuarios reales (ML-07).
- Republicar microdatos crudos o procesados (solo se publican agregados con atribución).
- Laboratorio de microdatos de INEGI o procesamiento remoto: solo se usan productos de descarga pública.

## 6. Diseño

- **Ponderación obligatoria:** toda estimación de encuesta usa el factor de expansión de la fuente (Censo muestra, ENVI, ENIGH); los conteos sin ponderar solo se usan para decidir si la muestra alcanza.
- **Regla de suficiencia:** una estimación municipal se usa solo si tiene al menos 30 viviendas rentadas en la muestra y coeficiente de variación ≤ 30 %; si no, se usa la del estado, y si tampoco alcanza, la nacional. Cada valor guarda de qué nivel salió.
- **Bases nacionales (salida `national_baselines.parquet`):** percentiles de renta mensual (p10, p25, p50, p75, p90) por tipo de vivienda y tamaño de localidad; proporción de viviendas rentadas; prevalencia y co-ocurrencia de servicios (agua entubada, drenaje, electricidad, internet, gas); todo con fecha de la fuente.
- **Actualización de montos:** las rentas de 2020 se llevan a pesos del año de ejecución con el INPC de INEGI; el índice y la fecha quedan en el manifiesto.
- **Calibración regional (salida `regional_calibration.parquet`):** para los 7 municipios, proporción de viviendas rentadas y servicios desde el Censo (nivel municipal con respaldo); el monto de renta solo existe a nivel estatal (ENVI: Hidalgo y Estado de México), así que la renta regional usa la forma nacional reescalada a la mediana estatal y se contrasta con el seed de TheHouse. Cada valor guarda su nivel.
- **Entorno DENUE (salida `campus_context.parquet`):** conteos por categoría y radio para cada universidad; alimenta el factor de extras en ML-03.
- **Cuartos (salida `rooms.parquet`): DESCARTADA la T06 (M-ML-019).** No se descarga Inside Airbnb ni se produce este Parquet en ML-02. ML-03 generará un catálogo **sintético** de cuartos con semilla fija, distribuciones documentadas y catálogos cerrados de NV-11; la renta mensual se calibrará con `national_baselines.parquet` y `regional_calibration.parquet`, los servicios con el Censo 2020 y el entorno con `campus_context.parquet`. Todo resultado se etiquetará como sintético (constitution §8) y nunca se presentará como desempeño real.
- **Nota para ML-03:** el mapeo de servicios al catálogo de NV-11 (`wifi`, `water`, `electricity`, `gas`, `hot_water`, `laundry`, `kitchen`, `furnished`, `parking`, `security`, `cleaning`) y de `room_type`/`property_type` a `placeType` y `propertyTypes` se define en ML-03 sobre el catálogo sintético; las tablas de mapeo versionadas viven en `src/thehouse_ml/data/mappings/`. En ML-02 no se leen ni se guardan datos de anfitriones.
- **Universidades (salida `universities.csv`, en git):** código del seed, latitud, longitud, id del elemento OSM, fuente y fecha de verificación; diferencias mayores a 200 m respecto al seed se reportan para corregir NV-11.
- Todas las salidas en `data/processed/` (fuera de git) salvo `universities.csv`, `sources.toml`, manifiestos y tablas de mapeo.

## 6.1 Fuentes

| Fuente | Editor | Uso | Nivel | Licencia |
| --- | --- | --- | --- | --- |
| Censo de Población y Vivienda 2020, muestra (cuestionario ampliado, tabla `Viviendas_CA`) | INEGI | Tenencia (`TENENCIA`), servicios, factor de expansión (`FACTOR`) y municipio (`MUN`); no incluye monto de renta | Nacional, estatal, municipal | Términos de libre uso de INEGI: citar la fuente |
| Encuesta Nacional de Vivienda (ENVI) 2020 | INEGI, Infonavit, SHF | Monto y motivo de renta, gasto en vivienda | Nacional, estatal | Términos de libre uso de INEGI |
| ENIGH 2022 | INEGI | Validación del gasto en renta de inquilinos (no el alquiler imputado a propietarios) | Nacional, estatal | Términos de libre uso de INEGI |
| DENUE (edición vigente) | INEGI | Establecimientos con coordenadas alrededor de cada campus | Hasta localidad | Términos de libre uso de INEGI |
| INPC | INEGI | Actualizar montos de 2020 a pesos del año de ejecución | Nacional | Términos de libre uso de INEGI |
| ~~Inside Airbnb, Ciudad de México (instantánea 2026-09-23, `listings.csv.gz`)~~ | Inside Airbnb | **DESCARTADA (M-ML-019)**: no cubre los 7 municipios, mide renta turística por noche y trae datos personales; el catálogo será sintético en ML-03 | — | CC BY 4.0 verificada, pero sin uso: no se descarga ni se verifica |
| OpenStreetMap | Colaboradores de OSM | Verificar coordenadas de campus | Punto | ODbL: atribución y misma licencia para bases derivadas |
| Seed de TheHouse (NV-07, NV-08, NV-11) | Propio | Universidades, rangos de referencia | Región | Propio |

## 6.2 Resultados esperados

| Salida | Ubicación | En git | Cómo se verifica |
| --- | --- | --- | --- |
| Registro de fuentes | `data/sources.toml` | Sí | Prueba: toda fuente tiene licencia, URL, fecha, atribución, `dictionary_url` y `variables_status` válido |
| Manifiestos SHA-256 | `data/manifests/*.json` | Sí | `download` falla si un hash no coincide |
| Bases nacionales | `data/processed/national_baselines.parquet` | No | Prueba de esquema, percentiles ordenados (p10 ≤ … ≤ p90), proporciones entre 0 y 1 |
| Calibración regional | `data/processed/regional_calibration.parquet` | No | 7 municipios presentes, cada valor con su nivel de respaldo |
| Entorno de campus | `data/processed/campus_context.parquet` | No | 10 universidades × 3 radios, conteos no negativos y crecientes con el radio |
| ~~Catálogo de cuartos~~ | ~~`data/processed/rooms.parquet`~~ | No | **DESCARTADA (M-ML-019)**: no se descarga ni se produce; se traslada a ML-03 como catálogo sintético etiquetado como tal |
| Universidades verificadas | `data/universities.csv` | Sí | 10 filas con id OSM; diferencias > 200 m reportadas |
| Data card | `reports/data_card.md` | Sí | Fuentes, atribuciones, ponderación, tamaños de muestra y limitaciones |


## 7. Reglas

1. Una edición por fuente y por versión de datos; cambiarla crea una versión nueva.
2. Ninguna variable se usa sin confirmarla; en T01 cada variable queda registrada con `variables_status = "pending"` y `dictionary_url`, y en T02 se verifica automáticamente contra el diccionario o las columnas del archivo descargado (solo entonces pasa a `"confirmed"`). El nombre exacto queda en `data/sources.toml`.
3. Toda estimación de encuesta se pondera con su factor de expansión.
4. Nunca se presenta una estimación municipal que no cumple la regla de suficiencia.
5. Toda fila con datos personales residuales se descarta, no se corrige.
6. Se cita a INEGI como autor y el nombre del producto en la data card, el README y los reportes, sin sugerir que INEGI respalda el proyecto.

## 8. Seguridad, datos y privacidad

- Ningún dato crudo o procesado entra a git; solo agregados, manifiestos y tablas de mapeo.
- Los microdatos públicos de INEGI son anónimos; aun así no se publican celdas con menos de 30 viviendas.
- Atribución a INEGI y OpenStreetMap (ODbL) en la data card y el README. Inside Airbnb no se usa (M-ML-019), por lo que no se descarga, se verifica ni se atribuye como fuente de datos.
- Sin scraping: la única fuente bajo demanda es OpenStreetMap/Nominatim, con sus 10 consultas autorizadas (M-ML-016).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-02-R01 | Registro de fuentes y licencias | `data/sources.toml` con URL, edición, fecha, licencia, atribución, `dictionary_url`, `variables_status` y variables registradas; prueba que falla si falta un campo. | F1 | T01 |
| ML-02-R02 | Descarga verificable | `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores; verifica que cada variable registrada exista en el archivo descargado y la marca `confirmed`. | F2 | T02 |
| ML-02-R03 | Bases nacionales ponderadas | `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles. | F2 | T03 |
| ML-02-R04 | Calibración regional con respaldo | `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla. | F2 | T04 |
| ML-02-R05 | Entorno de campus | `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios. | F2 | T05 |
| ML-02-R06 | Inside Airbnb descartado (M-ML-019) | `data/sources.toml` marca la fuente con `variables_status = "discarded"` y `status_reference = "M-ML-019"`, sin la URL no verificada de `listings.csv.gz`; `download` la omite con el aviso `descartada: M-ML-019` sin descargarla ni verificarla, y `reports/variables_check.md` la muestra como descartada conservando las demás secciones. No se produce `rooms.parquet`: el catálogo sintético de cuartos, sin datos personales y etiquetado como sintético, se define en ML-03. | F2 | T06 (DESCARTADA) |
| ML-02-R07 | Universidades verificadas | `data/universities.csv` con las 10 universidades, id OSM y fecha de verificación; las 3 aproximadas revisadas; diferencias mayores a 200 m reportadas para el backend. | F3 | T07 |
| ML-02-R08 | Data card | `reports/data_card.md` con fuentes, licencias, atribuciones, ponderación, niveles de respaldo, tamaños de muestra, histogramas y limitaciones. | F3 | T08 |
| ML-02-R09 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T09 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Monto de renta en el Censo 2020 (muestra) | Verificado el 2026-11-10: el cuestionario ampliado solo pregunta tenencia (pregunta 23) y el diccionario de `Viviendas_CA` (83 variables) no tiene monto. La renta sale de ENVI 2020 (estatal y nacional); el Censo aporta tenencia y servicios por municipio | RESUELTO |
| Región | Los 7 municipios de las universidades del seed (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón) | APROBADO |
| Regla de suficiencia | ≥ 30 viviendas rentadas en muestra y CV ≤ 30 %; si no, respaldo estado → país | APROBADO |
| Edición del DENUE | La vigente al ejecutar (basada en los Censos Económicos más recientes) | APROBADO |
| Inside Airbnb | **DESCARTADO por M-ML-019** (2026-11-18): solo CDMX, renta turística por noche, datos personales de anfitriones y cobertura ya resuelta por ENVI/ENIGH, Censo y DENUE. Se retira la URL de `listings.csv.gz`; no se descarga nada | DESCARTADO (M-ML-019) |
| Datos de matrícula (ANUIES) | — | DESCARTADO (M-ML-012): no es indispensable y su licencia no está verificada |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-02-RK01 | Muestra municipal insuficiente | Estimaciones regionales inestables | Regla de suficiencia con respaldo estado → país y reporte del nivel usado |
| ML-02-RK02 | Sin renta a nivel municipal (el Censo 2020 no la pregunta) | La renta regional es estatal, no municipal | Usar ENVI estatal, contrastar con el seed de TheHouse y, después, con cuartos reales (ML-07) |
| ML-02-RK03 | Datos de 2020 desactualizados | Rentas bajas respecto a hoy | Actualizar con INPC y validar contra el seed de TheHouse |
| ML-02-RK04 | La Ciudad de México no representa a la región | ~~Atributos de cuarto sesgados~~ | **RESUELTO por M-ML-019:** Inside Airbnb se descarta; ML-03 genera el catálogo sintético calibrado con INEGI regional y DENUE, etiquetado como sintético |
| ML-02-RK05 | Coordenadas equivocadas | Distancias erróneas en backend y modelo | Verificación punto por punto contra OSM |

## 13. Dependencias

- ML-01.
- Seed de universidades de NV-11 (backend).
- Licencias de M-ML-003 aprobadas.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 9 |
| Tasks | 9 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 8 |
| Tasks descartadas | 1 (T06, M-ML-019) |
| Estado documental | **✅ APROBADA** |
| Estado de ejecución | **✅ COMPLETA — T01–T05 y T07–T09 completas (8 tasks); T06 DESCARTADA (M-ML-019)** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.

El 2026-11-18 el responsable aprobó M-ML-019 y dio «EJECUTA ML-02 1.0.1, task T06, en la rama docs/ml-02-t06-descartada», con rutas permitidas `data/sources.toml`, `memoria.md`, `specs/ml-02-data-sources/`, `reports/variables_check.md`, `src/thehouse_ml/` y `tests/` (estas últimas solo para que la fuente descartada no se descargue ni se verifique). Sin dependencias nuevas y sin descargar nada de Inside Airbnb.

El 2026-11-18 César Yael López Trujillo dio «EJECUTA ML-02 1.0.1, task T08 (ML-02-R08), en la rama feat/ml-02-data-card», con rutas permitidas `src/thehouse_ml/`, `tests/`, `reports/`, `memoria.md` y `specs/ml-02-data-sources/`, sin dependencias nuevas. La data card se genera desde código como parte de `prepare`; la evidencia está en `evidence.md`.

El 2026-11-19 César Yael López Trujillo dio «EJECUTA ML-02 1.0.1, task T09 (ML-02-R09), en la rama feat/ml-02-cierre», con rutas permitidas `src/thehouse_ml/`, `tests/`, `data/manifests/` (solo para conservar fechas originales), `data/universities.csv`, `reports/`, `README.md` (solo la sección de atribución de datos), `memoria.md` y `specs/ml-02-data-sources/`, sin dependencias nuevas ni descargas nuevas. T09 cierra la spec con las pruebas, la evidencia y la revisión de coherencia; la decisión de cierre se registra como M-ML-020. No se sube la versión de la spec: ni la constitución ni `AGENTS.md` exigen un bump al cerrar, y cambiar la versión aprobada 1.0.1 requeriría una aprobación nueva («Apruebo X.Y.Z»).
