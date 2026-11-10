# ML-02 — Datos: bases nacionales, calibración regional y licencias

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-10 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-10 («pues si te apruebo que revises si el censo trae lo de las rentas, pues la region si esta bien, pues si en el municipio creo que hay varias rentas las suficientes y si se usa airbnb usa la licencia y lo demas tambien en pocas palabras si a todo»): ML-02 1.0.0 y sus decisiones de la §10 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-01 completa; fuentes de M-ML-003 aprobadas |
| Relación con el backend | Verifica las coordenadas de las 10 universidades del seed de NV-11 (3 están aproximadas al centro del municipio) y entrega las correcciones |

Spec aprobada. No autoriza código ni descargas hasta recibir «EJECUTA» con tasks y rutas.

## 1. Contexto

No existe un dataset público de estudiantes buscando cuarto en México, y las encuestas tienen pocas viviendas por municipio. Por eso los datos se trabajan en dos niveles: **nacional** (INEGI) para obtener bases y proporciones con muestra suficiente, y **regional** para calibrarlas a los 7 municipios de las universidades del seed (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón). INEGI describe hogares y viviendas, no cuartos; los atributos a nivel cuarto (servicios, tipo de cuarto) vienen de Inside Airbnb Ciudad de México, y más adelante de los cuartos reales de la app (ML-07).

## 2. Objetivo

Obtener con licencia verificada: (1) bases nacionales de renta, tenencia y servicios de vivienda; (2) su calibración a la región; (3) un catálogo de cuartos sin datos personales mapeado a los catálogos de NV-11; (4) el entorno de cada universidad (DENUE); y (5) las coordenadas verificadas de las 10 universidades.

## 3. Precondiciones

1. ML-01 completa.
2. Fuentes y licencias registradas como APROBADO en M-ML-003 (términos de libre uso de INEGI, Inside Airbnb, OpenStreetMap).
3. Variables confirmadas en los diccionarios oficiales antes de escribir código; las del Censo 2020 ya se confirmaron (§10).
4. Autorización «EJECUTA».

## 4. Alcance

1. Registro de fuentes: URL, edición, fecha, licencia, atribución y variables usadas de cada fuente, en `data/sources.toml` (en git).
2. `download`: descarga reproducible de cada fuente con manifiesto SHA-256 en `data/manifests/`.
3. Bases nacionales (INEGI): Censo de Población y Vivienda 2020, muestra del cuestionario ampliado (`Viviendas_CA`: `TENENCIA`, servicios, `FACTOR` y `MUN`), que **no** pregunta el monto de renta; Encuesta Nacional de Vivienda (ENVI) 2020 como fuente del monto y motivo de renta; ENIGH 2022 solo como validación del gasto en vivienda de inquilinos.
4. Calibración regional: estimaciones de los 7 municipios (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón) con respaldo jerárquico municipio → estado → nacional cuando la muestra no alcanza.
5. Entorno de las universidades: conteos del DENUE por categoría SCIAN (abarrotes, lavanderías, farmacias, papelerías, transporte, comida) en radios de 1, 2 y 5 km de cada campus.
6. Catálogo de cuartos: Inside Airbnb Ciudad de México (una instantánea) para atributos a nivel cuarto, sin datos personales y mapeado a NV-11.
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
- **Cuartos (salida `rooms.parquet`):** columnas conservadas de Inside Airbnb `room_type`, `property_type`, `accommodates`, `bedrooms`, `bathrooms_text`, `amenities`, `price`, `latitude`, `longitude`; se eliminan nombres, descripciones, URLs, fotos y todo dato del anfitrión; coordenadas redondeadas a 3 decimales; precio por noche convertido a percentil dentro de su tipo y luego a renta mensual con la calibración regional (nunca se usa el precio por noche como renta).
- Mapeo de `amenities` al catálogo de NV-11 (`wifi`, `water`, `electricity`, `gas`, `hot_water`, `laundry`, `kitchen`, `furnished`, `parking`, `security`, `cleaning`) y de `room_type`/`property_type` a `placeType` y `propertyTypes`; tablas de mapeo versionadas en `src/thehouse_ml/data/mappings/`.
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
| Inside Airbnb, Ciudad de México (instantánea 2026-09-23, `listings.csv.gz`) | Inside Airbnb | Atributos a nivel cuarto | Ciudad | CC BY 4.0, verificada el 2026-11-10; cita formal de dataset, una descarga, sin scraping ni republicar |
| OpenStreetMap | Colaboradores de OSM | Verificar coordenadas de campus | Punto | ODbL: atribución y misma licencia para bases derivadas |
| Seed de TheHouse (NV-07, NV-08, NV-11) | Propio | Universidades, rangos de referencia | Región | Propio |

## 6.2 Resultados esperados

| Salida | Ubicación | En git | Cómo se verifica |
| --- | --- | --- | --- |
| Registro de fuentes | `data/sources.toml` | Sí | Prueba: toda fuente tiene licencia, URL, fecha y atribución |
| Manifiestos SHA-256 | `data/manifests/*.json` | Sí | `download` falla si un hash no coincide |
| Bases nacionales | `data/processed/national_baselines.parquet` | No | Prueba de esquema, percentiles ordenados (p10 ≤ … ≤ p90), proporciones entre 0 y 1 |
| Calibración regional | `data/processed/regional_calibration.parquet` | No | 7 municipios presentes, cada valor con su nivel de respaldo |
| Entorno de campus | `data/processed/campus_context.parquet` | No | 10 universidades × 3 radios, conteos no negativos y crecientes con el radio |
| Catálogo de cuartos | `data/processed/rooms.parquet` | No | Sin columnas prohibidas, catálogos cerrados de NV-11, rentas dentro del rango regional |
| Universidades verificadas | `data/universities.csv` | Sí | 10 filas con id OSM; diferencias > 200 m reportadas |
| Data card | `reports/data_card.md` | Sí | Fuentes, atribuciones, ponderación, tamaños de muestra y limitaciones |


## 7. Reglas

1. Una edición por fuente y por versión de datos; cambiarla crea una versión nueva.
2. Ninguna variable se usa sin confirmarla en el diccionario oficial de su fuente; el nombre exacto queda en `data/sources.toml`.
3. Toda estimación de encuesta se pondera con su factor de expansión.
4. Nunca se presenta una estimación municipal que no cumple la regla de suficiencia.
5. Toda fila con datos personales residuales se descarta, no se corrige.
6. Se cita a INEGI como autor y el nombre del producto en la data card, el README y los reportes, sin sugerir que INEGI respalda el proyecto.

## 8. Seguridad, datos y privacidad

- Ningún dato crudo o procesado entra a git; solo agregados, manifiestos y tablas de mapeo.
- Los microdatos públicos de INEGI son anónimos; aun así no se publican celdas con menos de 30 viviendas.
- Atribución a INEGI, Inside Airbnb (CC BY 4.0) y OpenStreetMap (ODbL) en la data card y el README.
- Se respeta la política de Inside Airbnb: una descarga, sin scraping del sitio.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-02-R01 | Registro de fuentes y licencias | `data/sources.toml` con URL, edición, fecha, licencia, atribución y variables confirmadas en el diccionario oficial; prueba que falla si falta un campo. | F1 | T01 |
| ML-02-R02 | Descarga verificable | `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores. | F2 | T02 |
| ML-02-R03 | Bases nacionales ponderadas | `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles. | F2 | T03 |
| ML-02-R04 | Calibración regional con respaldo | `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla. | F2 | T04 |
| ML-02-R05 | Entorno de campus | `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios. | F2 | T05 |
| ML-02-R06 | Catálogo de cuartos sin datos personales | `rooms.parquet` sin columnas prohibidas (prueba automática), con coordenadas redondeadas, servicios y tipos mapeados a NV-11 y renta mensual calibrada; los valores no mapeados se cuentan y se reportan. | F2 | T06 |
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
| Inside Airbnb | Instantánea de Ciudad de México del 2026-09-23 (17.2 MB); CC BY 4.0 verificada el 2026-11-10; cita formal de dataset; sin republicar ni subir el CSV | APROBADO |
| Datos de matrícula (ANUIES) | Usarlos solo si su licencia lo permite; se verifica en la T01 y, si no, no se usan | APROBADO con condición |

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
| ML-02-RK04 | La Ciudad de México no representa a la región | Atributos de cuarto sesgados | Usar solo formas relativas de Inside Airbnb y calibrar montos con INEGI regional |
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
| Tasks completadas | 0 |
| Estado documental | **✅ APROBADA** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
