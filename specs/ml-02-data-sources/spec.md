# ML-02 — Datos: fuentes, licencias y preparación

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-01 completa; fuentes de M-ML-003 aprobadas |
| Relación con el backend | Produce las coordenadas verificadas de las 10 universidades que necesita el seed de NV-11 |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

No hay un dataset público de estudiantes y cuartos en México. Inside Airbnb publica listados de la Ciudad de México con servicios, tipo de cuarto, precio y ubicación; OpenStreetMap tiene la ubicación de los campus; el seed de TheHouse da los rangos regionales de precio de Tula–Tepeji–Jilotepec.

## 2. Objetivo

Descargar, verificar y transformar las fuentes permitidas en un catálogo de cuartos sin datos personales y en un catálogo de universidades con coordenadas verificadas.

## 3. Precondiciones

1. ML-01 completa.
2. Licencia de Inside Airbnb verificada en su propia página y registrada en M-ML-003.
3. Autorización «EJECUTA».

## 4. Alcance

1. `download`: descarga de `listings` de Inside Airbnb (Ciudad de México, una sola instantánea fechada) con manifiesto SHA-256.
2. Catálogo de universidades: las 10 del seed de TheHouse con coordenadas del campus tomadas de OpenStreetMap (id del elemento OSM) y verificadas a mano, más campus de la Ciudad de México para medir distancias reales.
3. `prepare`: limpieza, eliminación de datos personales y mapeo al catálogo de servicios de NV-11.
4. Reescalado de precios a rentas mensuales regionales usando el seed de TheHouse y rangos documentados.
5. Data card con fuentes, licencias, atribuciones, columnas, transformaciones y limitaciones.

## 5. Fuera de alcance

- Scraping de portales inmobiliarios (M-ML-004).
- Datos de usuarios reales (ML-07).
- Republicar los datos crudos o procesados.

## 6. Diseño

- Columnas conservadas de Inside Airbnb: `room_type`, `property_type`, `accommodates`, `bedrooms`, `bathrooms_text`, `amenities`, `price`, `latitude`, `longitude`, `neighbourhood_cleansed`. Se eliminan nombres, descripciones, URLs, fotos y todo dato del anfitrión.
- Coordenadas redondeadas a 3 decimales (unos 100 m).
- Mapeo de `amenities` al catálogo de NV-11: `wifi`, `water`, `electricity`, `gas`, `hot_water`, `laundry`, `kitchen`, `furnished`, `parking`, `security`, `cleaning`; tabla de mapeo versionada en el repositorio.
- `room_type`/`property_type` → `placeType` (`room`, `room_private_bath`, `studio`, `shared`) y `propertyTypes` (`house`, `apartment`, `student_residence`).
- Precio: se usa el percentil del precio dentro de su tipo y se convierte a renta mensual con la distribución regional; el precio por noche nunca se usa como renta.
- Distancias: distribución de la distancia de cada listado al campus más cercano de la Ciudad de México, reutilizada para ubicar cuartos alrededor de las universidades regionales.
- Salida en Parquet en `data/processed/` con su manifiesto.

## 7. Reglas

1. Una sola instantánea por versión de datos; cambiarla crea una versión nueva.
2. Toda fila con datos personales residuales se descarta, no se corrige.
3. Las coordenadas de universidades se revisan una por una contra el mapa antes de entregarlas al backend.

## 8. Seguridad, datos y privacidad

- Ningún dato crudo o procesado entra a git.
- Atribución a Inside Airbnb (CC BY 4.0) y a OpenStreetMap (ODbL) en la data card y el README.
- Se respeta la política de Inside Airbnb: una descarga, sin scraping del sitio.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-02-R01 | Descarga verificable | `download` obtiene la instantánea definida, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores. | F2 | T01 |
| ML-02-R02 | Universidades con coordenadas | Las 10 universidades del seed tienen latitud, longitud e id OSM verificados a mano; se entregan en un CSV versionado para el seed de NV-11. | F2 | T02 |
| ML-02-R03 | Preparación sin datos personales | `prepare` produce el catálogo de cuartos sin columnas personales (prueba automática de columnas prohibidas) y con coordenadas redondeadas. | F2 | T03 |
| ML-02-R04 | Mapeo a catálogos de NV-11 | Cada cuarto tiene `placeType`, `propertyTypes` y servicios del catálogo cerrado; los valores no mapeados se cuentan y se reportan. | F2 | T04 |
| ML-02-R05 | Precios y distancias regionales | Rentas mensuales dentro de los rangos regionales documentados y distancias con la distribución medida; histogramas en la data card. | F3 | T05 |
| ML-02-R06 | Data card y licencias | `reports/data_card.md` con fuentes, licencias, atribuciones, transformaciones, conteos y limitaciones. | F3 | T06 |
| ML-02-R07 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T07 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Instantánea de Inside Airbnb | La más reciente de la Ciudad de México al momento de ejecutar | PROPUESTO |
| Licencia de Inside Airbnb | CC BY 4.0, verificada en su página antes de descargar | PENDIENTE |
| Rangos regionales de renta | Derivados del seed de TheHouse y validados por el equipo | PROPUESTO |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-02-RK01 | La Ciudad de México no representa a Tula–Tepeji | Distribuciones sesgadas | Usar solo formas relativas (percentiles, co-ocurrencia de servicios) y reescalar con datos regionales |
| ML-02-RK02 | Licencia distinta a la esperada | Fuente no utilizable | Verificar antes de descargar; si no es compatible, BLOCKED y solo datos sintéticos |
| ML-02-RK03 | Coordenadas equivocadas | Distancias erróneas en backend y modelo | Verificación manual punto por punto |

## 13. Dependencias

- ML-01.
- Seed de TheHouse (NV-07, NV-08).

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 7 |
| Tasks | 7 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
