# DATOS SINTÉTICOS — Catálogo sintético de cuartos (ML-03 T05)

> **DATOS SINTÉTICOS.** Ningún cuarto de este reporte es un anuncio real. Los cuartos se generan con
> distribuciones documentadas y semilla fija (constitución §8), calibrados con agregados públicos de INEGI;
> las métricas que se calculen con ellos no son desempeño real.

Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.rooms_stage`). Parámetros:
`src/thehouse_ml/synthetic/rooms_params.json` versión 1.0.0 (decisión M-ML-026 (APROBADO)).
Salida: `data/processed/rooms.parquet` (fuera de git, columna `synthetic = true`).
Esquema de features 2.0.0.

## 1. Resumen

- Cuartos: **3,000** (300 por campus, 10 campus; semilla 20261905).
- Ids sintéticos sin datos personales (`R00001` a `R03000`); sin títulos, descripciones, nombres ni datos de anfitriones.
- Sin ningún campo de género (constitución §11): `livingWith` es la política que declara el cuarto.
- Todos a ≤ 30 km de su campus (distancia haversine, radio 6,371 km).
- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017; sus distancias son orientativas): `TESJI`, `UMB-VDC`, `UCAN`.
- Rentas en pesos de julio de 2024 (M-ML-014), igual que la renta regional y el presupuesto de los
  estudiantes.

## 2. Calibración de la renta

La forma de la renta es el histograma ENVI 2020 del estado del campus (`rent_histogram.parquet`), escalado
para que su mediana sea la renta p50 regional (`regional_calibration.parquet`, indicador `renta`; el nivel
indica el respaldo de M-ML-015). Los multiplicadores por `placeType` se normalizan para que la mediana
general de cada campus siga en esa p50; tolerancia ±10 %.

| Campus | Municipio | Nivel | Renta p50 regional | p25 | p50 | p75 | Desviación p50 | Dentro |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| UTTT | Tula de Allende | estado | $1,872 | $1,200 | $1,925 | $3,100 | +2.8 % | sí |
| UAEH-ESTE | Tepeji del Río de Ocampo | estado | $1,872 | $1,200 | $1,850 | $3,050 | -1.2 % | sí |
| UAEH-ESTL | Tlahuelilpan | estado | $1,872 | $1,150 | $1,800 | $2,925 | -3.9 % | sí |
| UAEH-ESAT | Atotonilco de Tula | estado | $1,872 | $1,150 | $1,850 | $2,962 | -1.2 % | sí |
| ITSOEH | Mixquiahuala de Juárez | estado | $1,872 | $1,200 | $1,900 | $3,012 | +1.5 % | sí |
| TESJI | Jilotepec | estado | $2,496 | $1,650 | $2,500 | $4,012 | +0.2 % | sí |
| UMB-JILO | Jilotepec | estado | $2,496 | $1,650 | $2,500 | $4,062 | +0.2 % | sí |
| UMB-VDC | Villa del Carbón | estado | $2,496 | $1,550 | $2,400 | $3,900 | -3.9 % | sí |
| UCAN | Tula de Allende | estado | $1,872 | $1,150 | $1,850 | $3,012 | -1.2 % | sí |
| UNE-TULA | Tula de Allende | estado | $1,872 | $1,150 | $1,850 | $2,962 | -1.2 % | sí |

## 3. Renta por tipo de lugar (`placeType`)

La renta de la ENVI es de vivienda completa, no de cuarto; cada `placeType` aplica un multiplicador
SUPUESTO (orden `shared` < `room` < `room_private_bath` < `studio`). El multiplicador efectivo ya incluye el
factor común que mantiene la mediana general del campus en la renta p50 regional.

| `placeType` | Multiplicador | Efectivo (entidad 13) | Efectivo (entidad 15) |
| --- | ---: | ---: | ---: |
| `shared` | 0.60 | 0.622 | 0.622 |
| `room` | 0.90 | 0.933 | 0.933 |
| `room_private_bath` | 1.15 | 1.192 | 1.192 |
| `studio` | 1.50 | 1.555 | 1.554 |

Mediana de la renta por campus y `placeType`:

| Campus | `shared` | `room` | `room_private_bath` | `studio` | Orden |
| --- | ---: | ---: | ---: | ---: | --- |
| UTTT | $1,150 | $1,750 | $2,225 | $2,950 | sí |
| UAEH-ESTE | $1,175 | $1,750 | $2,250 | $2,900 | sí |
| UAEH-ESTL | $1,150 | $1,750 | $2,225 | $2,900 | sí |
| UAEH-ESAT | $1,150 | $1,750 | $2,250 | $2,950 | sí |
| ITSOEH | $1,150 | $1,750 | $2,250 | $2,925 | sí |
| TESJI | $1,550 | $2,350 | $2,950 | $3,925 | sí |
| UMB-JILO | $1,575 | $2,300 | $2,975 | $3,850 | sí |
| UMB-VDC | $1,550 | $2,300 | $3,000 | $3,900 | sí |
| UCAN | $1,150 | $1,750 | $2,250 | $2,925 | sí |
| UNE-TULA | $1,150 | $1,750 | $2,250 | $2,925 | sí |
| **Total** | $1,250 | $1,900 | $2,450 | $3,150 | sí |

## 4. Distancia al campus

Rumbo uniforme y distancia por bandas SUPUESTAS (uniforme dentro de cada banda). Probabilidad de cada
banda: 0-1 km 25 % · 1-2 km 20 % · 2-5 km 25 % · 5-10 km 15 % · 10-20 km 10 % · 20-30 km 5 %.

| Campus | Verificado | 0-1 km | 1-2 km | 2-5 km | 5-10 km | 10-20 km | 20-30 km | p50 | Máxima |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UTTT | sí | 30.7 % | 19.3 % | 17.7 % | 15.3 % | 12.7 % | 4.3 % | 1.99 km | 29.04 km |
| UAEH-ESTE | sí | 24.3 % | 21.0 % | 25.7 % | 14.0 % | 8.7 % | 6.3 % | 2.33 km | 29.94 km |
| UAEH-ESTL | sí | 25.0 % | 19.0 % | 25.7 % | 15.0 % | 12.3 % | 3.0 % | 2.82 km | 29.76 km |
| UAEH-ESAT | sí | 27.0 % | 19.0 % | 27.0 % | 12.7 % | 11.7 % | 2.7 % | 2.41 km | 29.66 km |
| ITSOEH | sí | 28.7 % | 18.0 % | 22.7 % | 15.7 % | 11.7 % | 3.3 % | 2.45 km | 28.32 km |
| TESJI | **no** | 24.3 % | 22.7 % | 26.3 % | 14.3 % | 7.0 % | 5.3 % | 2.29 km | 29.00 km |
| UMB-JILO | sí | 24.3 % | 18.7 % | 24.3 % | 16.3 % | 11.0 % | 5.3 % | 2.68 km | 29.90 km |
| UMB-VDC | **no** | 24.7 % | 14.7 % | 27.0 % | 17.7 % | 10.7 % | 5.3 % | 3.05 km | 29.71 km |
| UCAN | **no** | 24.7 % | 18.0 % | 27.0 % | 17.0 % | 8.7 % | 4.7 % | 2.70 km | 29.62 km |
| UNE-TULA | sí | 22.0 % | 19.0 % | 27.0 % | 15.3 % | 11.0 % | 5.7 % | 2.85 km | 29.24 km |
| **Total** | — | 25.6 % | 18.9 % | 25.0 % | 15.3 % | 10.5 % | 4.6 % | 2.59 km | 29.94 km |

## 5. Servicios frente al Censo 2020

Agua, electricidad, gas e internet (→ `wifi`) siguen la proporción ponderada del Censo 2020 del municipio
del campus (`regional_calibration.parquet`; drenaje no tiene equivalente en NV-11). Tolerancia: ±3 errores estándar binomiales con n = 300 (mínimo ±2 puntos).

| Campus | Servicio | Indicador del Censo | Nivel | Censo | Generado | Tolerancia | Dentro |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| UTTT | `water` | `agua_entubada` | municipio | 98.6 % | 99.0 % | ±2.0 | sí |
| UTTT | `electricity` | `electricidad` | municipio | 99.3 % | 99.0 % | ±2.0 | sí |
| UTTT | `gas` | `gas` | municipio | 95.5 % | 96.3 % | ±3.6 | sí |
| UTTT | `wifi` | `internet` | municipio | 46.9 % | 42.0 % | ±8.6 | sí |
| UAEH-ESTE | `water` | `agua_entubada` | municipio | 98.8 % | 99.7 % | ±2.0 | sí |
| UAEH-ESTE | `electricity` | `electricidad` | municipio | 99.5 % | 100.0 % | ±2.0 | sí |
| UAEH-ESTE | `gas` | `gas` | municipio | 92.6 % | 93.0 % | ±4.5 | sí |
| UAEH-ESTE | `wifi` | `internet` | municipio | 37.6 % | 36.3 % | ±8.4 | sí |
| UAEH-ESTL | `water` | `agua_entubada` | municipio | 99.5 % | 99.0 % | ±2.0 | sí |
| UAEH-ESTL | `electricity` | `electricidad` | municipio | 99.5 % | 99.3 % | ±2.0 | sí |
| UAEH-ESTL | `gas` | `gas` | municipio | 96.6 % | 94.7 % | ±3.2 | sí |
| UAEH-ESTL | `wifi` | `internet` | municipio | 42.6 % | 42.0 % | ±8.6 | sí |
| UAEH-ESAT | `water` | `agua_entubada` | municipio | 98.2 % | 98.0 % | ±2.3 | sí |
| UAEH-ESAT | `electricity` | `electricidad` | municipio | 99.9 % | 100.0 % | ±2.0 | sí |
| UAEH-ESAT | `gas` | `gas` | municipio | 94.8 % | 95.3 % | ±3.9 | sí |
| UAEH-ESAT | `wifi` | `internet` | municipio | 44.7 % | 44.7 % | ±8.6 | sí |
| ITSOEH | `water` | `agua_entubada` | municipio | 96.4 % | 96.7 % | ±3.2 | sí |
| ITSOEH | `electricity` | `electricidad` | municipio | 97.4 % | 99.0 % | ±2.7 | sí |
| ITSOEH | `gas` | `gas` | municipio | 91.2 % | 94.0 % | ±4.9 | sí |
| ITSOEH | `wifi` | `internet` | municipio | 43.0 % | 40.0 % | ±8.6 | sí |
| TESJI | `water` | `agua_entubada` | municipio | 93.0 % | 95.3 % | ±4.4 | sí |
| TESJI | `electricity` | `electricidad` | municipio | 97.8 % | 99.3 % | ±2.5 | sí |
| TESJI | `gas` | `gas` | municipio | 90.2 % | 90.3 % | ±5.1 | sí |
| TESJI | `wifi` | `internet` | municipio | 35.1 % | 36.0 % | ±8.3 | sí |
| UMB-JILO | `water` | `agua_entubada` | municipio | 93.0 % | 90.3 % | ±4.4 | sí |
| UMB-JILO | `electricity` | `electricidad` | municipio | 97.8 % | 97.7 % | ±2.5 | sí |
| UMB-JILO | `gas` | `gas` | municipio | 90.2 % | 87.0 % | ±5.1 | sí |
| UMB-JILO | `wifi` | `internet` | municipio | 35.1 % | 33.7 % | ±8.3 | sí |
| UMB-VDC | `water` | `agua_entubada` | municipio | 97.7 % | 98.0 % | ±2.6 | sí |
| UMB-VDC | `electricity` | `electricidad` | municipio | 97.5 % | 98.7 % | ±2.7 | sí |
| UMB-VDC | `gas` | `gas` | municipio | 72.7 % | 77.3 % | ±7.7 | sí |
| UMB-VDC | `wifi` | `internet` | municipio | 18.8 % | 18.3 % | ±6.8 | sí |
| UCAN | `water` | `agua_entubada` | municipio | 98.6 % | 98.0 % | ±2.0 | sí |
| UCAN | `electricity` | `electricidad` | municipio | 99.3 % | 99.0 % | ±2.0 | sí |
| UCAN | `gas` | `gas` | municipio | 95.5 % | 94.3 % | ±3.6 | sí |
| UCAN | `wifi` | `internet` | municipio | 46.9 % | 45.7 % | ±8.6 | sí |
| UNE-TULA | `water` | `agua_entubada` | municipio | 98.6 % | 99.0 % | ±2.0 | sí |
| UNE-TULA | `electricity` | `electricidad` | municipio | 99.3 % | 99.3 % | ±2.0 | sí |
| UNE-TULA | `gas` | `gas` | municipio | 95.5 % | 96.3 % | ±3.6 | sí |
| UNE-TULA | `wifi` | `internet` | municipio | 46.9 % | 50.0 % | ±8.6 | sí |

Servicios que el Censo no mide: probabilidad SUPUESTA por `placeType` y proporción generada en todo el
catálogo.

| Servicio | `shared` (prior / generado) | `room` (prior / generado) | `room_private_bath` (prior / generado) | `studio` (prior / generado) | Total generado |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hot_water` | 75 % / 78.2 % | 75 % / 76.9 % | 85 % / 86.7 % | 85 % / 85.7 % | 80.9 % |
| `laundry` | 50 % / 47.8 % | 45 % / 44.6 % | 45 % / 46.9 % | 35 % / 33.9 % | 44.3 % |
| `kitchen` | 80 % / 75.9 % | 60 % / 58.8 % | 60 % / 61.2 % | 90 % / 88.9 % | 67.3 % |
| `furnished` | 70 % / 69.4 % | 55 % / 56.6 % | 60 % / 60.2 % | 50 % / 50.9 % | 59.3 % |
| `parking` | 20 % / 20.0 % | 25 % / 25.6 % | 30 % / 30.2 % | 35 % / 32.1 % | 26.5 % |
| `security` | 30 % / 25.2 % | 30 % / 32.1 % | 35 % / 34.7 % | 40 % / 37.1 % | 32.1 % |
| `cleaning` | 20 % / 19.5 % | 15 % / 15.3 % | 20 % / 21.4 % | 10 % / 10.6 % | 17.0 % |

## 6. Tipo de lugar (`placeType`)

| Campus | `room` | `room_private_bath` | `studio` | `shared` |
| --- | ---: | ---: | ---: | ---: |
| UTTT | 40.0 % | 26.7 % | 16.3 % | 17.0 % |
| UAEH-ESTE | 42.0 % | 23.7 % | 15.0 % | 19.3 % |
| UAEH-ESTL | 41.3 % | 21.3 % | 14.3 % | 23.0 % |
| UAEH-ESAT | 41.7 % | 21.0 % | 15.7 % | 21.7 % |
| ITSOEH | 40.0 % | 29.0 % | 14.0 % | 17.0 % |
| TESJI | 42.3 % | 24.3 % | 14.7 % | 18.7 % |
| UMB-JILO | 37.7 % | 25.3 % | 17.0 % | 20.0 % |
| UMB-VDC | 37.7 % | 23.7 % | 13.0 % | 25.7 % |
| UCAN | 43.3 % | 22.3 % | 14.0 % | 20.3 % |
| UNE-TULA | 38.7 % | 25.7 % | 13.3 % | 22.3 % |
| **Total** | 40.5 % | 24.3 % | 14.7 % | 20.5 % |

## 7. Tipo de inmueble (`propertyType`)

| Campus | `house` | `apartment` | `student_residence` |
| --- | ---: | ---: | ---: |
| UTTT | 50.0 % | 30.0 % | 20.0 % |
| UAEH-ESTE | 48.0 % | 31.7 % | 20.3 % |
| UAEH-ESTL | 51.3 % | 34.0 % | 14.7 % |
| UAEH-ESAT | 47.0 % | 31.7 % | 21.3 % |
| ITSOEH | 54.3 % | 28.7 % | 17.0 % |
| TESJI | 53.0 % | 31.0 % | 16.0 % |
| UMB-JILO | 51.0 % | 32.0 % | 17.0 % |
| UMB-VDC | 55.0 % | 33.3 % | 11.7 % |
| UCAN | 53.0 % | 30.0 % | 17.0 % |
| UNE-TULA | 47.3 % | 35.0 % | 17.7 % |
| **Total** | 51.0 % | 31.7 % | 17.3 % |

## 8. Mascotas (`pets_allowed`)

| Campus | `true` | `false` |
| --- | ---: | ---: |
| UTTT | 25.0 % | 75.0 % |
| UAEH-ESTE | 25.3 % | 74.7 % |
| UAEH-ESTL | 27.0 % | 73.0 % |
| UAEH-ESAT | 29.3 % | 70.7 % |
| ITSOEH | 28.3 % | 71.7 % |
| TESJI | 22.7 % | 77.3 % |
| UMB-JILO | 31.0 % | 69.0 % |
| UMB-VDC | 30.0 % | 70.0 % |
| UCAN | 32.7 % | 67.3 % |
| UNE-TULA | 27.7 % | 72.3 % |
| **Total** | 27.9 % | 72.1 % |

## 9. Política de convivencia (`livingWith`)

El cuarto declara quién vive ahí. Con HC02 (M-ML-024) los estudiantes `women` y `men` solo ven cuartos de
su mismo valor, así que cada campus necesita al menos 40 cuartos `women` y 40 `men`
(nota para T05, M-ML-025).

| Campus | `women` | `men` | `any` |
| --- | ---: | ---: | ---: |
| UTTT | 66 (22.0 %) | 79 (26.3 %) | 155 (51.7 %) |
| UAEH-ESTE | 82 (27.3 %) | 72 (24.0 %) | 146 (48.7 %) |
| UAEH-ESTL | 77 (25.7 %) | 84 (28.0 %) | 139 (46.3 %) |
| UAEH-ESAT | 63 (21.0 %) | 79 (26.3 %) | 158 (52.7 %) |
| ITSOEH | 79 (26.3 %) | 74 (24.7 %) | 147 (49.0 %) |
| TESJI | 85 (28.3 %) | 72 (24.0 %) | 143 (47.7 %) |
| UMB-JILO | 67 (22.3 %) | 83 (27.7 %) | 150 (50.0 %) |
| UMB-VDC | 77 (25.7 %) | 69 (23.0 %) | 154 (51.3 %) |
| UCAN | 80 (26.7 %) | 78 (26.0 %) | 142 (47.3 %) |
| UNE-TULA | 62 (20.7 %) | 62 (20.7 %) | 176 (58.7 %) |
| **Total** | 738 (24.6 %) | 752 (25.1 %) | 1510 (50.3 %) |

## 10. Recámaras, baños y superficie

Recámaras y baños son del inmueble; la superficie es la del espacio que se ofrece. Rangos SUPUESTOS del
esquema (M-ML-022): recámaras 1-10, baños 1-6, superficie 6-400 m².

| `placeType` | n | Recámaras (mín-máx) | Baños (mín-máx) | Superficie p50 | Superficie (mín-máx) |
| --- | ---: | --- | --- | ---: | --- |
| `shared` | 615 | 2-5 | 1-3 | 18 m² | 12-24 m² |
| `room` | 1214 | 2-5 | 1-3 | 12 m² | 9-16 m² |
| `room_private_bath` | 729 | 2-5 | 2-4 | 17 m² | 12-22 m² |
| `studio` | 442 | 1-1 | 1-1 | 32 m² | 20-45 m² |

## 11. Privacidad y limitaciones

- Sin nombres, títulos, descripciones, fotos ni identificadores reales; los ids son consecutivos
  (constitución §7). Sin género, edad ni origen (constitución §11).
- La renta regional es estatal (no municipal) y de vivienda completa (no de cuarto): los multiplicadores
  por `placeType` son SUPUESTOS y las proporciones del Censo son de todas las viviendas del municipio,
  no solo de las rentadas.
- Ubicación, tipo de lugar, tipo de inmueble, mascotas, convivencia y los servicios que el Censo no mide
  son supuestos documentados en `rooms_params.json`; validar con el equipo y reemplazar con señales
  reales en ML-07 (riesgo RK02).
