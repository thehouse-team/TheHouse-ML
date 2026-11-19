# DATOS SINTÉTICOS — Estudiantes sintéticos (ML-03 T04)

> **DATOS SINTÉTICOS.** Ningún perfil de este reporte corresponde a una persona real. Los estudiantes se
> generan con distribuciones documentadas y semilla fija (constitución §8); las métricas que se calculen
> con ellos no son desempeño real.

Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.students_stage`). Parámetros:
`src/thehouse_ml/synthetic/students_params.json` versión 1.0.0 (decisión M-ML-025 (APROBADO)).
Salida: `data/processed/students.parquet` (fuera de git, columna `synthetic = true`).
Esquema de features 1.1.0.

## 1. Resumen

- Estudiantes: **2,000** (200 por campus, 10 campus; semilla 20261119).
- Ids sintéticos sin datos personales (`S0001` a `S2000`).
- Sin ningún campo de género (constitución §11): `livingWith` es la elección del estudiante.
- Quick setup (`budget.max`, `livingWith`, `placeType`) nunca nulo; los opcionales llevan nulos para ejercitar las reglas para faltantes (M-ML-022).
- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017): `TESJI`, `UMB-VDC`, `UCAN`.
- Presupuestos en pesos de julio de 2024 (M-ML-014), igual que la renta regional.

## 2. Calibración del presupuesto

Mediana objetivo de `budget.max` = renta p50 regional por 1.3 (`budget.max.median_factor`); tolerancia ±10 %. La renta p50 sale de
`data/processed/regional_calibration.parquet` (indicador `renta`, ENVI 2020 deflactada); el nivel indica el
respaldo de M-ML-015 (`estado` = entidad completa).

| Campus | Municipio | Nivel | Renta p50 regional | Objetivo | Mediana budget.max | Desviación | Dentro |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| UTTT | Tula de Allende | estado | $1,872 | $2,434 | $2,500 | +2.7 % | sí |
| UAEH-ESTE | Tepeji del Río de Ocampo | estado | $1,872 | $2,434 | $2,400 | -1.4 % | sí |
| UAEH-ESTL | Tlahuelilpan | estado | $1,872 | $2,434 | $2,400 | -1.4 % | sí |
| UAEH-ESAT | Atotonilco de Tula | estado | $1,872 | $2,434 | $2,400 | -1.4 % | sí |
| ITSOEH | Mixquiahuala de Juárez | estado | $1,872 | $2,434 | $2,400 | -1.4 % | sí |
| TESJI | Jilotepec | estado | $2,496 | $3,245 | $3,300 | +1.7 % | sí |
| UMB-JILO | Jilotepec | estado | $2,496 | $3,245 | $3,450 | +6.3 % | sí |
| UMB-VDC | Villa del Carbón | estado | $2,496 | $3,245 | $3,400 | +4.8 % | sí |
| UCAN | Tula de Allende | estado | $1,872 | $2,434 | $2,600 | +6.8 % | sí |
| UNE-TULA | Tula de Allende | estado | $1,872 | $2,434 | $2,400 | -1.4 % | sí |

## 3. Presupuesto por universidad

| Campus | Verificado | n | budget.max p25 | budget.max p50 | budget.max p75 | budget.min p50 | budget.min nulo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UTTT | sí | 200 | $1,900 | $2,500 | $3,200 | $1,400 | 27.5 % |
| UAEH-ESTE | sí | 200 | $1,900 | $2,400 | $3,000 | $1,200 | 28.5 % |
| UAEH-ESTL | sí | 200 | $1,900 | $2,400 | $3,200 | $1,300 | 30.5 % |
| UAEH-ESAT | sí | 200 | $1,900 | $2,400 | $3,000 | $1,400 | 27.0 % |
| ITSOEH | sí | 200 | $1,900 | $2,400 | $3,100 | $1,300 | 27.0 % |
| TESJI | **no** | 200 | $2,500 | $3,300 | $4,100 | $1,800 | 28.5 % |
| UMB-JILO | sí | 200 | $2,800 | $3,450 | $4,300 | $2,000 | 26.5 % |
| UMB-VDC | **no** | 200 | $2,600 | $3,400 | $4,200 | $1,900 | 28.5 % |
| UCAN | **no** | 200 | $1,900 | $2,600 | $3,200 | $1,400 | 34.0 % |
| UNE-TULA | sí | 200 | $1,900 | $2,400 | $3,200 | $1,400 | 26.5 % |

## 4. Transporte

| Campus | `walk` | `bike` | `public` | `car` | nulo |
| --- | ---: | ---: | ---: | ---: | ---: |
| UTTT | 18.0 % | 5.0 % | 56.0 % | 12.0 % | 9.0 % |
| UAEH-ESTE | 14.0 % | 7.0 % | 57.0 % | 14.0 % | 8.0 % |
| UAEH-ESTL | 18.5 % | 6.5 % | 54.0 % | 12.5 % | 8.5 % |
| UAEH-ESAT | 17.0 % | 4.0 % | 56.0 % | 14.5 % | 8.5 % |
| ITSOEH | 16.0 % | 2.0 % | 59.0 % | 12.0 % | 11.0 % |
| TESJI | 16.5 % | 5.5 % | 55.5 % | 13.0 % | 9.5 % |
| UMB-JILO | 14.5 % | 6.5 % | 58.0 % | 12.0 % | 9.0 % |
| UMB-VDC | 18.5 % | 3.5 % | 54.0 % | 11.5 % | 12.5 % |
| UCAN | 16.5 % | 2.5 % | 58.5 % | 13.5 % | 9.0 % |
| UNE-TULA | 23.0 % | 4.0 % | 48.5 % | 13.0 % | 11.5 % |
| **Total** | 17.2 % | 4.7 % | 55.6 % | 12.8 % | 9.7 % |

## 5. Con quién vivir (`livingWith`)

| Campus | `women` | `men` | `any` |
| --- | ---: | ---: | ---: |
| UTTT | 28.5 % | 29.5 % | 42.0 % |
| UAEH-ESTE | 28.0 % | 27.0 % | 45.0 % |
| UAEH-ESTL | 27.0 % | 26.5 % | 46.5 % |
| UAEH-ESAT | 28.0 % | 26.0 % | 46.0 % |
| ITSOEH | 31.5 % | 20.5 % | 48.0 % |
| TESJI | 29.5 % | 23.5 % | 47.0 % |
| UMB-JILO | 42.5 % | 19.0 % | 38.5 % |
| UMB-VDC | 30.5 % | 25.5 % | 44.0 % |
| UCAN | 24.0 % | 29.0 % | 47.0 % |
| UNE-TULA | 32.5 % | 22.5 % | 45.0 % |
| **Total** | 30.2 % | 24.9 % | 44.9 % |

## 6. Tipo de lugar (`placeType`)

| Campus | `room` | `room_private_bath` | `studio` | `shared` |
| --- | ---: | ---: | ---: | ---: |
| UTTT | 39.5 % | 25.5 % | 18.5 % | 16.5 % |
| UAEH-ESTE | 44.0 % | 22.0 % | 14.5 % | 19.5 % |
| UAEH-ESTL | 44.0 % | 24.0 % | 18.0 % | 14.0 % |
| UAEH-ESAT | 47.5 % | 28.0 % | 15.0 % | 9.5 % |
| ITSOEH | 46.0 % | 20.5 % | 19.5 % | 14.0 % |
| TESJI | 53.0 % | 24.5 % | 13.0 % | 9.5 % |
| UMB-JILO | 40.5 % | 26.5 % | 14.0 % | 19.0 % |
| UMB-VDC | 40.5 % | 28.0 % | 12.5 % | 19.0 % |
| UCAN | 48.0 % | 18.5 % | 15.0 % | 18.5 % |
| UNE-TULA | 42.0 % | 29.5 % | 15.0 % | 13.5 % |
| **Total** | 44.5 % | 24.7 % | 15.5 % | 15.3 % |

## 7. Servicios esenciales

Porcentaje de estudiantes que marca cada servicio como esencial (sobre el total del campus; los nulos
cuentan como no marcado y se resuelven a conjunto vacío).

| Campus | `wifi` | `water` | `electricity` | `gas` | `hot_water` | `laundry` | `kitchen` | `furnished` | `parking` | `security` | `cleaning` | nulo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UTTT | 76.5 % | 80.0 % | 77.5 % | 52.0 % | 54.5 % | 31.0 % | 44.0 % | 30.0 % | 14.5 % | 39.5 % | 16.5 % | 11.5 % |
| UAEH-ESTE | 77.0 % | 83.5 % | 81.0 % | 52.5 % | 63.0 % | 29.0 % | 38.0 % | 38.0 % | 18.0 % | 36.0 % | 15.5 % | 9.5 % |
| UAEH-ESTL | 74.0 % | 78.5 % | 76.5 % | 56.0 % | 58.0 % | 32.5 % | 55.5 % | 36.0 % | 15.0 % | 39.5 % | 14.5 % | 10.0 % |
| UAEH-ESAT | 69.5 % | 80.0 % | 77.5 % | 54.5 % | 61.5 % | 34.0 % | 46.5 % | 37.5 % | 17.5 % | 38.0 % | 12.0 % | 10.5 % |
| ITSOEH | 79.5 % | 83.0 % | 80.0 % | 52.5 % | 66.0 % | 33.0 % | 50.5 % | 37.5 % | 14.0 % | 41.0 % | 9.0 % | 8.5 % |
| TESJI | 82.0 % | 84.0 % | 83.0 % | 60.5 % | 59.0 % | 32.0 % | 44.5 % | 38.0 % | 17.5 % | 46.0 % | 16.5 % | 6.0 % |
| UMB-JILO | 78.0 % | 83.0 % | 83.0 % | 57.0 % | 52.5 % | 36.0 % | 46.0 % | 34.0 % | 14.5 % | 45.0 % | 13.5 % | 7.5 % |
| UMB-VDC | 74.5 % | 81.5 % | 80.0 % | 53.5 % | 57.5 % | 34.5 % | 41.5 % | 36.5 % | 12.5 % | 43.0 % | 13.5 % | 10.0 % |
| UCAN | 78.5 % | 82.5 % | 82.0 % | 57.0 % | 61.5 % | 30.5 % | 43.5 % | 34.0 % | 11.5 % | 43.0 % | 12.0 % | 9.0 % |
| UNE-TULA | 77.0 % | 78.0 % | 77.0 % | 53.5 % | 56.5 % | 30.0 % | 48.0 % | 31.5 % | 14.5 % | 40.0 % | 10.0 % | 12.0 % |
| **Total** | 76.7 % | 81.4 % | 79.8 % | 54.9 % | 59.0 % | 32.2 % | 45.8 % | 35.3 % | 14.9 % | 41.1 % | 13.3 % | 9.4 % |

## 8. Nulos

Proporción de nulos por campo frente a la probabilidad de `students_params.json`. Los del quick setup
deben ser 0.

| Campo | Probabilidad (parámetro) | Observado | Regla para faltantes |
| --- | ---: | ---: | --- |
| `budget.min` | 30.0 % | 28.4 % | budget.min nulo → 0 (sin piso de presupuesto). |
| `budget.max` | 0.0 % | 0.0 % | Quick setup: budget.max nulo → no se calcula el puntaje (perfil incompleto). |
| `maxDistanceKm` | 15.0 % | 14.7 % | maxDistanceKm nulo → 30 (máximo de NV-11). |
| `essentialServices` | 10.0 % | 9.4 % | essentialServices nulo → conjunto vacío (no exige servicios). |
| `livingWith` | 0.0 % | 0.0 % | Quick setup: livingWith nulo → no se calcula el puntaje (perfil incompleto). |
| `placeType` | 0.0 % | 0.0 % | Quick setup: placeType nulo → no se calcula el puntaje (perfil incompleto). |
| `pets` | 20.0 % | 20.2 % | pets nulo → false (no lleva mascota). |
| `transport` | 10.0 % | 9.7 % | transport nulo → "public". |
| `propertyTypes` | 25.0 % | 24.0 % | propertyTypes nulo o vacío → acepta todos los tipos de inmueble. |

## 9. Hallazgo: estudiantes limitados por `livingWith`

1,102 de 2,000 estudiantes (55.1 %) piden `women` o `men`. Con HC02
(M-ML-024) solo ven cuartos con el mismo `livingWith`: **no ven cuartos `any`**. Por eso la
distribución de `livingWith` de los cuartos (T05) debe tener suficientes cuartos `women` y `men`
en cada campus para el umbral de ≥ 90 % de T06; la recomendación está en `evidence.md` de ML-03 (T04).

## 10. Privacidad

- Sin nombres, correos, teléfonos ni identificadores reales; los ids son consecutivos (constitución §7).
- Sin género, edad ni origen (constitución §11).
