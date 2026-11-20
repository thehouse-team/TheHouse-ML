# DATOS SINTÉTICOS — Conjunto de pares estudiante-cuarto (ML-03 T06)

> **DATOS SINTÉTICOS.** Los pares combinan estudiantes sintéticos (T04) con cuartos sintéticos (T05); ninguno
> es una persona ni un anuncio real (constitución §8). Las métricas que se calculen con ellos no son
> desempeño real.

Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.pairs_stage`). Parámetros:
`src/thehouse_ml/synthetic/pairs_params.json` versión 1.0.0 (decisión M-ML-027 (APROBADO)).
Entradas: `data/processed/students.parquet` y `data/processed/rooms.parquet`. Salida: `data/processed/pairs.parquet` (fuera de git, columna `synthetic = true`). Esquema de features 2.0.0.

## 1. Resumen

- Estudiantes: **2,000**; pares: **196,605** (semilla 20261120).
- **Sin etiquetas:** el conjunto no lleva relevancia ni puntaje (ML-04 y ML-05). Columnas: `student_id`, `room_id`, `university_code`, `seed_verified`, `split`, `distance_km`, `effective_distance_km`, `price_to_budget`, `services_coverage`, `place_type_match`, `property_type_match`, `pets_compatible`, `distance_ratio`, `price_below_min`, `extra_services_ratio`, `synthetic`.
- Umbral de M-ML-021 (≥ 90 % de los estudiantes con al menos un cuarto de su campus que pasa las restricciones duras, en total y por campus, con y sin las `seed_sin_verificar`): **cumple**.
- Estudiantes sin candidatos (no generan pares): **0**.
- Estudiantes con más de 200 cuartos compatibles (muestra uniforme de 200): **420**.
- Pares con `distance_km` mayor que el `maxDistanceKm` resuelto del estudiante: **51,794** (26.3 %). La distancia no es restricción dura (M-ML-023): esos cuartos sí son candidatos y ML-04 aprende a penalizarlos.
- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017; distancias orientativas): `TESJI`, `UCAN`, `UMB-VDC`. Cada tabla trae el total con y sin ellos.

## 2. Regla de candidatos (M-ML-027)

Para cada estudiante, los candidatos son los cuartos **de su campus** que pasan las restricciones duras
(`HC01_budget`, `HC02_living_with`, `HC03_pets`; `filter_rooms`, M-ML-024), hasta 200. Si hay más,
se toma una muestra uniforme sin reemplazo con `random.Random("pairs:candidates:<seed>:<student_id>")`.
Los cuartos que no pasan las restricciones no entran. Las 10 features se calculan con
`compute_pair_features` (T02, una sola implementación) en el orden de `feature_order` del esquema.

## 3. Partición por estudiante

Entrenamiento 70 %, validación 15 % y prueba 15 %, estratificada por campus (cuotas por mayor residuo) con
`random.Random("pairs:split:<seed>:<code>")`. Cada estudiante está en una sola partición (sin fuga). La
partición no depende de los candidatos: un estudiante sin candidatos conserva su partición y no aporta pares.
Los cuartos son el catálogo, como en la app, y **sí se repiten entre particiones**: 2,751 de 2,893 cuartos con algún par aparecen en más de una partición.

| Campus | Verificado | Estudiantes `train` | Estudiantes `validation` | Estudiantes `test` | Pares `train` | Pares `validation` | Pares `test` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UTTT | sí | 140 | 30 | 30 | 12,107 | 3,261 | 2,784 |
| UAEH-ESTE | sí | 140 | 30 | 30 | 13,470 | 2,857 | 3,377 |
| UAEH-ESTL | sí | 140 | 30 | 30 | 14,810 | 3,399 | 2,718 |
| UAEH-ESAT | sí | 140 | 30 | 30 | 13,741 | 2,858 | 2,791 |
| ITSOEH | sí | 140 | 30 | 30 | 13,722 | 3,082 | 3,120 |
| TESJI | **no** | 140 | 30 | 30 | 13,982 | 2,715 | 3,214 |
| UMB-JILO | sí | 140 | 30 | 30 | 12,505 | 2,718 | 3,472 |
| UMB-VDC | **no** | 140 | 30 | 30 | 14,061 | 2,985 | 3,297 |
| UCAN | **no** | 140 | 30 | 30 | 14,255 | 2,988 | 3,690 |
| UNE-TULA | sí | 140 | 30 | 30 | 13,340 | 2,994 | 2,292 |
| **Total** | — | 1,400 | 300 | 300 | 135,993 | 29,857 | 30,755 |
| **Total sin `seed_sin_verificar`** | — | 980 | 210 | 210 | 93,695 | 21,169 | 20,554 |

## 4. Candidatos por estudiante

Número de candidatos (después del máximo de 200); percentiles con interpolación lineal.
«Compatibles máx.» es el mayor número de cuartos que pasan las restricciones antes del máximo.

| Campus | Verificado | Estudiantes | Mín. | p25 | Mediana | p75 | Máx. | Compatibles máx. | Con muestra (> 200) | Sin candidatos |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UTTT | sí | 200 | 5 | 41 | 58 | 169 | 200 | 299 | 35 | 0 |
| UAEH-ESTE | sí | 200 | 6 | 44.8 | 58 | 191 | 200 | 291 | 43 | 0 |
| UAEH-ESTL | sí | 200 | 8 | 50 | 70 | 186 | 200 | 290 | 43 | 0 |
| UAEH-ESAT | sí | 200 | 7 | 39 | 61 | 175 | 200 | 282 | 40 | 0 |
| ITSOEH | sí | 200 | 4 | 46 | 65 | 172.2 | 200 | 291 | 41 | 0 |
| TESJI | **no** | 200 | 1 | 43 | 66 | 181 | 200 | 280 | 44 | 0 |
| UMB-JILO | sí | 200 | 10 | 44 | 60 | 178.5 | 200 | 280 | 39 | 0 |
| UMB-VDC | **no** | 200 | 8 | 42.8 | 61.5 | 198.5 | 200 | 295 | 50 | 0 |
| UCAN | **no** | 200 | 8 | 52 | 71.5 | 183 | 200 | 288 | 44 | 0 |
| UNE-TULA | sí | 200 | 5 | 36 | 55 | 175 | 200 | 295 | 41 | 0 |
| **Total** | — | 2,000 | 1 | 44 | 62 | 182 | 200 | 299 | 420 | 0 |
| **Total sin `seed_sin_verificar`** | — | 1,400 | 4 | 44 | 61 | 177 | 200 | 299 | 282 | 0 |

## 5. Exclusiones por restricción dura

Pares evaluados = estudiantes por cuartos de su campus. Un cuarto que viola varias restricciones cuenta en
cada una, así que la suma por restricción puede ser mayor que los excluidos.

| Campus | Verificado | Pares evaluados | Excluidos | `HC01_budget` | `HC02_living_with` | `HC03_pets` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UTTT | sí | 60,000 | 40,461 (67.4 %) | 21,709 (36.2 %) | 26,377 (44.0 %) | 4,950 (8.2 %) |
| UAEH-ESTE | sí | 60,000 | 38,811 (64.7 %) | 22,015 (36.7 %) | 24,520 (40.9 %) | 4,256 (7.1 %) |
| UAEH-ESTL | sí | 60,000 | 37,259 (62.1 %) | 20,529 (34.2 %) | 23,490 (39.1 %) | 3,723 (6.2 %) |
| UAEH-ESAT | sí | 60,000 | 39,176 (65.3 %) | 21,631 (36.1 %) | 24,764 (41.3 %) | 4,452 (7.4 %) |
| ITSOEH | sí | 60,000 | 38,790 (64.7 %) | 22,771 (38.0 %) | 23,189 (38.6 %) | 3,870 (6.5 %) |
| TESJI | **no** | 60,000 | 38,605 (64.3 %) | 22,861 (38.1 %) | 23,401 (39.0 %) | 4,408 (7.3 %) |
| UMB-JILO | sí | 60,000 | 39,859 (66.4 %) | 20,787 (34.6 %) | 28,051 (46.8 %) | 3,519 (5.9 %) |
| UMB-VDC | **no** | 60,000 | 37,841 (63.1 %) | 20,342 (33.9 %) | 25,384 (42.3 %) | 2,940 (4.9 %) |
| UCAN | **no** | 60,000 | 37,308 (62.2 %) | 20,128 (33.5 %) | 23,436 (39.1 %) | 3,636 (6.1 %) |
| UNE-TULA | sí | 60,000 | 39,844 (66.4 %) | 21,187 (35.3 %) | 26,180 (43.6 %) | 4,991 (8.3 %) |
| **Total** | — | 600,000 | 387,954 (64.7 %) | 213,960 (35.7 %) | 248,792 (41.5 %) | 40,745 (6.8 %) |
| **Total sin `seed_sin_verificar`** | — | 420,000 | 274,200 (65.3 %) | 150,629 (35.9 %) | 176,571 (42.0 %) | 29,761 (7.1 %) |

## 6. Estudiantes con al menos un candidato

Umbral de M-ML-021: ≥ 90 % en total y en cada campus.

| Campus | Verificado | Estudiantes | Con ≥ 1 candidato | Porcentaje | ≥ 90 % |
| --- | --- | ---: | ---: | ---: | ---: |
| UTTT | sí | 200 | 200 | 100.0 % | sí |
| UAEH-ESTE | sí | 200 | 200 | 100.0 % | sí |
| UAEH-ESTL | sí | 200 | 200 | 100.0 % | sí |
| UAEH-ESAT | sí | 200 | 200 | 100.0 % | sí |
| ITSOEH | sí | 200 | 200 | 100.0 % | sí |
| TESJI | **no** | 200 | 200 | 100.0 % | sí |
| UMB-JILO | sí | 200 | 200 | 100.0 % | sí |
| UMB-VDC | **no** | 200 | 200 | 100.0 % | sí |
| UCAN | **no** | 200 | 200 | 100.0 % | sí |
| UNE-TULA | sí | 200 | 200 | 100.0 % | sí |
| **Total** | — | 2,000 | 2,000 | 100.0 % | sí |
| **Total sin `seed_sin_verificar`** | — | 1,400 | 1,400 | 100.0 % | sí |

## 7. Features de los pares

Valores calculados sobre todos los pares (sin etiqueta). `training_range` del esquema: `distance_km` y
`effective_distance_km` 0-30 km y `price_to_budget` 0-1 (los pares pasan HC01).

| Feature | Tipo | Mín. | Mediana | Máx. | Media |
| --- | --- | ---: | ---: | ---: | ---: |
| `distance_km` | number | 0.001 | 2.656 | 29.936 | 5.215 |
| `effective_distance_km` | number | 0.000 | 1.202 | 29.899 | 2.574 |
| `price_to_budget` | number | 0.036 | 0.543 | 1.000 | 0.557 |
| `services_coverage` | number | 0.000 | 0.714 | 1.000 | 0.733 |
| `place_type_match` | integer | 0.000 | 0.000 | 1.000 | 0.301 |
| `property_type_match` | integer | 0.000 | 1.000 | 1.000 | 0.700 |
| `pets_compatible` | integer | 1.000 | 1.000 | 1.000 | 1.000 |
| `distance_ratio` | number | 0.000 | 0.333 | 59.797 | 1.357 |
| `price_below_min` | number | 0.000 | 0.000 | 0.931 | 0.137 |
| `extra_services_ratio` | number | 0.000 | 0.500 | 1.000 | 0.483 |

## 8. Privacidad y limitaciones

- Ids sintéticos (`S…`, `R…`) sin datos personales (constitución §7); sin género, edad ni origen
  (constitución §11): `livingWith` es la elección del estudiante y la política del cuarto.
- Los candidatos son solo cuartos que pasan las restricciones duras: el conjunto no trae negativos por
  restricción y ML-04 decide el puntaje dentro de los compatibles.
- Las distribuciones de estudiantes y cuartos son supuestos documentados (M-ML-025 y M-ML-026); validar con
  el equipo y reemplazar con señales reales en ML-07 (riesgo RK02).
