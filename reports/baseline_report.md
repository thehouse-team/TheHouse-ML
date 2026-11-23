# DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS — Reporte del modelo base baseline-1.0.0 (esquema 2.0.0)

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Estudiantes, cuartos y etiquetas son **sintéticos**. Las
> etiquetas de relevancia **no son preferencias reales**: salen de una utilidad oculta documentada más ruido
> (constitución §8.2, M-ML-032), con riesgo de **circularidad** (ML-04-RK02). Ninguna métrica de este reporte
> es desempeño real (constitución §20.3); se reemplazan con señales reales en ML-07.

Generado por `uv run python -m thehouse_ml evaluate` (`thehouse_ml.models.baseline_report`). Modelo `baseline-1.0.0` (budget 30 · distance 25 · essential_services 20 · place_type 10 · property_type 10 · extras 5); esquema de features 2.0.0; etiquetas `labels_params.json` 1.0.0; parámetros de evaluación `baseline_eval_params.json` 1.0.0 (decisión M-ML-034 (APROBADO)). Métricas exactas en `reports/baseline_metrics.json`.

## 1. Resumen

- Evaluación en la partición **`test`**: 300 estudiantes; evaluados **300**; fuera por tener menos de 2 candidatos: **0**; fuera por no tener ningún cuarto relevante: **0**.
- **NDCG@10 (principal): 0.527 [0.503, 0.550]** frente a 0.197 [0.182, 0.214] con un orden al azar; diferencia +0.330 [+0.302, +0.355] (bootstrap pareado).
- Estas cifras son la **referencia que ML-05 debe superar** (constitución §10.1), sobre las mismas etiquetas sintéticas y los mismos 300 estudiantes.
- Entradas: `data/processed/pairs.parquet`, `data/processed/labels.parquet`, `data/processed/students.parquet`, `data/processed/rooms.parquet`. **No** se leen `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet`.

## 2. Distribución de `matchScore`

Todos los pares candidatos de `pairs.parquet` (las tres particiones; **196,605** pares que pasan HC01-HC03). Histograma por decenas (la última barra incluye 100):

```text
  0-9   |                                                0 (0.0 %)
 10-19  |                                               48 (0.0 %)
 20-29  | #                                          1,050 (0.5 %)
 30-39  | ####                                       4,719 (2.4 %)
 40-49  | #############                             14,856 (7.6 %)
 50-59  | ##########################                31,206 (15.9 %)
 60-69  | ###############################           37,064 (18.9 %)
 70-79  | ########################################  47,339 (24.1 %)
 80-89  | ########################################  46,974 (23.9 %)
 90-100 | ###########                               13,349 (6.8 %)
```

| Mín. | p5 | p10 | p25 | p50 | p75 | p90 | p95 | Máx. | Media |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 43 | 49 | 59 | 72 | 81 | 88 | 91 | 100 | 69.8 |

Percentiles con interpolación lineal. Puntos por factor (sin redondear) en los mismos pares:

| Factor | Máximo | Media | p25 | Mediana | p75 | Con puntos completos | Con 0 puntos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Presupuesto (`budget`) | 30 | 25.9 | 22.5 | 30.0 | 30.0 | 62.5 % | 0.0 % |
| Distancia (`distance`) | 25 | 16.8 | 0.0 | 25.0 | 25.0 | 58.8 % | 26.3 % |
| Servicios esenciales (`essential_services`) | 20 | 14.7 | 12.0 | 14.3 | 17.1 | 20.7 % | 0.0 % |
| Tipo de lugar (`place_type`) | 10 | 3.0 | 0.0 | 0.0 | 10.0 | 30.1 % | 69.9 % |
| Tipo de inmueble (`property_type`) | 10 | 7.0 | 0.0 | 10.0 | 10.0 | 70.0 % | 30.0 % |
| Extras (`extras`) | 5 | 2.4 | 1.7 | 2.5 | 3.2 | 2.1 % | 5.4 % |

## 3. Ejemplo explicado: un estudiante con su top 5

Regla fija (M-ML-034): el primer estudiante de `test` por `student_id`, de un campus verificado, con al menos 10 candidatos. Estudiante y cuartos **sintéticos**; se omiten `livingWith` y mascotas (restricciones duras que ya filtraron a los candidatos).

| Campo | Valor |
| --- | --- |
| Estudiante | `S0003` (campus `UTTT`) |
| Presupuesto | $1,200 a $2,200 |
| Distancia máxima | 0.5 km |
| Transporte | `walk` |
| Tipo de lugar | `room_private_bath` |
| Tipo de inmueble | `student_residence` |
| Servicios esenciales | `wifi`, `water`, `electricity`, `gas`, `kitchen`, `parking`, `security` |
| Candidatos | 178 (relevantes con relevancia ≥ 2: 36) |

Top 5 del modelo base (orden por la suma sin redondear; puntos del desglose con 1 decimal) y la relevancia sintética de cada cuarto:

| # | Cuarto | Renta | `matchScore` | budget (30) | distance (25) | essential_services (20) | place_type (10) | property_type (10) | extras (5) | Suma sin redondear | Relevancia |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `R00175` | $2,000 | **79** | 30.0 | 25.0 | 8.6 | 10.0 | 0.0 | 5.0 | 78.571429 | 2 |
| 2 | `R00086` | $1,850 | **77** | 30.0 | 17.6 | 17.1 | 0.0 | 10.0 | 2.5 | 77.196833 | 1 |
| 3 | `R00277` | $1,500 | **77** | 30.0 | 23.2 | 11.4 | 10.0 | 0.0 | 2.5 | 77.097606 | 3 |
| 4 | `R00019` | $1,400 | **76** | 30.0 | 25.0 | 20.0 | 0.0 | 0.0 | 1.3 | 76.250000 | 1 |
| 5 | `R00225` | $1,600 | **73** | 30.0 | 25.0 | 14.3 | 0.0 | 0.0 | 3.8 | 73.035714 | 3 |

Por qué recibe esos puntos cada cuarto:

1. **`R00175` — `matchScore` 79** (relevancia sintética 2). Presupuesto 30.0/30: renta $2,000 dentro de su rango ($1,200 a $2,200); Distancia 25.0/25: 0.1 km del campus, `distance_ratio` 0.12 (≤ 0.5 de su máximo: puntos completos); Servicios esenciales 8.6/20: cubre 3 de 7 servicios pedidos; Tipo de lugar 10.0/10: `room_private_bath` coincide con lo pedido (`room_private_bath`); Tipo de inmueble 0.0/10: `house` no está en lo pedido (`student_residence`); Extras 5.0/5: 4 servicios extra no pedidos (`extra_services_ratio` 1.00).
2. **`R00086` — `matchScore` 77** (relevancia sintética 1). Presupuesto 30.0/30: renta $1,850 dentro de su rango ($1,200 a $2,200); Distancia 17.6/25: 0.3 km del campus, `distance_ratio` 0.65 (entre 0.5 y 1 de su máximo: decae); Servicios esenciales 17.1/20: cubre 6 de 7 servicios pedidos; Tipo de lugar 0.0/10: `room` no coincide con lo pedido (`room_private_bath`); Tipo de inmueble 10.0/10: `student_residence` está en lo pedido (`student_residence`); Extras 2.5/5: 2 servicios extra no pedidos (`extra_services_ratio` 0.50).
3. **`R00277` — `matchScore` 77** (relevancia sintética 3). Presupuesto 30.0/30: renta $1,500 dentro de su rango ($1,200 a $2,200); Distancia 23.2/25: 0.3 km del campus, `distance_ratio` 0.54 (entre 0.5 y 1 de su máximo: decae); Servicios esenciales 11.4/20: cubre 4 de 7 servicios pedidos; Tipo de lugar 10.0/10: `room_private_bath` coincide con lo pedido (`room_private_bath`); Tipo de inmueble 0.0/10: `apartment` no está en lo pedido (`student_residence`); Extras 2.5/5: 2 servicios extra no pedidos (`extra_services_ratio` 0.50).
4. **`R00019` — `matchScore` 76** (relevancia sintética 1). Presupuesto 30.0/30: renta $1,400 dentro de su rango ($1,200 a $2,200); Distancia 25.0/25: 0.1 km del campus, `distance_ratio` 0.27 (≤ 0.5 de su máximo: puntos completos); Servicios esenciales 20.0/20: cubre 7 de 7 servicios pedidos; Tipo de lugar 0.0/10: `room` no coincide con lo pedido (`room_private_bath`); Tipo de inmueble 0.0/10: `house` no está en lo pedido (`student_residence`); Extras 1.3/5: 1 servicio extra no pedido (`extra_services_ratio` 0.25).
5. **`R00225` — `matchScore` 73** (relevancia sintética 3). Presupuesto 30.0/30: renta $1,600 dentro de su rango ($1,200 a $2,200); Distancia 25.0/25: 0.0 km del campus, `distance_ratio` 0.05 (≤ 0.5 de su máximo: puntos completos); Servicios esenciales 14.3/20: cubre 5 de 7 servicios pedidos; Tipo de lugar 0.0/10: `room` no coincide con lo pedido (`room_private_bath`); Tipo de inmueble 0.0/10: `house` no está en lo pedido (`student_residence`); Extras 3.8/5: 3 servicios extra no pedidos (`extra_services_ratio` 0.75).

Sus cuartos con relevancia 3 quedan en las posiciones 3, 5, 14, 17, 21, 30, 47, 51, 59, 62, 68, 72, 83, 113, 115, 119, 172, 177 del ranking del modelo base. Métricas de este estudiante: NDCG@10 0.323, NDCG@5 0.498, Precision@5 0.600, AP 0.358.

Salida del primer cuarto en el formato de la spec §6.4 (`matchScore` + `breakdown`):

```json
{
  "modelVersion": "baseline-1.0.0",
  "featureSchemaVersion": "2.0.0",
  "matchScore": 79,
  "breakdown": [
    {
      "factor": "budget",
      "points": 30.0,
      "maxPoints": 30,
      "feature": "price_below_min",
      "value": 0
    },
    {
      "factor": "distance",
      "points": 25.0,
      "maxPoints": 25,
      "feature": "distance_ratio",
      "value": 0.11619321885016323
    },
    {
      "factor": "essential_services",
      "points": 8.6,
      "maxPoints": 20,
      "feature": "services_coverage",
      "value": 0.42857142857142855
    },
    {
      "factor": "place_type",
      "points": 10.0,
      "maxPoints": 10,
      "feature": "place_type_match",
      "value": 1
    },
    {
      "factor": "property_type",
      "points": 0.0,
      "maxPoints": 10,
      "feature": "property_type_match",
      "value": 0
    },
    {
      "factor": "extras",
      "points": 5.0,
      "maxPoints": 5,
      "feature": "extra_services_ratio",
      "value": 1
    }
  ]
}
```

El desglose suma el total antes de redondear (constitución §12): la suma de los puntos con 1 decimal puede diferir de `matchScore` hasta en 0.8 (spec §6.3).

## 4. Métricas de ranking por segmento

Media por estudiante e **intervalo percentil del 95 %** por bootstrap: se remuestrean **estudiantes** con reemplazo (1,000 remuestreos por segmento, semilla 20261122, `random.Random("bootstrap:<seed>:<segment_id>")`). NDCG con ganancia `2^rel - 1` y descuento `log2(posición + 1)`; Precision@5 y MAP con **relevante = relevancia ≥ 2**. «Fuera» = estudiantes del segmento excluidos (menos de 2 candidatos o sin cuarto relevante).

| Segmento | Estudiantes | Fuera | NDCG@10 [IC 95 %] | NDCG@5 [IC 95 %] | Precision@5 [IC 95 %] | MAP [IC 95 %] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total | 300 | 0 | 0.527 [0.503, 0.550] | 0.535 [0.507, 0.560] | 0.586 [0.557, 0.615] | 0.503 [0.485, 0.522] |
| UTTT | 30 | 0 | 0.515 [0.441, 0.581] | 0.515 [0.433, 0.592] | 0.573 [0.473, 0.667] | 0.497 [0.437, 0.552] |
| UAEH-ESTE | 30 | 0 | 0.497 [0.415, 0.584] | 0.500 [0.402, 0.597] | 0.587 [0.487, 0.687] | 0.493 [0.429, 0.558] |
| UAEH-ESTL | 30 | 0 | 0.514 [0.443, 0.580] | 0.533 [0.456, 0.606] | 0.600 [0.500, 0.700] | 0.502 [0.442, 0.560] |
| UAEH-ESAT | 30 | 0 | 0.465 [0.385, 0.549] | 0.443 [0.356, 0.532] | 0.507 [0.407, 0.620] | 0.485 [0.417, 0.552] |
| ITSOEH | 30 | 0 | 0.478 [0.405, 0.559] | 0.508 [0.421, 0.594] | 0.580 [0.473, 0.680] | 0.448 [0.389, 0.516] |
| TESJI (sin verificar) | 30 | 0 | 0.581 [0.511, 0.650] | 0.577 [0.487, 0.658] | 0.580 [0.493, 0.673] | 0.534 [0.476, 0.598] |
| UMB-JILO | 30 | 0 | 0.536 [0.457, 0.618] | 0.544 [0.453, 0.630] | 0.560 [0.473, 0.647] | 0.482 [0.423, 0.546] |
| UMB-VDC (sin verificar) | 30 | 0 | 0.576 [0.509, 0.638] | 0.579 [0.497, 0.662] | 0.607 [0.507, 0.700] | 0.508 [0.459, 0.556] |
| UCAN (sin verificar) | 30 | 0 | 0.540 [0.478, 0.601] | 0.590 [0.518, 0.657] | 0.640 [0.567, 0.707] | 0.532 [0.476, 0.591] |
| UNE-TULA | 30 | 0 | 0.570 [0.502, 0.641] | 0.561 [0.481, 0.648] | 0.627 [0.540, 0.720] | 0.553 [0.496, 0.618] |
| Sin `seed_sin_verificar` | 210 | 0 | 0.511 [0.480, 0.539] | 0.515 [0.480, 0.548] | 0.576 [0.536, 0.613] | 0.494 [0.469, 0.519] |
| `budget.max` ≤ $2,000 | 73 | 0 | 0.568 [0.525, 0.608] | 0.583 [0.536, 0.631] | 0.666 [0.608, 0.723] | 0.561 [0.523, 0.601] |
| `budget.max` $2,001–$3,500 | 158 | 0 | 0.534 [0.503, 0.566] | 0.535 [0.496, 0.571] | 0.570 [0.533, 0.606] | 0.494 [0.469, 0.519] |
| `budget.max` > $3,500 | 69 | 0 | 0.468 [0.410, 0.518] | 0.485 [0.419, 0.545] | 0.539 [0.469, 0.606] | 0.463 [0.421, 0.501] |

Los segmentos de un solo campus tienen pocos estudiantes (unos 30): sus intervalos son anchos y no permiten concluir diferencias entre campus. Los rangos de presupuesto usan `budget.max` del estudiante.

## 5. Comparación contra el azar y el techo

Piso: el mismo cálculo con los candidatos en un orden al azar (semilla 20261123, `random.Random("random_order:<seed>:<student_id>")`). Techo: el orden ideal según las etiquetas (NDCG = 1). «Brecha cerrada» = (baseline - azar) / (ideal - azar). La diferencia usa el mismo remuestreo que las medias (pareado).

### 5.1 NDCG@10 (principal)

| Segmento | Azar (piso) | Baseline | Ideal (techo) | Baseline - azar [IC 95 %] | Brecha cerrada |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total | 0.197 [0.182, 0.214] | 0.527 [0.503, 0.550] | 1.000 | +0.330 [+0.302, +0.355] | 41 % |
| UTTT | 0.187 [0.143, 0.234] | 0.515 [0.441, 0.581] | 1.000 | +0.328 [+0.236, +0.416] | 40 % |
| UAEH-ESTE | 0.162 [0.126, 0.202] | 0.497 [0.415, 0.584] | 1.000 | +0.335 [+0.237, +0.429] | 40 % |
| UAEH-ESTL | 0.197 [0.139, 0.254] | 0.514 [0.443, 0.580] | 1.000 | +0.317 [+0.218, +0.418] | 39 % |
| UAEH-ESAT | 0.249 [0.196, 0.306] | 0.465 [0.385, 0.549] | 1.000 | +0.217 [+0.135, +0.306] | 29 % |
| ITSOEH | 0.181 [0.143, 0.222] | 0.478 [0.405, 0.559] | 1.000 | +0.298 [+0.215, +0.384] | 36 % |
| TESJI (sin verificar) | 0.202 [0.161, 0.245] | 0.581 [0.511, 0.650] | 1.000 | +0.379 [+0.300, +0.456] | 47 % |
| UMB-JILO | 0.168 [0.121, 0.215] | 0.536 [0.457, 0.618] | 1.000 | +0.368 [+0.287, +0.449] | 44 % |
| UMB-VDC (sin verificar) | 0.214 [0.164, 0.268] | 0.576 [0.509, 0.638] | 1.000 | +0.362 [+0.282, +0.434] | 46 % |
| UCAN (sin verificar) | 0.190 [0.149, 0.233] | 0.540 [0.478, 0.601] | 1.000 | +0.350 [+0.279, +0.419] | 43 % |
| UNE-TULA | 0.225 [0.176, 0.279] | 0.570 [0.502, 0.641] | 1.000 | +0.345 [+0.266, +0.423] | 45 % |
| Sin `seed_sin_verificar` | 0.195 [0.178, 0.213] | 0.511 [0.480, 0.539] | 1.000 | +0.315 [+0.279, +0.349] | 39 % |
| `budget.max` ≤ $2,000 | 0.206 [0.175, 0.236] | 0.568 [0.525, 0.608] | 1.000 | +0.362 [+0.315, +0.415] | 46 % |
| `budget.max` $2,001–$3,500 | 0.204 [0.181, 0.229] | 0.534 [0.503, 0.566] | 1.000 | +0.330 [+0.292, +0.367] | 41 % |
| `budget.max` > $3,500 | 0.173 [0.148, 0.197] | 0.468 [0.410, 0.518] | 1.000 | +0.295 [+0.235, +0.352] | 36 % |

### 5.2 NDCG@5

| Segmento | Azar (piso) | Baseline | Ideal (techo) | Baseline - azar [IC 95 %] | Brecha cerrada |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total | 0.165 [0.148, 0.182] | 0.535 [0.507, 0.560] | 1.000 | +0.370 [+0.338, +0.399] | 44 % |
| UTTT | 0.160 [0.113, 0.215] | 0.515 [0.433, 0.592] | 1.000 | +0.355 [+0.257, +0.450] | 42 % |
| UAEH-ESTE | 0.137 [0.094, 0.188] | 0.500 [0.402, 0.597] | 1.000 | +0.363 [+0.248, +0.466] | 42 % |
| UAEH-ESTL | 0.178 [0.113, 0.249] | 0.533 [0.456, 0.606] | 1.000 | +0.355 [+0.236, +0.467] | 43 % |
| UAEH-ESAT | 0.212 [0.162, 0.267] | 0.443 [0.356, 0.532] | 1.000 | +0.231 [+0.144, +0.324] | 29 % |
| ITSOEH | 0.157 [0.112, 0.208] | 0.508 [0.421, 0.594] | 1.000 | +0.351 [+0.241, +0.452] | 42 % |
| TESJI (sin verificar) | 0.158 [0.108, 0.209] | 0.577 [0.487, 0.658] | 1.000 | +0.418 [+0.304, +0.524] | 50 % |
| UMB-JILO | 0.133 [0.089, 0.179] | 0.544 [0.453, 0.630] | 1.000 | +0.412 [+0.316, +0.509] | 47 % |
| UMB-VDC (sin verificar) | 0.190 [0.131, 0.247] | 0.579 [0.497, 0.662] | 1.000 | +0.390 [+0.280, +0.493] | 48 % |
| UCAN (sin verificar) | 0.168 [0.118, 0.221] | 0.590 [0.518, 0.657] | 1.000 | +0.421 [+0.331, +0.511] | 51 % |
| UNE-TULA | 0.161 [0.115, 0.214] | 0.561 [0.481, 0.648] | 1.000 | +0.400 [+0.311, +0.494] | 48 % |
| Sin `seed_sin_verificar` | 0.163 [0.143, 0.182] | 0.515 [0.480, 0.548] | 1.000 | +0.352 [+0.312, +0.390] | 42 % |
| `budget.max` ≤ $2,000 | 0.163 [0.127, 0.201] | 0.583 [0.536, 0.631] | 1.000 | +0.420 [+0.358, +0.483] | 50 % |
| `budget.max` $2,001–$3,500 | 0.169 [0.145, 0.195] | 0.535 [0.496, 0.571] | 1.000 | +0.366 [+0.319, +0.410] | 44 % |
| `budget.max` > $3,500 | 0.159 [0.131, 0.191] | 0.485 [0.419, 0.545] | 1.000 | +0.325 [+0.250, +0.389] | 39 % |

### 5.3 Precision@5

| Segmento | Azar (piso) | Baseline | Ideal (techo) | Baseline - azar [IC 95 %] | Brecha cerrada |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total | 0.204 [0.185, 0.223] | 0.586 [0.557, 0.615] | 0.983 | +0.382 [+0.347, +0.416] | 49 % |
| UTTT | 0.200 [0.140, 0.267] | 0.573 [0.473, 0.667] | 0.967 | +0.373 [+0.247, +0.493] | 49 % |
| UAEH-ESTE | 0.180 [0.120, 0.240] | 0.587 [0.487, 0.687] | 1.000 | +0.407 [+0.280, +0.527] | 50 % |
| UAEH-ESTL | 0.240 [0.160, 0.327] | 0.600 [0.500, 0.700] | 0.993 | +0.360 [+0.227, +0.500] | 48 % |
| UAEH-ESAT | 0.240 [0.180, 0.293] | 0.507 [0.407, 0.620] | 0.987 | +0.267 [+0.153, +0.387] | 36 % |
| ITSOEH | 0.207 [0.153, 0.267] | 0.580 [0.473, 0.680] | 1.000 | +0.373 [+0.247, +0.493] | 47 % |
| TESJI (sin verificar) | 0.207 [0.147, 0.273] | 0.580 [0.493, 0.673] | 0.953 | +0.373 [+0.280, +0.467] | 50 % |
| UMB-JILO | 0.160 [0.107, 0.213] | 0.560 [0.473, 0.647] | 0.987 | +0.400 [+0.307, +0.500] | 48 % |
| UMB-VDC (sin verificar) | 0.187 [0.133, 0.240] | 0.607 [0.507, 0.700] | 0.993 | +0.420 [+0.307, +0.533] | 52 % |
| UCAN (sin verificar) | 0.240 [0.180, 0.300] | 0.640 [0.567, 0.707] | 0.987 | +0.400 [+0.320, +0.487] | 54 % |
| UNE-TULA | 0.180 [0.133, 0.233] | 0.627 [0.540, 0.720] | 0.960 | +0.447 [+0.360, +0.533] | 57 % |
| Sin `seed_sin_verificar` | 0.201 [0.177, 0.225] | 0.576 [0.536, 0.613] | 0.985 | +0.375 [+0.326, +0.420] | 48 % |
| `budget.max` ≤ $2,000 | 0.211 [0.167, 0.258] | 0.666 [0.608, 0.723] | 0.978 | +0.455 [+0.392, +0.521] | 59 % |
| `budget.max` $2,001–$3,500 | 0.201 [0.176, 0.228] | 0.570 [0.533, 0.606] | 0.978 | +0.368 [+0.323, +0.413] | 47 % |
| `budget.max` > $3,500 | 0.203 [0.168, 0.243] | 0.539 [0.469, 0.606] | 0.997 | +0.336 [+0.252, +0.414] | 42 % |

### 5.4 MAP

| Segmento | Azar (piso) | Baseline | Ideal (techo) | Baseline - azar [IC 95 %] | Brecha cerrada |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total | 0.251 [0.242, 0.260] | 0.503 [0.485, 0.522] | 1.000 | +0.252 [+0.233, +0.272] | 34 % |
| UTTT | 0.251 [0.219, 0.289] | 0.497 [0.437, 0.552] | 1.000 | +0.245 [+0.179, +0.312] | 33 % |
| UAEH-ESTE | 0.228 [0.212, 0.245] | 0.493 [0.429, 0.558] | 1.000 | +0.265 [+0.201, +0.327] | 34 % |
| UAEH-ESTL | 0.274 [0.237, 0.313] | 0.502 [0.442, 0.560] | 1.000 | +0.228 [+0.159, +0.299] | 31 % |
| UAEH-ESAT | 0.263 [0.235, 0.293] | 0.485 [0.417, 0.552] | 1.000 | +0.222 [+0.148, +0.296] | 30 % |
| ITSOEH | 0.248 [0.229, 0.271] | 0.448 [0.389, 0.516] | 1.000 | +0.200 [+0.140, +0.270] | 27 % |
| TESJI (sin verificar) | 0.232 [0.212, 0.251] | 0.534 [0.476, 0.598] | 1.000 | +0.303 [+0.238, +0.370] | 39 % |
| UMB-JILO | 0.238 [0.222, 0.256] | 0.482 [0.423, 0.546] | 1.000 | +0.244 [+0.186, +0.309] | 32 % |
| UMB-VDC (sin verificar) | 0.253 [0.222, 0.287] | 0.508 [0.459, 0.556] | 1.000 | +0.254 [+0.194, +0.308] | 34 % |
| UCAN (sin verificar) | 0.250 [0.227, 0.279] | 0.532 [0.476, 0.591] | 1.000 | +0.283 [+0.222, +0.343] | 38 % |
| UNE-TULA | 0.274 [0.247, 0.311] | 0.553 [0.496, 0.618] | 1.000 | +0.279 [+0.210, +0.350] | 38 % |
| Sin `seed_sin_verificar` | 0.254 [0.243, 0.265] | 0.494 [0.469, 0.519] | 1.000 | +0.241 [+0.215, +0.265] | 32 % |
| `budget.max` ≤ $2,000 | 0.261 [0.242, 0.283] | 0.561 [0.523, 0.601] | 1.000 | +0.300 [+0.260, +0.345] | 41 % |
| `budget.max` $2,001–$3,500 | 0.255 [0.242, 0.269] | 0.494 [0.469, 0.519] | 1.000 | +0.240 [+0.213, +0.268] | 32 % |
| `budget.max` > $3,500 | 0.233 [0.222, 0.244] | 0.463 [0.421, 0.501] | 1.000 | +0.231 [+0.187, +0.270] | 30 % |

Precision@5 divide siempre entre 5: con menos de 5 relevantes el techo es menor que 1 (con los cortes de M-ML-032 hay ⌈0.2·n⌉ relevantes por estudiante).

## 6. Cómo leer estas cifras

- El modelo base **no** se ajustó con estas etiquetas (spec §7 regla 3): sus pesos se fijaron antes de generarlas.
- Las etiquetas premian lo mismo que las reglas en lo general, pero con otros pesos y otras curvas (correlación de Spearman en `reports/labels_synthetic.md` §6): el margen entre el baseline y el techo es lo que un modelo entrenado (ML-05) podría recuperar **de esta función sintética**, no de preferencias reales.
- Sin atributos sensibles: los segmentos son campus y presupuesto (constitución §11.3); ningún factor ni segmento usa género, edad u origen.
