# ML-05 T05 — Puntaje 0-100 y desglose de la regresión logística

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Los coeficientes se aprendieron de etiquetas sintéticas (M-ML-032) y la distribución y los ejemplos son de pares sintéticos: no describen preferencias ni desempeño reales (constitución §8.2 y §20.3; ML-07).

Generado por `python -m thehouse_ml evaluate` (ML-05 1.0.0, T05 (ML-05-R05)). Modelo `logreg-1.0.0`, esquema 2.0.0. Perfil: **`full`** (corrida oficial).

## 1. Coeficientes de la regresión logística

Modelo final de T02 (`models/logreg.json`, transformación `standard`, C = 0.01); se lee con su SHA-256 verificado contra `train_manifest.json` y **no se reentrena**. Cada feature se recorta a su `training_range` (o `range`) y entra estandarizada: el término del logit es `coef · (clip(x) - media) / escala`, así que «logit por unidad» = `coef / escala`. Intercepto `b0` = -4.133224.

| Feature | Factor | Recorte | coef (estandarizado) | Escala | Logit por unidad | Esperado | Signo coherente |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `distance_km` | `distance` | [0, 30] | -3.349741 | 6.303973 | -0.531370 | ↑ no suma | **sí** |
| `effective_distance_km` | `distance` | [0, 30] | -1.869938 | 3.697290 | -0.505759 | ↑ no suma | **sí** |
| `price_to_budget` | `budget` | [0, 1] | -0.996049 | 0.232874 | -4.277202 | ↑ no suma | **sí** |
| `services_coverage` | `essential_services` | [0, 1] | +0.565531 | 0.185477 | +3.049062 | ↑ suma | **sí** |
| `place_type_match` | `place_type` | [0, 1] | +1.006235 | 0.457783 | +2.198060 | ↑ suma | **sí** |
| `property_type_match` | `property_type` | [0, 1] | +0.359535 | 0.460339 | +0.781024 | ↑ suma | **sí** |
| `pets_compatible` | término fijo | [0, 1] | +0.000000 | 1.000000 | +0.000000 | — | — |
| `distance_ratio` | `distance` | [0, 60] | -0.100585 | 3.421818 | -0.029395 | ↑ no suma | **sí** |
| `price_below_min` | `budget` | [0, 1] | -0.012314 | 0.219381 | -0.056130 | ↑ suma | **no** |
| `extra_services_ratio` | `extras` | [0, 1] | +0.439171 | 0.215048 | +2.042204 | ↑ suma | **sí** |

«Esperado»: «↑ suma» = más valor no debe restar puntos; «↑ no suma» = más valor no debe sumar puntos (más renta o más distancia). `price_below_min` crece cuando la renta **baja** por debajo de `budget.min`, así que su signo «esperado» respecto a la renta es positivo. `pets_compatible` es constante (1) en los candidatos (HC03): su coeficiente es 0 y su término va al término fijo con `b0`, no a un factor.

## 2. Calibración 0-100 (spec §6.6)

Transformación afín por factor sobre el recorte: con `C_k` la suma de los términos del factor `k` y `[min_k, max_k]` su rango alcanzable dentro de los `training_range`,

```text
puntos_k   = 100 · (C_k - min_k) / S        S = Σ_k (max_k - min_k)
maxPoints_k = 100 · (max_k - min_k) / S      Σ_k maxPoints_k = 100
matchScore = floor(round6(Σ_k puntos_k) + 0.5)   (M-ML-029/031)
logit      = b0 + término fijo + Σ_k min_k + S · Σ_k puntos_k / 100
```

S = 45.281261 unidades de logit; Σ_k min_k = -35.138603. El total es una función afín **creciente** del logit, así que el orden coincide con el de `decision_function` para entradas dentro del rango (fuera del rango se recortan). `maxPoints` se emite con el redondeo por factor (1 decimal); los puntos sin redondear de un factor nunca superan su máximo sin redondear.

| Factor | Features | min_k (logit) | max_k (logit) | maxPoints (sin redondear) | maxPoints | Peso del baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Presupuesto (`budget`) | `price_to_budget`, `price_below_min` | -1.9443 | +2.3890 | 9.5698 | **9.6** | 30 |
| Distancia (`distance`) | `distance_km`, `effective_distance_km`, `distance_ratio` | -28.7731 | +4.1045 | 72.6075 | **72.6** | 25 |
| Servicios esenciales (`essential_services`) | `services_coverage` | -2.2348 | +0.8142 | 6.7336 | **6.7** | 20 |
| Tipo de lugar (`place_type`) | `place_type_match` | -0.6570 | +1.5410 | 4.8542 | **4.9** | 10 |
| Tipo de inmueble (`property_type`) | `property_type_match` | -0.5429 | +0.2381 | 1.7248 | **1.7** | 10 |
| Extras (`extras`) | `extra_services_ratio` | -0.9863 | +1.0559 | 4.5100 | **4.5** | 5 |
| **Total** |  |  |  | 100.0000 | **100** | 100 |

**Lectura:** `maxPoints` mide cuánto puede mover el logit cada factor **dentro de todo su `training_range`**, no su importancia en los candidatos típicos. La distancia domina porque su rango llega a 30 km (`distance_km`, `effective_distance_km`) y 60 (`distance_ratio`), mientras la mayoría de los candidatos están a pocos kilómetros: casi todos reciben una parte grande de sus puntos de distancia y las diferencias entre cuartos cercanos se ven en décimas. Comparado con el baseline (30/25/20/10/10/5), presupuesto y servicios pesan mucho menos. Ver el hallazgo T05-H2 de la evidencia.

## 3. Verificación del desglose en test

Todos los pares de la partición **test** que puntuó T03 (**30,755** pares de **300** estudiantes; misma medición, sin volver a evaluar el modelo):

| Comprobación | Resultado | Criterio |
| --- | --- | --- |
| Total contra la suma sin redondear | máx. 0.499973; fuera de tolerancia: 0 | `|Σ sin redondear - matchScore| ≤ 0.5` |
| Suma reportada contra la suma sin redondear | máx. 0.212896; fuera de tolerancia: 0 | `|Σ reportada - Σ sin redondear| ≤ 0.3` |
| Cada factor en [0, `maxPoints`] | pares fuera: 0 | sin redondear y redondeado |
| `matchScore` en [0, 100] | pares fuera: 0 | entero |
| Logit reconstruido desde el puntaje | máx. |diferencia| 2.31e-14 | ≤ 1e-09 (transformación afín exacta salvo binary64) |
| Orden por `raw_total` = orden del logit | 300/300 estudiantes idénticos | desempate por `room_id` ascendente |
| Métricas con el puntaje 0-100 | idénticas a las de T03 | NDCG@10, NDCG@5, Precision@5 y MAP por estudiante |

**El desglose suma el total:** **sí**.

## 4. Distribución de `matchScore` en test

**30,755** pares candidatos de test. Histograma por decenas (la última barra incluye 100):

```text
  0-9   |                                                1 (0.0 %)
 10-19  |                                               45 (0.1 %)
 20-29  |                                              108 (0.4 %)
 30-39  | #                                            319 (1.0 %)
 40-49  | ##                                           963 (3.1 %)
 50-59  | ####                                       1,444 (4.7 %)
 60-69  | ######                                     2,491 (8.1 %)
 70-79  | ##################                         7,407 (24.1 %)
 80-89  | ########################################  16,140 (52.5 %)
 90-100 | #####                                      1,837 (6.0 %)
```

| Mín. | p5 | p10 | p25 | p50 | p75 | p90 | p95 | Máx. | Media |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 50 | 61 | 74 | 81 | 85 | 88 | 90 | 97 | 77.6 |

Puntos por factor (sin redondear) en los mismos pares:

| Factor | maxPoints | Media | p25 | Mediana | p75 | Media / maxPoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Presupuesto (`budget`) | 9.6 | 4.27 | 2.49 | 4.40 | 6.04 | 45 % |
| Distancia (`distance`) | 72.6 | 63.42 | 60.55 | 68.04 | 70.90 | 87 % |
| Servicios esenciales (`essential_services`) | 6.7 | 4.94 | 4.04 | 4.81 | 5.77 | 73 % |
| Tipo de lugar (`place_type`) | 4.9 | 1.52 | 0.00 | 0.00 | 4.85 | 31 % |
| Tipo de inmueble (`property_type`) | 1.7 | 1.25 | 0.00 | 1.72 | 1.72 | 73 % |
| Extras (`extras`) | 4.5 | 2.17 | 1.50 | 2.26 | 2.87 | 48 % |

## 5. Monotonía por factor (hallazgo obligatorio)

Coeficientes con signo «incorrecto» respecto a su variable natural: `price_below_min`. Como pueden compensarse con otra feature del mismo factor (colinealidad), la comprobación decisiva es **a nivel de factor**: una rejilla que construye pares reales con `compute_pair_features` (ML-03) y recorre la variable natural; el puntaje del factor no puede subir más de 1e-09 puntos entre dos pasos consecutivos.

| Factor y variable | Curvas | Pasos | Pasos en que sube | Mayor subida | Monótono |
| --- | ---: | ---: | ---: | --- | --- |
| Presupuesto al subir la renta | 100 | 10,204 | 46 | +0.1238 pts (budget.min = 1, budget.max = 50000, rent_mxn_from = 0.0, rent_mxn_to = 1.0; 9.4459 → 9.5696) | **no** |
| Distancia al aumentar la distancia | 32 | 4,608 | 0 | — | **sí** |

- **Rejilla de presupuesto:** NV-11: budget.min ∈ {nulo, 1, 10, 50, 100, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000, 25000, 50000} ≤ budget.max ∈ {500, 1000, 2000, 2700, 3500, 5000, 10000, 25000, 50000}; renta entera de 0 a budget.max (101 pasos más budget.min ± 1): price_to_budget recorre su training_range [0, 1] y price_below_min su range [0, 1].
- **Rejilla de distancia:** transporte ∈ {walk, bike, public, car} x maxDistanceKm ∈ {nulo→30, 0.5, 1, 2, 5, 10, 20, 30}; distancia de 0 a 36 km cada 0.25 km (incluye > 30 km para comprobar el recorte).
- **Zona exacta:** con renta < `budget.min`, al subir la renta `price_to_budget` sube (resta puntos) y `price_below_min` baja (con su coeficiente negativo, **suma** puntos); el factor sube si y solo si `budget.max / budget.min > 76.20` (cociente de los logits por unidad de `price_to_budget` y `price_below_min`). La subida máxima posible es el término completo de `price_below_min`: 0.1240 puntos sobre 100.
- **Datos sintéticos:** 73,815 pares con renta < `budget.min`; el mayor `budget.max / budget.min` implícito (`(1 - price_below_min) / price_to_budget`) es 2.83, **por debajo** de 76.20: ningún par observado cae en la zona.

**Monotonía a nivel de factor:** **no**.

**Hallazgo T05-H1.** El coeficiente de `price_below_min` es negativo (-0.0123 estandarizado): el modelo «penaliza» quedar por debajo del mínimo como el baseline, pero eso significa que, por debajo de `budget.min`, más renta suma puntos por esa feature. A nivel de factor lo compensa `price_to_budget` salvo cuando `budget.max` es más de 76.2 veces `budget.min` (perfiles válidos en NV-11, ausentes de los datos sintéticos; efecto ≤ 0.124 puntos sobre 100; 0 pares observados en la zona).

**Desviación aceptada (M-ML-041, APROBADO por el responsable el 2026-11-24).** La condición 3 de la spec §6.5 es «desglose aditivo que suma el total», y se cumple; la verificación de monotonía del presupuesto fue una protección adicional del «EJECUTA» de T05, no parte del criterio. La no monotonía se acepta como desviación conocida **después de ver el resultado de esa verificación**, y no se corrige en este modelo. Trabajo futuro: `logreg-1.1.0` sin `price_below_min` o con su coeficiente ≥ 0, evaluado con datos nuevos en ML-07 (test ya se usó, spec §6.2). La model card (T06) y NV-23 deben mencionarla.

## 6. Ejemplos explicados

Regla fija (definida antes de ver los resultados): para cada rango de `budget.max` (≤ $2,000, $2,001–$3,500, > $3,500), el primer estudiante de test por `student_id` de un campus verificado con al menos 10 candidatos; su top 5 por `raw_total` (desempate `room_id`). Ids sintéticos; sin datos personales. «Relevancia sintética» es la etiqueta 0-3 del simulador (M-ML-032) y «Baseline» el `matchScore` de `baseline-1.0.0` para el mismo par, como referencia.

### 6.1 `S0031` — UTTT, `budget.max` $1,500 (≤ $2,000), 112 candidatos

| # | Cuarto | matchScore | Presup. (/9.6) | Dist. (/72.6) | Serv. (/6.7) | Lugar (/4.9) | Inmueble (/1.7) | Extras (/4.5) | Relevancia sintética | Baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `R00261` | **94** | 7.3 | 70.6 | 6.7 | 4.9 | 1.7 | 2.6 | 3 | 81 |
| 2 | `R00101` | **93** | 6.4 | 72.2 | 5.1 | 4.9 | 1.7 | 3.2 | 3 | 82 |
| 3 | `R00069` | **93** | 7.0 | 72.2 | 5.1 | 4.9 | 1.7 | 2.6 | 3 | 78 |
| 4 | `R00282` | **91** | 6.1 | 70.7 | 5.1 | 4.9 | 1.7 | 2.6 | 3 | 83 |
| 5 | `R00193` | **90** | 5.5 | 71.1 | 5.1 | 4.9 | 0.0 | 3.2 | 2 | 78 |

Primer lugar `R00261`: Presup. 7.3/9.6, Dist. 70.6/72.6, Serv. 6.7/6.7, Lugar 4.9/4.9, Inmueble 1.7/1.7, Extras 2.6/4.5 → `matchScore` **94** (suma sin redondear 93.747352; probabilidad 0.960, dato informativo). Renta = 23% de `budget.max`, 0.89 km (efectiva 0.89 km, 0.03 de su máximo), cobertura de servicios 100%, extras 0.57.

Salida JSON de ese cuarto (formato del baseline con `maxPoints` de la calibración y lista `features` en los factores de varias features; `value` = valor de entrada):

```json
{
  "modelVersion": "logreg-1.0.0",
  "featureSchemaVersion": "2.0.0",
  "matchScore": 94,
  "breakdown": [
    {
      "factor": "budget",
      "points": 7.3,
      "maxPoints": 9.6,
      "features": [
        {
          "feature": "price_to_budget",
          "value": 0.23333333333333334
        },
        {
          "feature": "price_below_min",
          "value": 0.5625
        }
      ]
    },
    {
      "factor": "distance",
      "points": 70.6,
      "maxPoints": 72.6,
      "features": [
        {
          "feature": "distance_km",
          "value": 0.8924504599656791
        },
        {
          "feature": "effective_distance_km",
          "value": 0.8924504599656791
        },
        {
          "feature": "distance_ratio",
          "value": 0.029748348665522634
        }
      ]
    },
    {
      "factor": "essential_services",
      "points": 6.7,
      "maxPoints": 6.7,
      "feature": "services_coverage",
      "value": 1
    },
    {
      "factor": "place_type",
      "points": 4.9,
      "maxPoints": 4.9,
      "feature": "place_type_match",
      "value": 1
    },
    {
      "factor": "property_type",
      "points": 1.7,
      "maxPoints": 1.7,
      "feature": "property_type_match",
      "value": 1
    },
    {
      "factor": "extras",
      "points": 2.6,
      "maxPoints": 4.5,
      "feature": "extra_services_ratio",
      "value": 0.5714285714285714
    }
  ]
}
```

### 6.2 `S0003` — UTTT, `budget.max` $2,200 ($2,001–$3,500), 178 candidatos

| # | Cuarto | matchScore | Presup. (/9.6) | Dist. (/72.6) | Serv. (/6.7) | Lugar (/4.9) | Inmueble (/1.7) | Extras (/4.5) | Relevancia sintética | Baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `R00244` | **89** | 6.3 | 70.7 | 3.8 | 4.9 | 0.0 | 3.4 | 3 | 44 |
| 2 | `R00088` | **88** | 6.5 | 66.6 | 5.8 | 4.9 | 0.0 | 4.5 | 3 | 50 |
| 3 | `R00101` | **88** | 7.4 | 72.2 | 3.8 | 0.0 | 0.0 | 4.5 | 3 | 54 |
| 4 | `R00193` | **88** | 6.7 | 71.0 | 4.8 | 0.0 | 1.7 | 3.4 | 3 | 44 |
| 5 | `R00075` | **88** | 5.0 | 70.6 | 4.8 | 4.9 | 0.0 | 2.3 | 3 | 53 |

Primer lugar `R00244`: Presup. 6.3/9.6, Dist. 70.7/72.6, Serv. 3.8/6.7, Lugar 4.9/4.9, Inmueble 0/1.7, Extras 3.4/4.5 → `matchScore` **89** (suma sin redondear 89.101906; probabilidad 0.745, dato informativo). Renta = 34% de `budget.max`, 0.78 km (efectiva 0.78 km, 1.56 de su máximo), cobertura de servicios 57%, extras 0.75.

### 6.3 `S0044` — UTTT, `budget.max` $5,400 (> $3,500), 200 candidatos

| # | Cuarto | matchScore | Presup. (/9.6) | Dist. (/72.6) | Serv. (/6.7) | Lugar (/4.9) | Inmueble (/1.7) | Extras (/4.5) | Relevancia sintética | Baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `R00088` | **94** | 8.2 | 69.0 | 6.7 | 4.9 | 1.7 | 3.4 | 3 | 74 |
| 2 | `R00175` | **92** | 6.0 | 72.5 | 4.8 | 4.9 | 1.7 | 2.3 | 3 | 78 |
| 3 | `R00235` | **92** | 6.9 | 70.6 | 6.7 | 4.9 | 1.7 | 1.1 | 3 | 78 |
| 4 | `R00156` | **91** | 7.0 | 72.0 | 5.8 | 4.9 | 1.7 | 0.0 | 3 | 74 |
| 5 | `R00269` | **91** | 6.4 | 70.9 | 4.8 | 4.9 | 1.7 | 2.3 | 3 | 76 |

Primer lugar `R00088`: Presup. 8.2/9.6, Dist. 69/72.6, Serv. 6.7/6.7, Lugar 4.9/4.9, Inmueble 1.7/1.7, Extras 3.4/4.5 → `matchScore` **94** (suma sin redondear 93.931952; probabilidad 0.963, dato informativo). Renta = 13% de `budget.max`, 2.48 km (efectiva 0.62 km, 0.17 de su máximo), cobertura de servicios 100%, extras 0.75.

## 7. Condición 3 de promoción y decisión

| Condición 3 (regresión logística) | Resultado |
| --- | --- |
| El desglose suma el total (tolerancias, rangos, orden y métricas) | **sí** |
| Monotonía a nivel de factor (presupuesto y distancia) | **no** |
| Desviaciones aceptadas (no bloquean) | `monotonicity_budget` (M-ML-041) |
| **Condición 3** (spec §6.5: desglose aditivo que suma el total) | **sí** |

**Decisión:** La regresión logística cumple las 3 condiciones: queda **PROMOVIDA** como modelo de ML-05 (`logreg-1.0.0`, puntaje 0-100 con desglose por factor); `baseline-1.0.0` queda como respaldo. Desviación conocida aceptada (M-ML-041): `monotonicity_budget` (ver `reports/trained_breakdown.md`).

## Advertencias

- **DATOS SINTÉTICOS:** etiquetas de un simulador con circularidad (ML-05-RK01); no es desempeño real.
- Test se midió una sola vez en T03; T05 reutiliza esos pares y no ajusta nada con sus resultados.
- El formato de `breakdown` (lista `features`, `maxPoints` variables) y el recorte al `training_range` se coordinan con `TheHouse-GenAI` y NV-23 (responsable).
