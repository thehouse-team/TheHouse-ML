# ML-05 T04 — Ablación, segmentos e hipótesis del segmento > $3,500

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Las etiquetas son sintéticas (utilidad oculta documentada más ruido, M-ML-032): estas métricas **no** son desempeño real ni preferencias reales (constitución §8.2 y §20.3). El desempeño real solo se afirma con señales reales en **ML-07**.

Generado por `python -m thehouse_ml evaluate` (ML-05 1.0.0, T04 (ML-05-R04)). Métricas completas en `reports/ablation_metrics.json`; la tabla por pliegue con las 4 métricas está también en `reports/trained_report.md`.

Perfil: **`full`** (corrida oficial).

## Grupos de features

Los 6 factores del desglose (spec §6.6, mismos `id` que `baseline-1.0.0`). Cada feature de `feature_order` está en un solo grupo.

| Grupo | Factor | Features |
| --- | --- | --- |
| `budget` | Presupuesto | `price_to_budget`, `price_below_min` |
| `distance` | Distancia | `distance_km`, `effective_distance_km`, `distance_ratio` |
| `essential_services` | Servicios esenciales | `services_coverage` |
| `place_type` | Tipo de lugar | `place_type_match` |
| `property_type` | Tipo de inmueble | `property_type_match` |
| `extras` | Extras | `extra_services_ratio` |
| — | (fuera de los grupos) | `pets_compatible`: constante (1) en los candidatos (HC03 excluye los incompatibles): no aporta señal; su término se suma a la constante del modelo (spec §6.6) y se queda en todos los modelos de la ablación |

## Métricas por pliegue (traspaso T03-H2)

5 pliegues de T02 (misma asignación de `models/train_manifest.json`), 1,699 estudiantes y 165,849 pares de train + validación; hiperparámetros elegidos en T02 (`reports/train_cv_metrics.json`), sin búsqueda nueva. El NDCG@10 por pliegue reproduce exactamente el de T02. Test no participa.

| Métrica | Modelo | P0 | P1 | P2 | P3 | P4 | Media ± desv. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NDCG@10 | Baseline | 0.5396 | 0.5470 | 0.5453 | 0.5241 | 0.5345 | 0.5381 ± 0.0093 |
| NDCG@10 | Regresión logística | 0.8664 | 0.8570 | 0.8616 | 0.8626 | 0.8640 | 0.8623 ± 0.0035 |
| NDCG@10 | LightGBM | 0.8733 | 0.8597 | 0.8651 | 0.8661 | 0.8684 | 0.8665 ± 0.0049 |
| NDCG@5 | Baseline | 0.5399 | 0.5512 | 0.5583 | 0.5185 | 0.5282 | 0.5392 ± 0.0163 |
| NDCG@5 | Regresión logística | 0.8683 | 0.8659 | 0.8747 | 0.8674 | 0.8648 | 0.8682 ± 0.0039 |
| NDCG@5 | LightGBM | 0.8773 | 0.8675 | 0.8775 | 0.8735 | 0.8708 | 0.8733 ± 0.0043 |
| Precision@5 | Baseline | 0.5765 | 0.5935 | 0.5882 | 0.5617 | 0.5576 | 0.5755 ± 0.0158 |
| Precision@5 | Regresión logística | 0.8906 | 0.8888 | 0.8900 | 0.8897 | 0.8800 | 0.8878 ± 0.0044 |
| Precision@5 | LightGBM | 0.8959 | 0.8871 | 0.8994 | 0.8973 | 0.8865 | 0.8932 ± 0.0060 |
| MAP | Baseline | 0.5114 | 0.5161 | 0.5095 | 0.5012 | 0.5103 | 0.5097 ± 0.0054 |
| MAP | Regresión logística | 0.8342 | 0.8244 | 0.8275 | 0.8230 | 0.8278 | 0.8274 ± 0.0043 |
| MAP | LightGBM | 0.8408 | 0.8275 | 0.8309 | 0.8318 | 0.8333 | 0.8329 ± 0.0049 |

## Ablación por grupo (solo pliegues)

Cada modelo se reajusta en cada pliegue sin las features del grupo, con los mismos hiperparámetros. Δ = NDCG@10 sin el grupo - NDCG@10 del modelo completo; media ± desviación muestral entre pliegues y pliegues en los que empeora.

| Grupo quitado | Regresión logística: NDCG@10 sin el grupo | Δ NDCG@10 media ± desv. | Pliegues peor | LightGBM: NDCG@10 sin el grupo | Δ NDCG@10 media ± desv. | Pliegues peor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (modelo completo) | 0.8623 | — | — | 0.8665 | — | — |
| Presupuesto (`budget`) | 0.7783 | -0.0840 ± 0.0090 | 5/5 | 0.7843 | -0.0822 ± 0.0076 | 5/5 |
| Distancia (`distance`) | 0.6078 | -0.2545 ± 0.0065 | 5/5 | 0.6152 | -0.2514 ± 0.0074 | 5/5 |
| Servicios esenciales (`essential_services`) | 0.8239 | -0.0384 ± 0.0036 | 5/5 | 0.8281 | -0.0384 ± 0.0040 | 5/5 |
| Tipo de lugar (`place_type`) | 0.7505 | -0.1118 ± 0.0036 | 5/5 | 0.7519 | -0.1146 ± 0.0048 | 5/5 |
| Tipo de inmueble (`property_type`) | 0.8486 | -0.0137 ± 0.0015 | 5/5 | 0.8530 | -0.0136 ± 0.0025 | 5/5 |
| Extras (`extras`) | 0.8429 | -0.0195 ± 0.0021 | 5/5 | 0.8475 | -0.0191 ± 0.0021 | 5/5 |

## Segmentos en test (medición única de T03)

Métricas por estudiante de la medición única de T03 (modelos oficiales, sin reentrenar ni volver a puntuar test). Bootstrap pareado por estudiante, 1,000 remuestreos, IC 95 %, semilla `bootstrap:20261122:<segment_id>` (la del baseline en cada segmento: sus intervalos coinciden con `baseline_metrics.json`). Las medias por campus y por rango de presupuesto se agregan exactamente a las del total.

| Segmento | Estudiantes | Baseline NDCG@10 | Regresión logística | Δ LR [IC 95 %] | LightGBM | Δ LightGBM [IC 95 %] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total | 300 | 0.527 [0.503, 0.550] | 0.862 [0.852, 0.873] | +0.335 [+0.312, +0.360] | 0.863 [0.853, 0.872] | +0.336 [+0.313, +0.360] |
| UTTT | 30 | 0.515 [0.441, 0.581] | 0.837 [0.803, 0.871] | +0.322 [+0.261, +0.389] | 0.832 [0.792, 0.870] | +0.317 [+0.259, +0.380] |
| UAEH-ESTE | 30 | 0.497 [0.415, 0.584] | 0.870 [0.832, 0.904] | +0.373 [+0.287, +0.460] | 0.877 [0.841, 0.907] | +0.380 [+0.297, +0.465] |
| UAEH-ESTL | 30 | 0.514 [0.443, 0.580] | 0.851 [0.818, 0.882] | +0.337 [+0.263, +0.415] | 0.854 [0.828, 0.880] | +0.340 [+0.270, +0.413] |
| UAEH-ESAT | 30 | 0.465 [0.385, 0.549] | 0.845 [0.819, 0.872] | +0.379 [+0.290, +0.467] | 0.837 [0.810, 0.865] | +0.371 [+0.282, +0.458] |
| ITSOEH | 30 | 0.478 [0.405, 0.559] | 0.865 [0.831, 0.895] | +0.386 [+0.300, +0.464] | 0.866 [0.840, 0.891] | +0.387 [+0.312, +0.461] |
| TESJI (sin verificar) | 30 | 0.581 [0.511, 0.650] | 0.861 [0.830, 0.891] | +0.280 [+0.214, +0.346] | 0.864 [0.835, 0.894] | +0.283 [+0.219, +0.351] |
| UMB-JILO | 30 | 0.536 [0.457, 0.618] | 0.890 [0.857, 0.919] | +0.354 [+0.270, +0.441] | 0.895 [0.861, 0.924] | +0.359 [+0.280, +0.445] |
| UMB-VDC (sin verificar) | 30 | 0.576 [0.509, 0.638] | 0.886 [0.861, 0.910] | +0.310 [+0.243, +0.381] | 0.890 [0.867, 0.911] | +0.314 [+0.249, +0.383] |
| UCAN (sin verificar) | 30 | 0.540 [0.478, 0.601] | 0.856 [0.821, 0.891] | +0.316 [+0.260, +0.370] | 0.863 [0.826, 0.898] | +0.323 [+0.265, +0.382] |
| UNE-TULA | 30 | 0.570 [0.502, 0.641] | 0.863 [0.832, 0.893] | +0.293 [+0.222, +0.359] | 0.855 [0.823, 0.888] | +0.286 [+0.215, +0.352] |
| Sin `seed_sin_verificar` | 210 | 0.511 [0.480, 0.539] | 0.860 [0.849, 0.871] | +0.349 [+0.319, +0.381] | 0.859 [0.848, 0.871] | +0.349 [+0.319, +0.379] |
| `budget.max` ≤ $2,000 | 73 | 0.568 [0.525, 0.608] | 0.857 [0.837, 0.875] | +0.289 [+0.243, +0.336] | 0.865 [0.847, 0.882] | +0.298 [+0.254, +0.343] |
| `budget.max` $2,001–$3,500 | 158 | 0.534 [0.503, 0.566] | 0.867 [0.854, 0.880] | +0.332 [+0.300, +0.364] | 0.864 [0.850, 0.877] | +0.330 [+0.298, +0.361] |
| `budget.max` > $3,500 | 69 | 0.468 [0.410, 0.518] | 0.859 [0.834, 0.880] | +0.391 [+0.338, +0.446] | 0.859 [0.835, 0.881] | +0.391 [+0.343, +0.446] |

NDCG@5, Precision@5 y MAP por segmento, con sus diferencias pareadas, en `reports/ablation_metrics.json`.

## Hipótesis del segmento `budget.max` > $3,500 (M-ML-035, herencia 4)

Hipótesis: el baseline es más débil en > $3,500 porque puntúa igual cualquier renta dentro del presupuesto, mientras la utilidad oculta prefiere lo más barato.

Regla: Confirmada si (1) la Δ NDCG@10 media de la regresión logística contra el baseline en `budget.max` > $3,500 es estrictamente la mayor de los tres rangos, (2) su IC 95 % está por encima de 0 y (3) quitar el grupo presupuesto baja el NDCG@10 medio entre pliegues; refutada si falla (1) o (3) o si el IC está entero por debajo de 0; no concluyente si se cumplen (1) y (3) pero el IC incluye el 0 (M-ML-040, fijada antes de calcular los segmentos). Los rangos se comparan por su media puntual: sus intervalos pueden solaparse.

- (1) Δ NDCG@10 en budget:budget_gt_3500 = +0.391 es la mayor de los rangos de presupuesto
- (2) IC 95 % [+0.338, +0.446] excluye el 0 (por encima)
- (3) quitar el grupo presupuesto cambia el NDCG@10 en -0.0840 de media (5/5 pliegues por debajo): lo empeora

**Veredicto: CONFIRMADA.**

## Advertencias

- **DATOS SINTÉTICOS:** con etiquetas del simulador, la ablación mide qué features necesita el modelo para imitar la utilidad oculta (circularidad, ML-05-RK01), no qué valoran los estudiantes reales (ML-07).
- Los segmentos reutilizan la medición única de T03: no se ajustó nada con ellos (spec §6.2).
- Los segmentos pequeños (p. ej. un campus, ~30 estudiantes) tienen intervalos anchos.
