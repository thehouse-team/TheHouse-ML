# ML-05 T03 — Evaluación en test y decisión de promoción

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Las etiquetas son sintéticas (utilidad oculta documentada más ruido, M-ML-032): los modelos aprenden la función del simulador y estas métricas **no** son desempeño real ni preferencias reales (constitución §8.2 y §20.3). El desempeño real solo se afirma con señales reales en **ML-07**.

Generado por `python -m thehouse_ml evaluate` (ML-05 1.0.0, T03 (ML-05-R03)). Métricas completas, por estudiante y con las condiciones en `reports/trained_metrics.json`; ablación, segmentos e hipótesis > $3,500 (T04) en `reports/ablation_report.md`.

Perfil: **`full`** (corrida oficial).

## Medición única en test

La partición test se evaluó una sola vez con los modelos finales de T02 (SHA-256 verificados); ningún hiperparámetro, feature, calibración, umbral ni decisión se ajustó con estos resultados (spec ML-05 §6.2 y §7 regla 2). Cambiar algo del modelo después invalida esta medición.

| Modelo | Archivo (fuera de git) | Configuración | SHA-256 (verificado contra `train_manifest.json`) |
| --- | --- | --- | --- |
| Regresión logística | `logreg.json` | `logreg:standard:C=0.01` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` |
| LightGBM | `lightgbm_lambdarank.txt` | `lightgbm:leaves=15:lr=0.05:trees=300` | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` |

Partición **test** de ML-03: **300 estudiantes** con 2 o más candidatos y **30,755 pares**. Cada modelo ordena los candidatos de cada estudiante por puntaje descendente con desempate por `room_id` ascendente; NDCG@10, NDCG@5, Precision@5 y MAP con `thehouse_ml.evaluation.ranking` (M-ML-034, relevante si `relevance ≥ 2`). El baseline recalculado reproduce exactamente `per_student_baseline` de `reports/baseline_metrics.json` (mismos 300 estudiantes y mismas métricas).

## Test: baseline, regresión logística y LightGBM

Media por estudiante con IC 95 % por bootstrap pareado (1,000 remuestreos de estudiantes, percentil, semilla `bootstrap:20261122:total`, la misma que el total del baseline: su intervalo coincide con el de `baseline_metrics.json`).

| Modelo | NDCG@10 [IC 95 %] | NDCG@5 [IC 95 %] | Precision@5 [IC 95 %] | MAP [IC 95 %] |
| --- | ---: | ---: | ---: | ---: |
| Baseline `baseline-1.0.0` | 0.527 [0.503, 0.550] | 0.535 [0.507, 0.560] | 0.586 [0.557, 0.615] | 0.503 [0.485, 0.522] |
| Regresión logística | 0.862 [0.852, 0.873] | 0.874 [0.861, 0.886] | 0.916 [0.900, 0.931] | 0.831 [0.821, 0.839] |
| LightGBM | 0.863 [0.853, 0.872] | 0.877 [0.864, 0.890] | 0.911 [0.894, 0.927] | 0.835 [0.825, 0.844] |

## Diferencia pareada contra el baseline (test)

Modelo - baseline por estudiante (mismos estudiantes y mismos índices de remuestreo), media con IC 95 %.

| Modelo | Δ NDCG@10 [IC 95 %] | Δ NDCG@5 [IC 95 %] | Δ Precision@5 [IC 95 %] | Δ MAP [IC 95 %] |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | +0.335 [+0.312, +0.360] | +0.339 [+0.313, +0.367] | +0.330 [+0.299, +0.361] | +0.327 [+0.307, +0.347] |
| LightGBM | +0.336 [+0.313, +0.360] | +0.342 [+0.315, +0.370] | +0.325 [+0.295, +0.358] | +0.331 [+0.312, +0.350] |

## Validación cruzada por pliegue (NDCG@10)

De `reports/train_cv_metrics.json` (T02, train + validación; test no participa): mismos estudiantes retenidos en cada pliegue para los tres modelos. Media ± desviación estándar muestral entre pliegues.

| Pliegue | Baseline | Regresión logística | Δ | LightGBM | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| P0 | 0.5396 | 0.8664 | +0.3267 | 0.8733 | +0.3336 |
| P1 | 0.5470 | 0.8570 | +0.3100 | 0.8597 | +0.3128 |
| P2 | 0.5453 | 0.8616 | +0.3163 | 0.8651 | +0.3198 |
| P3 | 0.5241 | 0.8626 | +0.3386 | 0.8661 | +0.3420 |
| P4 | 0.5345 | 0.8640 | +0.3295 | 0.8684 | +0.3339 |
| **Media ± desv.** | 0.5381 ± 0.0093 | 0.8623 ± 0.0035 | +0.3242 | 0.8665 ± 0.0049 | +0.3284 |

### Las 4 métricas por pliegue (T04, traspaso T03-H2)

Recalculadas en T04 con la misma asignación de pliegues de `train_manifest.json` y los hiperparámetros elegidos en T02 (sin búsqueda nueva); el NDCG@10 reproduce exactamente la tabla anterior. Media ± desviación estándar muestral entre pliegues.

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

## Criterio de promoción (spec §6.5)

Condición 1 con los 5 pliegues del perfil completo. Solo la regresión logística (aditiva) puede cumplir la condición 3, que se verifica en T05; LightGBM es referencia del potencial y no se promueve en v1 aunque cumpla 1 y 2 (decisión 5).

| Condición | Regresión logística | LightGBM |
| --- | --- | --- |
| 1. Validación cruzada: NDCG@10 > baseline en los 5 pliegues | **sí** (5/5 pliegues) | **sí** (5/5 pliegues) |
| 2. Test: IC 95 % de Δ NDCG@10 pareada con límite inferior > 0 | **sí** (+0.335 [+0.312, +0.360]) | **sí** (+0.336 [+0.313, +0.360]) |
| 3. Desglose aditivo que suma el total (constitución §12) | pendiente (T05) | no (no aditivo) |
| **Resultado** | **candidata a promoción, sujeta a T05** | **referencia (no se promueve en v1)** |

**Decisión:** La regresión logística cumple las condiciones 1 y 2: queda candidata a promoción, sujeta a T05 (condición 3, desglose aditivo que suma el total). Hasta entonces `baseline-1.0.0` sigue vigente.

## Advertencias

- **DATOS SINTÉTICOS:** etiquetas generadas por un simulador (M-ML-032), con circularidad (ML-05-RK01): superar al baseline aquí significa aprender mejor la utilidad oculta del simulador, no recomendar mejor a estudiantes reales. No es desempeño real; eso es de ML-07.
- La promoción no es definitiva hasta T05 (condición 3) y la model card (T06); ML-06 exporta el modelo que resulte.
- Las métricas por segmento (campus, sin `seed_sin_verificar`, rango de presupuesto) y la hipótesis del segmento > $3,500 son de T04, con esta misma medición (sin volver a medir en test).
