# ML-05 — Modelo entrenado y evaluación

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-04 completa |
| Relación con el backend | El modelo promovido se exporta en ML-06 para NV-23 |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

Un modelo de ranking puede aprender interacciones que las reglas no capturan (por ejemplo, que la distancia importa menos si el estudiante tiene auto). Sin señales reales todavía, se entrena con etiquetas de una función de utilidad oculta y documentada para validar el pipeline completo.

## 2. Objetivo

Entrenar un modelo de ranking, evaluarlo con rigor contra el modelo base y documentar con honestidad qué significan sus métricas.

## 3. Precondiciones

1. ML-04 completa.
2. Autorización «EJECUTA».

## 4. Alcance

1. Etiquetas de relevancia graduada (0–3) desde una utilidad oculta con interacciones y ruido.
2. Modelos: regresión logística y LightGBM con objetivo de ranking (LambdaRank).
3. Validación cruzada agrupada por estudiante.
4. Comparación con el modelo base, ablación de factores y análisis por segmentos.
5. Model card.

## 5. Fuera de alcance

- Exportación a ONNX (ML-06).
- Afirmaciones de desempeño real (ML-07).

## 6. Diseño

- La utilidad oculta usa pesos distintos a los del modelo base y efectos de interacción (transporte × distancia, presupuesto × servicios) más ruido; queda documentada en el reporte.
- Métricas: NDCG@10, Precision@5 y MAP por estudiante, con media y desviación entre particiones.
- Restricciones monotónicas en LightGBM para precio (no aumenta el puntaje al subir la renta) y distancia.
- Calibración del puntaje del modelo a 0–100 para mostrarlo en la app.
- Promoción solo si supera al modelo base en NDCG@10 de forma consistente entre particiones.

## 7. Reglas

1. Toda métrica sobre datos sintéticos se etiqueta como tal.
2. Sin búsqueda de hiperparámetros sobre la partición de prueba.

## 8. Seguridad, datos y privacidad

- El análisis por segmentos (universidad, rango de presupuesto) se incluye en la model card para detectar sesgos.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-05-R01 | Etiquetas documentadas | Generador de etiquetas reproducible con la utilidad oculta descrita en el reporte. | F1 | T01 |
| ML-05-R02 | Entrenamiento reproducible | `train` produce el mismo modelo y las mismas métricas con la misma semilla y los mismos datos. | F2 | T02 |
| ML-05-R03 | Evaluación contra el modelo base | Tabla con NDCG@10, Precision@5 y MAP (media ± desviación) del modelo base, la regresión logística y LightGBM. | F2 | T03 |
| ML-05-R04 | Ablación y segmentos | Impacto de quitar cada grupo de features y métricas por universidad y rango de presupuesto. | F3 | T04 |
| ML-05-R05 | Model card | `reports/model_card.md` con uso previsto, datos, métricas, limitaciones (datos sintéticos) y decisión de promoción. | F3 | T05 |
| ML-05-R06 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T06 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Modelo candidato | LightGBM LambdaRank con restricciones monotónicas | PROPUESTO |
| Desglose del modelo entrenado | Puntaje del modelo entrenado con el desglose del modelo base, o modelo aditivo explicable | PENDIENTE |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-05-RK01 | El modelo solo aprende la utilidad sintética | Métricas sin valor real | Declararlo en la model card y validar con señales reales (ML-07) |
| ML-05-RK02 | Sobreajuste | Mal desempeño en producción | Validación cruzada por estudiante y restricciones monotónicas |

## 13. Dependencias

- ML-04.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 6 |
| Tasks | 6 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
