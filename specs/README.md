# TheHouse ML — Índice de specs

Orden de bloques (mismo criterio que el backend): base → datos → modelo → integración → señales reales. Cada spec requiere su propia aprobación y «EJECUTA».

Para entender el proyecto completo (qué hace, cómo se conecta con el backend y con `TheHouse-GenAI`, el pipeline, la estructura del código y qué lleva cada spec) lee primero [`ARQUITECTURA.md`](ARQUITECTURA.md).

## 1. Mapa de specs

| ID | Spec | Estado documental | Estado de ejecución | Tasks |
| --- | --- | --- | --- | --- |
| ML-01 | [Base del proyecto](ml-01-foundation/spec.md) | **✅ APROBADA** | **✅ COMPLETA — T01–T05 (2026-11-10): PASS, ver `evidence.md`** | 5/5 |
| ML-02 | [Datos: bases nacionales, calibración regional y licencias](ml-02-data-sources/spec.md) | **✅ APROBADA** | **✅ COMPLETA — T01–T05 y T07–T09 (8 tasks); T06 DESCARTADA (M-ML-019); cierre M-ML-020, ver `evidence.md`** | 8 completadas + 1 descartada |
| ML-03 | [Esquema de features, estudiantes sintéticos y catálogo sintético de cuartos](ml-03-feature-schema/spec.md) | **✅ APROBADA — 1.0.0** | **✅ COMPLETA — T01–T07 (2026-11-19 y 2026-11-20); cierre M-ML-028, ver `evidence.md`** | 7/7 |
| ML-04 | [Modelo base por reglas](ml-04-baseline-model/spec.md) | **✅ APROBADA — 1.0.0** | **✅ COMPLETA — T01–T07 (2026-11-20 a 2026-11-23); esquema 2.0.0, `baseline-1.0.0`, puntaje con desglose, etiquetas sintéticas, 81 casos dorados y reporte de referencia para ML-05 (sintético); M-ML-030 a M-ML-034, cierre M-ML-035, ver `evidence.md`** | 7/7 |
| ML-05 | [Modelo entrenado y evaluación](ml-05-trained-model/spec.md) | **✅ APROBADA — 1.0.0** | **🟡 EN EJECUCIÓN — T01 (2026-11-23): `scikit-learn` 1.9.1 y `lightgbm` 4.7.0, cargador de pares y etiquetas sin archivos ocultos; PASS (libgomp confirmado en el CI del PR #31); M-ML-037. T02 (2026-11-24): 5 pliegues por estudiante, rejilla, regresión logística C=0.01 y LightGBM con restricciones monotónicas reproducibles (SHA-256 iguales), perfil `ci` para el CI; PASS, M-ML-038. T03 (2026-11-24): medición única en test (300 estudiantes) con SHA-256 de los modelos verificados; Δ NDCG@10 pareada vs baseline: regresión logística +0.335 [+0.312, +0.360], LightGBM +0.336 [+0.313, +0.360]; regresión logística candidata a promoción, sujeta a T05; PASS, M-ML-039. T04 (2026-11-24): 4 métricas por pliegue (traspaso T03-H2), ablación de los 6 grupos (distancia el más influyente: Δ NDCG@10 LR −0.255), segmentos en test e hipótesis > $3,500 **confirmada** (Δ LR +0.391 [+0.338, +0.446]; salvedades T04-H2/H3 para la model card); PASS, M-ML-040. T05 (2026-11-24): puntaje 0-100 y desglose de `logreg-1.0.0` (maxPoints 9.6/72.6/6.7/4.9/1.7/4.5) que suma el total en los 30,755 pares de test; **regresión logística PROMOVIDA** como modelo de ML-05 (`baseline-1.0.0` de respaldo); no monotonía del presupuesto con `budget.max/budget.min` > 76.2 aceptada como desviación (T05-H1, corrección en `logreg-1.1.0`/ML-07); PASS WITH WARNINGS, M-ML-041. T06 (2026-11-24): `reports/model_card.md` generada desde los JSON de `reports/` (10 secciones, «DATOS SINTÉTICOS», salvedades T04-H2/H3, T05-H1 y nota de `maxPoints` para `TheHouse-GenAI`); PASS, M-ML-042 (T06-H1 traspasado a T07); ver `evidence.md` (sintético)** | 6/7 |
| ML-06 | [Artefacto ONNX y contrato con el backend](ml-06-onnx-export/spec.md) | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** | **⛔ NO EJECUTAR HASTA «EJECUTA»** | 0/6 |
| ML-07 | [Señales reales de TheHouse](ml-07-real-signals/spec.md) | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** | **⛔ NO EJECUTAR HASTA «EJECUTA»** | 0/4 |

## 2. Orden recomendado

| Prioridad | Specs | Motivo |
| --- | --- | --- |
| 1 | ML-01 | Base del proyecto: estructura, herramientas y CI |
| 2 | ML-02, ML-03 | Bases nacionales de INEGI, calibración regional y esquema de features compartido con NV-11 |
| 3 | ML-04 | Modelo base por reglas: referencia y respaldo |
| 4 | ML-05, ML-06 | Modelo entrenado, evaluación y artefacto ONNX para el backend |
| 5 | ML-07 | Señales reales de TheHouse cuando haya usuarios |

## 3. Relación con TheHouse-Backend

- **NV-11** define el esquema de preferencias que ML-03 usa como entrada del modelo.
- **NV-23** carga el artefacto ONNX de ML-06 detrás de `ICompatibilityModel`.
- **TheHouse-GenAI** (servicio de IA generativa) explica el desglose por factor; no es parte de este repositorio.
- **ML-07** necesita una spec del backend para exportar señales anonimizadas.
