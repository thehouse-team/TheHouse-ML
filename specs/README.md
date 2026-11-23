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
| ML-05 | [Modelo entrenado y evaluación](ml-05-trained-model/spec.md) | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** | **⛔ NO EJECUTAR HASTA «EJECUTA»** | 0/6 |
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
