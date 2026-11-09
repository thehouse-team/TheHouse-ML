# ML-06 — Artefacto ONNX y contrato con el backend

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-05 con modelo promovido (o el modelo base si ninguno lo supera) |
| Relación con el backend | NV-23 carga el artefacto con `Microsoft.ML.OnnxRuntime` detrás de `ICompatibilityModel` y corre los casos dorados en .NET |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

El backend no corre Python. El modelo debe llegar como un archivo portable, verificable y versionado, junto con todo lo necesario para que el backend calcule las mismas features y compruebe que obtiene los mismos puntajes.

## 2. Objetivo

Exportar el modelo promovido a ONNX y publicar un paquete de entrega verificable para el backend.

## 3. Precondiciones

1. ML-05 con decisión de promoción.
2. Autorización «EJECUTA».
3. Permiso explícito para publicar el release en GitHub.

## 4. Alcance

1. Exportación a ONNX y verificación con onnxruntime.
2. Paquete de entrega: `model.onnx`, `feature_schema.json`, `golden_cases.json`, `model_card.md`, `metrics.json` y `SHA256SUMS`.
3. Pruebas de paridad Python ↔ ONNX.
4. Prueba de rendimiento.
5. Release versionado en GitHub.

## 5. Fuera de alcance

- Código del backend (NV-23).
- Servir el modelo como API.

## 6. Diseño

- LightGBM → ONNX con `onnxmltools`; regresión logística o modelo base con `skl2onnx` o como reglas documentadas.
- Los casos dorados incluyen entradas crudas, features calculadas, puntaje y desglose, para probar en .NET tanto las features como la inferencia.
- Tolerancia de paridad: diferencia absoluta ≤ 1e-4 en el puntaje del modelo y 0 en el total redondeado.
- Rendimiento: puntuar 200 cuartos en menos de 20 ms en CPU (constitution §13).
- Versión `modelVersion` semántica; release inmutable con hashes.

## 7. Reglas

1. Sin pickle en el paquete de entrega.
2. Un release publicado no se modifica; una corrección es una versión nueva.

## 8. Seguridad, datos y privacidad

- Hashes SHA-256 de todos los archivos del paquete.
- El backend verifica el hash antes de cargar el modelo.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-06-R01 | Exportación verificada | `export` produce `model.onnx` que onnxruntime carga y ejecuta sobre la partición de prueba. | F2 | T01 |
| ML-06-R02 | Paridad | Puntajes de Python y ONNX dentro de la tolerancia en todos los casos dorados. | F2 | T02 |
| ML-06-R03 | Rendimiento | 200 cuartos en menos de 20 ms en CPU (mediana de 100 corridas). | F3 | T03 |
| ML-06-R04 | Paquete y release | Release `model-vX.Y.Z` con los seis archivos y sus hashes, aprobado por el responsable. | F3 | T04 |
| ML-06-R05 | Guía de integración | `docs/backend-integration.md` con el contrato de entrada y salida, el cálculo de features y cómo correr los casos dorados en .NET. | F3 | T05 |
| ML-06-R06 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T06 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Runtime en el backend | `Microsoft.ML.OnnxRuntime` en NV-23 (reemplaza a ML.NET) | PROPUESTO |
| Distribución del artefacto | Release de GitHub descargado y verificado en el build del backend | PENDIENTE |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-06-RK01 | Diferencias de cálculo entre Python y .NET | Puntajes distintos | Casos dorados con features y pruebas en ambos lados |
| ML-06-RK02 | Artefacto alterado | Comportamiento inesperado | Hashes verificados al cargar |

## 13. Dependencias

- ML-05.
- NV-23 (backend) para la integración.

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
