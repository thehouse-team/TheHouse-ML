# ML-04 — Modelo base por reglas

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-03 completa |
| Relación con el backend | Es el respaldo determinista de NV-23 y la base del desglose que explica `TheHouse-GenAI` |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

Smart Match debe ser explicable y funcionar desde el primer día. Un modelo por reglas con pesos documentados da un puntaje 0–100 con desglose y sirve de referencia contra la que se mide cualquier modelo entrenado.

## 2. Objetivo

Implementar un puntaje determinista 0–100 con desglose por factor y publicarlo como referencia y respaldo.

## 3. Precondiciones

1. ML-03 completa.
2. Autorización «EJECUTA».

## 4. Alcance

1. Función de puntaje por factor y total.
2. Pesos y curvas en un archivo versionado.
3. Casos dorados de referencia.
4. Reporte del comportamiento del modelo base.

## 5. Fuera de alcance

- Modelos entrenados (ML-05).

## 6. Diseño

- Factores y pesos iniciales: presupuesto 30, distancia 25, servicios esenciales 20, tipo de lugar 10, tipo de inmueble y modalidad 10, extras 5 (total 100).
- Presupuesto: 1.0 si la renta está entre `min` y `max`, decae linealmente por debajo de `min`; fuera de `max` el cuarto ya fue excluido.
- Distancia: 1.0 hasta la mitad de `maxDistanceKm` efectiva, decae a 0 en `maxDistanceKm`.
- Servicios: proporción de servicios esenciales cubiertos.
- Desglose: cada factor reporta sus puntos; la suma es el total redondeado.

## 7. Reglas

1. Mismo resultado para la misma entrada en Python y en el backend (casos dorados).
2. Los pesos solo cambian con una versión nueva del modelo base.

## 8. Seguridad, datos y privacidad

- Sin atributos sensibles como factor (constitution §11).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-04-R01 | Puntaje y desglose | Puntaje 0–100 con desglose por factor que suma el total; pruebas de propiedades (monotonía en precio y distancia). | F2 | T01 |
| ML-04-R02 | Configuración versionada | Pesos y curvas en un archivo con versión `baseline-x.y.z`; el reporte muestra la versión usada. | F1 | T02 |
| ML-04-R03 | Casos dorados | Al menos 50 casos (entradas, features, desglose, total) exportados para la prueba de paridad del backend. | F2 | T03 |
| ML-04-R04 | Reporte del modelo base | Distribución de puntajes, ejemplos explicados y métricas de ranking sobre la partición de prueba de ML-03 (marcadas como sintéticas). | F3 | T04 |
| ML-04-R05 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T05 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Pesos iniciales | Los de la §6, validados con el equipo | PROPUESTO |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-04-RK01 | Pesos arbitrarios | Recomendaciones poco útiles | Validación con el equipo y comparación posterior con señales reales |

## 13. Dependencias

- ML-03.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 5 |
| Tasks | 5 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
