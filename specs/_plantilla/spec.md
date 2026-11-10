# ML-XX — Título

| Campo | Valor |
| --- | --- |
| ID | ML-XX |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | AAAA-MM-DD |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | Specs o decisiones que deben estar completas |
| Relación con el backend | Qué usa o qué entrega a `TheHouse-Backend` (o "Ninguna directa") |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

Situación actual y por qué hace falta esta spec.

## 2. Objetivo

Una o dos frases con el resultado medible.

## 3. Precondiciones

1. …
2. Autorización «EJECUTA».

## 4. Alcance

1. …

## 5. Fuera de alcance

- …

## 6. Diseño

- Decisiones técnicas: datos, algoritmos, formatos, rutas.

## 6.1 Resultados esperados

| Salida | Ubicación | En git | Cómo se verifica |
| --- | --- | --- | --- |
| … | … | Sí / No | Prueba o comando |

## 7. Reglas

1. …

## 8. Seguridad, datos y privacidad

- Licencias, datos personales, secretos.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-XX-R01 | … | Criterio comprobable con una prueba o un comando | F1 | T01 |
| ML-XX-RNN | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | TNN |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| … | … | PROPUESTO / PENDIENTE |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-XX-RK01 | … | … | … |

## 13. Dependencias

- …

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | N |
| Tasks | N |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
