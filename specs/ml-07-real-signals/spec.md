# ML-07 — Señales reales de TheHouse

| Campo | Valor |
| --- | --- |
| ID | ML-07 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | Spec del backend que exporte señales anonimizadas; volumen mínimo de datos |
| Relación con el backend | Requiere una spec del backend para exportar favoritos, visitas y retroalimentación de Smart Match con ids anonimizados |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

La única forma de saber si Smart Match ayuda es medirlo con el comportamiento real de los estudiantes: cuartos que guardan en favoritos, visitas que solicitan y retroalimentación útil/no útil.

## 2. Objetivo

Evaluar y reentrenar el modelo con señales reales anonimizadas, y reportar por primera vez desempeño real.

## 3. Precondiciones

1. Spec del backend para la exportación anonimizada, aprobada y desplegada.
2. Al menos 200 estudiantes con 3 o más interacciones (si no, BLOCKED).
3. Autorización «EJECUTA».

## 4. Alcance

1. Ingesta de la exportación anonimizada.
2. Etiquetas reales: visita > favorito > retroalimentación útil > vista; descarte como negativo.
3. Evaluación del modelo vigente y del modelo base con datos reales.
4. Reentrenamiento y nueva versión si mejora.

## 5. Fuera de alcance

- Acceso directo a la base de datos de producción.
- Datos personales (correos, nombres, coordenadas del usuario).

## 6. Diseño

- La exportación solo trae ids anonimizados (hash con sal que no sale del backend), preferencias, ids de cuartos, tipo de señal y fecha.
- Evaluación temporal: entrenar con semanas anteriores y probar con la más reciente.

## 7. Reglas

1. Sin volumen mínimo no se reporta desempeño real.
2. Las señales se borran del entorno local al terminar cada evaluación; solo quedan métricas agregadas.

## 8. Seguridad, datos y privacidad

- Sin datos personales ni forma de reidentificar usuarios.
- La exportación la ejecuta el backend con autorización humana; `TheHouse-ML` nunca se conecta a producción.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-07-R01 | Ingesta anonimizada | Lector de la exportación con validación de esquema y prueba de ausencia de datos personales. | F2 | T01 |
| ML-07-R02 | Evaluación real | Métricas del modelo base y del vigente sobre datos reales con partición temporal. | F2 | T02 |
| ML-07-R03 | Reentrenamiento | Nuevo modelo solo si mejora sobre datos reales; nueva versión por ML-06. | F3 | T03 |
| ML-07-R04 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T04 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Volumen mínimo | 200 estudiantes con 3 o más interacciones | PROPUESTO |
| Spec de exportación en el backend | Por definir | PENDIENTE |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-07-RK01 | Pocos usuarios | Sin señal suficiente | BLOCKED hasta alcanzar el volumen mínimo |
| ML-07-RK02 | Reidentificación | Privacidad | Hash con sal en el backend y sin coordenadas del usuario |

## 13. Dependencias

- ML-06.
- NV-15, NV-17 y NV-23 en producción.
- Spec de exportación del backend.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 4 |
| Tasks | 4 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
