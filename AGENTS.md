# TheHouse ML — Instrucciones para agentes

| Campo | Valor |
| --- | --- |
| Versión | 1.1.0 |
| Fecha | 2026-11-11 |
| Estado | **✅ APROBADO** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-11 (M-ML-011) |
| Decisión | M-ML-011 |

## Autoridad y lectura obligatoria

Antes de cada tarea, leer en este orden: `constitution.md` → `AGENTS.md` → `memoria.md` → la spec, el plan y las tasks de la tarea. Si hay conflicto, manda el documento de mayor autoridad; si la duda persiste, preguntar al humano.

## Estado vigente

- Constitution 1.0.0 aprobada (2026-11-09, M-ML-001). AGENTS 1.1.0 aprobado (2026-11-11, M-ML-011). El responsable de `TheHouse-ML` es César Yael López Trujillo.
- ML-01 está **completa**. ML-02 está **aprobada sin ejecutar** (**⛔ NO EJECUTAR HASTA «EJECUTA»**). El resto de specs sigue en borrador y no autorizadas para ejecución.
- Relación con el backend: `TheHouse-Backend` consume el artefacto ONNX en NV-23; el esquema de entrada viene de NV-11.
- Relación con `TheHouse-GenAI`: ese servicio explica el puntaje y su desglose; nunca lo calcula. Este repositorio no llama a ningún LLM.

## Protocolo de autorización

| Acción | Requiere |
| --- | --- |
| Aprobar constitution, AGENTS o una spec | «Apruebo X.Y.Z» del responsable |
| Implementar tasks | «EJECUTA» con spec, versión, tasks y rutas |
| Descargar datos de terceros | Spec de datos aprobada y fuente registrada en `memoria.md` con su licencia |
| Agregar dependencias | Decisión M-ML o tabla de stack de la constitution |
| Publicar un artefacto de modelo (release) | Evidencia de evaluación y aprobación humana de la versión |
| Cambios en GitHub (repositorio, ruleset, secretos) | Permiso explícito para esa acción |
| Merge de un pull request | Aprobación de otra persona en GitHub; el agente no mezcla por iniciativa propia |

Un plan aprobado no habilita tareas fuera de él. Un cambio material requiere reaprobar.

## Flujo de trabajo obligatorio

1. Spec primero; implementación solo con autorización.
2. Cada iteración: implementar → probar → revisar coherencia con la constitution y la memoria → revisar seguridad y licencias de datos.
3. Un control no ejecutado no se reporta como PASS.
4. Cerrar una task solo con aceptación cumplida y evidencia reproducible (comando, resultado, commit).

## Datos

- Nunca subir `data/raw/`, `data/processed/` ni `models/` al repositorio.
- Antes de descargar, comprobar que la fuente y su licencia están en `memoria.md`.
- Eliminar datos personales en la primera transformación; nunca mostrarlos en salidas ni reportes.
- Registrar SHA-256 de cada archivo descargado en `data/manifests/`.

## Modelos y métricas

- Todo modelo se compara con el modelo base por reglas.
- Las métricas sobre datos sintéticos se etiquetan como tales en todo reporte.
- No usar pickle para artefactos de intercambio; solo ONNX.
- Cada artefacto lleva `feature_schema.json`, `model_card.md`, `metrics.json` y SHA-256.

## Rutas protegidas y acciones prohibidas

- No editar `constitution.md` ni `AGENTS.md` sin aprobación de la nueva versión.
- No leer ni mostrar secretos.
- No hacer scraping no autorizado ni usar fuentes sin licencia registrada.
- No usar servicios de pago (COST-001).
- No hacer commit directo a `main` después del commit inicial.

## Resultado de revisión

Cada evidencia termina con **PASS**, **PASS WITH WARNINGS** o **BLOCKED**, con la lista de hallazgos y su clasificación.

## Convenciones documentales

- Documentos en español; código, nombres de archivos y mensajes técnicos en inglés.
- Decisiones `M-ML-XXX` en `memoria.md` con estado APROBADO, PROPUESTO, PENDIENTE o DESCARTADO.
- Specs en `specs/ml-XX-<nombre>/` con `spec.md`, `plan.md`, `task.md` y, al ejecutar, `evidence.md`.

## Cierre de una tarea

Reportar qué se hizo, qué se verificó (con comandos y resultados), qué queda pendiente y qué decisiones esperan aprobación.
