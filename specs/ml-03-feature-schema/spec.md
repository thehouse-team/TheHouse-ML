# ML-03 — Esquema de features y estudiantes sintéticos

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | ML-02 completa; NV-11 1.1.0 aprobada en el backend |
| Relación con el backend | El esquema de preferencias de NV-11 es la entrada del modelo; `feature_schema.json` es el contrato con NV-23 |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

El modelo necesita la misma información que la app guarda: las preferencias de NV-11 y los datos públicos del cuarto. Sin usuarios todavía, los perfiles de estudiante se generan de forma sintética y documentada.

## 2. Objetivo

Definir el esquema de features compartido con el backend y generar estudiantes sintéticos realistas y reproducibles.

## 3. Precondiciones

1. ML-02 completa.
2. NV-11 1.1.0 aprobada (catálogos cerrados).
3. Autorización «EJECUTA».

## 4. Alcance

1. `feature_schema.json` versionado: entradas del estudiante, del cuarto y features del par.
2. Restricciones duras definidas en el esquema.
3. Generador de estudiantes sintéticos con semilla.
4. Pares estudiante–cuarto candidatos para entrenamiento y evaluación.

## 5. Fuera de alcance

- Etiquetas y modelos (ML-04, ML-05).
- Datos personales de cualquier tipo.

## 6. Diseño

- Estudiante (de NV-11): `budget.min`, `budget.max`, `maxDistanceKm`, `essentialServices` (multi-hot de `wifi`, `water`, `electricity`, `gas`, `hot_water`, `laundry`, `kitchen`, `furnished`, `parking`, `security`, `cleaning`), `livingWith`, `placeType`, `pets`, `transport`, `propertyTypes`, universidad (coordenadas).
- Cuarto: renta mensual, coordenadas, servicios (mismo catálogo), `placeType`, tipo de inmueble, admite mascotas, política de convivencia (`women`, `men`, `mixed`), recámaras, baños y área.
- Par: distancia en km (haversine), distancia efectiva según `transport`, razón precio/presupuesto, cobertura de servicios esenciales, coincidencia de tipo de lugar y de inmueble, compatibilidad de mascotas.
- Restricciones duras: renta mayor a `budget.max`, `livingWith` incompatible o mascota no admitida → el cuarto se excluye.
- Estudiantes sintéticos: distribuciones por universidad documentadas (presupuesto, transporte, servicios) con semilla fija; parámetros en un archivo versionado.

## 7. Reglas

1. El esquema usa exactamente los catálogos de NV-11; un cambio en NV-11 obliga a una versión mayor del esquema.
2. Las features se calculan con una sola implementación de referencia que también genera los casos dorados para el backend.

## 8. Seguridad, datos y privacidad

- Los perfiles sintéticos no imitan personas reales.
- El género solo interviene en la restricción `livingWith` que elige el estudiante (constitution §11).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-03-R01 | Esquema de features | `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba. | F1 | T01 |
| ML-03-R02 | Cálculo de features de referencia | Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios). | F2 | T02 |
| ML-03-R03 | Restricciones duras | Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla. | F2 | T03 |
| ML-03-R04 | Estudiantes sintéticos | Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones. | F2 | T04 |
| ML-03-R05 | Conjunto de pares | Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga. | F3 | T05 |
| ML-03-R06 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T06 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Número de estudiantes sintéticos | 2 000 estudiantes, 200 candidatos por estudiante | PROPUESTO |
| Distancia efectiva por transporte | Factores documentados: caminar 1.0, bici 0.5, transporte público 0.4, auto 0.25 | PROPUESTO |
| Política de convivencia de cuartos | Campo nuevo del cuarto en NV-12; en datos de Airbnb se asigna de forma sintética | PENDIENTE |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-03-RK01 | Esquema distinto al de la app | Modelo inutilizable | Prueba contra NV-11 y versión mayor ante cambios |
| ML-03-RK02 | Sintéticos poco realistas | Modelo que no generaliza | Validar distribuciones con el equipo y reemplazar con señales reales en ML-07 |

## 13. Dependencias

- ML-02.
- NV-11 1.1.0 (backend).
- NV-12 para la política de convivencia real del cuarto.

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
