# TheHouse ML — Memoria de decisiones

Este documento no es un diario. Registra decisiones estables y pendientes bajo `constitution.md` y `AGENTS.md`. **APROBADO** significa directriz aceptada, no implementación realizada. **PROPUESTO** requiere revisión; **PENDIENTE** carece de resolución o evidencia; **DESCARTADO** no se adopta.

## M-ML-001 — Proyecto separado para Smart Match

- **Decisión:** crear `TheHouse-ML` como repositorio separado, con constitution, AGENTS, memoria y specs propias, que produce un artefacto de modelo y no se despliega.
- **Estado:** APROBADO — Victor Manuel Jose Jose — 2026-11-09 («si crealo», en respuesta a «¿Creo el repo TheHouse-ML con la constitución 1.0.0?»). Constitution y AGENTS 1.0.0.
- **Justificación:** el profesor pide proyectos separados para ML e IA generativa; un repositorio que entrega un artefacto cumple eso sin agregar un servicio en producción ni romper el monolito modular del backend.
- **Impacto:** requiere la decisión M-BE-062 en el backend (repositorios satélite sin despliegue) y la versión 1.2.0 de su constitution.

## M-ML-002 — Stack

- **Decisión:** Python 3.12, `uv`, pandas, scikit-learn, LightGBM, ONNX con onnxruntime, pytest, ruff, mypy, pip-audit y gitleaks.
- **Estado:** APROBADO con la constitution 1.0.0 (§5).
- **Justificación:** herramientas estándar de ML, gratuitas y con exportación a ONNX, que el backend .NET puede cargar.
- **Impacto:** el backend agrega `Microsoft.ML.OnnxRuntime` en NV-23 en lugar de entrenar con ML.NET.

## M-ML-003 — Fuentes de datos permitidas

- **Decisión:**

| Fuente | Uso | Licencia y condiciones | Estado |
| --- | --- | --- | --- |
| Inside Airbnb — Ciudad de México (`listings`) | Distribuciones realistas de servicios (amenities), tipo de cuarto, precio relativo y ubicación de los cuartos | CC BY 4.0 según terceros; su página de políticas pide descargar solo lo necesario, no hacer scraping del sitio y no republicar los datos. Se verifica la licencia en la propia página antes de descargar | PENDIENTE de verificación |
| OpenStreetMap | Coordenadas de los campus de las 10 universidades regionales y puntos de interés | ODbL: atribución y obras derivadas bajo la misma licencia | PROPUESTO |
| Seed de TheHouse (12 propiedades, 10 universidades) | Rangos regionales de precio y ubicación (Tula–Tepeji–Jilotepec) para reescalar | Datos propios | PROPUESTO |
| Estudiantes sintéticos | Perfiles generados desde el esquema de NV-11 | Propios, con distribuciones documentadas | PROPUESTO |
| Señales reales de TheHouse (favoritos y visitas) | Etiquetas reales de interés | Exportación anonimizada definida por una spec del backend | PENDIENTE |

- **Estado:** PROPUESTO.
- **Justificación:** no existe un dataset público de estudiantes, preferencias e interés en cuartos en México; se combinan datos reales de inmuebles con perfiles sintéticos y, cuando existan, señales reales de la app.
- **Impacto:** los datos crudos nunca se suben al repositorio (constitution §7).

## M-ML-004 — Fuentes descartadas

- **Decisión:** no usar:
  - EconLab de Banxico: acceso solo por proyecto de investigación con convenio.
  - Scraping de Inmuebles24, Vivanuncios, Propiedades.com o Pincali: sus términos no lo autorizan.
  - Properati: licencia y cobertura de rentas en México no confirmadas.
  - Datasets de citas o perfiles personales (por ejemplo OkCupid): datos personales sensibles.
- **Estado:** PROPUESTO.
- **Justificación:** licencias, privacidad o acceso incompatibles con la constitution §7.

## M-ML-005 — Método de etiquetas y evaluación

- **Decisión:** etapa 1, modelo base por reglas y explicable; etapa 2, modelo de ranking entrenado con etiquetas de una función de utilidad documentada más ruido, comparado contra la base; etapa 3, reentrenamiento y evaluación con señales reales cuando existan.
- **Estado:** PROPUESTO.
- **Justificación:** permite construir y validar todo el pipeline desde ahora, sin presentar como real un desempeño sintético (constitution §8).
- **Impacto:** las afirmaciones de desempeño real quedan para ML-07.

## M-ML-006 — Contrato del artefacto con el backend

- **Decisión:** ONNX con `feature_schema.json`, `model_card.md`, `metrics.json`, SHA-256 y casos dorados de paridad; versión semántica `modelVersion`.
- **Justificación:** formato portable sin deserializar código, verificable en .NET.
- **Estado:** APROBADO con la constitution 1.0.0 (§13).
- **Impacto:** NV-23 del backend pasa de "ML.NET entrenado en una herramienta separada" a "carga ONNX producido por `TheHouse-ML`".

## M-ML-007 — Costo

- **Decisión:** $0: CPU local y GitHub Actions en repositorio público; sin servicios de ML administrados ni GPU en la nube.
- **Estado:** APROBADO con la constitution 1.0.0 (§15). El repositorio `thehouse-team/TheHouse-ML` se creó privado el 2026-11-09 y Victor lo hizo **público** el 2026-11-10 («si haslo publico»), igual que `TheHouse-Backend` y `TheHouse-Mobile`: CI sin límite de minutos y ruleset de `main` disponible en el plan Free.

## M-ML-008 — Separación con TheHouse-GenAI

- **Decisión:** `TheHouse-ML` calcula el puntaje de Smart Match y su desglose; `TheHouse-GenAI` (repositorio de Isai Axel Paniagua Francisco) solo los explica en lenguaje natural y nunca los calcula ni modifica; `TheHouse-Backend` orquesta ambos.
- **Estado:** APROBADO — coherente con el Artículo 2 de la constitución de `TheHouse-GenAI` y con la constitution 1.0.0 §2 de este repositorio.
- **Fecha:** 2026-11-09.
- **Impacto:** la spec NV-24 del backend debe ajustarse para consumir `TheHouse-GenAI` (decisión del backend, pendiente).
