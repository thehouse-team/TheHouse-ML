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

## M-ML-003 — Fuentes de datos: bases nacionales y calibración regional

- **Decisión:** los datos se trabajan en dos niveles. **Nacional** para obtener bases y proporciones con muestra suficiente; **regional** para calibrarlas a los 7 municipios de las universidades del seed (Hidalgo: Tula de Allende, Tepeji del Río de Ocampo, Tlahuelilpan, Atotonilco de Tula y Mixquiahuala de Juárez; Estado de México: Jilotepec y Villa del Carbón), con respaldo municipio → estado → nacional cuando la muestra no alcanza.

| Fuente | Uso | Licencia y condiciones | Estado |
| --- | --- | --- | --- |
| INEGI — Censo de Población y Vivienda 2020, muestra del cuestionario ampliado (`Viviendas_CA`) | Tenencia (`TENENCIA`), servicios, factor de expansión (`FACTOR`) y municipio (`MUN`). No pregunta el monto de renta (verificado el 2026-11-10) | Términos de libre uso de la información del INEGI: copiar, adaptar y publicar citando a INEGI como autor y el producto; no sugerir respaldo de INEGI | APROBADO |
| INEGI — Encuesta Nacional de Vivienda (ENVI) 2020 (con Infonavit y SHF) | Monto y motivo de renta, gasto en vivienda; niveles nacional y estatal | Términos de libre uso de INEGI | APROBADO |
| INEGI — ENIGH 2022 | Validación del gasto en renta de inquilinos; no se usa el alquiler imputado a propietarios | Términos de libre uso de INEGI | APROBADO |
| INEGI — DENUE (edición vigente) | Establecimientos con coordenadas alrededor de cada campus (factor de extras) | Términos de libre uso de INEGI | APROBADO |
| INEGI — INPC | Actualizar montos de 2020 a pesos del año de ejecución | Términos de libre uso de INEGI | APROBADO |
| Inside Airbnb — Ciudad de México (`listings.csv.gz`, instantánea 2026-09-23) | Atributos a nivel cuarto: servicios, tipo de cuarto, precio relativo | CC BY 4.0, verificada el 2026-11-10 en su página de políticas: cita formal de dataset con ciudad y fecha, descargar una sola vez, sin scraping, sin republicar los datos (se publican solo agregados con atribución); los datos archivados requieren solicitud | APROBADO |
| OpenStreetMap | Verificar coordenadas de campus, en especial las 3 aproximadas del seed (UAEH-ESTL, UMB-VDC, UCAN) | ODbL: atribución y misma licencia para bases derivadas | APROBADO |
| Seed de TheHouse (NV-07, NV-08, NV-11) | Universidades y rangos de referencia de la región | Datos propios | APROBADO |
| ANUIES — matrícula por institución | Posible ponderación de universidades | Licencia no verificada y fuente no indispensable por ahora | DESCARTADO (M-ML-012) |
| Señales reales de TheHouse (favoritos y visitas) | Etiquetas reales de interés | Exportación anonimizada definida por una spec del backend | PENDIENTE |

- **Reglas:** toda estimación de encuesta usa su factor de expansión; una estimación municipal solo se usa con al menos 30 viviendas rentadas en la muestra y coeficiente de variación ≤ 30 %; solo se usan productos de descarga pública (no el laboratorio de microdatos); ninguna variable se usa sin confirmarla en el diccionario oficial.
- **Estado:** APROBADO — Victor Manuel Jose Jose — 2026-11-10 («pues si te apruebo que revises si el censo trae lo de las rentas, pues la region si esta bien, pues si en el municipio creo que hay varias rentas las suficientes y si se usa airbnb usa la licencia y lo demas tambien en pocas palabras si a todo»): ML-02 1.0.0 y sus decisiones de la §10. Pedido original de Victor (2026-11-10): «que los datos sean a nivel México para sacar el porcentaje o las bases del proyecto y para el final solo se base en una sola área».
- **Justificación:** en pocos municipios las encuestas tienen muy pocas viviendas; usar lo nacional como base y lo regional como ajuste da estimaciones estables y a la vez locales. INEGI describe viviendas, no cuartos, por eso se conserva una fuente a nivel cuarto.
- **Impacto:** ML-02 1.0.0-borrador revisada el 2026-11-10 (9 requisitos, 9 tasks). Los datos crudos nunca se suben al repositorio (constitution §7).

## M-ML-004 — Fuentes descartadas

- **Decisión:** no usar:
  - EconLab de Banxico: acceso solo por proyecto de investigación con convenio.
  - Scraping de Inmuebles24, Vivanuncios, Propiedades.com o Pincali: sus términos no lo autorizan.
  - Properati: licencia y cobertura de rentas en México no confirmadas.
  - Datasets de citas o perfiles personales (por ejemplo OkCupid): datos personales sensibles.
- **Estado:** APROBADO — Victor, 2026-11-10, junto con ML-02 (M-ML-010).
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

## M-ML-009 — ML-01 aprobada: base del proyecto y licencia MIT

- **Decisión:** aprobar ML-01 1.0.0 (paquete con `uv`, CLI de cinco etapas, ruff, mypy estricto, pytest con cobertura ≥ 80 %, pip-audit, gitleaks, CI en Linux y Windows y ruleset de `main`) y licenciar el código con MIT. La licencia no cubre los datos de terceros, que conservan la suya.
- **Estado:** APROBADO — Victor Manuel Jose Jose — 2026-11-10 («Apruebo y EJECUTA»: ML-01 1.0.0 con licencia MIT; tasks T01–T05; rutas `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `src/`, `tests/`, `.github/workflows/`, `README.md`, `INSTALACION.txt` y `specs/ml-01-foundation/`; ruleset de `main`).
- **Fecha:** 2026-11-10.
- **Impacto:** el repositorio pasa de solo documentos a un paquete instalable; ML-02 puede ejecutarse cuando se aprueben ella y las fuentes de M-ML-003.

## M-ML-010 — ML-02 aprobada: datos nacionales de INEGI y calibración regional

- **Decisión:** aprobar ML-02 1.0.0 con sus decisiones: región de 7 municipios; regla de suficiencia (≥ 30 viviendas rentadas y CV ≤ 30 %, respaldo estado → país); DENUE vigente; Inside Airbnb Ciudad de México 2026-09-23 bajo CC BY 4.0; ANUIES solo si su licencia lo permite.
- **Verificaciones del 2026-11-10:** el Censo 2020 (muestra) no pregunta el monto de renta: la pregunta 23 «Tenencia» solo pregunta si se paga renta y el diccionario de `Viviendas_CA` (83 variables) no tiene monto; sí tiene `TENENCIA`, `FACTOR` y `MUN`. Por eso la renta sale de ENVI 2020 (estatal) y el Censo aporta la proporción de renta y los servicios por municipio. Inside Airbnb publica bajo CC BY 4.0 y su instantánea de Ciudad de México del 2026-09-23 está disponible (17.2 MB).
- **Estado:** APROBADO — Victor Manuel Jose Jose — 2026-11-10 («pues si te apruebo que revises si el censo trae lo de las rentas, pues la region si esta bien, pues si en el municipio creo que hay varias rentas las suficientes y si se usa airbnb usa la licencia y lo demas tambien en pocas palabras si a todo»): ML-02 1.0.0 y sus decisiones de la §10. La ejecución requiere «EJECUTA».
- **Fecha:** 2026-11-10.
- **Impacto:** la renta regional queda a nivel estatal, no municipal; se contrasta con el seed de TheHouse y, más adelante, con cuartos reales (ML-07).

## M-ML-012 — Verificación automática de variables y retiro de ANUIES

- **Decisión:** (a) en la T01 cada variable queda registrada con `variables_status = "pending"` y `dictionary_url`; en la T02 una prueba falla si alguna variable registrada no existe en el archivo descargado, y solo entonces pasa a `"confirmed"`. (b) ANUIES queda **DESCARTADO** por ahora, porque no es indispensable y su licencia no está verificada.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-12.
- **Fecha:** 2026-11-12.
- **Impacto:** la confirmación de variables se automatiza en T02 contra el diccionario o las columnas de los archivos descargados; se actualiza M-ML-003 y se retira ANUIES de ML-02 1.0.1.
