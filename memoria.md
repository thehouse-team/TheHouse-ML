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
| ~~Inside Airbnb — Ciudad de México (`listings.csv.gz`, instantánea 2026-09-23)~~ | Atributos a nivel cuarto: servicios, tipo de cuarto, precio relativo | CC BY 4.0 verificada el 2026-11-10, pero sin uso: no cubre los 7 municipios, mide renta turística por noche y trae datos personales de anfitriones | **DESCARTADO (M-ML-019, 2026-11-18)** |
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

## M-ML-011 — Cambio de responsable

- **Decisión:** César Yael López Trujillo pasa a ser el responsable de `TheHouse-ML` (aprueba specs, da «EJECUTA» y aprueba releases). Victor Manuel Jose Jose queda como revisor.
- **Estado:** APROBADO — Victor Manuel Jose Jose — 2026-11-11. Aprobación humana por WhatsApp el 2026-11-11 a las 23:06–23:11. A la pregunta «¿te parece si me pasas el rol de responsable del repo? Lo registro en memoria.md como M-ML-011 y actualizo CONTRIBUTING y AGENTS» respondió: «simon» y «a eso solo dile a tu claude mandale un mensaje que yo los apruebo».
- **Fecha:** 2026-11-11.
- **Impacto:** el responsable vigente pasa a ser César Yael López Trujillo para aprobaciones, autorizaciones de ejecución y releases; Victor Manuel Jose Jose queda como revisor. Las aprobaciones anteriores de Victor siguen vigentes. Se actualizan `AGENTS.md` y `CONTRIBUTING.md`.

## M-ML-012 — Verificación automática de variables y retiro de ANUIES

- **Decisión:** (a) en la T01 cada variable queda registrada con `variables_status = "pending"` y `dictionary_url`; en la T02 una prueba falla si alguna variable registrada no existe en el archivo descargado, y solo entonces pasa a `"confirmed"`. (b) ANUIES queda **DESCARTADO** por ahora, porque no es indispensable y su licencia no está verificada.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-12.
- **Fecha:** 2026-11-12.
- **Impacto:** la confirmación de variables se automatiza en T02 contra el diccionario o las columnas de los archivos descargados; se actualiza M-ML-003 y se retira ANUIES de ML-02 1.0.1.

## M-ML-013 — Variables de ENVI y ENIGH confirmadas

- **Decisión:** confirmar, abriendo los archivos oficiales el 2026-11-17, las variables que se usarán de dos fuentes:
  - **ENVI 2020** (tabla `Bases de datos/TVIVIENDA.csv`, codificación latin-1): `P5_1` (tenencia; 1 = rentada), `P5_2` (cuánto se paga al mes), `P5_5` (motivo de renta), `ENT`, `TLOC`, `FACTOR`, `EST_DIS` y `UPM_DIS`. Queda **excluida** `P5_8` (renta imputada si se rentara), porque no es una renta efectivamente pagada. INEGI reemplazó `TVIVIENDA` el 2024-04-03 por una corrección de `P4_25_6`.
  - **ENIGH 2022** (tabla `viviendas.csv`, UTF-8 con BOM): `tenencia`, `renta`, `factor`, `ubica_geo`, `tam_loc`, `est_dis`, `upm`, `disp_agua`, `drenaje`, `disp_elect` y `combustible`. No se usa el alquiler imputado a propietarios.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-17. Verificación realizada por el responsable abriendo los archivos oficiales; registrada en `data/sources.toml` con `variables_status = "confirmed"`.
- **Fecha:** 2026-11-17.
- **Impacto:** T02 descarga los dos ZIP y el diccionario oficial de ENVI (`envi_2020_fd.xlsx`), verifica SHA-256 y contrasta los encabezados de los CSV; una variable confirmada ausente hace fallar el comando. El Censo 2020 se descargará **solo para Hidalgo (13) y Estado de México (15)**; Censo, DENUE e INPC quedan con `download_url` vacío hasta tener URL directa y hash verificados.

## M-ML-014 — Pesos de julio de 2024, deflactor INPC y alcance de la T03

- **Decisión:**
  - **(a)** Las rentas se expresan en **pesos de julio de 2024**, último mes disponible en el INPC de datos abiertos; es una **limitación documentada** (si la serie se actualiza, se cambia el mes de referencia en una versión nueva).
  - **(b)** El deflactor es el **INPC general** (la fila cuyo `CONCEPTO` termina en «Precios al Consumidor (INPC)», no la subyacente). ENVI 2020 usa el **promedio oct-dic 2020** (levantamiento del 26-oct al 18-dic-2020); ENIGH 2022 usa el **promedio ago-nov 2022** (levantamiento del 21-ago al 28-nov-2022). Factor = `INPC(2024-07) / promedio del periodo`. El archivo `inpc_indicador_mensual_csv.zip` tiene SHA-256 verificado y, aunque su diccionario dice que `VALOR` es «variación porcentual anual», los valores son niveles del índice base 2Q jul 2018 = 100 (2024-07 = 136.003).
  - **(c)** La prevalencia de servicios por municipio sale del **Censo 2020 en la T04**; la **T03 cubre renta y proporción de viviendas rentadas** (percentiles ponderados por entidad, tamaño de localidad y clase/tipo de vivienda).
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-17.
- **Fecha:** 2026-11-17.
- **Impacto:** `inegi_inpc` queda registrado en `data/sources.toml` con URL, hash y `variables_status = "confirmed"`; se agregan `P4_3` a ENVI y `tipo_viv` a ENIGH. La etapa `prepare` genera `national_baselines.parquet` y su reporte; la calibración regional (T04) queda para después.

## M-ML-015 — Censo 2020: fuente, archivos y códigos confirmados para la T04

- **Decisión:** registrar y descargar la fuente `inegi_censo_2020_viviendas_ca` del Censo de Población y Vivienda 2020, muestra del cuestionario ampliado, **solo para Hidalgo (13) y Estado de México (15)** y solo la tabla `Viviendas`, con cuatro archivos verificados por el responsable el 2026-11-17 abriendo los archivos oficiales:
  - `Censo2020_CA_hgo_csv.zip` (17,225,011 bytes, SHA-256 `5d0ff2b338acb93f8558ef2fed0500c1b4af004cdfc5ee34fb1fec7faa21e865`, tabla `Viviendas13.CSV`, latin-1);
  - `Censo2020_CA_mex_csv.zip` (41,189,169 bytes, SHA-256 `178a2a5c3a0b7b08a1ac878a5e94ca4e6c9b9d5851d98b8b07fe2f9d35672c2b`, tabla `Viviendas15.CSV`, latin-1);
  - `diccionario_cuestionario_ampliado_cpv2020.xlsx` (95,642 bytes, SHA-256 `e728cdac1872be856ba15c2902a77fb6c03e60c2d98619b7ae8988cc82afb6dc`, diccionario oficial);
  - `Censo2020_clasificaciones_CPV.xlsx` (210,951 bytes, SHA-256 `64c7d6974392132258d77b8697e92cb9f39ba4cd303fc5a9c0f32861f174f381`, catálogos oficiales).
- **Variables usadas:** `ENT`, `MUN`, `FACTOR`, `ESTRATO`, `UPM`, `TENENCIA`, `AGUA_ENTUBADA`, `DRENAJE`, `ELECTRICIDAD`, `INTERNET`, `COMBUSTIBLE`. Se retira `AGUA_ENT` (no existe en el diccionario) y se confirman las anteriores con `variables_status = "confirmed"` y `dictionary_url` apuntando al diccionario ampliado.
- **Códigos confirmados con el diccionario oficial:** `TENENCIA` 2 = «se paga renta» (1 propia, 3 prestada, 4 otra); `INTERNET` 7 sí / 8 no; `ELECTRICIDAD` 1 sí / 3 no; `AGUA_ENTUBADA` 1 dentro, 2 solo patio, 3 no tiene; `DRENAJE` 1–4 tiene / 5 no tiene; `COMBUSTIBLE` 1 leña o carbón, 2 gas, 3 electricidad, 4 otro, 5 no cocinan. El código 9 y el blanco significan «no especificado» y se excluyen del denominador de cada indicador.
- **Claves de los 7 municipios (catálogo oficial):** Hidalgo 013 Atotonilco de Tula, 041 Mixquiahuala de Juárez, 063 Tepeji del Río de Ocampo, 070 Tlahuelilpan, 076 Tula de Allende; Estado de México 045 Jilotepec, 112 Villa del Carbón.
- **Renta regional:** el Censo no trae monto. Para cada municipio se copian los percentiles p10–p90 de renta en pesos de julio-2024 de su estado según ENVI 2020 (de `national_baselines`, T03), con `level = "estado"` y `source = ENVI`.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-17.
- **Fecha:** 2026-11-17.
- **Impacto:** habilita la T04 (calibración regional con respaldo municipio → estado → nacional) y confirma la fuente y los códigos del Censo en `data/sources.toml`. La varianza usa linealización de Taylor para razón con UPM con reemplazo dentro de cada estrato. Se mantiene la licencia de libre uso de INEGI citando al autor y el producto, sin sugerir respaldo de INEGI.

## M-ML-016 — ML-02 T07: verificación de coordenadas con OpenStreetMap

- **Decisión:** ejecutar la **T07 antes que la T05**, porque la T05 (entorno de campus con DENUE) necesita las coordenadas verificadas que entrega la T07.
- **Fuente del seed:** las 10 universidades se copian del backend `TheHouse-Backend`, archivo `TheHouse.Application/Common/UniversitySeedData.cs`, commit `b7d75955513320a0a97aa5daf6b3bbe92e9514d2`, a `data/seed/universities_seed.csv` (en git). Procedencia registrada también en `data/sources.toml` (fuente `thehouse_seed_nv07_nv08_nv11`).
- **Uso de Nominatim:** una sola consulta por universidad a `https://nominatim.openstreetmap.org/search` con `format=jsonv2`, `limit=5`, `countrycodes=mx`, viewbox de ±0.1° alrededor de la coordenada del seed y `bounded=1`. Se respeta su política de uso: User-Agent propio `TheHouse-ML/0.1 (proyecto escolar UTTT)`, **máximo 1 petición por segundo**, **solo estas 10 consultas** y sin scraping. Cada respuesta se guarda en `data/raw/openstreetmap/<code>.json` (fuera de git) con su SHA-256 en `data/manifests/openstreetmap.json`; si el JSON ya existe, no se vuelve a pedir. Atribución `© OpenStreetMap contributors (ODbL)` en `reports/universities_check.md` y en `data/sources.toml`.
- **Regla de selección y umbral:** se elige el elemento OSM más cercano al seed cuyo tipo sea educativo (`class=amenity` y `type` en `university`, `college` o `school`). Distancia haversine: `status = "ok"` si es ≤ 200 m; `"revisar"` si es > 200 m; `"sin_coincidencia"` si no hay candidato educativo (columnas `osm_*` vacías).
- **Confirmación humana:** las coincidencias dudosas (estado `revisar` o `sin_coincidencia`, en particular las 3 coordenadas aproximadas del seed: `UAEH-ESTL`, `UMB-VDC` y `UCAN`) las confirma un humano en openstreetmap.org **antes de usarlas en la T05**.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-17. Autorización: «Apruebo M-ML-016 y EJECUTA ML-02 1.0.1, task T07 (antes que T05, porque T05 necesita coordenadas verificadas), en la rama feat/ml-02-universidades».
- **Fecha:** 2026-11-17.
- **Impacto:** habilita la T07 y entrega a `TheHouse-Backend` (NV-11) la tabla de coordenadas y la lista de universidades a corregir en `reports/universities_check.md`. Sin dependencias nuevas: HTTP y JSON con la biblioteca estándar.

## M-ML-017 — Verificación humana de coordenadas y correcciones a NV-11

- **Decisión:** continuar la T07 con la revisión humana de los casos dudosos, hecha en **openstreetmap.org el 2026-11-18** por César Yael López Trujillo y registrada en `data/seed/universities_manual.csv` (en git): 3 `ok_seed` (el marcador del seed cae sobre el nombre correcto en OSM), 2 `corregida` (ITSOEH y UAEH-ESTL) y 3 `no_localizada` (TESJI, UMB-VDC y UCAN).
- **Origen de coordenadas:** solo se guardan coordenadas tomadas de **OpenStreetMap** (Nominatim o referencia `way/…`), nunca de Google Maps, por su licencia. Las filas `corregida` conservan la referencia OSM que anotó el revisor en `osm_ref`.
- **Reglas de la coordenada final en `data/universities.csv`:** `ok` de T07 → coordenada OSM y `final_status = ok_osm`; fila manual `ok_seed` → coordenada del seed y `final_status = ok_manual`; `corregida` → lat/lon manuales y `final_status = corregida_manual`; `no_localizada` → coordenada del seed marcada `final_status = seed_sin_verificar`. Se conservan las columnas originales de la T07 y se agregan `final_lat`, `final_lon`, `final_status` y `final_source`.
- **No localizadas:** TESJI, UMB-VDC y UCAN conservan el seed como `seed_sin_verificar`. **T05 debe reportarlas por separado** y no tratarlas como coordenadas verificadas. En particular, el nombre de **UCAN** es dudoso: no hay elemento en OSM y posiblemente corresponda a «Instituto Canadiense A.C.» (carretera Tepetitlán–Tula, El Tesoro); debe confirmarlo el equipo del backend.
- **Aviso al backend:** hay que enviar a NV-11 las correcciones de `ITSOEH` (20.2333739, -99.2096330) y `UAEH-ESTL` (20.132062, -99.235535), la confirmación de `UAEH-ESTE`, `UAEH-ESAT` y `UMB-JILO`, y la duda de `UCAN`. La lista definitiva está en `reports/universities_check.md`.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-18. Autorización: «Apruebo M-ML-017 y continúa T07 de ML-02 1.0.1 en la rama feat/ml-02-universidades».
- **Fecha:** 2026-11-18.
- **Impacto:** cierra la verificación de coordenadas de la T07 sin consultas nuevas a Nominatim y sin dependencias nuevas; habilita a T05 para usar coordenadas finales y a `TheHouse-Backend` para corregir NV-11.

## M-ML-018 — DENUE 05/2026 para el entorno de campus (T05)

- **Decisión:** usar el **DENUE 05/2026** (modificado 2026-05-20) de los estados **13 (Hidalgo)** y **15 (Estado de México)** para la T05, con `variables_status = "confirmed"` y licencia «Términos de Libre Uso de la Información del INEGI». Archivos verificados/registrados: `denue_13_csv.zip` (15,059,273 bytes, SHA-256 `d0747958bc364177ea8d281047334cffd4d3505247f43967e08da2ca9c1d9ca5`, tabla `conjunto_de_datos/denue_inegi_13_.csv`, latin-1, 160,666 filas) y Edomex 05/2026 en dos ZIP del catálogo oficial, `denue_15_1_csv.zip` y `denue_15_2_csv.zip` (tamaños y SHA-256 los escribe `download` en `data/manifests/inegi_denue_vigente.json` en la primera descarga; se leen todos los CSV dentro de `conjunto_de_datos/` de cada uno). La URL originalmente indicada `denue_15_csv.zip` devuelve 404 de INEGI; el catálogo de descarga masiva 05/2026 publica Edomex en dos partes, cambio registrado como desviación en `evidence.md`.
- **Variables permitidas:** solo `id`, `codigo_act`, `per_ocu`, `cve_ent`, `cve_mun`, `latitud`, `longitud` y `fecha_alta`, leídas con `usecols` y por chunks. **Nunca** se leen ni guardan `nom_estab`, `raz_social`, `telefono`, `correoelec`, `www` ni campos de dirección (`tipo_vial`, `nom_vial`, `numero_ext`, etc.). Solo se publican **conteos agregados por campus, radio y categoría**.
- **Categorías y radios:** categorías SCIAN por prefijo de `codigo_act` — comida 722; abarrotes_super 4611 y 4621; farmacia 464111 y 464112; salud 621 y 622; papeleria_copias 465311 y 561431; lavanderia 812210; gimnasio 713943; transporte 485; bancos 522; total (todos los negocios). Radios de 1, 2 y 5 km con distancia haversine. Se usan ambos estados porque los campus de Tepeji y Jilotepec están cerca de la frontera Hidalgo–Edomex.
- **Coordenadas sin verificar:** TESJI, UMB-VDC y UCAN quedan con `final_status = seed_sin_verificar` y se reportan en una sección aparte, con la advertencia de que su coordenada no está verificada (M-ML-017).
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-18. Autorización: «Apruebo M-ML-018 y EJECUTA ML-02 1.0.1, task T05, en la rama feat/ml-02-denue. Rutas permitidas: src/thehouse_ml/, tests/, data/sources.toml, data/manifests/, reports/, memoria.md y specs/ml-02-data-sources/. Sin dependencias nuevas».
- **Fecha:** 2026-11-18.
- **Impacto:** habilita la T05 con los tres ZIP registrados en `data/manifests/inegi_denue_vigente.json`; la salida es `data/processed/campus_context.parquet` (nombre de la spec) y `reports/denue_campus.md` lleva la atribución «Fuente: INEGI, DENUE 05/2026»; sin dependencias nuevas (pandas, pyarrow y biblioteca estándar).
- **Adenda del 2026-11-18 (aprobación de hallazgos 16 y 17):** el responsable aprueba (a) sustituir la URL caída `denue_15_csv.zip` por los dos ZIP oficiales `denue_15_1_csv.zip` (50,572,588 B; SHA-256 `23021fc24a2046e1aa3ad8c21137ca57988ded26b6f293ec0c929f59dd2cb1ae`) y `denue_15_2_csv.zip` (30,236,017 B; SHA-256 `ad53e5fb65f15b491e140a80e79b207b95eb0320e4a69822d5a3cefb7652bc20`), y (b) usar el nombre de la spec, `data/processed/campus_context.parquet`, en lugar de `denue_campus.parquet`. Aprobación: «Apruebo los hallazgos: Hallazgo 16… Hallazgo 17: usa el nombre de la spec» — César Yael López Trujillo, 2026-11-18. Además, se corrigió la reescritura innecesaria de `data/universities.csv` por CRLF/LF: `write_universities_csv` escribe con LF y no toca el archivo si el contenido no cambia.

## M-ML-019 — Inside Airbnb descartado; catálogo de cuartos sintético en ML-03

- **Decisión:** **DESCARTAR** Inside Airbnb como fuente de ML-02 (T06). Motivos: (1) en México solo publica datos descargables de Ciudad de México y no cubre Tula, Tepeji, Atotonilco, Mixquiahuala, Tlahuelilpan, Jilotepec ni Villa del Carbón; (2) mide rentas turísticas por noche, no cuartos para estudiantes por mes; (3) sus archivos traen datos personales de anfitriones (nombres, fotos), contrarios a la constitución §7; (4) las rentas ya están cubiertas por ENVI 2020 y ENIGH 2022 (M-ML-014), los servicios de la vivienda por el Censo 2020 (M-ML-015) y el entorno por el DENUE (M-ML-018).
- **Consecuencia:** el catálogo de cuartos se generará **sintético en ML-03**, calibrado con esas fuentes, sin datos personales, mapeado a NV-11 y etiquetado como sintético en todo reporte (constitution §8). No se descarga ni se verifica ningún archivo de Inside Airbnb y se retira la URL no verificada de `listings.csv.gz` de `data/sources.toml`.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-18. Autorización: «Apruebo M-ML-019 y EJECUTA ML-02 1.0.1, task T06, en la rama docs/ml-02-t06-descartada. Rutas permitidas: data/sources.toml, memoria.md, specs/ml-02-data-sources/, reports/variables_check.md, src/thehouse_ml/ y tests/ (solo si hace falta para que la fuente descartada no se descargue ni se verifique). Sin dependencias nuevas y sin descargar nada de Inside Airbnb».
- **Fecha:** 2026-11-18.
- **Impacto:** T06 queda DESCARTADA, no completada; R06 se reorienta a verificar el descarte y la matriculación en el registro; se agrega el estado `discarded` con `status_reference` en `data/sources.toml`, el cual `download` omite con `descartada: M-ML-019` y el reporte de variables muestra como descartado sin borrar las otras secciones. Sin dependencias nuevas; se actualizan spec, plan, task y evidence de ML-02 1.0.1.
