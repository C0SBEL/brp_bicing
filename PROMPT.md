# EDA Profesional — Bicing Barcelona (Optimización de Rutas de Redistribución)

## Contexto del proyecto
Proyecto académico de máster (La Salle BCN - Data Science) sobre optimización
de rutas de redistribución del Bicing de Barcelona (Bike Rebalancing Problem).
Este notebook es el EDA completo que precede al entrenamiento de modelos LSTM
y XGBoost/LightGBM para predecir disponibilidad de bicicletas por estación,
siguiendo el paradigma predict-then-optimize.

El notebook debe ser visualmente atractivo, narrativo y académicamente riguroso.
Cada sección debe contar una historia coherente: qué se analiza, por qué es
relevante para el problema del BRP, qué dice la literatura al respecto y qué
revelan nuestros datos. Las celdas Markdown no son títulos secos — son texto
académico que contextualiza, cita papers y saca conclusiones. El objetivo final
es justificar qué variables entrarán en los modelos LSTM y XGBoost.

## Fuentes de datos y descarga

### Dataset 1 — Información de estaciones (estático)
- Fuente: https://opendata-ajuntament.barcelona.cat/data/es/dataset/informacio-estacions-bicing
- Formato: archivos .7z mensuales con CSV dentro
- Usar el archivo más reciente disponible (Oct 2025 o el último disponible)
- Columnas esperadas: station_id, name, lat, lon, altitude, capacity,
  nearby_distance, groups, post_code (algunas pueden estar vacías)
- Descargar con requests + py7zr
- Inspeccionar columnas reales antes de asumir nombres
- Descartar columnas completamente vacías o con un único valor constante

### Dataset 2 — Estado de las estaciones (serie temporal)
- Fuente: https://opendata-ajuntament.barcelona.cat/data/es/dataset/estat-estacions-bicing
- Período: Noviembre 2023 – Octubre 2025 (24 archivos .7z mensuales)
- Formato: CSV dentro de .7z, un registro por estación cada ~5 minutos
- Columnas esperadas: station_id, num_bikes_available,
  num_bikes_available_types (mecánicas/eléctricas), num_docks_available,
  last_reported (Epoch Unix timestamp), status, is_charging_station
- IMPORTANTE: last_reported está en formato Epoch Unix — convertir a datetime UTC
  y luego a Europa/Madrid
- IMPORTANTE: el volumen es ~250M filas en bruto. Estrategia obligatoria:
  procesar mes a mes, agregar a nivel horario (media por station_id y hora
  redondeada), concatenar y guardar como parquet antes de continuar.
  Nunca cargar los 24 meses en memoria en bruto.

### Dataset 3 — Meteorología horaria
- Usar la API gratuita de Open-Meteo (sin API key necesaria)
- Coordenadas Barcelona: lat=41.3874, lon=2.1686
- Variables: temperature_2m, precipitation, relative_humidity_2m,
  wind_speed_10m, surface_pressure, weather_code
- Período: 2023-11-01 a 2025-10-31, frecuencia horaria, timezone Europa/Madrid
- Endpoint:
  https://archive-api.open-meteo.com/v1/archive?latitude=41.3874&longitude=2.1686
  &start_date=2023-11-01&end_date=2025-10-31
  &hourly=temperature_2m,precipitation,relative_humidity_2m,
  wind_speed_10m,surface_pressure,weather_code&timezone=Europe/Madrid
- Si la API falla: reintentar 3 veces con backoff exponencial

### Dataset 4 — Festivos y calendario laboral
- Fuente: https://opendata-ajuntament.barcelona.cat/data/es/dataset/calendari-festes-laborals
- Descargar versión en español (CSV)
- Extraer: fecha, tipo_dia, descripción del festivo si aplica
- Derivar con pandas: is_festivo, is_weekend, is_laborable, day_of_week, month

## Estructura del notebook

### Sección 0 — Setup y configuración
- Librerías: pandas, numpy, matplotlib, seaborn, plotly (express + graph_objects),
  folium, scipy, scikit-learn (PCA, RandomForestRegressor, StandardScaler,
  OrdinalEncoder), statsmodels, missingno, py7zr, requests, pyarrow, tqdm
- Celda inicial que verifique e instale automáticamente las librerías faltantes
- Paleta de colores: BICING_RED = "#E3000B" usada consistentemente
- Configurar matplotlib rcParams para estilo limpio y profesional
- Funciones auxiliares: epoch_to_datetime(), save_figure(), cyclic_encode()
- Crear directorios: data/, outputs/figures/

### Sección 1 — Descarga y carga de datos
- Descarga automatizada de los 24 archivos .7z con barra de progreso (tqdm).
  Si los archivos ya existen en disco, no re-descargar.
- Procesamiento mes a mes: descomprimir en memoria → leer CSV → convertir Epoch →
  redondear timestamp a hora → agregar (mean) por station_id y hora →
  concatenar → guardar como data/bicing_hourly.parquet
- Descargar dataset estaciones → data/estaciones.parquet
- Descargar meteorología via Open-Meteo → data/meteo_barcelona.parquet
- Descargar festivos → data/festivos_barcelona.csv
- Mostrar shape, dtypes y head(3) de cada dataset antes del merge
- Merge final:
    bicing_hourly
    LEFT JOIN meteo      ON hora redondeada
    LEFT JOIN festivos   ON fecha (sin hora)
    LEFT JOIN estaciones ON station_id
  → guardar como data/bicing_full.parquet
- Celda Markdown que explique la lógica del merge, los posibles registros
  perdidos en cada join y por qué se elige LEFT JOIN

### Sección 2 — Exploración inicial y calidad de datos
- df.describe() con percentiles [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
- Heatmap missingno + tabla de % nulos por columna ordenada descendente
- Rango temporal: primera fecha, última fecha, cobertura mensual en heatmap
  estilo GitHub (meses en eje x, % datos disponibles en color)
- Tabla de gaps temporales detectados (> 1 hora) con fecha inicio, fin y duración.
  La celda Markdown debe mencionar que el dataset tiene incidencias técnicas
  conocidas documentadas en Open Data BCN y que pueden existir gaps similares
  en nuestro período 2023-2025
- Distribución de estaciones por tipo: pie chart + barplot
- Verificar cuántas estaciones únicas aparecen en los datos vs las 557 oficiales
  del servicio Bicing (fuente: bicing.barcelona)

### Sección 3 — EDA geoespacial

La celda Markdown introductoria debe explicar que la dimensión espacial del
Bicing no es homogénea: Barcelona tiene un relieve pronunciado que genera
asimetrías sistemáticas en el uso. Mencionar que Cortez-Ordoñez et al. (2024,
Heliyon) identifican la altitud como una de las variables más influyentes para
el Bicing Barcelona, y que Cipriano et al. (2021, Applied Sciences) proponen
la tasa de ocupación relativa respecto al vecindario como métrica clave para
detectar estaciones crónicamente desequilibradas.

- Mapa interactivo Folium:
    · Marcadores circulares, tamaño proporcional a capacidad
    · Color en gradiente según altitud (azul=bajo → rojo=alto)
    · Tooltip: nombre, capacidad, altitud, tipo
    · LayerControl: capa mecánicas y capa eléctricas separadas
- Histograma + KDE de altitud con línea vertical en la mediana
- Histograma superpuesto de capacidad por tipo de estación + boxplot comparativo
- Scatter altitud vs tasa_ocupacion media histórica con regresión OLS,
  banda de confianza IC 95% y R² anotado. La celda Markdown posterior debe
  interpretar el signo y significatividad de la pendiente: si es negativa,
  confirma el efecto gravitacional de Cortez-Ordoñez et al.; si no es
  significativa, discutir por qué podría diferir del paper.
- Calcular tasa_ocupacion_relativa por estación: para cada estación,
  media histórica de (tasa_ocupacion_estacion − media_tasa_vecinas_500m).
  Valor positivo = emisora neta crónica; negativo = receptora neta crónica.
  Visualizar en mapa Folium coloreado por esta métrica (rojo=emisora, azul=receptora).
  La celda Markdown debe explicar que esta métrica proviene de Cipriano et al.
  (2021) y que será relevante para el módulo de optimización de rutas del pipeline.

### Sección 4 — EDA series temporales de uso

#### 4.1 Variables derivadas
La celda Markdown debe explicar que estas variables son las candidatas a features
del modelo predictivo, justificando cada una con la literatura correspondiente.

Calcular y añadir al dataframe (ordenar siempre por station_id y hora antes
de aplicar diff/shift):

tasa_ocupacion = num_bikes_available / capacity
demanda_neta   = num_bikes_available.diff()  # por station_id

# Lags de persistencia — justificados por Peláez-Rodríguez et al. (2024):
bikes_lag_1h   = num_bikes_available.shift(1)
bikes_lag_2h   = num_bikes_available.shift(2)
bikes_lag_24h  = num_bikes_available.shift(24)
bikes_lag_168h = num_bikes_available.shift(168)  # mismo slot semana anterior

# Variable instancias analógicas — Peláez-Rodríguez et al. (2024):
# Media histórica de num_bikes_available en registros con mismo day_of_week,
# mismo hour, temperatura en bin de 5°C y precipitación en bin (lluvia/no lluvia).
# Implementar con groupby sobre bins + expanding mean para evitar data leakage.
media_analogos = ...

# Codificación cíclica — Giner Fabregat et al. (2025):
hour_sin  = np.sin(2 * np.pi * hour / 24)
hour_cos  = np.cos(2 * np.pi * hour / 24)
dow_sin   = np.sin(2 * np.pi * day_of_week / 7)
dow_cos   = np.cos(2 * np.pi * day_of_week / 7)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

Incluir un gráfico explicativo de la codificación cíclica: dos subplots lado
a lado mostrando la variable hour como entero (izquierda) vs como par sin/cos
(derecha), ilustrando por qué la representación cíclica es superior para modelos
secuenciales como LSTM. La celda Markdown debe explicar este concepto citando
a Giner Fabregat et al. (2025).

#### 4.2 Evolución temporal del sistema
- Línea de disponibilidad media diaria Nov 2023–Oct 2025 con banda ±1 std,
  coloreada por año (plotly con rangeslider interactivo)
- Descomposición STL de la serie agregada diaria (statsmodels):
  subplots de tendencia, estacionalidad y residuo.
  Identificar y anotar en el gráfico semanas con residuos anómalos.

#### 4.3 Patrones temporales
La celda Markdown introductoria debe citar a Bustamante et al. (2022, Sustainable
Cities and Society), que documentan que el Bicing se usa predominantemente en días
laborables con pico los viernes, confirmando su función principal como transporte
de commuting. Presentar los siguientes gráficos como la verificación de esa
hipótesis con datos 2023-2025.

- Dos heatmaps seaborn lado a lado: hora del día × día de la semana con
  tasa_ocupacion media, uno para días laborables y otro para festivos.
  Estos son los gráficos más informativos del EDA — deben ser grandes y claros.
- Barplot de uso medio por día de la semana con la barra del viernes destacada
  en BICING_RED. La celda Markdown posterior debe indicar si se confirma o no
  el pico de viernes documentado por Bustamante et al.
- Barplot de diferencia porcentual festivo–laborable por franja horaria
  (madrugada 0-6h / mañana 6-12h / tarde 12-18h / noche 18-24h)
- Boxplot de tasa_ocupacion por hora del día (24 cajas) con líneas horizontales
  en umbrales de estación vacía (<10%) y saturada (>90%)
- Barplot de % de registros con tasa < 0.1 y tasa > 0.9 por hora del día,
  para mostrar en qué momentos el problema de redistribución es más crítico
- Top 10 estaciones con mayor frecuencia de vaciado: barplot horizontal
- Top 10 estaciones con mayor frecuencia de saturación: barplot horizontal

#### 4.4 Análisis gravitacional por altitud
- Violin plot de demanda_neta por cuartil de altitud (Q1 más baja → Q4 más alta)
  con medias marcadas y test Kruskal-Wallis con p-value anotado
- Scatter lat/lon coloreado por tasa_ocupacion media histórica:
  mapa de calor de zonas de déficit y exceso crónicos sobre Barcelona

### Sección 5 — EDA meteorología

La celda Markdown introductoria debe explicar que la meteorología es uno de
los predictores más robustos del uso del Bicing. Citar a Cortez-Ordoñez et al.
(2024) que identifican temperatura y presión atmosférica entre las variables
más influyentes, y a Peláez-Rodríguez et al. (2024) cuya variable de instancias
analógicas usa precisamente temperatura y precipitación como criterios de similitud.

#### 5.1 Series y distribuciones
- Serie temporal de temperatura, precipitación y viento (3 subplots plotly,
  rangeslider interactivo)
- Histograma + KDE por variable meteorológica
- Barplot de frecuencia de weather_code agrupado en categorías:
  soleado / nublado / lluvia / tormenta

#### 5.2 Relación meteo–demanda
- Scatter plots (plotly, 10k puntos muestreados con random_state=42):
  tasa_ocupacion media vs temperatura / precipitación / viento / presión,
  con línea LOWESS superpuesta y R² de regresión lineal anotado
- Boxplot de demanda_neta por bins de temperatura cada 5°C
- Boxplot de demanda_neta por condición lluvia (precipitación > 0.1mm: sí/no)
- Heatmap de correlación entre variables meteorológicas y tasa_ocupacion /
  demanda_neta. La celda Markdown posterior debe discutir si la jerarquía de
  correlaciones observada coincide con la de Cortez-Ordoñez et al. (2024):
  ¿es la presión realmente más relevante que la humedad o el viento en
  nuestros datos?

### Sección 6 — EDA calendario y festivos

La celda Markdown introductoria debe mencionar que Peláez-Rodríguez et al.
(2024) reportan que la selección de características mejora el R² de los modelos
ML un 149% de media para el Bicing Barcelona, siendo las variables calendarias
parte esencial del conjunto seleccionado. Esta sección cuantifica ese impacto.

- Violin plot de tasa_ocupacion por tipo de día (laborable / fin de semana /
  festivo) con test Kruskal-Wallis y p-value anotado
- Heatmap de demanda media por mes × hora del día, que captura simultáneamente
  la estacionalidad anual y el patrón diario intrahorario
- Top 5 festivos con mayor desviación respecto al patrón medio de su día de
  semana, para identificar qué festivos tienen comportamiento más anómalo y
  deben ser capturados explícitamente por el modelo

### Sección 7 — Selección de variables (análisis multivariante)

La celda Markdown introductoria debe enmarcar esta sección como el núcleo
analítico del EDA: aquí se decide qué entra al modelo y por qué, combinando
evidencia estadística propia con los hallazgos de la literatura.

#### 7.1 Correlación y colinealidad
Variables candidatas:
[num_bikes_available, tasa_ocupacion, tasa_ocupacion_relativa,
 bikes_lag_1h, bikes_lag_2h, bikes_lag_24h, bikes_lag_168h, media_analogos,
 temperature_2m, precipitation, relative_humidity_2m, wind_speed_10m,
 surface_pressure, altitude, capacity, hour, day_of_week, month,
 hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_festivo]

- Heatmap triángulo superior con anotaciones, paleta divergente RdBu_r,
  triángulo inferior enmascarado
- Identificar con recuadro rojo los pares con |r| > 0.85: son colineales.
  La celda Markdown debe documentar cuál de cada par se descarta y por qué.
- Scatter matrix de las 6 variables con mayor correlación con tasa_ocupacion.
  La celda Markdown debe comparar si temperatura y altitud aparecen entre
  las más correlacionadas, como predice Cortez-Ordoñez et al. (2024).

#### 7.2 PCA
- StandardScaler → PCA completo sobre todas las variables candidatas numéricas
  (imputar nulos con mediana antes de escalar)
- Gráfico de varianza explicada acumulada con línea discontinua en 95%
- Biplot PC1 vs PC2: 10k puntos coloreados por tasa_ocupacion,
  flechas de loadings con etiquetas para las 8 variables de mayor contribución
- Tabla de loadings de las 3 primeras componentes ordenada por |loading| desc
- La celda Markdown de conclusiones debe responder: ¿los lags temporales
  forman una componente separada de la meteorología? ¿La codificación cíclica
  sin/cos aparece como más informativa que los enteros? ¿Qué grupos de
  variables aportan información realmente independiente?

#### 7.3 ANOVA para variables categóricas
Para cada variable categórica [hour_bin (4 franjas), day_of_week, is_festivo,
is_weekend, station_type, season, temp_bin (quintiles), lluvia_bin]:
  1. Violin plot de demanda_neta por categoría con medias marcadas en BICING_RED
  2. Test ANOVA (scipy.stats.f_oneway) + Kruskal-Wallis como alternativa
     no paramétrica
  3. F-statistic y p-value mostrados bajo el gráfico con interpretación en texto

Tabla resumen al final: variable → F-statistic → p-value → decisión (incluir/excluir)

#### 7.4 Feature importance proxy (Random Forest)
- RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
- Target: tasa_ocupacion
- Features: todas las candidatas (imputar nulos con mediana, OrdinalEncoder
  para categóricas)
- Muestra estratificada de 200k filas (estratificar por station_id y mes)
- Barplot horizontal de importancias ordenadas descendente con línea vertical
  en el umbral de importancia acumulada del 80%
- La celda Markdown debe comparar los resultados con PCA y ANOVA, señalando
  coincidencias y discrepancias. Debe también comparar con Peláez-Rodríguez
  et al. (2024), que reportan R²=0.86 para RF en el Bicing Barcelona, y discutir
  si media_analogos aparece entre las top features tal como predice ese paper
  (mejora del 29.75% en R² al añadirla).

### Sección 8 — Conclusiones y variables seleccionadas

- Tabla resumen final como pandas Styler exportado a PNG:
  | Variable | Tipo | Fuente | Correlación | PCA | ANOVA F | RF Importance | Decisión | Justificación |

- Texto narrativo académico de 250-300 palabras en español que cubra:
  1. Qué patrones principales se han descubierto y si confirman o matizan
     los hallazgos del estado del arte
  2. Qué variables entran al modelo y por qué, con respaldo estadístico
  3. Diferencias de tratamiento entre LSTM y XGBoost:
     LSTM prioriza lags temporales + codificación cíclica sin/cos + media_analogos;
     XGBoost puede usar enteros para calendarias y aprovecha mejor features
     estáticas de estación como altitud y capacidad
  4. Variables descartadas y razón (colinealidad, baja importancia, redundancia)
  5. Consideraciones de preprocesamiento para el siguiente módulo

- Guardar outputs/selected_features.json con estructura:
  {
    "lstm_features": [...],
    "xgboost_features": [...],
    "target": "tasa_ocupacion_t+h",
    "notes": "El horizonte h se define en el módulo de modelado."
  }
- Guardar data/bicing_full.parquet con todas las variables derivadas incluidas
- Guardar todos los gráficos en outputs/figures/ en PNG dpi=150

## Requisitos técnicos

### Rendimiento
- Procesar estado de estaciones mes a mes, nunca en bruto los 24 meses juntos
- Parquet con compresión snappy para persistencia intermedia
- Scatter plots: máximo 10k puntos muestreados con random_state=42
- Dtypes eficientes: station_id como int32, tasa_ocupacion como float32
- Si un archivo .7z falla al descargar: registrar el error y continuar

### Estilo visual
- BICING_RED = "#E3000B" como color primario en todos los gráficos
- Todos los gráficos: título descriptivo en negrita, ejes etiquetados con
  unidades, fuente de datos en pie de figura (tamaño 8, color gris)
- Plotly para series temporales, mapas y scatter interactivos
- Seaborn/matplotlib para matrices, boxplots, violins y barplots
- Tamaños estándar: (14, 6) para series temporales, (10, 8) para matrices
- Guardar cada figura en outputs/figures/ como PNG dpi=150

### Estructura del notebook
- Cada sección abre con una celda Markdown que contextualiza el análisis,
  menciona la relevancia para el BRP y cita los papers pertinentes de forma
  natural, como haría un investigador al redactar un artículo
- Cada sección cierra con una celda Markdown de conclusiones con 3-5 bullets
  concretos sobre los hallazgos, redactados en tono académico
- Añadir # %% entre secciones para compatibilidad con VSCode
- El notebook debe ejecutarse de principio a fin sin errores con kernel restart limpio

## Papers de referencia para citar en las celdas Markdown
- Peláez-Rodríguez et al. (2024). Expert Systems with Applications, 238, 122264.
  Aportación clave: RF y CNN mejores modelos; variable análogos meteorológicos;
  selección de características mejora R² un 149% de media.
- Cortez-Ordoñez et al. (2024). Heliyon, 10, e37608.
  Aportación clave: variables más influyentes son temperatura, mes, presión,
  altitud y festivos; impacto modesto de variables espaciales de equipamientos.
- Bustamante et al. (2022). Sustainable Cities and Society, 83, 103929.
  Aportación clave: uso predominante en laborables; pico viernes; efecto
  de la infraestructura ciclista en la demanda.
- Cipriano et al. (2021). Applied Sciences, 11(15), 6967.
  Aportación clave: tasa de ocupación relativa respecto al vecindario (radio 500m)
  como métrica de desequilibrio; partición por día de semana como mejor estrategia.
- Giner Fabregat et al. (2025). Sustainability, 17, 4316.
  Aportación clave: codificación trigonométrica para variables cíclicas;
  pipeline integrado descriptivo + predictivo + optimización para el Bicing.