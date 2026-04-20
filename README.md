# Optimización de Rutas de Redistribución de Bicicletas del Bicing de Barcelona

**Máster Universitario en Data Science — La Salle, Universitat Ramon Llull**  
Agustí Costabella · Blanca Gallardo · Ferran Tubert · Abril 2026

**Asignaturas:** MD008 – Inteligencia artificial para la ciencia de los datos · MD009 – Herramientas avanzadas de análisis de datos

## Descripción

Este proyecto aborda el **Bike Rebalancing Problem (BRP)** del sistema Bicing de Barcelona mediante un pipeline integrado de predicción y optimización. El enfoque combina modelos supervisados para estimar la disponibilidad neta de bicicletas por estación a corto plazo con un módulo de optimización de rutas de recogida y entrega sujeto a restricciones operativas, siguiendo el paradigma **predict-then-optimize**.

Los resultados muestran que los modelos basados en *gradient boosting* (XGBoost) ofrecen el mejor equilibrio entre precisión, estabilidad e interpretabilidad, y que el esquema predict-then-optimize mejora la capacidad de anticipación frente a estrategias reactivas.

## Estructura del repositorio

```
brp_bicing/
├── data/
│   ├── raw/                    # Archivos .7z mensuales de Open Data BCN (oct 2023 – sep 2025)
│   └── processed/              # Parquets generados por el pipeline
│       ├── bicing_hourly.parquet   # Serie horaria agregada por estación
│       ├── bicing_stations.parquet # Metadatos estáticos de estaciones
│       ├── meteo_barcelona.parquet # Variables meteorológicas horarias
│       ├── bicing_clean.parquet    # Dataset limpio post-EDA
│       └── bicing_full.parquet     # Dataset completo con features derivadas
├── models/
│   ├── lstm_bicing.keras       # Modelo LSTM entrenado
│   └── xgboost_bicing.ubj      # Modelo XGBoost entrenado
├── notebooks/
│   ├── EDA.ipynb               # Análisis exploratorio de datos
│   ├── XGBoost.ipynb           # Predicción de demanda con XGBoost
│   ├── LSTM.ipynb              # Predicción de demanda con LSTM
│   └── BRP.ipynb               # Optimización de rutas de redistribución
├── src/
│   └── dataset_builder.py      # Pipeline de construcción del dataset
└── requirements.txt
```

## Fuentes de datos

| Dataset | Fuente | Período | Descripción |
|---|---|---|---|
| Estado de estaciones | [Open Data BCN](https://opendata-ajuntament.barcelona.cat/data/es/dataset/estat-estacions-bicing) | Oct 2023 – Sep 2025 | ~250M filas brutas; registro por estación cada ~5 min |
| Información de estaciones | [Open Data BCN](https://opendata-ajuntament.barcelona.cat/data/es/dataset/informacio-estacions-bicing) | Estático | Metadatos: nombre, coordenadas, altitud, capacidad |
| Meteorología horaria | [Open-Meteo Archive API](https://open-meteo.com/) | Oct 2023 – Sep 2025 | Temperatura, precipitación, humedad, viento, presión |
| Festivos y calendario | Librería `holidays` + 6 festivos locales BCN | 2023–2025 | Festivos estatal y autonómico (Cataluña) calculados automáticamente; La Mercè y Pascua Granada añadidos manualmente |

## Pipeline de datos (`src/dataset_builder.py`)

El módulo `dataset_builder.py` construye el dataset de forma incremental y reproducible:

1. **`build_hourly()`** — Lee los 24 archivos `.7z` mensuales uno a uno, convierte los timestamps Unix a `Europe/Madrid`, agrega a resolución horaria por `station_id` (media) y guarda `bicing_hourly.parquet`. Nunca carga los 24 meses en bruto en memoria simultáneamente.
2. **`build_stations()`** — Lee el archivo de información estática de estaciones, descarta columnas completamente vacías o con valor constante y guarda `bicing_stations.parquet`.
3. **`build_meteo()`** — Descarga las variables meteorológicas desde la API de Open-Meteo con reintentos exponenciales (3 intentos) y guarda `meteo_barcelona.parquet`.
4. **`build_full()`** — Realiza los tres `LEFT JOIN` (meteo por hora, estaciones por `station_id`) y guarda `bicing_full.parquet` con dtypes eficientes (`float32`, `int32`, `category`).

Todos los parquets se comprimen con **Snappy**. Si el archivo ya existe en disco, se carga directamente sin reprocesar.

### Ejecución

```bash
pip install -r requirements.txt
python src/dataset_builder.py
```

Los archivos `.7z` deben estar descargados previamente en `data/raw/`.

## Notebooks

### `EDA.ipynb` — Análisis Exploratorio de Datos

| Sección | Contenido |
|---|---|
| § 0. Configuración del entorno | Librerías, paleta de colores, funciones auxiliares |
| § 1. Carga de datos | Procesamiento mensual de los `.7z`, merge con meteo y estaciones |
| § 2. EDA Dataset Estaciones | Análisis geoespacial: altitud, capacidad, distribución geográfica |
| § 3. EDA Dataset Estado (Horario) | Patrones temporales de uso, meteorología, calendario, selección de features |
| § 4. Dataset Completo | Variables derivadas, lags, codificación cíclica, análisis multivariante, selección final |

### `XGBoost.ipynb` — Predicción de demanda con XGBoost

Predice la tasa de ocupación (`occupancy_rate`) por estación y horizonte horario usando 9 features numéricas. Incluye búsqueda bayesiana de hiperparámetros con Optuna.

| Sección | Contenido |
|---|---|
| 1. Carga de librerías | Dependencias y configuración |
| 2. Carga del dataset model-ready | Carga de `bicing_model_ready.parquet` generado por el EDA |
| 3. Preparación para modelado | Eliminación de filas con nulos en features o target |
| 4. Split temporal Train / Val / Test | Corte temporal estricto (~80/10/10 %) sin shuffle |
| 5. Baseline: modelo de persistencia | Predictor naive `bikes_lag_1h / capacity` como umbral mínimo |
| 6. XGBoost | Entrenamiento inicial con parámetros de partida |
| 7. Optimización con Optuna | 50 trials de búsqueda bayesiana (TPE) sobre RMSE en validación |
| 8. Modelo final | Entrenamiento con la configuración óptima encontrada |
| 9. Evaluación completa | Tabla comparativa de todos los modelos en val y test |
| 10. Importancia de features | `weight`, `gain` y `cover` de XGBoost |
| 11. Análisis de residuos | Errores sistemáticos por franja horaria, día y nivel de ocupación |
| 12. Análisis por estación | Estaciones con mayor y menor error |
| 13. Visualización temporal | Serie real vs predicción en una semana de test (4 estaciones) |
| 14. Conclusiones | MAE, R² y comparativa frente al baseline |

### `LSTM.ipynb` — Predicción de demanda con LSTM

Predice la tasa de ocupación usando secuencias temporales. Incluye codificación cíclica sin/cos para variables temporales y búsqueda de hiperparámetros con Optuna.

| Sección | Contenido |
|---|---|
| 1. Carga de librerías | Dependencias y configuración |
| 2. Carga y preparación del dataset | Carga de `bicing_model_ready.parquet` |
| 3. Preprocesamiento y split temporal | Escalado, corte temporal estricto |
| 4. Baseline: modelo de persistencia | Mismo umbral que XGBoost para comparabilidad |
| 5. Pipeline de secuencias (tf.data) | Construcción de ventanas temporales para la LSTM |
| 6. Arquitectura LSTM | Definición de capas y función de pérdida |
| 7. LSTM v0 — parámetros iniciales | Primer entrenamiento de referencia |
| 8. Optimización con Optuna | Búsqueda bayesiana de arquitectura e hiperparámetros |
| 9. Modelo final | Entrenamiento con la configuración óptima |
| 10. Evaluación completa | Métricas MAE, RMSE y R² en val y test |
| 11. Análisis de residuos | Errores sistemáticos por franja horaria y día |
| 12. Análisis por estación | Estaciones con mayor y menor error |
| 13. Visualización temporal | Serie real vs predicción en una semana de test |
| 14. Comparación LSTM vs XGBoost | Tabla de métricas y análisis cualitativo |
| 15. Conclusiones | Resumen de resultados y recomendaciones |

### `BRP.ipynb` — Optimización de Rutas de Redistribución

Usa las predicciones de disponibilidad para planificar rutas de camiones que minimizan el desequilibrio del sistema.

| Sección | Contenido |
|---|---|
| 1. Carga de librerías y configuración | Dependencias y parámetros del escenario |
| 2. EDA de desequilibrio | Métricas de desequilibrio por estación, mapas interactivos, heatmaps |
| 3. Formulación y resolución del modelo VRP | Algoritmo greedy nearest-neighbor con restricciones de capacidad y paradas |
| 4. Evaluación y visualización de rutas | Métricas antes/después, mapa interactivo de rutas, detalle por camión |


## Librerías

Todas las dependencias están en [requirements.txt](requirements.txt). Instalación:

```bash
pip install -r requirements.txt
```

| Categoría | Librería | Uso principal |
|---|---|---|
| Manipulación de datos | `pandas`, `numpy` | DataFrames, operaciones vectoriales |
| Serialización | `pyarrow` | Lectura/escritura de archivos Parquet |
| Modelos ML | `xgboost`, `scikit-learn` | Gradient boosting, preprocesado, métricas |
| Optimización de hiperparámetros | `optuna` | Búsqueda bayesiana (TPE) en XGBoost y LSTM |
| Deep Learning | `tensorflow` / `keras` | Arquitectura y entrenamiento LSTM |
| Visualización estática | `matplotlib`, `seaborn` | Gráficos, heatmaps, boxplots |
| Visualización interactiva | `plotly`, `folium` | Series temporales, mapas geoespaciales |
| Series temporales | `statsmodels` | Descomposición STL, ANOVA |
| Estadística | `scipy` | Tests Kruskal-Wallis, correlaciones |
| Festivos | `holidays` | Calendario estatal + autonómico (Cataluña) |
| Compresión | `py7zr` | Descompresión de archivos `.7z` de Open Data BCN |
| Descarga de datos | `requests`, `tqdm` | API Open-Meteo, barras de progreso |

## Referencias

- **Peláez-Rodríguez et al. (2024)**. *Expert Systems with Applications*, 238, 122264.
- **Cortez-Ordoñez et al. (2024)**. *Heliyon*, 10, e37608.
- **Bustamante et al. (2022)**. *Sustainable Cities and Society*, 83, 103929.
- **Cipriano et al. (2021)**. *Applied Sciences*, 11(15), 6967.
- **Giner Fabregat et al. (2025)**. *Sustainability*, 17, 4316.

## Repositorio

[https://gitlab.com/agusticostabellamoreno/brp_bicing](https://gitlab.com/agusticostabellamoreno/brp_bicing)
