"""
Pipeline de construcción del dataset para el proyecto BRP Bicing Barcelona.

Descarga, descomprime y agrega los datos brutos del Bicing (estado de estaciones,
metadatos e información meteorológica) y los combina en un único parquet listo para
el análisis exploratorio y el entrenamiento de modelos.

Uso:
    python src/dataset_builder.py

Los archivos .7z mensuales deben estar previamente descargados en data/raw/.
Fuente: https://opendata-ajuntament.barcelona.cat/data/es/dataset/estat-estacions-bicing
"""

import shutil
import tempfile
import time
from pathlib import Path

import holidays
import pandas as pd
import py7zr
import requests
from tqdm.auto import tqdm

# Constants

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = RAW_DIR

HOURLY_PARQUET = PROC_DIR / "bicing_hourly.parquet"
INFO_PARQUET   = PROC_DIR / "bicing_stations.parquet"
METEO_PARQUET  = PROC_DIR / "meteo_barcelona.parquet"
FULL_PARQUET   = PROC_DIR / "bicing_full.parquet"

MONTH_NAMES_CAT = {
    1: "Gener",    2: "Febrer",   3: "Marc",    4: "Abril",
    5: "Maig",     6: "Juny",     7: "Juliol",  8: "Agost",
    9: "Setembre", 10: "Octubre", 11: "Novembre", 12: "Desembre",
}

# Período de estudio: oct 2023 – sep 2025
TARGET_MONTHS = (
    [(2023, m) for m in range(10, 13)] +
    [(2024, m) for m in range(1,  13)] +
    [(2025, m) for m in range(1,  10)]
)

# Festivos estatales + autonómicos (Cataluña) calculados automáticamente.
_YEARS = range(2023, 2026)
_cat_holidays = holidays.Spain(subdiv="CT", years=_YEARS)

# Festivos locales de Barcelona no cubiertos por la librería.
# La Mercè es siempre el 24 de septiembre; Pascua Granada es el lunes de Pentecostés.
_BCN_LOCAL = {
    "2023-06-05": "Pascua Granada", "2023-09-24": "La Mercè",
    "2024-06-09": "Pascua Granada", "2024-09-24": "La Mercè",
    "2025-06-09": "Pascua Granada", "2025-09-24": "La Mercè",
}

# Conjunto unificado para lookup O(1) en enrich_tipo_dia
festivos_set = (
    {pd.Timestamp(d) for d in _cat_holidays}
    | {pd.Timestamp(d) for d in _BCN_LOCAL}
)

# Funciones auxiliares

def parse_7z(path: Path) -> "pd.DataFrame | None":
    """
    Lee un archivo .7z mensual del Bicing y devuelve la telemetría agregada a nivel horario.

    Descomprime el archivo en un directorio temporal, lee el CSV interior,
    convierte el timestamp Unix a hora local (Europe/Madrid) y agrega por
    (station_id, hora) calculando la media de las columnas numéricas disponibles.

    Args:
        path: Ruta al archivo .7z mensual.

    Returns:
        DataFrame con columnas [station_id, hora, num_bikes_available,
        num_docks_available, num_bikes_available_types.*], o None si hay error.
    """
    try:
        with py7zr.SevenZipFile(path, mode="r") as z:
            csv_name = next((n for n in z.getnames() if n.endswith(".csv")), None)
            if not csv_name:
                print(f"Sin CSV dentro de {path.name}")
                return None
            tmp_dir = tempfile.mkdtemp()
            try:
                z.extractall(path=tmp_dir)
                csv_path = next(Path(tmp_dir).rglob("*.csv"))
                df_ = pd.read_csv(csv_path, low_memory=False)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Error leyendo {path.name}: {e}")
        return None

    df_.columns = [c.strip().lower() for c in df_.columns]

    ts_col = next(
        (c for c in df_.columns if "last_reported" in c or "timestamp" in c), None
    )
    if ts_col is None:
        print(f"Sin columna timestamp en {path.name}")
        return None

    df_["hora"] = (
        pd.to_datetime(df_[ts_col], unit="s", utc=True)
        .dt.tz_convert("Europe/Madrid")
        .dt.tz_localize(None)
        .dt.floor("h")
    )

    if "station_id" in df_.columns:
        df_ = df_.dropna(subset=["station_id"])
        df_["station_id"] = df_["station_id"].astype("int32")

    num_cols = [
        c for c in [
            "num_bikes_available", "num_docks_available",
            "num_bikes_available_types.mechanical",
            "num_bikes_available_types.electrical",
            "mechanical", "electrical",
        ]
        if c in df_.columns
    ]

    agg = (
        df_.groupby(["station_id", "hora"], observed=True)[num_cols]
        .mean()
        .reset_index()
    )

    # Normaliza nombres de columna si el CSV usa alias cortos
    for alias, canon in [
        ("mechanical", "num_bikes_available_types.mechanical"),
        ("electrical", "num_bikes_available_types.electrical"),
    ]:
        if alias in agg.columns and canon not in agg.columns:
            agg.rename(columns={alias: canon}, inplace=True)

    return agg

def enrich_tipo_dia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de clasificación temporal a un DataFrame con columna 'hora'.

    Columnas añadidas: fecha, dia_semana (0=lun…6=dom), es_finde, es_festivo,
    es_laboral, tipo_dia ('Laborable' | 'Fin de semana' | 'Festivo').

    Args:
        df: DataFrame con columna 'hora' de tipo datetime.

    Returns:
        El mismo DataFrame con las columnas de tipo de día añadidas in-place.
    """
    df["fecha"]      = df["hora"].dt.normalize()
    df["dia_semana"] = df["hora"].dt.dayofweek
    df["es_finde"]   = df["dia_semana"] >= 5
    df["es_festivo"] = df["fecha"].isin(festivos_set)
    df["es_laboral"] = ~df["es_finde"] & ~df["es_festivo"]
    df["tipo_dia"]   = "Laborable"
    df.loc[df["es_finde"],   "tipo_dia"] = "Fin de semana"
    df.loc[df["es_festivo"], "tipo_dia"] = "Festivo"
    return df

# Funciones de construcción de parquets

def build_hourly() -> pd.DataFrame:
    """
    Construye bicing_hourly.parquet procesando los .7z mensuales uno a uno.

    Si el parquet ya existe en disco lo carga directamente sin reprocesar.
    Procesa cada mes en memoria de forma independiente para evitar cargar
    los ~250M de registros brutos simultáneamente.

    Returns:
        DataFrame horario con columnas de telemetría y tipo de día,
        acotado al período oct 2023 – sep 2025.

    Raises:
        RuntimeError: si no se encuentra ningún archivo .7z en data/raw/.
    """
    if HOURLY_PARQUET.exists():
        print(f"Cargando {HOURLY_PARQUET} existente...")
        return enrich_tipo_dia(pd.read_parquet(HOURLY_PARQUET))

    monthly_dfs: list[pd.DataFrame] = []
    errors: list[str] = []

    for year, month in tqdm(TARGET_MONTHS, desc="Meses"):
        mn       = MONTH_NAMES_CAT[month]
        filename = f"{year}_{month:02d}_{mn}_BicingNou_ESTACIONS.7z"
        dest     = DATA_DIR / filename

        if not dest.exists():
            print(f"No encontrado: {filename}")
            errors.append(filename)
            continue

        agg = parse_7z(dest)
        if agg is not None:
            monthly_dfs.append(agg)
        else:
            errors.append(filename)

    if not monthly_dfs:
        raise RuntimeError(
            "Sin datos disponibles.\n"
            "Descarga los .7z desde:\n"
            "  https://opendata-ajuntament.barcelona.cat/data/es/dataset/"
            "estat-estacions-bicing\n"
            "y cópialos en la carpeta data/raw/"
        )

    bicing_hourly = pd.concat(monthly_dfs, ignore_index=True)

    mask = bicing_hourly["hora"].between("2023-10-01", "2025-09-30")
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"Descartadas {n_dropped:,} filas con timestamps fuera de rango")
    bicing_hourly = bicing_hourly[mask]

    bicing_hourly.to_parquet(HOURLY_PARQUET, index=False, compression="snappy")
    print(f"bicing_hourly.parquet guardado ({len(bicing_hourly):,} filas)")
    if errors:
        print(f"Meses con error ({len(errors)}): {errors}")

    return enrich_tipo_dia(bicing_hourly)

def build_stations() -> pd.DataFrame:
    """
    Construye bicing_stations.parquet a partir del .7z de información estática.

    Lee el archivo *INFORMACIO*.7z más reciente disponible en data/raw/,
    descarta columnas vacías o con valor constante y elimina duplicados de
    station_id conservando el registro más reciente.

    Returns:
        DataFrame con metadatos de estación (station_id, nombre, coordenadas,
        altitud, capacidad, etc.).

    Raises:
        FileNotFoundError: si no hay ningún *INFORMACIO*.7z en data/raw/.
        ValueError: si el .7z no contiene ningún CSV.
    """
    if INFO_PARQUET.exists():
        print(f"Cargando {INFO_PARQUET} existente...")
        return pd.read_parquet(INFO_PARQUET)

    candidatos = sorted(DATA_DIR.glob("*INFORMACIO*.7z"))
    if not candidatos:
        raise FileNotFoundError(
            "No se encontró ningún fichero *INFORMACIO*.7z en data/raw/\n"
            "Descárgalo desde:\n"
            "  https://opendata-ajuntament.barcelona.cat/data/es/dataset/"
            "informacio-estacions-bicing\n"
            "y cópialo en la carpeta data/raw/"
        )

    info_path = candidatos[-1]
    print(f"Leyendo {info_path.name}...")

    with py7zr.SevenZipFile(info_path, mode="r") as z:
        csv_name = next((n for n in z.getnames() if n.endswith(".csv")), None)
        if not csv_name:
            raise ValueError(f"No se encontró CSV dentro de {info_path.name}")
        tmp_dir = tempfile.mkdtemp()
        try:
            z.extractall(path=tmp_dir)
            csv_path = next(Path(tmp_dir).rglob("*.csv"))
            estaciones = pd.read_csv(csv_path, low_memory=False)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    estaciones.columns = [c.strip().lower() for c in estaciones.columns]

    n_before = estaciones.shape[1]
    estaciones.dropna(axis=1, how="all", inplace=True)
    single_val = [c for c in estaciones.columns if estaciones[c].nunique() <= 1]
    estaciones.drop(columns=single_val, inplace=True)
    print(f"Columnas descartadas (vacías/constantes): {n_before - estaciones.shape[1]}")

    if "station_id" in estaciones.columns:
        estaciones["station_id"] = estaciones["station_id"].astype("int32")

    n_before = len(estaciones)
    estaciones = estaciones.drop_duplicates(subset="station_id", keep="last")
    print(f"Duplicados de station_id eliminados: {n_before - len(estaciones):,}")

    estaciones.to_parquet(INFO_PARQUET, index=False, compression="snappy")
    print(f"bicing_stations.parquet guardado ({len(estaciones)} estaciones)")
    return estaciones

def build_meteo() -> pd.DataFrame:
    """
    Descarga datos meteorológicos horarios de Open-Meteo y crea meteo_barcelona.parquet.

    Consulta la API de archivo de Open-Meteo para Barcelona (lat=41.3874, lon=2.1686)
    con reintentos exponenciales (hasta 3 intentos). Variables descargadas:
    temperature_2m, precipitation, relative_humidity_2m, wind_speed_10m,
    surface_pressure, weather_code.

    Returns:
        DataFrame con columna 'hora' (datetime) y las seis variables meteorológicas.

    Raises:
        RuntimeError: si la descarga falla tras 3 intentos.
    """
    if METEO_PARQUET.exists():
        print(f"Cargando {METEO_PARQUET} existente...")
        return pd.read_parquet(METEO_PARQUET)

    METEO_URL = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=41.3874&longitude=2.1686"
        "&start_date=2023-10-01&end_date=2025-09-30"
        "&hourly=temperature_2m,precipitation,relative_humidity_2m,"
        "wind_speed_10m,surface_pressure,weather_code"
        "&timezone=Europe%2FMadrid"
    )

    print("Descargando meteorología de Open-Meteo...")
    for attempt in range(1, 4):
        try:
            r = requests.get(METEO_URL, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"Intento {attempt}/3: {e}. Reintentando en {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError("No se pudo descargar meteorología tras 3 intentos.")

    meteo = pd.DataFrame(data["hourly"])
    meteo["hora"] = pd.to_datetime(meteo["time"])
    meteo.drop(columns=["time"], inplace=True)

    meteo.to_parquet(METEO_PARQUET, index=False, compression="snappy")
    print(f"meteo_barcelona.parquet guardado ({len(meteo):,} filas)")
    return meteo

def build_full(
    bicing_hourly: pd.DataFrame,
    estaciones: pd.DataFrame,
    meteo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina los tres datasets en bicing_full.parquet mediante LEFT JOINs sucesivos.

    Orden de joins: bicing_hourly → meteo (por hora) → estaciones (por station_id).
    Elimina filas sin metadatos de estación (name nulo) y optimiza dtypes a
    float32 para columnas numéricas y category para tipo_dia.

    Args:
        bicing_hourly: Telemetría horaria agregada (salida de build_hourly).
        estaciones:    Metadatos estáticos de estación (salida de build_stations).
        meteo:         Variables meteorológicas horarias (salida de build_meteo).

    Returns:
        DataFrame integrado guardado en bicing_full.parquet.
    """
    if FULL_PARQUET.exists():
        print(f"Cargando {FULL_PARQUET} existente...")
        return pd.read_parquet(FULL_PARQUET)

    df = bicing_hourly.merge(meteo, on="hora", how="left")
    print(f"Tras merge meteo     : {df.shape[0]:,} filas, {df.shape[1]} cols")

    df = df.merge(estaciones, on="station_id", how="left")
    print(f"Tras merge estaciones: {df.shape[0]:,} filas, {df.shape[1]} cols")

    n_before = len(df)
    df = df.dropna(subset=["name"])
    print(f"Filas eliminadas sin metadatos: {n_before - len(df):,}")

    float_cols = [
        "num_bikes_available", "num_docks_available",
        "num_bikes_available_types.mechanical",
        "num_bikes_available_types.electrical",
        "temperature_2m", "precipitation", "relative_humidity_2m",
        "wind_speed_10m", "surface_pressure",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if "tipo_dia" in df.columns:
        df["tipo_dia"] = df["tipo_dia"].astype("category")

    df.to_parquet(FULL_PARQUET, index=False, compression="snappy")
    print(f"bicing_full.parquet guardado ({len(df):,} filas)")
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("Preparando datos de Bicing para análisis y modelado...")
    print("=" * 60)

    bicing_hourly = build_hourly()
    print(f"  Shape: {bicing_hourly.shape} | "
          f"Rango: {bicing_hourly['hora'].min()} → {bicing_hourly['hora'].max()}\n")

    estaciones = build_stations()
    print(f"  Shape: {estaciones.shape}\n")

    meteo = build_meteo()
    print(f"  Shape: {meteo.shape}\n")

    df = build_full(bicing_hourly, estaciones, meteo)
    print(f"  Shape final: {df.shape}")
    print(f"  Columnas: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("Parquets generados correctamente en data/processed/")
    print("=" * 60)
