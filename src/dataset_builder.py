import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import py7zr
import requests
from tqdm.auto import tqdm

ROOT      = Path(__file__).resolve().parent.parent  # src/ -> project root
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Keep DATA_DIR pointing to raw for .7z discovery
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

TARGET_MONTHS = (
    [(2023, m) for m in range(10, 13)] +  # oct-dic 2023
    [(2024, m) for m in range(1,  13)] +  # ene-dic 2024
    [(2025, m) for m in range(1,  10)]    # ene-sep 2025
)

FESTIVOS_BCN = {
    # 2023
    "2023-01-06": ("Reyes",                       "E"),
    "2023-04-07": ("Viernes Santo",               "E"),
    "2023-04-10": ("Lunes de Pascua",             "A"),
    "2023-05-01": ("Día del Trabajo",             "E"),
    "2023-06-05": ("Pascua Granada",              "L"),
    "2023-06-24": ("Sant Joan",                   "A"),
    "2023-08-15": ("Asunción",                    "E"),
    "2023-09-11": ("Diada Nacional de Catalunya", "A"),
    "2023-09-24": ("La Mercè",                    "L"),
    "2023-10-12": ("Fiesta Nacional de España",   "E"),
    "2023-11-01": ("Todos los Santos",            "E"),
    "2023-12-06": ("Día de la Constitución",      "E"),
    "2023-12-08": ("La Inmaculada",               "E"),
    "2023-12-25": ("Navidad",                     "E"),
    "2023-12-26": ("Sant Esteve",                 "A"),
    # 2024
    "2024-01-01": ("Año Nuevo",                   "E"),
    "2024-01-06": ("Reyes",                       "E"),
    "2024-03-29": ("Viernes Santo",               "E"),
    "2024-04-01": ("Lunes de Pascua",             "A"),
    "2024-05-01": ("Día del Trabajo",             "E"),
    "2024-06-09": ("Pascua Granada",              "L"),
    "2024-06-24": ("Sant Joan",                   "A"),
    "2024-08-15": ("Asunción",                    "E"),
    "2024-09-11": ("Diada Nacional de Catalunya", "A"),
    "2024-09-24": ("La Mercè",                    "L"),
    "2024-11-01": ("Todos los Santos",            "E"),
    "2024-12-06": ("Día de la Constitución",      "E"),
    "2024-12-25": ("Navidad",                     "E"),
    "2024-12-26": ("Sant Esteve",                 "A"),
    # 2025
    "2025-01-01": ("Año Nuevo",                   "E"),
    "2025-01-06": ("Reyes",                       "E"),
    "2025-04-18": ("Viernes Santo",               "E"),
    "2025-04-21": ("Lunes de Pascua",             "A"),
    "2025-05-01": ("Día del Trabajo",             "E"),
    "2025-06-09": ("Pascua Granada",              "L"),
    "2025-06-24": ("Sant Joan",                   "A"),
    "2025-08-15": ("Asunción",                    "E"),
    "2025-09-11": ("Diada Nacional de Catalunya", "A"),
    "2025-09-24": ("La Mercè",                    "L"),
    "2025-10-13": ("Fiesta Nacional de España",   "E"),
    "2025-11-01": ("Todos los Santos",            "E"),
    "2025-12-08": ("La Inmaculada",               "E"),
    "2025-12-25": ("Navidad",                     "E"),
    "2025-12-26": ("Sant Esteve",                 "A"),
}

festivos_set = {pd.Timestamp(d) for d in FESTIVOS_BCN}


def parse_7z(path: Path) -> "pd.DataFrame | None":
    """Descomprime .7z, lee CSV y agrega a nivel horario por station_id."""
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

    for alias, canon in [
        ("mechanical", "num_bikes_available_types.mechanical"),
        ("electrical", "num_bikes_available_types.electrical"),
    ]:
        if alias in agg.columns and canon not in agg.columns:
            agg.rename(columns={alias: canon}, inplace=True)

    return agg


def enrich_tipo_dia(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de tipo de día a un DataFrame con columna 'hora'."""
    df["fecha"]      = df["hora"].dt.normalize()
    df["dia_semana"] = df["hora"].dt.dayofweek   # 0=lun … 6=dom
    df["es_finde"]   = df["dia_semana"] >= 5
    df["es_festivo"] = df["fecha"].isin(festivos_set)
    df["es_laboral"] = ~df["es_finde"] & ~df["es_festivo"]
    df["tipo_dia"]   = "Laborable"
    df.loc[df["es_finde"],   "tipo_dia"] = "Fin de semana"
    df.loc[df["es_festivo"], "tipo_dia"] = "Festivo"
    return df

def build_hourly() -> pd.DataFrame:
    """Crea bicing_hourly.parquet a partir de los .7z mensuales."""
    if HOURLY_PARQUET.exists():
        print(f"Cargando {HOURLY_PARQUET} existente...")
        bicing_hourly = pd.read_parquet(HOURLY_PARQUET)
        return enrich_tipo_dia(bicing_hourly)

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
            "y cópialos en la carpeta data/"
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
    """Crea bicing_stations.parquet a partir del .7z de INFORMACIO."""
    if INFO_PARQUET.exists():
        print(f"Cargando {INFO_PARQUET} existente...")
        return pd.read_parquet(INFO_PARQUET)

    candidatos = sorted(DATA_DIR.glob("*INFORMACIO*.7z"))
    if not candidatos:
        raise FileNotFoundError(
            "No se encontró ningún fichero *INFORMACIO*.7z en data/\n"
            "Descárgalo desde:\n"
            "  https://opendata-ajuntament.barcelona.cat/data/es/dataset/"
            "informacio-estacions-bicing\n"
            "y cópialo en la carpeta data/"
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
    """Descarga meteorología de Open-Meteo y crea meteo_barcelona.parquet."""
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


def build_full(bicing_hourly: pd.DataFrame,
               estaciones: pd.DataFrame,
               meteo: pd.DataFrame) -> pd.DataFrame:
    if FULL_PARQUET.exists():
        print(f"Cargando {FULL_PARQUET} existente...")
        return pd.read_parquet(FULL_PARQUET)

    df = bicing_hourly.merge(meteo, on="hora", how="left")
    print(f"Tras merge meteo    : {df.shape[0]:,} filas, {df.shape[1]} cols")

    df = df.merge(estaciones, on="station_id", how="left")
    print(f"Tras merge estacion : {df.shape[0]:,} filas, {df.shape[1]} cols")

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
    print(f"✔ bicing_full.parquet guardado ({len(df):,} filas)")
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
    print("Parquets generados correctamente en data/")
    print("=" * 60)
