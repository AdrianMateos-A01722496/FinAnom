"""Consolidación de la base FINANOM para detección de anomalías financieras.

Construye UNA sola tabla analítica a grano de transacción (`hottra`), limpia y
enriquecida con el contexto de la reservación (`hothsp`). Es la base que consume
el modelo no supervisado (Isolation Forest).

Decisiones de diseño (ver diccionario_base_consolidada.md para el detalle):

- Grano: una fila = una transacción de `hottra` = una predicción del modelo.
- Enriquecimiento: LEFT JOIN con `hothsp` por `t_cve_res == h_res_cve`.
  El 78.5% de las transacciones tienen reservación; el resto (t_cve_res vacío)
  son walk-ins / depósitos / cargos de centro de consumo y se conservan.
- Se descartan: columnas muertas (un solo valor), PII (nombres, correos,
  tarjetas), y campos confirmados como no aplicables en México
  (t_ya_facturada, h_ya_fact, h_status_pago, h_num_fac, ...).
- `hotcag` NO se usa: no es el catálogo de `t_codigo` (solo 43/1.14M filas
  cruzan; 107 de 108 códigos quedan huérfanos). Los 108 códigos de transacción
  no tienen catálogo oficial entre los datos entregados.
- Se descarta `t_tipo_cambio` (siempre 0) y NO se hace conversión de moneda:
  los datos son de un hotel en México, prácticamente todo en MXN.

Uso:
    python data_consolidation/consolidate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "tablas_parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "transacciones_consolidado.parquet"

# --------------------------------------------------------------------------- #
# Selección de columnas
# --------------------------------------------------------------------------- #
# Transacciones (hottra) — columnas relevantes para anomalías financieras.
HOTTRA_COLS = [
    # Identificadores / trazabilidad
    "t_folio", "t_folio_ext", "t_referencia", "t_transaccion", "t_codigo",
    "t_cve_res", "t_cuarto", "t_centro_consumo",
    # Temporales
    "t_fecha", "t_tra_hra", "t_tra_mto",
    # Monetarias
    "t_monto", "t_impuesto", "t_impuesto2", "t_propina",
    # Control / cancelación / autoría
    "t_carabo", "t_tra_cancelada", "t_can_dia", "t_can_mes", "t_can_hra",
    "t_can_mto", "t_usuario", "t_usuario_mod",
    # Cargos partidos (split) → se colapsan en el booleano `es_split`
    "t_genero_split", "t_folio_split", "t_usr_split",
    # Contexto de ocupación / tarifa a nivel cargo
    "t_num_adu", "t_num_per", "t_noches", "t_inc_tfa", "t_renta",
    # Texto operativo (útil para reasons del auditor)
    "t_observaciones",
]

# Columnas de split que se colapsan en `es_split` y luego se eliminan.
SPLIT_COLS = ["t_genero_split", "t_folio_split", "t_usr_split"]

# Reservación (hothsp) — contexto que se une a cada transacción.
HOTHSP_COLS = [
    "h_res_cve",                       # llave de join
    "h_status",                        # ciclo de vida de la reservación
    "h_tpo_hab", "h_tpo_hsp",          # tipo de habitación / tipo de huésped (VIP)
    "h_seg_mer", "h_cod_age", "h_tpo_plan",  # segmento / agencia / plan tarifario
    "h_for_pgo", "h_tpo_mon",          # forma de pago / moneda
    "h_num_per", "h_num_adu", "h_num_men", "h_num_noc",  # ocupación esperada
    "h_tfa", "h_tfa_total", "h_tfa_renta", "h_tfa_impuestos", "h_tfa_extras",
    "h_tarifa_forzada", "h_dep_sol", "h_lim_cre",  # tarifas / depósito / límite crédito
    "h_fec_lld", "h_fec_sda", "h_fec_reg",  # llegada / salida / registro
    "h_res_usr", "h_rec_usr",          # usuario que reserva / registra
]

# Columnas tipo fecha YYYYMMDD que se parsean a datetime.
HOTHSP_DATE_COLS = ["h_fec_lld", "h_fec_sda", "h_fec_reg"]

# Columnas string que representan enteros con blancos → Int64 nullable.
HOTTRA_INT_COLS = ["t_num_adu", "t_num_per"]


# --------------------------------------------------------------------------- #
# Utilidades de limpieza
# --------------------------------------------------------------------------- #
def clean_str(s: pd.Series) -> pd.Series:
    """Normaliza una columna string: strip y blancos → <NA>."""
    out = s.astype("string").str.strip()
    return out.replace("", pd.NA)


def parse_yyyymmdd(s: pd.Series) -> pd.Series:
    """Parsea fechas en formato 'YYYYMMDD' (string). '00000000'/blanco → NaT."""
    raw = s.astype("string").str.strip()
    raw = raw.where(~raw.isin(["", "00000000", "0"]), other=pd.NA)
    return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")


def to_int_nullable(s: pd.Series) -> pd.Series:
    """Convierte string numérico con blancos a Int64 nullable."""
    return pd.to_numeric(clean_str(s), errors="coerce").astype("Int64")


def build_timestamp(fecha: pd.Series, hra: pd.Series, mto: pd.Series) -> pd.Series:
    """Combina fecha (YYYYMMDD) + hora (HH) + minuto (MM) en un Timestamp."""
    base = parse_yyyymmdd(fecha)
    h = pd.to_numeric(clean_str(hra), errors="coerce").fillna(0).clip(0, 23)
    m = pd.to_numeric(clean_str(mto), errors="coerce").fillna(0).clip(0, 59)
    delta = pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m")
    return base + delta


# --------------------------------------------------------------------------- #
# Carga + limpieza por tabla
# --------------------------------------------------------------------------- #
def load_hottra(parquet_dir: Path | str = PARQUET_DIR) -> pd.DataFrame:
    """Carga y limpia la tabla de transacciones."""
    parquet_dir = Path(parquet_dir)
    df = pd.read_parquet(parquet_dir / "hottra.parquet", columns=HOTTRA_COLS)

    # Timestamp del cargo (fecha + hora + minuto).
    df["t_timestamp"] = build_timestamp(df["t_fecha"], df["t_tra_hra"], df["t_tra_mto"])
    df["t_fecha"] = parse_yyyymmdd(df["t_fecha"])

    # Ocupación a nivel cargo → enteros nullable.
    for c in HOTTRA_INT_COLS:
        df[c] = to_int_nullable(df[c])

    # Strings restantes → normalizados.
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = clean_str(df[c])

    # Cargos partidos: 3 columnas casi vacías (77 filas) → un solo booleano.
    df["es_split"] = df[SPLIT_COLS].notna().any(axis=1)
    df = df.drop(columns=SPLIT_COLS)

    # t_renta es un flag "S"/vacío (cargo de renta / night audit) → booleano.
    df["es_renta"] = (df["t_renta"] == "S").fillna(False).astype(bool)
    df = df.drop(columns=["t_renta"])

    # t_noches ya es int64 nativo; t_monto/impuestos/propina ya son float64.
    return df


def load_hothsp(parquet_dir: Path | str = PARQUET_DIR) -> pd.DataFrame:
    """Carga y limpia el contexto de reservación."""
    parquet_dir = Path(parquet_dir)
    df = pd.read_parquet(parquet_dir / "hothsp.parquet", columns=HOTHSP_COLS)

    for c in HOTHSP_DATE_COLS:
        df[c] = parse_yyyymmdd(df[c])

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = clean_str(df[c])

    # Llave de join limpia y única.
    df["h_res_cve"] = clean_str(df["h_res_cve"])
    df = df.drop_duplicates(subset="h_res_cve", keep="first")
    return df


# --------------------------------------------------------------------------- #
# Consolidación
# --------------------------------------------------------------------------- #
def consolidate(raw_dir: Path | str = PARQUET_DIR) -> pd.DataFrame:
    """Construye la tabla consolidada transacción + contexto de reservación."""
    raw_dir = Path(raw_dir)
    print("Cargando hottra (transacciones)...")
    tra = load_hottra(raw_dir)
    print(f"  hottra: {len(tra):,} filas, {tra.shape[1]} columnas")

    print("Cargando hothsp (reservaciones)...")
    hsp = load_hothsp(raw_dir)
    print(f"  hothsp: {len(hsp):,} reservaciones, {hsp.shape[1]} columnas")

    print("Uniendo (LEFT JOIN t_cve_res == h_res_cve)...")
    out = tra.merge(
        hsp,
        how="left",
        left_on="t_cve_res",
        right_on="h_res_cve",
        validate="many_to_one",
    )
    out["tiene_reservacion"] = out["h_res_cve"].notna()
    out = out.drop(columns=["h_res_cve"])  # redundante con t_cve_res

    cobertura = 100 * out["tiene_reservacion"].mean()
    print(f"  consolidado: {len(out):,} filas, {out.shape[1]} columnas")
    print(f"  con reservación enlazada: {cobertura:.1f}%")
    return out


def run_consolidation(
    raw_dir: Path | str = PARQUET_DIR,
    output_file: Path | str = OUTPUT_FILE,
) -> Path:
    """Ejecuta la fase de consolidacion y devuelve el parquet generado."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = consolidate(raw_dir)
    df.to_parquet(output_file, index=False)
    size_mb = output_file.stat().st_size / 1e6
    print(f"\nGuardado: {output_file}  ({size_mb:.1f} MB)")
    return output_file


def main() -> None:
    run_consolidation()


if __name__ == "__main__":
    main()
