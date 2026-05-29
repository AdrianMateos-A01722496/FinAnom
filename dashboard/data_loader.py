"""Carga, filtrado y cálculo de KPIs para el dashboard FinAnom."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Raíz del proyecto y carpeta del dashboard
ROOT   = Path(__file__).resolve().parent.parent
DASHBD = Path(__file__).resolve().parent

# Rutas de archivos de datos (orden de prioridad en load_alertas)
PATHS: dict[str, Path] = {
    "operativas":           ROOT   / "model_Tony" / "output_alertas_operativas.csv",
    "contexto":             ROOT   / "model_Tony" / "output_senales_contexto.csv",
    "demo":                 ROOT   / "model_Tony" / "output_demo_sintetico.csv",
    "sample_alertas":       DASHBD / "data" / "sample_alertas.csv",
    "sample_transacciones": DASHBD / "data" / "sample_transacciones.csv",
}

# Columnas visibles en la tabla de alertas
COLS_ALERTAS: list[str] = [
    "fecha", "id_transaccion", "folio", "codigo", "monto",
    "cluster_anomalia", "nivel_riesgo", "score_riesgo", "mensaje_alerta",
]

# Columnas visibles en la tabla de todas las transacciones
COLS_TRANSACCIONES: list[str] = [
    "fecha", "id_transaccion", "folio", "codigo", "monto",
    "es_anomalia", "es_alerta_operativa", "cluster_anomalia",
    "nivel_riesgo", "score_riesgo",
]

_ORDEN_NIVELES = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAJO": 3}


# ── Helpers internos ──────────────────────────────────────────────────────────

def _parse_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna 'fecha' a tipo datetime.date para poder filtrar."""
    if "fecha" in df.columns:
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    return df


def _normalizar_score(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que score_riesgo sea int (puede llegar como float desde CSV)."""
    if "score_riesgo" in df.columns:
        df = df.copy()
        df["score_riesgo"] = pd.to_numeric(df["score_riesgo"], errors="coerce").fillna(0).astype(int)
    return df


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_alertas(use_sample: bool = True) -> tuple[pd.DataFrame | None, str]:
    """
    Carga el dataset de alertas operativas.

    Returns
    -------
    (DataFrame, descripcion_fuente)  si hay datos.
    (None, mensaje_de_error)          si no se encontró ningún archivo.
    """
    if use_sample:
        p = PATHS["sample_alertas"]
        if p.exists():
            df = _normalizar_score(_parse_fechas(pd.read_csv(p)))
            return df, f"Muestra de prueba · {len(df):,} alertas"
        # La muestra no existe → instrucciones claras
        return None, (
            "**No existe la muestra de prueba** (`dashboard/data/sample_alertas.csv`).\n\n"
            "**Paso 1** — genera los datos del modelo:\n"
            "```\nuv run python anomaly_detection/run_demo.py\n```\n\n"
            "**Paso 2** — genera la muestra para el dashboard:\n"
            "```\nuv run python dashboard/create_sample.py\n```"
        )

    # Modo completo: prioridad operativas → demo sintético
    for key in ("operativas", "demo"):
        p = PATHS[key]
        if p.exists():
            df = _normalizar_score(_parse_fechas(pd.read_csv(p)))
            return df, f"{p.name} · {len(df):,} alertas"

    return None, (
        "**No se encontraron archivos de salida del modelo.**\n\n"
        "Genera los datos con:\n```\nuv run python anomaly_detection/run_demo.py\n```"
    )


def load_transacciones(use_sample: bool = True) -> pd.DataFrame | None:
    """
    Carga el dataset completo (alertas + señales de contexto si existe).
    Devuelve None si no hay archivo disponible; el dashboard cae back a solo alertas.
    """
    if use_sample:
        p = PATHS["sample_transacciones"]
        if p.exists():
            return _normalizar_score(_parse_fechas(pd.read_csv(p)))
        return None

    dfs: list[pd.DataFrame] = []
    for key in ("operativas", "contexto", "demo"):
        p = PATHS[key]
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    if "id_transaccion" in df.columns:
        df = df.drop_duplicates(subset="id_transaccion")
    return _normalizar_score(_parse_fechas(df))


# ── Filtrado ──────────────────────────────────────────────────────────────────

def filter_by_date(
    df: pd.DataFrame,
    preset: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Filtra el DataFrame según el período seleccionado en el sidebar."""
    if "fecha" not in df.columns or df.empty:
        return df

    today = date.today()

    if preset == "Hoy":
        mask = df["fecha"] == today
    elif preset == "Esta semana":
        lunes = today - timedelta(days=today.weekday())
        mask = (df["fecha"] >= lunes) & (df["fecha"] <= today)
    elif preset == "Este mes":
        inicio = today.replace(day=1)
        mask = (df["fecha"] >= inicio) & (df["fecha"] <= today)
    elif preset == "Este año":
        inicio = today.replace(month=1, day=1)
        mask = (df["fecha"] >= inicio) & (df["fecha"] <= today)
    elif preset == "Rango personalizado" and start is not None and end is not None:
        mask = (df["fecha"] >= start) & (df["fecha"] <= end)
    else:
        return df  # "Todas"

    return df[mask].copy()


def get_clusters_disponibles(df: pd.DataFrame) -> list[str]:
    """Lista de clusters únicos presentes en el dataset, ordenados."""
    if df.empty or "cluster_anomalia" not in df.columns:
        return []
    clusters: set[str] = set()
    for val in df["cluster_anomalia"].dropna():
        for c in str(val).split(" | "):
            c = c.strip()
            if c:
                clusters.add(c)
    return sorted(clusters)


# ── KPIs ──────────────────────────────────────────────────────────────────────

def compute_kpis(alertas: pd.DataFrame, todas: pd.DataFrame | None) -> dict:
    """
    Calcula los indicadores principales para la fila superior del dashboard.

    Parameters
    ----------
    alertas : DataFrame filtrado de alertas operativas (ya con filtros aplicados).
    todas   : DataFrame completo de transacciones (puede ser None).
    """
    total = len(todas) if todas is not None else len(alertas)
    n_op  = len(alertas)

    niveles: pd.Series = (
        alertas["nivel_riesgo"].value_counts()
        if not alertas.empty and "nivel_riesgo" in alertas.columns
        else pd.Series(dtype=int)
    )

    def _cnt_cluster(nombre: str) -> int:
        if alertas.empty or "cluster_anomalia" not in alertas.columns:
            return 0
        return int(
            alertas["cluster_anomalia"]
            .str.contains(nombre, na=False, regex=False)
            .sum()
        )

    return {
        "total":              total,
        "alertas_operativas": n_op,
        "pct_alerta":         n_op / total if total > 0 else 0.0,
        "criticas":           int(niveles.get("CRITICO", 0)),
        "altas":              int(niveles.get("ALTO",    0)),
        "medias":             int(niveles.get("MEDIO",   0)),
        "duplicados":         _cnt_cluster("DUPLICADO"),
        "pagos_sospechosos":  _cnt_cluster("PAGO_PROVEEDOR_SOSPECHOSO"),
        "fuera_estancia":     _cnt_cluster("FUERA_DE_ESTANCIA"),
        "montos_atipicos":    _cnt_cluster("MONTO_ATIPICO"),
        "cancelaciones":      _cnt_cluster("CANCELACION_SOSPECHOSA"),
    }
