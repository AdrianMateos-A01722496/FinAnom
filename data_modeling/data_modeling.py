"""Modelado de datos FINANOM para deteccion no supervisada.

Toma la base limpia (`data_cleaning/output/transacciones_limpio.parquet`) y
produce un dataset listo para Isolation Forest:

- Una fila = una transaccion (mismo grano que hottra).
- Columnas `trace_*`: trazabilidad para auditoria, NO se usan como features.
- Columnas `feat_*`: matriz numerica, sin nulos, codificada y escalada cuando
  aporta al problema.

La evaluacion de calidad se ejecuta dentro del pipeline para que el equipo pueda
recrear desde cero no solo el parquet final, sino tambien las decisiones:

1. Diagnosticos estadisticos: nulos, varianza, correlacion alta.
2. Eliminacion de redundancias obvias generadas en el primer pase.
3. Isolation Forest proxy sobre muestra para inspeccionar senal por feature.

Uso:
    uv run python data_modeling/data_modeling.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# --------------------------------------------------------------------------- #
# Rutas y parametros reproducibles
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data_cleaning" / "output" / "transacciones_limpio.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TRAINING_DATA_DIR = ROOT / "training_data"
OUTPUT_FILE = TRAINING_DATA_DIR / "transacciones_modelado.parquet"
FEATURE_MATRIX_FILE = TRAINING_DATA_DIR / "X_modelo.parquet"
DICT_FILE = Path(__file__).resolve().parent / "diccionario_modelado.md"
QUALITY_REPORT_FILE = OUTPUT_DIR / "reporte_calidad_modelado.md"
QUALITY_JSON_FILE = OUTPUT_DIR / "diagnosticos_modelado.json"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "proxy_feature_importance.csv"
ANOMALY_SAMPLE_FILE = OUTPUT_DIR / "proxy_anomaly_sample.csv"
IMPORTANCE_PLOT_FILE = OUTPUT_DIR / "proxy_feature_importance_top20.png"
SCORE_PLOT_FILE = OUTPUT_DIR / "proxy_anomaly_score_distribution.png"

RANDOM_STATE = 42
PROXY_SAMPLE_SIZE = 120_000
PERMUTATION_SAMPLE_SIZE = 20_000
PROXY_CONTAMINATION = 0.02


@dataclass(frozen=True)
class ModelingPaths:
    """Rutas de artefactos generados por la fase de modelado."""

    output_dir: Path
    training_data_dir: Path
    output_file: Path
    feature_matrix_file: Path
    quality_report_file: Path
    quality_json_file: Path
    feature_importance_file: Path
    anomaly_sample_file: Path
    importance_plot_file: Path
    score_plot_file: Path

    @classmethod
    def from_dirs(
        cls,
        output_dir: Path | str = OUTPUT_DIR,
        training_data_dir: Path | str = TRAINING_DATA_DIR,
    ) -> "ModelingPaths":
        output_dir = Path(output_dir)
        training_data_dir = Path(training_data_dir)
        return cls(
            output_dir=output_dir,
            training_data_dir=training_data_dir,
            output_file=training_data_dir / "transacciones_modelado.parquet",
            feature_matrix_file=training_data_dir / "X_modelo.parquet",
            quality_report_file=output_dir / "reporte_calidad_modelado.md",
            quality_json_file=output_dir / "diagnosticos_modelado.json",
            feature_importance_file=output_dir / "proxy_feature_importance.csv",
            anomaly_sample_file=output_dir / "proxy_anomaly_sample.csv",
            importance_plot_file=output_dir / "proxy_feature_importance_top20.png",
            score_plot_file=output_dir / "proxy_anomaly_score_distribution.png",
        )


# --------------------------------------------------------------------------- #
# Columnas de entrada por rol
# --------------------------------------------------------------------------- #
TRACE_SOURCE_COLS = [
    "t_folio",
    "t_folio_ext",
    "t_referencia",
    "t_transaccion",
    "t_cve_res",
    "t_cuarto",
    "t_codigo",
    "t_timestamp",
]

CATEGORICAL_COLS = [
    "t_codigo",
    "t_folio_ext",
    "t_usuario",
    "t_usuario_mod",
    "h_tpo_hab",
    "h_tpo_hsp",
    "h_seg_mer",
    "h_cod_age",
    "h_tpo_plan",
    "h_for_pgo",
]

RESERVATION_NUMERIC_COLS = [
    "h_num_per",
    "h_num_noc",
    "h_tfa",
    "h_tfa_total",
    "h_tarifa_forzada",
    "h_dep_sol",
]

MONEY_BY_CODE_COLS = ["t_monto", "t_impuesto", "t_propina"]


# --------------------------------------------------------------------------- #
# Descripciones del diccionario final
# --------------------------------------------------------------------------- #
TRACE_DESCRIPTIONS = {
    "trace_row_id": (
        "Trazabilidad",
        "Identificador estable: posicion de la fila en la base limpia.",
        "Derivado como indice entero; no entra al modelo.",
    ),
    "trace_t_folio": (
        "Trazabilidad",
        "Folio de la cuenta donde vive la transaccion.",
        "Copia de `t_folio`; no entra al modelo.",
    ),
    "trace_t_folio_ext": (
        "Trazabilidad",
        "Extension/subcuenta del folio.",
        "Copia de `t_folio_ext`; no entra al modelo.",
    ),
    "trace_t_referencia": (
        "Trazabilidad",
        "Referencia operativa del movimiento.",
        "Copia de `t_referencia`; no entra al modelo.",
    ),
    "trace_t_transaccion": (
        "Trazabilidad",
        "Numero correlativo de la transaccion.",
        "Copia de `t_transaccion`; no entra al modelo.",
    ),
    "trace_t_cve_res": (
        "Trazabilidad",
        "Llave de reservacion enlazada, cuando existe.",
        "Copia de `t_cve_res`; no entra al modelo.",
    ),
    "trace_t_cuarto": (
        "Trazabilidad",
        "Cuarto/habitacion asociado al cargo.",
        "Copia de `t_cuarto`; no entra al modelo.",
    ),
    "trace_t_codigo": (
        "Trazabilidad",
        "Codigo original del concepto financiero.",
        "Copia de `t_codigo`; el modelo usa su frecuencia y escalados por codigo.",
    ),
    "trace_t_timestamp": (
        "Trazabilidad",
        "Timestamp original del cargo.",
        "Copia de `t_timestamp`; el modelo usa features temporales derivadas.",
    ),
}

FEATURE_DESCRIPTIONS = {
    # Flags contables / operativos
    "feat_es_abono": (
        "Flag contable",
        "Movimiento marcado como abono (`t_carabo == 1`).",
        "Binario 0/1.",
    ),
    "feat_cargo_cancelado": (
        "Flag operativo",
        "Transaccion marcada como cancelada.",
        "Binario 0/1 desde `t_tra_cancelada == 1`.",
    ),
    "feat_cancelacion_sin_marca": (
        "Flag de calidad",
        "Transaccion sin marca explicita de cancelacion.",
        "Binario 0/1 desde nulo en `t_tra_cancelada`.",
    ),
    "feat_es_split": (
        "Flag operativo",
        "Cargo generado como split.",
        "Binario 0/1 desde `es_split`; se conserva aunque sea raro por relevancia de doble conteo.",
    ),
    "feat_es_renta": (
        "Flag financiero",
        "Cargo de renta/hospedaje.",
        "Binario 0/1 desde `es_renta`.",
    ),
    "feat_tiene_reservacion": (
        "Flag de contexto",
        "La transaccion tiene contexto de reservacion enlazado.",
        "Binario 0/1 desde `tiene_reservacion`.",
    ),
    "feat_usuario_modificado": (
        "Flag operativo",
        "La transaccion tiene usuario modificador.",
        "Binario 0/1 desde `t_usuario_mod` no nulo.",
    ),
    "feat_usuario_mod_distinto": (
        "Flag operativo",
        "El usuario modificador existe y es distinto al usuario creador.",
        "Binario 0/1 comparando `t_usuario_mod` contra `t_usuario`.",
    ),
    "feat_monto_negativo_sin_abono": (
        "Regla financiera",
        "Monto negativo en movimiento que no esta marcado como abono.",
        "Binario 0/1; captura inconsistencias de signo contable.",
    ),
    "feat_monto_positivo_en_abono": (
        "Regla financiera",
        "Monto positivo en movimiento marcado como abono.",
        "Binario 0/1; captura inconsistencias de signo contable.",
    ),
    "feat_impuesto_signo_distinto_monto": (
        "Regla fiscal",
        "El impuesto y el monto tienen signos distintos.",
        "Binario 0/1 con monto e impuesto diferentes de cero.",
    ),
    "feat_propina_mayor_monto": (
        "Regla financiera",
        "La propina absoluta supera el monto absoluto.",
        "Binario 0/1; senal fuerte de captura rara.",
    ),
    # Monetarias
    "feat_monto_abs_log": (
        "Monto",
        "Magnitud del monto sin signo.",
        "`log1p(abs(t_monto))`; reduce cola larga sin perder orden.",
    ),
    "feat_impuesto_abs_log": (
        "Monto",
        "Magnitud del impuesto sin signo.",
        "`log1p(abs(t_impuesto))`; reduce cola larga.",
    ),
    "feat_propina_abs_log": (
        "Monto",
        "Magnitud de la propina sin signo.",
        "`log1p(abs(t_propina))`; reduce cola larga.",
    ),
    "feat_monto_z_codigo_carabo": (
        "Monto escalado",
        "Monto relativo a su concepto y naturaleza contable.",
        "Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10].",
    ),
    "feat_impuesto_z_codigo_carabo": (
        "Monto escalado",
        "Impuesto relativo a su concepto y naturaleza contable.",
        "Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10].",
    ),
    "feat_propina_z_codigo_carabo": (
        "Monto escalado",
        "Propina relativa a su concepto y naturaleza contable.",
        "Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10].",
    ),
    "feat_impuesto_ratio_abs": (
        "Ratio financiero",
        "Impuesto absoluto contra monto absoluto.",
        "`abs(t_impuesto) / abs(t_monto)`, imputado a 0 si monto=0, clip [0, 2].",
    ),
    "feat_propina_ratio_abs": (
        "Ratio financiero",
        "Propina absoluta contra monto absoluto.",
        "`abs(t_propina) / abs(t_monto)`, imputado a 0 si monto=0, clip [0, 2].",
    ),
    "feat_desvio_iva_16_abs": (
        "Ratio fiscal",
        "Distancia absoluta contra una tasa de referencia de 16%.",
        "`abs(abs(impuesto)/abs(monto) - 0.16)`, clip [0, 2]; no fuerza regla, solo senal.",
    ),
    "feat_monto_vs_tarifa_ratio": (
        "Ratio reserva",
        "Monto absoluto comparado contra tarifa diaria de la reserva.",
        "`abs(t_monto) / abs(h_tfa)`, imputado a 0 si no aplica, clip [0, 10].",
    ),
    "feat_monto_vs_tarifa_total_ratio": (
        "Ratio reserva",
        "Monto absoluto comparado contra tarifa total de la reserva.",
        "`abs(t_monto) / abs(h_tfa_total)`, imputado a 0 si no aplica, clip [0, 10].",
    ),
    # Numericas de reservacion / estancia
    "feat_t_noches_scaled": (
        "Estancia",
        "Noches asociadas al cargo.",
        "Robust scale global de `t_noches`, clip [-10, 10].",
    ),
    "feat_h_num_per_scaled": (
        "Reservacion",
        "Personas esperadas en la reservacion.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_h_num_noc_scaled": (
        "Reservacion",
        "Noches esperadas en la reservacion.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_h_tfa_scaled": (
        "Reservacion",
        "Tarifa diaria de la reservacion.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_h_tfa_total_scaled": (
        "Reservacion",
        "Tarifa total de la reservacion.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_h_tarifa_forzada_scaled": (
        "Reservacion",
        "Tarifa forzada manualmente.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_h_dep_sol_scaled": (
        "Reservacion",
        "Deposito solicitado.",
        "Robust scale global con nulos imputados a mediana antes de escalar.",
    ),
    "feat_noches_delta_scaled": (
        "Consistencia",
        "Diferencia entre noches del cargo y noches de la reservacion.",
        "Robust scale global de `t_noches - h_num_noc`; nulos imputados a 0.",
    ),
    # Temporales
    "feat_hora_sin": (
        "Temporal",
        "Hora del cargo en ciclo diario.",
        "`sin(2*pi*hora_decimal/24)`.",
    ),
    "feat_hora_cos": (
        "Temporal",
        "Hora del cargo en ciclo diario.",
        "`cos(2*pi*hora_decimal/24)`.",
    ),
    "feat_dia_semana_sin": (
        "Temporal",
        "Dia de semana en ciclo semanal.",
        "`sin(2*pi*dia_semana/7)`.",
    ),
    "feat_dia_semana_cos": (
        "Temporal",
        "Dia de semana en ciclo semanal.",
        "`cos(2*pi*dia_semana/7)`.",
    ),
    "feat_mes_sin": (
        "Temporal",
        "Mes del cargo en ciclo anual.",
        "`sin(2*pi*mes/12)`.",
    ),
    "feat_mes_cos": (
        "Temporal",
        "Mes del cargo en ciclo anual.",
        "`cos(2*pi*mes/12)`.",
    ),
    "feat_es_fin_semana": (
        "Temporal",
        "Cargo registrado en sabado o domingo.",
        "Binario 0/1 desde `t_timestamp.dayofweek >= 5`.",
    ),
    "feat_es_madrugada": (
        "Temporal",
        "Cargo registrado entre 00:00 y 05:59.",
        "Binario 0/1 desde hora del timestamp.",
    ),
    "feat_dias_desde_llegada_scaled": (
        "Temporal reserva",
        "Dias entre llegada y cargo.",
        "Robust scale global, nulos imputados a 0, clip [-10, 10].",
    ),
    "feat_dias_hasta_salida_scaled": (
        "Temporal reserva",
        "Dias entre cargo y salida.",
        "Robust scale global, nulos imputados a 0, clip [-10, 10].",
    ),
    "feat_cargo_antes_llegada": (
        "Temporal reserva",
        "Cargo fechado antes de la llegada.",
        "Binario 0/1 solo si existe reservacion.",
    ),
    "feat_cargo_despues_salida": (
        "Temporal reserva",
        "Cargo fechado despues de la salida.",
        "Binario 0/1 solo si existe reservacion.",
    ),
    "feat_cargo_fuera_estancia": (
        "Temporal reserva",
        "Cargo antes de llegada o despues de salida.",
        "Binario 0/1; senal directa para revision.",
    ),
    # Duplicados / densidad
    "feat_dup_mismo_dia_flag": (
        "Duplicados",
        "Existe mas de un cargo con mismo folio, subfolio, codigo, monto y dia.",
        "Binario 0/1; captura el dolor principal de cargos duplicados.",
    ),
    "feat_dup_mismo_dia_log": (
        "Duplicados",
        "Intensidad de duplicados exactos por dia.",
        "`log1p(conteo - 1)` por folio/subfolio/codigo/monto/dia.",
    ),
    "feat_dup_mismo_minuto_flag": (
        "Duplicados",
        "Existe mas de un cargo identico incluso en el mismo minuto.",
        "Binario 0/1 por folio/subfolio/codigo/monto/timestamp.",
    ),
    "feat_folio_codigo_dia_count_log": (
        "Densidad folio",
        "Repeticion del mismo concepto en el folio durante el dia.",
        "`log1p(conteo)` por folio/subfolio/codigo/dia.",
    ),
    "feat_folio_dia_movimientos_log": (
        "Densidad folio",
        "Cantidad total de movimientos del folio en el dia.",
        "`log1p(conteo)` por folio/subfolio/dia.",
    ),
    "feat_folio_total_movimientos_log": (
        "Densidad folio",
        "Cantidad total de movimientos historicos del folio.",
        "`log1p(conteo)` por folio/subfolio.",
    ),
    # Texto operacional
    "feat_obs_missing": (
        "Texto derivado",
        "Observaciones vacias.",
        "Binario 0/1; no se exporta texto crudo por posible PII.",
    ),
    "feat_obs_len_log": (
        "Texto derivado",
        "Longitud del texto de observaciones.",
        "`log1p(numero de caracteres)`.",
    ),
    "feat_obs_kw_ajuste": (
        "Texto derivado",
        "Observacion menciona ajuste.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
    "feat_obs_kw_cancelacion": (
        "Texto derivado",
        "Observacion menciona cancelacion.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
    "feat_obs_kw_error": (
        "Texto derivado",
        "Observacion menciona error.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
    "feat_obs_kw_cortesia": (
        "Texto derivado",
        "Observacion menciona cortesia/compensacion.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
    "feat_obs_kw_deposito": (
        "Texto derivado",
        "Observacion menciona deposito.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
    "feat_obs_kw_reembolso": (
        "Texto derivado",
        "Observacion menciona reembolso/devolucion.",
        "Binario 0/1 por keyword en `t_observaciones`.",
    ),
}


@dataclass(frozen=True)
class DiagnosticResult:
    """Resultado de los diagnosticos de calidad."""

    dropped_features: dict[str, str]
    high_correlation_pairs: list[dict[str, object]]
    near_zero_features: list[dict[str, object]]
    feature_importance: pd.DataFrame
    anomaly_score_summary: dict[str, float]
    proxy_sample_size: int


# --------------------------------------------------------------------------- #
# Utilidades numericas
# --------------------------------------------------------------------------- #
def safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    """Division estable: evita inf/NaN y devuelve fill cuando no aplica."""
    out = num.astype("float64") / den.replace(0, np.nan).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def clip_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    """Clip numerico preservando Series."""
    return s.astype("float64").clip(lower=lower, upper=upper)


def robust_scale_global(s: pd.Series, fill_value: float | None = None) -> pd.Series:
    """Escala con mediana/IQR, imputa nulos a la mediana y limita extremos."""
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    median = float(x.median()) if fill_value is None else float(fill_value)
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or abs(iqr) < 1e-9:
        std = float(x.std())
        iqr = std if np.isfinite(std) and std > 1e-9 else 1.0
    out = (x.fillna(median) - median) / iqr
    return clip_series(out, -10, 10)


def robust_scale_by_group(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.Series:
    """Robust z por grupo usando mediana/IQR, con fallback global."""
    value = pd.to_numeric(df[value_col], errors="coerce").astype("float64")
    global_median = float(value.median())
    global_iqr = float(value.quantile(0.75) - value.quantile(0.25))
    if not np.isfinite(global_iqr) or abs(global_iqr) < 1e-9:
        global_iqr = float(value.std()) if float(value.std()) > 1e-9 else 1.0

    stats = (
        df[group_cols + [value_col]]
        .assign(**{value_col: value})
        .groupby(group_cols, dropna=False)[value_col]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            count="count",
        )
        .reset_index()
    )
    stats["scale"] = stats["q3"] - stats["q1"]
    stats.loc[~np.isfinite(stats["scale"]) | (stats["scale"].abs() < 1e-9), "scale"] = global_iqr
    stats.loc[~np.isfinite(stats["median"]), "median"] = global_median

    joined = df[group_cols].merge(stats[group_cols + ["median", "scale"]], on=group_cols, how="left")
    median = joined["median"].fillna(global_median).to_numpy()
    scale = joined["scale"].fillna(global_iqr).to_numpy()
    out = (value.fillna(global_median).to_numpy() - median) / scale
    return pd.Series(np.clip(out, -10, 10), index=df.index)


def frequency_encode(s: pd.Series) -> pd.Series:
    """Encoding por frecuencia relativa. Los nulos se tratan como categoria."""
    filled = s.astype("string").fillna("__MISSING__")
    freq = filled.value_counts(normalize=True, dropna=False)
    return filled.map(freq).astype("float64")


def add_prefixed_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas de trazabilidad que no entran al modelo."""
    missing = [c for c in TRACE_SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de trazabilidad en la base limpia: {missing}")

    trace = df[TRACE_SOURCE_COLS].copy()
    trace.columns = [f"trace_{c}" for c in trace.columns]
    trace.insert(0, "trace_row_id", np.arange(len(df), dtype=np.int64))
    return trace


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features candidatas numericas para Isolation Forest."""
    out = pd.DataFrame(index=df.index)

    # Flags contables / operativos.
    out["feat_es_abono"] = df["t_carabo"].astype("string").eq("1").fillna(False).astype("int8")
    out["feat_cargo_cancelado"] = (
        df["t_tra_cancelada"].astype("string").eq("1").fillna(False).astype("int8")
    )
    out["feat_cancelacion_sin_marca"] = df["t_tra_cancelada"].isna().astype("int8")
    out["feat_es_split"] = df["es_split"].astype("int8")
    out["feat_es_renta"] = df["es_renta"].astype("int8")
    out["feat_tiene_reservacion"] = df["tiene_reservacion"].astype("int8")

    # Estas dos se generan en el primer pase; el diagnostico elimina sus inversos redundantes.
    out["feat_falta_contexto_reserva"] = (~df["tiene_reservacion"]).astype("int8")
    out["feat_usuario_modificado"] = df["t_usuario_mod"].notna().astype("int8")
    out["feat_t_usuario_mod_missing"] = df["t_usuario_mod"].isna().astype("int8")
    usuario_mod = df["t_usuario_mod"].astype("string")
    usuario = df["t_usuario"].astype("string")
    out["feat_usuario_mod_distinto"] = (
        usuario_mod.notna() & usuario_mod.ne(usuario).fillna(False)
    ).astype("int8")

    monto = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0)
    impuesto = pd.to_numeric(df["t_impuesto"], errors="coerce").fillna(0.0)
    propina = pd.to_numeric(df["t_propina"], errors="coerce").fillna(0.0)
    monto_abs = monto.abs()

    out["feat_monto_negativo_sin_abono"] = ((monto < 0) & (out["feat_es_abono"] == 0)).astype("int8")
    out["feat_monto_positivo_en_abono"] = ((monto > 0) & (out["feat_es_abono"] == 1)).astype("int8")
    out["feat_impuesto_signo_distinto_monto"] = (
        (monto != 0) & (impuesto != 0) & (np.sign(monto) != np.sign(impuesto))
    ).astype("int8")
    out["feat_propina_mayor_monto"] = ((propina.abs() > monto_abs) & (monto_abs > 0)).astype("int8")

    # Montos: log para magnitud, robust z por concepto/naturaleza para comparabilidad.
    out["feat_monto_abs_log"] = np.log1p(monto_abs)
    out["feat_impuesto_abs_log"] = np.log1p(impuesto.abs())
    out["feat_propina_abs_log"] = np.log1p(propina.abs())
    for source_col, feature_col in [
        ("t_monto", "feat_monto_z_codigo_carabo"),
        ("t_impuesto", "feat_impuesto_z_codigo_carabo"),
        ("t_propina", "feat_propina_z_codigo_carabo"),
    ]:
        out[feature_col] = robust_scale_by_group(df, source_col, ["t_codigo", "t_carabo"])

    out["feat_impuesto_ratio_abs"] = clip_series(safe_divide(impuesto.abs(), monto_abs), 0, 2)
    out["feat_propina_ratio_abs"] = clip_series(safe_divide(propina.abs(), monto_abs), 0, 2)
    out["feat_desvio_iva_16_abs"] = clip_series((out["feat_impuesto_ratio_abs"] - 0.16).abs(), 0, 2)
    out["feat_monto_vs_tarifa_ratio"] = clip_series(
        safe_divide(monto_abs, pd.to_numeric(df["h_tfa"], errors="coerce").abs()), 0, 10
    )
    out["feat_monto_vs_tarifa_total_ratio"] = clip_series(
        safe_divide(monto_abs, pd.to_numeric(df["h_tfa_total"], errors="coerce").abs()), 0, 10
    )

    # Numericas de reservacion y consistencia estancia-cargo.
    out["feat_t_noches_scaled"] = robust_scale_global(df["t_noches"])
    for c in RESERVATION_NUMERIC_COLS:
        out[f"feat_{c}_scaled"] = robust_scale_global(df[c])
    noches_delta = pd.to_numeric(df["t_noches"], errors="coerce") - pd.to_numeric(df["h_num_noc"], errors="coerce")
    out["feat_noches_delta_scaled"] = robust_scale_global(noches_delta.fillna(0.0))

    # Temporales ciclicas y relativas a la estancia.
    ts = pd.to_datetime(df["t_timestamp"])
    hour_decimal = ts.dt.hour + ts.dt.minute / 60
    out["feat_hora_sin"] = np.sin(2 * np.pi * hour_decimal / 24)
    out["feat_hora_cos"] = np.cos(2 * np.pi * hour_decimal / 24)
    day_of_week = ts.dt.dayofweek
    out["feat_dia_semana_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["feat_dia_semana_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    month = ts.dt.month
    out["feat_mes_sin"] = np.sin(2 * np.pi * month / 12)
    out["feat_mes_cos"] = np.cos(2 * np.pi * month / 12)
    out["feat_es_fin_semana"] = (day_of_week >= 5).astype("int8")
    out["feat_es_madrugada"] = ts.dt.hour.between(0, 5).astype("int8")

    cargo_date = ts.dt.normalize()
    llegada = pd.to_datetime(df["h_fec_lld"])
    salida = pd.to_datetime(df["h_fec_sda"])
    dias_desde_llegada = (cargo_date - llegada).dt.days
    dias_hasta_salida = (salida - cargo_date).dt.days
    out["feat_dias_desde_llegada_scaled"] = robust_scale_global(dias_desde_llegada.fillna(0.0))
    out["feat_dias_hasta_salida_scaled"] = robust_scale_global(dias_hasta_salida.fillna(0.0))
    has_res = df["tiene_reservacion"].astype(bool)
    out["feat_cargo_antes_llegada"] = (has_res & (dias_desde_llegada < 0)).fillna(False).astype("int8")
    out["feat_cargo_despues_salida"] = (has_res & (dias_hasta_salida < 0)).fillna(False).astype("int8")
    out["feat_cargo_fuera_estancia"] = (
        (out["feat_cargo_antes_llegada"] == 1) | (out["feat_cargo_despues_salida"] == 1)
    ).astype("int8")

    # Conteos de duplicados y densidad por folio.
    work = pd.DataFrame(
        {
            "folio": df["t_folio"].astype("string").fillna("__MISSING__"),
            "folio_ext": df["t_folio_ext"].astype("string").fillna("__MISSING__"),
            "codigo": df["t_codigo"].astype("string").fillna("__MISSING__"),
            "monto_centavos": np.rint(monto * 100).astype("int64"),
            "fecha": cargo_date,
            "timestamp": ts,
        },
        index=df.index,
    )
    dup_day = work.groupby(
        ["folio", "folio_ext", "codigo", "monto_centavos", "fecha"], dropna=False
    )["codigo"].transform("size")
    dup_minute = work.groupby(
        ["folio", "folio_ext", "codigo", "monto_centavos", "timestamp"], dropna=False
    )["codigo"].transform("size")
    folio_codigo_day = work.groupby(["folio", "folio_ext", "codigo", "fecha"], dropna=False)["codigo"].transform("size")
    folio_day = work.groupby(["folio", "folio_ext", "fecha"], dropna=False)["codigo"].transform("size")
    folio_total = work.groupby(["folio", "folio_ext"], dropna=False)["codigo"].transform("size")

    out["feat_dup_mismo_dia_flag"] = (dup_day > 1).astype("int8")
    out["feat_dup_mismo_dia_log"] = np.log1p((dup_day - 1).clip(lower=0))
    out["feat_dup_mismo_minuto_flag"] = (dup_minute > 1).astype("int8")
    out["feat_folio_codigo_dia_count_log"] = np.log1p(folio_codigo_day)
    out["feat_folio_dia_movimientos_log"] = np.log1p(folio_day)
    out["feat_folio_total_movimientos_log"] = np.log1p(folio_total)

    # Encoding por frecuencia de categoricas. No agrupa "otros" para no ocultar rarezas.
    for c in CATEGORICAL_COLS:
        out[f"feat_{c}_freq"] = frequency_encode(df[c])

    # Texto libre solo como senales derivadas; no exportamos el texto crudo.
    obs = df["t_observaciones"].astype("string")
    obs_upper = obs.fillna("").str.upper()
    out["feat_obs_missing"] = obs.isna().astype("int8")
    out["feat_obs_len_log"] = np.log1p(obs.fillna("").str.len().astype("float64"))
    keyword_patterns = {
        "feat_obs_kw_ajuste": r"AJUST|AJUS|AJSTE",
        "feat_obs_kw_cancelacion": r"CANCEL|CANC\.?",
        "feat_obs_kw_error": r"ERROR|ERRONE|ERR ",
        "feat_obs_kw_cortesia": r"CORTES|CORTESIA|COMPENSA|COMPLIMENT",
        "feat_obs_kw_deposito": r"DEPOS|DEP[OÓ]S",
        "feat_obs_kw_reembolso": r"REEMBOL|DEVOL|DEV ",
    }
    for feature, pattern in keyword_patterns.items():
        out[feature] = obs_upper.str.contains(pattern, regex=True, na=False).astype("int8")

    # Seguridad: todo feature debe ser finito y numerico.
    out = out.replace([np.inf, -np.inf], np.nan)
    if out.isna().any().any():
        null_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Features con NaN despues de feature engineering: {null_cols}")

    return downcast_features(out)


def downcast_features(features: pd.DataFrame) -> pd.DataFrame:
    """Reduce tamano del parquet sin perder compatibilidad con sklearn."""
    out = features.copy()
    for c in out.columns:
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype("int8") if out[c].min() >= -128 and out[c].max() <= 127 else out[c].astype("int32")
        else:
            out[c] = out[c].astype("float32")
    return out


# --------------------------------------------------------------------------- #
# Diagnosticos y proxy model
# --------------------------------------------------------------------------- #
def find_high_correlation_pairs(features: pd.DataFrame, threshold: float = 0.985) -> list[dict[str, object]]:
    """Encuentra pares de features con correlacion absoluta muy alta."""
    corr = features.corr(numeric_only=True).abs()
    pairs: list[dict[str, object]] = []
    cols = corr.columns.to_list()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr.loc[left, right])
            if np.isfinite(value) and value >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "abs_corr": round(value, 6)})
    return pairs


def near_zero_variance(features: pd.DataFrame) -> list[dict[str, object]]:
    """Lista features constantes o extremadamente raras para revisar manualmente."""
    out: list[dict[str, object]] = []
    n = len(features)
    for c in features.columns:
        s = features[c]
        nunique = int(s.nunique(dropna=False))
        if nunique <= 1:
            out.append({"feature": c, "reason": "constante", "nunique": nunique, "minority_pct": 0.0})
            continue
        vc = s.value_counts(dropna=False, normalize=True)
        majority = float(vc.iloc[0])
        minority_pct = 100 * (1 - majority)
        if nunique == 2 and minority_pct < 0.1:
            out.append(
                {
                    "feature": c,
                    "reason": "binaria muy rara, revisar pero no eliminar automaticamente",
                    "nunique": nunique,
                    "minority_pct": round(minority_pct, 4),
                    "minority_rows": int(round(n * (1 - majority))),
                }
            )
    return out


def select_final_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Elimina redundancias y features sin varianza detectadas en diagnostico."""
    dropped = {
        "feat_falta_contexto_reserva": "Inverso perfecto de `feat_tiene_reservacion`; se conserva la version positiva.",
        "feat_t_usuario_mod_missing": "Inverso perfecto de `feat_usuario_modificado`; se conserva la version positiva.",
    }

    for c in features.columns:
        if c in dropped:
            continue
        if features[c].nunique(dropna=False) <= 1:
            dropped[c] = "Constante en la base actual; no aporta separacion al Isolation Forest."

    corr_rules = [
        (
            "feat_t_usuario_mod_freq",
            "feat_usuario_modificado",
            "Correlacion >= 0.985 con `feat_usuario_modificado`; la frecuencia queda dominada por el nulo.",
        ),
        (
            "feat_desvio_iva_16_abs",
            "feat_impuesto_ratio_abs",
            "Correlacion >= 0.985 con `feat_impuesto_ratio_abs`; se conserva el ratio fiscal mas general.",
        ),
        (
            "feat_cargo_despues_salida",
            "feat_cargo_fuera_estancia",
            "Correlacion >= 0.985 con `feat_cargo_fuera_estancia`; se conserva la regla agregada.",
        ),
    ]
    remaining = features.drop(columns=[c for c in dropped if c in features.columns])
    for drop_col, keep_col, reason in corr_rules:
        if drop_col not in remaining.columns or keep_col not in remaining.columns:
            continue
        corr = remaining[[drop_col, keep_col]].corr().iloc[0, 1]
        if np.isfinite(corr) and abs(float(corr)) >= 0.985:
            dropped[drop_col] = reason
            remaining = remaining.drop(columns=[drop_col])

    existing = {c: r for c, r in dropped.items() if c in features.columns}
    return features.drop(columns=list(existing)), existing


def sample_features(features: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Muestra reproducible para diagnosticos costosos."""
    if len(features) <= sample_size:
        return features.copy()
    return features.sample(n=sample_size, random_state=random_state)


def proxy_feature_importance(model: IsolationForest, features: pd.DataFrame) -> pd.DataFrame:
    """Importancia ligera basada en frecuencia de splits dentro del Isolation Forest."""
    split_counts = np.zeros(features.shape[1], dtype="float64")
    for estimator in model.estimators_:
        tree_features = estimator.tree_.feature
        used = tree_features[tree_features >= 0]
        if len(used):
            counts = np.bincount(used, minlength=features.shape[1])
            split_counts += counts

    total_splits = split_counts.sum()
    if total_splits == 0:
        importance = np.zeros_like(split_counts)
    else:
        importance = split_counts / total_splits

    result = pd.DataFrame(
        {
            "feature": features.columns,
            "split_count": split_counts.astype(int),
            "split_importance": importance,
        }
    )
    return result.sort_values("split_importance", ascending=False).reset_index(drop=True)


def fit_proxy_iforest(features: pd.DataFrame) -> tuple[IsolationForest, pd.DataFrame, pd.Series]:
    """Entrena Isolation Forest rapido sobre muestra para evaluar senal."""
    sample = sample_features(features, PROXY_SAMPLE_SIZE, RANDOM_STATE)
    model = IsolationForest(
        n_estimators=120,
        max_samples=min(8192, len(sample)),
        contamination=PROXY_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(sample)
    # Mayor score = mas anomalo.
    score = pd.Series(-model.decision_function(sample), index=sample.index, name="proxy_anomaly_score")
    return model, sample, score


def write_proxy_plots(importance: pd.DataFrame, scores: pd.Series, paths: ModelingPaths) -> None:
    """Guarda graficas diagnosticas ligeras."""
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["split_importance"], color="#2f6f73")
    plt.xlabel("Importancia por frecuencia de splits")
    plt.title("Isolation Forest proxy - Top 20 features")
    plt.tight_layout()
    plt.savefig(paths.importance_plot_file, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=80, color="#3b6ea8", alpha=0.85)
    plt.axvline(scores.quantile(0.98), color="#b23b3b", linestyle="--", label="p98")
    plt.xlabel("Proxy anomaly score (mayor = mas anomalo)")
    plt.ylabel("Transacciones en muestra")
    plt.title("Distribucion de scores - Isolation Forest proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.score_plot_file, dpi=160)
    plt.close()


def run_diagnostics(features: pd.DataFrame, trace: pd.DataFrame, paths: ModelingPaths) -> DiagnosticResult:
    """Ejecuta la evaluacion recursiva de calidad y escribe artefactos auxiliares."""
    candidate_pairs = find_high_correlation_pairs(features)
    candidate_near_zero = near_zero_variance(features)
    final_features, dropped = select_final_features(features)
    final_pairs = find_high_correlation_pairs(final_features)
    final_near_zero = near_zero_variance(final_features)

    model, sample, scores = fit_proxy_iforest(final_features)
    importance = proxy_feature_importance(model, sample)
    importance.to_csv(paths.feature_importance_file, index=False)
    write_proxy_plots(importance, scores, paths)

    anomaly_cutoff = float(scores.quantile(1 - PROXY_CONTAMINATION))
    sample_out = (
        trace.loc[scores.index, ["trace_row_id", "trace_t_folio", "trace_t_codigo", "trace_t_timestamp"]]
        .assign(proxy_anomaly_score=scores)
        .sort_values("proxy_anomaly_score", ascending=False)
        .head(200)
    )
    sample_out.to_csv(paths.anomaly_sample_file, index=False)

    score_summary = {
        "min": float(scores.min()),
        "p50": float(scores.quantile(0.50)),
        "p95": float(scores.quantile(0.95)),
        "p98": float(scores.quantile(0.98)),
        "max": float(scores.max()),
        "cutoff_contamination_2pct": anomaly_cutoff,
    }

    diagnostics = {
        "input_rows": int(len(features)),
        "candidate_features": int(features.shape[1]),
        "final_features": int(final_features.shape[1]),
        "dropped_features": dropped,
        "candidate_high_correlation_pairs": candidate_pairs[:50],
        "final_high_correlation_pairs": final_pairs[:50],
        "candidate_near_zero_features": candidate_near_zero,
        "final_near_zero_features": final_near_zero,
        "proxy_sample_size": int(len(sample)),
        "proxy_contamination": PROXY_CONTAMINATION,
        "anomaly_score_summary": score_summary,
        "top_proxy_features": importance.head(20).to_dict(orient="records"),
    }
    paths.quality_json_file.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    result = DiagnosticResult(
        dropped_features=dropped,
        high_correlation_pairs=final_pairs,
        near_zero_features=final_near_zero,
        feature_importance=importance,
        anomaly_score_summary=score_summary,
        proxy_sample_size=len(sample),
    )
    write_quality_report(features, final_features, result, candidate_pairs, candidate_near_zero, paths)
    return result


# --------------------------------------------------------------------------- #
# Documentacion
# --------------------------------------------------------------------------- #
def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def display_path(path: Path) -> str:
    """Devuelve una ruta legible relativa al repo cuando se puede."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_quality_report(
    candidate_features: pd.DataFrame,
    final_features: pd.DataFrame,
    result: DiagnosticResult,
    candidate_pairs: list[dict[str, object]],
    candidate_near_zero: list[dict[str, object]],
    paths: ModelingPaths,
) -> None:
    """Escribe reporte markdown de la evaluacion iterativa."""
    lines: list[str] = []
    lines.append("# Reporte de calidad — Modelado FINANOM\n")
    lines.append("Generado por `data_modeling/data_modeling.py`.\n")
    lines.append("## Resumen\n")
    lines.append(f"- Filas evaluadas: {len(final_features):,}")
    lines.append(f"- Features candidatas: {candidate_features.shape[1]}")
    lines.append(f"- Features finales: {final_features.shape[1]}")
    lines.append(f"- Nulos en features finales: {int(final_features.isna().sum().sum())}")
    lines.append(f"- Muestra Isolation Forest proxy: {result.proxy_sample_size:,}")
    lines.append(f"- Contaminacion proxy: {PROXY_CONTAMINATION:.1%}\n")

    lines.append("## Iteracion 1 — diagnosticos estadisticos\n")
    lines.append("### Redundancias eliminadas\n")
    if result.dropped_features:
        lines.append("| Feature eliminada | Motivo |")
        lines.append("| ----------------- | ------ |")
        for c, reason in result.dropped_features.items():
            lines.append(f"| `{c}` | {reason} |")
    else:
        lines.append("No se eliminaron features por redundancia exacta.")
    lines.append("")

    lines.append("### Correlacion alta\n")
    lines.append(
        f"- Pares candidatos con |corr| >= 0.985 antes del ajuste: {len(candidate_pairs)}"
    )
    lines.append(
        f"- Pares con |corr| >= 0.985 despues del ajuste: {len(result.high_correlation_pairs)}"
    )
    if result.high_correlation_pairs:
        lines.append("\n| Feature A | Feature B | |corr| |")
        lines.append("| --------- | --------- | ------ |")
        for p in result.high_correlation_pairs[:15]:
            lines.append(f"| `{p['feature_a']}` | `{p['feature_b']}` | {p['abs_corr']} |")
    lines.append("")

    lines.append("### Varianza casi cero\n")
    lines.append(
        "Las binarias raras no se eliminan automaticamente: algunas son reglas de negocio "
        "con pocos casos pero alta utilidad para auditoria."
    )
    if candidate_near_zero:
        lines.append("\n| Feature | Motivo | Filas minoria | % minoria |")
        lines.append("| ------- | ------ | ------------- | --------- |")
        for item in candidate_near_zero:
            lines.append(
                f"| `{item['feature']}` | {item['reason']} | "
                f"{item.get('minority_rows', '')} | {item.get('minority_pct', '')} |"
            )
    else:
        lines.append("\nNo hay features constantes ni binarias extremadamente raras.")
    lines.append("")

    lines.append("## Iteracion 2 — Isolation Forest proxy\n")
    lines.append(
        "Se uso Isolation Forest como modelo proxy, no como modelo final. La importancia "
        "se calcula por frecuencia de uso en splits de los arboles: sirve para revisar "
        "senal y ruido, no como explicacion causal."
    )
    lines.append("")
    lines.append("| Percentil score | Valor |")
    lines.append("| --------------- | ----- |")
    for k, v in result.anomaly_score_summary.items():
        lines.append(f"| {k} | {v:.6f} |")
    lines.append("")
    lines.append("### Top 20 features por senal proxy\n")
    lines.append("| Rank | Feature | Importancia |")
    lines.append("| ---- | ------- | ----------- |")
    for i, row in result.feature_importance.head(20).iterrows():
        lines.append(f"| {i + 1} | `{row['feature']}` | {row['split_importance']:.4f} |")
    lines.append("")
    lines.append("## Conclusion de modelado\n")
    lines.append(
        "- El dataset final queda numerico y sin nulos en `feat_*`; los identificadores "
        "quedan aislados como `trace_*` para no contaminar el modelo."
    )
    lines.append(
        "- Se conserva encoding por frecuencia en lugar de one-hot porque mantiene el dataset "
        "compacto y deja visibles categorias raras."
    )
    lines.append(
        "- Se conserva escalado robusto por concepto/naturaleza para montos porque comparar "
        "RENHAB, PROPTI, depositos y devoluciones en escala cruda haria que el IF detecte "
        "solo tamanos grandes."
    )
    lines.append(
        "- VIF se descarta como criterio principal: las features mezclan ciclos, flags y "
        "transformaciones no lineales, y el modelo objetivo es basado en arboles. La "
        "correlacion alta cubre el riesgo practico de duplicar senal."
    )
    lines.append("")
    lines.append("## Artefactos\n")
    lines.append(f"- Dataset final: `{display_path(paths.output_file)}`")
    lines.append(f"- Matriz numerica pura: `{display_path(paths.feature_matrix_file)}`")
    lines.append(f"- Importancias proxy: `{display_path(paths.feature_importance_file)}`")
    lines.append(f"- Muestra top anomalias proxy: `{display_path(paths.anomaly_sample_file)}`")
    lines.append(f"- Grafica importancias: `{display_path(paths.importance_plot_file)}`")
    lines.append(f"- Grafica scores: `{display_path(paths.score_plot_file)}`")

    paths.quality_report_file.write_text("\n".join(lines), encoding="utf-8")


def write_dictionary(
    output_df: pd.DataFrame,
    feature_cols: list[str],
    dropped_features: dict[str, str],
    paths: ModelingPaths,
    dictionary_file: Path | str = DICT_FILE,
) -> None:
    """Genera diccionario de datos actualizado para el dataset modelado."""
    dictionary_file = Path(dictionary_file)
    lines: list[str] = []
    lines.append("# Diccionario — Base MODELADA FINANOM\n")
    lines.append(
        "Generado por `data_modeling/data_modeling.py` a partir de "
        "`data_cleaning/output/transacciones_limpio.parquet`.\n"
    )
    lines.append("## 1. Resumen\n")
    lines.append(f"- **Archivo principal**: `{display_path(paths.output_file)}`")
    lines.append(f"- **Matriz numerica pura**: `{display_path(paths.feature_matrix_file)}`")
    lines.append(f"- **Filas**: {len(output_df):,} (una por transaccion; no se eliminaron filas)")
    lines.append(f"- **Columnas totales**: {output_df.shape[1]}")
    lines.append(f"- **Columnas de trazabilidad**: {output_df.shape[1] - len(feature_cols)}")
    lines.append(f"- **Features finales (`feat_*`)**: {len(feature_cols)}")
    lines.append("- **Nulos en features**: 0")
    lines.append("- **Uso recomendado**: entrenar modelos solo con las columnas `feat_*` o con `X_modelo.parquet`.\n")

    lines.append("## 2. Decisiones principales\n")
    lines.append(
        "- No se codifican IDs crudos (`folio`, `referencia`, `transaccion`, `cuarto`) como features; "
        "solo se usan para trazabilidad o agregados derivados."
    )
    lines.append(
        "- Las categoricas se codifican por frecuencia relativa. Esto evita una matriz one-hot grande "
        "y permite que categorias raras sigan siendo visibles para Isolation Forest."
    )
    lines.append(
        "- Los montos se transforman con `log1p(abs(x))` y robust z por (`t_codigo`, `t_carabo`) "
        "para comparar conceptos financieros en su propio contexto."
    )
    lines.append(
        "- Los nulos de reservacion se imputan a valores neutros despues de generar flags de contexto; "
        "`feat_tiene_reservacion` explica cuando falta ese bloque."
    )
    lines.append(
        "- `t_observaciones` no se exporta crudo por posible PII; se transforma en longitud y keywords operativas.\n"
    )

    lines.append("## 3. Columnas de trazabilidad (NO features)\n")
    lines.append("| Columna | Tipo | Descripcion | Transformacion |")
    lines.append("| ------- | ---- | ----------- | -------------- |")
    for c in [c for c in output_df.columns if c.startswith("trace_")]:
        role, desc, transform = TRACE_DESCRIPTIONS.get(
            c, ("Trazabilidad", "Columna de trazabilidad.", "No entra al modelo.")
        )
        lines.append(f"| `{c}` | {output_df[c].dtype} | {desc} | {transform} |")
    lines.append("")

    lines.append("## 4. Features finales\n")
    lines.append("| Columna | Tipo | Grupo | Descripcion | Transformacion aplicada |")
    lines.append("| ------- | ---- | ----- | ----------- | ----------------------- |")
    for c in feature_cols:
        group, desc, transform = FEATURE_DESCRIPTIONS.get(
            c,
            (
                "Encoding",
                "Feature codificada por frecuencia relativa de la categoria original.",
                "Frecuencia relativa incluyendo nulos como categoria `__MISSING__`.",
            ),
        )
        lines.append(f"| `{c}` | {output_df[c].dtype} | {group} | {desc} | {transform} |")
    lines.append("")

    lines.append("## 5. Columnas descartadas en modelado\n")
    lines.append("| Columna/familia | Motivo |")
    lines.append("| --------------- | ------ |")
    lines.append(
        "| `t_observaciones` crudo | Puede contener nombres u otra PII; se reemplaza por longitud y keywords. |"
    )
    lines.append(
        "| IDs crudos como feature | Identifican cuentas/movimientos, pero no generalizan; se conservan solo en `trace_*`. |"
    )
    for feature, reason in dropped_features.items():
        lines.append(f"| `{feature}` | {reason} |")
    lines.append("")

    lines.append("## 6. Evaluacion recursiva\n")
    lines.append(
        f"Ver `{display_path(paths.quality_report_file)}` para correlaciones, "
        "features raras, Isolation Forest proxy e importancias."
    )

    dictionary_file.parent.mkdir(parents=True, exist_ok=True)
    dictionary_file.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def validate_input(df: pd.DataFrame) -> None:
    """Validaciones defensivas antes de modelar."""
    required = sorted(set(TRACE_SOURCE_COLS + CATEGORICAL_COLS + RESERVATION_NUMERIC_COLS + MONEY_BY_CODE_COLS + [
        "t_carabo",
        "t_tra_cancelada",
        "t_usuario",
        "es_split",
        "es_renta",
        "tiene_reservacion",
        "t_noches",
        "h_fec_lld",
        "h_fec_sda",
        "t_observaciones",
    ]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"La base limpia no tiene columnas requeridas para modelado: {missing}")


def build_modeled_dataset(df: pd.DataFrame, paths: ModelingPaths) -> tuple[pd.DataFrame, list[str], DiagnosticResult]:
    """Construye dataset final y devuelve columnas feature."""
    validate_input(df)
    trace = add_prefixed_trace(df)
    candidate_features = build_features(df)
    diagnostic = run_diagnostics(candidate_features, trace, paths)
    final_features, _ = select_final_features(candidate_features)

    feature_cols = final_features.columns.to_list()
    output = pd.concat([trace.reset_index(drop=True), final_features.reset_index(drop=True)], axis=1)
    return output, feature_cols, diagnostic


def run_modeling(
    input_file: Path | str = INPUT_FILE,
    output_dir: Path | str = OUTPUT_DIR,
    training_data_dir: Path | str = TRAINING_DATA_DIR,
    dictionary_file: Path | str = DICT_FILE,
) -> dict[str, Path]:
    """Ejecuta la fase de modelado y devuelve los artefactos generados."""
    warnings.filterwarnings("ignore", category=UserWarning)
    input_file = Path(input_file)
    dictionary_file = Path(dictionary_file)
    paths = ModelingPaths.from_dirs(output_dir, training_data_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.training_data_dir.mkdir(parents=True, exist_ok=True)
    for legacy_file in [
        paths.output_dir / "transacciones_modelado.parquet",
        paths.output_dir / "X_modelo.parquet",
    ]:
        if legacy_file.exists() and legacy_file not in {paths.output_file, paths.feature_matrix_file}:
            legacy_file.unlink()

    print(f"Leyendo base limpia: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  entrada: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    output, feature_cols, diagnostic = build_modeled_dataset(df, paths)
    print(f"  salida:  {output.shape[0]:,} filas, {output.shape[1]} columnas")
    print(f"  features finales: {len(feature_cols)}")
    print(f"  features eliminadas en diagnostico: {len(diagnostic.dropped_features)}")

    output.to_parquet(paths.output_file, index=False)
    output[feature_cols].to_parquet(paths.feature_matrix_file, index=False)
    write_dictionary(output, feature_cols, diagnostic.dropped_features, paths, dictionary_file)

    print(f"\nGuardado dataset: {paths.output_file} ({paths.output_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Guardada matriz X: {paths.feature_matrix_file} ({paths.feature_matrix_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Guardado diccionario: {dictionary_file}")
    print(f"Guardado reporte: {paths.quality_report_file}")

    return {
        "dataset": paths.output_file,
        "feature_matrix": paths.feature_matrix_file,
        "dictionary": dictionary_file,
        "quality_report": paths.quality_report_file,
        "quality_json": paths.quality_json_file,
        "feature_importance": paths.feature_importance_file,
        "anomaly_sample": paths.anomaly_sample_file,
        "importance_plot": paths.importance_plot_file,
        "score_plot": paths.score_plot_file,
    }


def main() -> None:
    run_modeling()


if __name__ == "__main__":
    main()
