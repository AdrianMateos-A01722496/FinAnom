"""Runtime scoring for FINANOM live demo.

The online path intentionally reuses the existing batch feature engineering and the
saved Isolation Forest. New transactions are scored by rebuilding features over the
recent window, not by retraining the model.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_modeling.data_modeling import add_prefixed_trace, build_features, validate_input  # noqa: E402
from model_final import model as M  # noqa: E402
from model_final.train_model import build_report, run_rule_engine, to_numpy_df  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORER_BUNDLE = HERE / "output" / "scorer_bundle.joblib"
WINDOW_START = pd.Timestamp("2025-03-11")

CLEAN_COLUMNS = [
    "t_folio",
    "t_folio_ext",
    "t_referencia",
    "t_transaccion",
    "t_cve_res",
    "t_cuarto",
    "t_codigo",
    "t_carabo",
    "t_usuario",
    "t_usuario_mod",
    "h_tpo_hab",
    "h_tpo_hsp",
    "h_seg_mer",
    "h_cod_age",
    "h_tpo_plan",
    "h_for_pgo",
    "t_monto",
    "t_impuesto",
    "t_propina",
    "t_noches",
    "h_num_per",
    "h_num_noc",
    "h_tfa",
    "h_tfa_total",
    "h_tarifa_forzada",
    "h_dep_sol",
    "t_tra_cancelada",
    "es_split",
    "es_renta",
    "tiene_reservacion",
    "t_timestamp",
    "h_fec_lld",
    "h_fec_sda",
    "t_observaciones",
]

RUNTIME_COLUMNS = ["tx_id", "es_nueva", "estado", "nota", "created_at"]
SCORE_COLUMNS = [
    "anomaly_score",
    "anomaly_pct",
    "score_samples",
    "is_anomaly_if",
    "rule_score",
    "rule_score_eff",
    "severidad",
    "is_anomaly",
    "tipo_inconsistencia",
    "motivos",
    "evidencia_shap",
    "requiere_aprobacion",
    "anomaly_rank",
    "dup_alta_confianza",
]


def load_clean_window(path: Path | str = M.CLEAN_FILE) -> pd.DataFrame:
    """Load the recent clean-data window used for live micro-batch scoring."""
    clean = pd.read_parquet(path)
    ts = pd.to_datetime(clean["t_timestamp"], errors="coerce")
    window = clean.loc[ts >= WINDOW_START, CLEAN_COLUMNS].copy().reset_index(drop=True)
    window.insert(0, "tx_id", [f"batch-{i}" for i in window.index])
    window["es_nueva"] = False
    window["estado"] = "Pendiente"
    window["nota"] = ""
    window["created_at"] = pd.Timestamp.now("UTC").isoformat()
    return window


def _load_bundle(path: Path | str = SCORER_BUNDLE) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(f"Bundle invalido: {path}")
    return bundle


def _threshold(bundle: dict[str, Any], state: dict | None) -> float:
    value = (state or {}).get("threshold_score_samples", bundle.get("threshold_score_samples"))
    if value is None:
        raise ValueError("No hay threshold_score_samples en feedback_state ni scorer_bundle.")
    return float(value)


def score_window(df_window: pd.DataFrame, state: dict | None = None) -> pd.DataFrame:
    """Score every row in the recent window and return clean + scoring columns.

    The DataFrame may include runtime columns such as `tx_id` and `es_nueva`; only
    the clean columns are sent through feature engineering and the rule engine.
    """
    if df_window.empty:
        return df_window.copy()

    runtime = df_window.reindex(columns=RUNTIME_COLUMNS).copy()
    clean = df_window[CLEAN_COLUMNS].copy().reset_index(drop=True)
    for col in ["t_timestamp", "h_fec_lld", "h_fec_sda"]:
        clean[col] = pd.to_datetime(clean[col], errors="coerce", format="mixed")
    runtime = runtime.reset_index(drop=True)
    validate_input(clean)

    bundle = _load_bundle()
    features = build_features(clean)
    feature_cols = list(bundle["feature_cols"])
    X = features.reindex(columns=feature_cols)
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Features requeridas faltantes para scoring: {missing}")

    score_samples = bundle["model"].score_samples(X.values)
    anomaly_score = -score_samples
    threshold = _threshold(bundle, state)
    is_if = score_samples <= threshold

    trace = add_prefixed_trace(clean)
    rules = run_rule_engine(to_numpy_df(clean))
    shap_top = np.full(len(clean), "", dtype=object)
    report, _ = build_report(trace, anomaly_score, is_if, rules, shap_top, score_samples)
    queued = M.compute_queue(report, state, clean_df=clean)

    out = pd.concat([runtime, clean, queued[SCORE_COLUMNS]], axis=1)
    out["estado"] = out["estado"].fillna("Pendiente").replace("", "Pendiente")
    out["nota"] = out["nota"].fillna("")
    out["es_nueva"] = out["es_nueva"].fillna(False).astype(bool)
    return out


def _next_transaccion(df: pd.DataFrame) -> int:
    values = pd.to_numeric(df.get("t_transaccion"), errors="coerce")
    return int(values.max()) + 1 if values.notna().any() else 1


def build_synthetic_transaction(df_window: pd.DataFrame, scenario: str = "anomala") -> dict[str, Any]:
    """Create a clean-level synthetic transaction using the latest window as context."""
    if df_window.empty:
        raise ValueError("No hay ventana base para sintetizar una transaccion.")

    work = df_window.copy()
    work["_ts"] = pd.to_datetime(work["t_timestamp"], errors="coerce", format="mixed")
    latest_ts = work["_ts"].max()
    if pd.isna(latest_ts):
        latest_ts = pd.Timestamp.now().floor("min")
    base = work.sort_values("_ts").iloc[-1][CLEAN_COLUMNS].to_dict()
    now_time = pd.Timestamp.now().floor("min")
    demo_ts = latest_ts.normalize() + pd.Timedelta(hours=now_time.hour, minutes=now_time.minute)
    tx_id = f"new-{uuid.uuid4().hex[:12]}"
    base.update(
        {
            "t_folio": int(pd.to_numeric(df_window["t_folio"], errors="coerce").max()) + 1,
            "t_folio_ext": 0,
            "t_referencia": f"SIM-{tx_id[-6:]}",
            "t_transaccion": _next_transaccion(df_window),
            "t_cuarto": "SIM",
            "t_usuario": "DEMO",
            "t_usuario_mod": "",
            "t_timestamp": demo_ts,
            "t_tra_cancelada": "",
            "es_split": False,
            "created_at": pd.Timestamp.now("UTC").isoformat(),
            "tx_id": tx_id,
            "es_nueva": True,
            "estado": "Pendiente",
            "nota": "",
        }
    )

    if scenario == "normal":
        base.update(
            {
                "t_codigo": "RENHAB",
                "t_carabo": "0",
                "t_monto": 1800.0,
                "t_impuesto": 288.0,
                "t_propina": 0.0,
                "t_noches": 1,
                "t_observaciones": "SIMULACION NORMAL",
                "es_renta": True,
            }
        )
    else:
        base.update(
            {
                "t_codigo": "RENHAB",
                "t_carabo": "0",
                "t_monto": -125000.0,
                "t_impuesto": 0.0,
                "t_propina": 9000.0,
                "t_noches": 1,
                "tiene_reservacion": True,
                "h_fec_lld": demo_ts + pd.Timedelta(days=10),
                "h_fec_sda": demo_ts + pd.Timedelta(days=12),
                "h_num_per": 2,
                "h_num_noc": 2,
                "h_tfa": 1800.0,
                "h_tfa_total": 3600.0,
                "h_tarifa_forzada": 0.0,
                "h_dep_sol": 0.0,
                "h_tpo_hab": "STD",
                "h_tpo_hsp": "REG",
                "h_seg_mer": "DIR",
                "h_cod_age": "WEB",
                "h_tpo_plan": "EP",
                "h_for_pgo": "TARCRE",
                "t_observaciones": "SIMULACION ANOMALA MONTO EXTREMO",
                "es_renta": True,
            }
        )
    return base


def dashboard_record(row: pd.Series) -> dict[str, Any]:
    """Map a stored scored row to the dashboard JSON schema."""
    is_anomaly = bool(row.get("is_anomaly"))
    if not is_anomaly and not str(row.get("tipo_inconsistencia") or "").strip():
        categories, reasons = "Transaccion normal", []
    else:
        categories, reasons = M._build_reasons(  # noqa: SLF001 - shared dashboard mapping
            pd.Series(
                {
                    "tipo_inconsistencia": row.get("tipo_inconsistencia") or "",
                    "anomaly_pct": row.get("anomaly_pct", 0.0),
                    "evidencia_shap": row.get("evidencia_shap") or "",
                    "motivos": row.get("motivos") or "",
                }
            )
        )
    severity = M.SEV_DASHBOARD.get(str(row.get("severidad") or "BAJO"), "Informativa")
    estado = row.get("estado") or "Pendiente"
    created_at = row.get("created_at")
    is_new = bool(row.get("es_nueva"))
    if is_new and created_at:
        created_ts = pd.to_datetime(created_at, errors="coerce", utc=True)
        if pd.notna(created_ts):
            is_new = (pd.Timestamp.now("UTC") - created_ts) < pd.Timedelta(minutes=10)
    return {
        "trace_row_id": str(row.get("tx_id")),
        "tx_id": str(row.get("tx_id")),
        "trace_t_folio": str(row.get("t_folio")),
        "trace_t_folio_ext": str(row.get("t_folio_ext")),
        "trace_t_transaccion": str(row.get("t_transaccion")),
        "trace_t_cuarto": None if pd.isna(row.get("t_cuarto")) else str(row.get("t_cuarto")),
        "trace_t_codigo": str(row.get("t_codigo")),
        "trace_t_timestamp": str(row.get("t_timestamp")),
        "trace_t_cve_res": None if pd.isna(row.get("t_cve_res")) else str(row.get("t_cve_res")),
        "anomaly_rank": int(row.get("anomaly_rank") or 0),
        "severity": severity,
        "n_reasons": len(reasons),
        "categories": categories or "Sin anomalía",
        "explanation": str(
            row.get("motivos")
            or (
                "Transaccion marcada para revision."
                if is_anomaly
                else "Transaccion dentro del comportamiento esperado."
            )
        ),
        "reasons_json": json.dumps(reasons, ensure_ascii=False),
        "monto": None if pd.isna(row.get("t_monto")) else float(row.get("t_monto")),
        "es_abono": int(str(row.get("t_carabo")).strip() == "1"),
        "estado": str(estado),
        "nota": str(row.get("nota") or ""),
        "is_anomaly": is_anomaly,
        "es_nueva": is_new,
        "created_at": None if pd.isna(created_at) else str(created_at),
        "score_samples": None if pd.isna(row.get("score_samples")) else float(row.get("score_samples")),
        "rule_score": None if pd.isna(row.get("rule_score_eff")) else float(row.get("rule_score_eff")),
    }
