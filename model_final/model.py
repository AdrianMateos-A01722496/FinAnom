"""Modelo FINAL FINANOM — consolidado + aprendizaje, y datos para el dashboard.

Reutiliza el reporte del modelo consolidado local (`model_final/output/reporte_revision.parquet`)
y APLICA el estado aprendido por el feedback del auditor (umbral del IF + factor de confianza
por regla) para regenerar la cola, SIN reentrenar el IF (adaptacion instantanea).

Convencion de severidad (espejo de `model_final/train_model.severidad`, reimplementada aqui
para no arrastrar imports pesados de SHAP/IF en un script que solo re-umbraliza):
    CRITICO  si rule_score>=90, o (IF y rule_score>=50), o metodo_pago
    ALTO     si rule_score>=60, o marcada por el IF
    MEDIO    si rule_score>0
    BAJO     en otro caso
La cola de revision = severidad ALTO o CRITICO.

`to_dashboard_records` mapea la cola al esquema que consume el dashboard HTML de Rogelio.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BASE_REPORT = HERE / "output" / "reporte_revision.parquet"
BASE_EVAL = HERE / "output" / "evaluacion_modelo.json"
CLEAN_FILE = HERE / "data" / "transacciones_limpio.parquet"
STATE_FILE = HERE / "output" / "feedback_state.json"
DASHBOARD_DATA = HERE / "dashboard" / "data" / "anomalies.json"
DASHBOARD_TOP_N = 300
DUPLICADO_SCORE = 40

# Cluster de regla -> categoria legible (nombres que el dashboard reconoce para iconos).
CLUSTER_CAT = {
    "MONTO_ATIPICO": "Monto atipico",
    "CANCELACION_SOSPECHOSA": "Cancelacion irregular",
    "FUERA_DE_ESTANCIA": "Fuera de estancia",
    "DUPLICADO": "Posible duplicado",
    "CONTEXTO_RESERVACION": "Inconsistencia de reservacion",
    "SIGNO_CONTABLE": "Inconsistencia de signo",
    "METODO_PAGO": "Metodo de pago",
    "PAGO_PROVEEDOR_SOSPECHOSO": "Egreso sospechoso",
    "ATIPICO_IF": "Patron atipico (modelo)",
}
CLUSTER_TEXTO = {
    "MONTO_ATIPICO": "El monto se desvia del rango tipico para este concepto",
    "CANCELACION_SOSPECHOSA": "La cancelacion o reposteo tiene caracteristicas atipicas",
    "FUERA_DE_ESTANCIA": "El cargo cae fuera de la ventana de estancia de la reserva",
    "DUPLICADO": "Se repite un cargo identico en el mismo minuto",
    "CONTEXTO_RESERVACION": "El cargo no cuadra con el contexto de la reserva enlazada",
    "SIGNO_CONTABLE": "El signo del monto es inesperado para el tipo de movimiento",
    "METODO_PAGO": "El metodo de pago del cargo no coincide con el de la reserva (Amex vs tarjeta generica)",
    "PAGO_PROVEEDOR_SOSPECHOSO": "Egreso a proveedor con un patron inusual",
    "ATIPICO_IF": "Patron globalmente atipico detectado por el modelo no supervisado",
}
SEV_DASHBOARD = {"CRITICO": "Alta", "ALTO": "Media", "MEDIO": "Informativa", "BAJO": "Informativa"}


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def load_base_report() -> pd.DataFrame:
    if not BASE_REPORT.exists():
        raise FileNotFoundError(
            f"No existe {BASE_REPORT}.\nGenera primero el modelo consolidado:\n"
            "  uv run python model_final/train_model.py")
    return pd.read_parquet(BASE_REPORT)


def base_threshold() -> float:
    """Umbral base del IF (score_samples) que dejo el modelo consolidado."""
    return float(json.loads(BASE_EVAL.read_text(encoding="utf-8"))["threshold_score_samples"])


def load_state(path: Path | str = STATE_FILE) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Aplicacion del estado aprendido -> cola
# --------------------------------------------------------------------------- #
def _clusters(tipo: str) -> list[str]:
    return [c for c in str(tipo).split(" | ") if c and c != "ATIPICO_IF"]


def _replace_clusters(tipo: str, clusters: list[str]) -> str:
    keep = set(clusters)
    return " | ".join(c for c in str(tipo).split(" | ") if c and c in keep)


def _duplicate_evidence(clean_df: pd.DataFrame | None = None) -> pd.Series | None:
    """Marca duplicados de alta confianza: mismo minuto con subfolio y naturaleza contable.

    El duplicado por dia tiene mucho recall, pero mezcla errores reales con cargos legitimos
    repetidos. Esta evidencia endurece la regla sin reentrenar el reporte ya generado.
    """
    cols = [
        "t_folio", "t_folio_ext", "t_cuarto", "t_codigo", "t_carabo", "t_monto", "t_timestamp",
    ]
    if clean_df is None and not CLEAN_FILE.exists():
        return None
    clean = clean_df[cols].copy() if clean_df is not None else pd.read_parquet(CLEAN_FILE, columns=cols)
    monto = pd.to_numeric(clean["t_monto"], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(clean["t_timestamp"], errors="coerce")

    work = pd.DataFrame({
        "folio": clean["t_folio"].astype("string").fillna("__NA__"),
        "folio_ext": clean["t_folio_ext"].astype("string").fillna("__NA__"),
        "codigo": clean["t_codigo"].astype("string").str.strip().fillna("__NA__"),
        "carabo": clean["t_carabo"].astype("string").str.strip().fillna("__NA__"),
        "cuarto": clean["t_cuarto"].astype("string").fillna("__NA__"),
        "monto_c": np.rint(monto * 100).astype("int64"),
        "minuto": ts.dt.floor("min"),
    }, index=clean.index)

    key = ["folio", "folio_ext", "cuarto", "codigo", "carabo", "monto_c", "minuto"]
    dup_min = work.groupby(key, dropna=False)["codigo"].transform("size") > 1
    return dup_min.rename("dup_alta_confianza")


def _degrade_low_confidence_duplicates(
    out: pd.DataFrame,
    clean_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Quita puntaje/categoria DUPLICADO cuando solo hay coincidencia por dia."""
    has_dup = out["tipo_inconsistencia"].str.contains("DUPLICADO", na=False)
    if not has_dup.any():
        return out["rule_score"].to_numpy(), pd.Series(False, index=out.index)

    evidence = _duplicate_evidence(clean_df)
    if evidence is None:
        return out["rule_score"].to_numpy(), pd.Series(False, index=out.index)

    row_ids = out["trace_row_id"].astype(int)
    hard_dup = pd.Series(evidence.reindex(row_ids).fillna(False).to_numpy(), index=out.index)
    weak_dup = has_dup & ~hard_dup
    if not weak_dup.any():
        return out["rule_score"].to_numpy(), hard_dup

    out.loc[weak_dup, "tipo_inconsistencia"] = out.loc[weak_dup, "tipo_inconsistencia"].map(
        lambda t: _replace_clusters(t, [c for c in _clusters(t) if c != "DUPLICADO"])
    )
    out.loc[weak_dup, "motivos"] = out.loc[weak_dup, "motivos"].str.replace(
        r"(?:^|; )duplicado_exacto:[^;]*(?:; )?",
        "",
        regex=True,
    ).str.strip(" ;")
    adjusted = out["rule_score"].to_numpy().astype(float)
    adjusted[weak_dup.to_numpy()] = np.maximum(0.0, adjusted[weak_dup.to_numpy()] - DUPLICADO_SCORE)
    return adjusted, hard_dup


def _severidad(rule_score: np.ndarray, is_if: np.ndarray, metodo_pago: np.ndarray) -> np.ndarray:
    sev = np.full(len(rule_score), "BAJO", dtype=object)
    sev[rule_score > 0] = "MEDIO"
    sev[(rule_score >= 60) | is_if] = "ALTO"
    sev[(rule_score >= 90) | (is_if & (rule_score >= 50)) | metodo_pago] = "CRITICO"
    return sev


def compute_queue(
    report: pd.DataFrame,
    state: dict | None,
    clean_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reaplica umbral del IF + factores de confianza por regla y recomputa la cola.

    Sin estado aprendido, reproduce la cola base. Con estado, endurece/afloja segun lo
    aprendido del auditor. No reentrena nada: solo re-umbraliza scores ya calculados.
    """
    out = report.copy()
    weights = (state or {}).get("rule_weights", {})
    threshold = (state or {}).get("threshold_score_samples")
    base_rule_score, hard_dup = _degrade_low_confidence_duplicates(out, clean_df)

    # IF: si hay umbral aprendido, re-marcar; si no, usar la marca base.
    is_if = (out["score_samples"].to_numpy() <= threshold) if threshold is not None \
        else out["is_anomaly_if"].to_numpy().astype(bool)

    # Factor de confianza por fila = el MINIMO de los factores de sus reglas (la mas
    # desconfiada manda). Si no hay pesos aprendidos, factor 1.0 (sin cambio).
    if weights:
        factor = out["tipo_inconsistencia"].map(
            lambda t: min([weights.get(c, 1.0) for c in _clusters(t)], default=1.0)).to_numpy()
    else:
        factor = np.ones(len(out))
    eff_rule = base_rule_score * factor

    metodo_pago = out["tipo_inconsistencia"].str.contains("METODO_PAGO", na=False).to_numpy()
    sev = _severidad(eff_rule, is_if, metodo_pago)

    out["is_anomaly_if"] = is_if
    out["dup_alta_confianza"] = hard_dup.to_numpy()
    out["rule_score_eff"] = np.round(eff_rule, 1)
    out["severidad"] = sev
    out["is_anomaly"] = np.isin(sev, ["ALTO", "CRITICO"])
    # Ranking: severidad (CRITICO primero) y luego score del IF.
    sev_rank = pd.Series(sev).map({"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAJO": 3}).to_numpy()
    order = np.lexsort((-out["anomaly_score"].to_numpy(), sev_rank))
    out["anomaly_rank"] = np.empty(len(out), dtype=int)
    out.iloc[order, out.columns.get_loc("anomaly_rank")] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- #
# Mapeo al esquema del dashboard
# --------------------------------------------------------------------------- #
def _build_reasons(row: pd.Series) -> tuple[str, list[dict]]:
    """Construye categorias + reasons_json (con peso relevancia) para una fila."""
    clusters = _clusters(row["tipo_inconsistencia"])
    cats, reasons = [], []
    base = max(0.2, float(row.get("anomaly_pct", 0.5)))
    for i, c in enumerate(clusters):
        cat = CLUSTER_CAT.get(c, c.title())
        cats.append(cat)
        reasons.append({"categoria": cat, "texto": CLUSTER_TEXTO.get(c, cat),
                        "peso_shap": round(base * (1 - 0.15 * i), 4)})
    # Evidencia SHAP del IF como una razon adicional (cuando exista).
    ev = str(row.get("evidencia_shap") or "").strip()
    if ev:
        cat = CLUSTER_CAT["ATIPICO_IF"]
        if cat not in cats:
            cats.append(cat)
        reasons.append({"categoria": cat,
                        "texto": f"Las features que mas la aislaron: {ev}",
                        "peso_shap": round(base * 0.5, 4)})
    if not reasons:  # solo IF, sin reglas ni evidencia
        cats = [CLUSTER_CAT["ATIPICO_IF"]]
        reasons = [{"categoria": cats[0], "texto": CLUSTER_TEXTO["ATIPICO_IF"],
                    "peso_shap": round(base, 4)}]
    return " | ".join(dict.fromkeys(cats)), reasons


def to_dashboard_records(queue: pd.DataFrame, clean: pd.DataFrame,
                         top_n: int = DASHBOARD_TOP_N) -> list[dict]:
    """Top-N de la cola -> lista de dicts con el esquema de anomalies.json."""
    cola = queue[queue["is_anomaly"]].sort_values("anomaly_rank").head(top_n).copy()
    rows = clean.iloc[cola["trace_row_id"].to_numpy()]
    montos = pd.to_numeric(rows["t_monto"], errors="coerce").to_numpy()
    es_abono = (rows["t_carabo"].astype("string").str.strip() == "1").to_numpy()

    records = []
    for (_, r), monto, abono in zip(cola.iterrows(), montos, es_abono):
        categories, reasons = _build_reasons(r)
        records.append({
            "trace_row_id": int(r["trace_row_id"]),
            "trace_t_folio": str(r["trace_t_folio"]),
            "trace_t_folio_ext": str(r["trace_t_folio_ext"]),
            "trace_t_transaccion": str(r["trace_t_transaccion"]),
            "trace_t_cuarto": None if pd.isna(r["trace_t_cuarto"]) else str(r["trace_t_cuarto"]),
            "trace_t_codigo": str(r["trace_t_codigo"]),
            "trace_t_timestamp": str(r["trace_t_timestamp"]),
            "trace_t_cve_res": None if pd.isna(r["trace_t_cve_res"]) else str(r["trace_t_cve_res"]),
            "anomaly_rank": int(r["anomaly_rank"]),
            "severity": SEV_DASHBOARD.get(r["severidad"], "Informativa"),
            "n_reasons": len(reasons),
            "categories": categories,
            "explanation": str(r["motivos"]),
            "reasons_json": json.dumps(reasons, ensure_ascii=False),
            "monto": None if np.isnan(monto) else float(monto),
            "es_abono": int(bool(abono)),
            "estado": "Pendiente",
        })
    return records


def write_dashboard_data(records: list[dict], path: Path | str = DASHBOARD_DATA) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
