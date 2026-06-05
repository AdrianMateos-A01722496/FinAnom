"""Servidor Flask para FINANOM final.

Uso:
    uv run python model_final/app.py
    abrir http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_final import adapt  # noqa: E402
from model_final import build_dashboard  # noqa: E402
from model_final import db  # noqa: E402
from model_final import feedback as fb  # noqa: E402
from model_final import model as M  # noqa: E402
from model_final import scoring  # noqa: E402

HERE = Path(__file__).resolve().parent
DASHBOARD_DIR = HERE / "dashboard"

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")


def _ensure_dashboard_data() -> None:
    if not M.DASHBOARD_DATA.exists():
        build_dashboard.build()


@app.get("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/api/anomalies")
def anomalies():
    _ensure_dashboard_data()
    return app.response_class(
        M.DASHBOARD_DATA.read_text(encoding="utf-8"),
        mimetype="application/json",
    )


@app.get("/api/transactions")
def transactions():
    engine = db.ensure_store()
    result = db.query_transactions(
        engine,
        filter_mode=request.args.get("filter", "anomalias"),
        severidad=request.args.get("severidad", ""),
        q=request.args.get("q", ""),
        scope=request.args.get("scope", "all"),
        date_from=request.args.get("date_from", ""),
        date_to=request.args.get("date_to", ""),
        folio=request.args.get("folio", ""),
        cuarto=request.args.get("cuarto", ""),
        concepto=request.args.get("concepto", ""),
        categoria=request.args.get("categoria", ""),
        monto_min=request.args.get("monto_min", ""),
        monto_max=request.args.get("monto_max", ""),
        sort_by=request.args.get("sort_by", "priority"),
        sort_dir=request.args.get("sort_dir", "asc"),
        page=int(request.args.get("page", "1")),
        page_size=int(request.args.get("page_size", "25")),
    )
    return jsonify(result)


@app.post("/api/transactions")
def create_transaction():
    payload = request.get_json(silent=True) or {}
    scenario = str(payload.get("scenario", "anomala")).lower()
    if scenario not in {"normal", "anomala"}:
        return jsonify({"ok": False, "message": "scenario debe ser normal o anomala."}), 400

    try:
        engine = db.ensure_store()
        context = db.load_recent_window(engine, limit=500)
        new_row = scoring.build_synthetic_transaction(context, scenario=scenario)
        expanded = pd.concat([context, pd.DataFrame([new_row])], ignore_index=True)
        state = db.load_feedback_state(engine)
        scored = scoring.score_window(expanded, state)
        tx = scored.loc[scored["tx_id"] == new_row["tx_id"]].iloc[0]
        db.append_transaction(engine, tx)
    except Exception as exc:  # pragma: no cover - surfaced to dashboard
        return jsonify({"ok": False, "message": str(exc)}), 500
    return jsonify({"ok": True, "transaction": scoring.dashboard_record(tx)})


@app.delete("/api/transactions/<tx_id>")
def delete_transaction(tx_id: str):
    try:
        engine = db.ensure_store()
        deleted = db.delete_transaction(engine, tx_id)
        if not deleted:
            return jsonify({"ok": False, "message": f"Transacción '{tx_id}' no encontrada."}), 404
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500
    return jsonify({"ok": True, "deleted": tx_id})


@app.post("/api/apply-corrections")
def apply_corrections():
    payload = request.get_json(silent=True) or {}
    reviews = payload.get("reviews", [])
    if not isinstance(reviews, list):
        return jsonify({"applied": False, "message": "reviews debe ser una lista."}), 400
    try:
        if os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or db.DEFAULT_DB.exists():
            result = _adapt_reviews_sql(pd.DataFrame(reviews), revisor="dashboard")
        else:
            result = adapt.adapt_reviews(pd.DataFrame(reviews), revisor="dashboard")
    except Exception as exc:  # pragma: no cover - se reporta al dashboard
        return jsonify({"applied": False, "message": str(exc)}), 500
    return jsonify(result)


@app.post("/api/reset-dashboard")
def reset_dashboard():
    build_dashboard.build()
    return jsonify({"ok": True, "message": "Dashboard regenerado con el estado actual."})


@app.get("/api/state")
def state():
    engine = db.ensure_store()
    state_data = db.load_feedback_state(engine) or {}
    return jsonify({
        "feedback_state": state_data,
        "dashboard_records": json.loads(M.DASHBOARD_DATA.read_text(encoding="utf-8"))
        if M.DASHBOARD_DATA.exists() else [],
    })


def _adapt_reviews_sql(reviews: pd.DataFrame, revisor: str = "dashboard") -> dict:
    rev = reviews.copy()
    if rev.empty:
        return {
            "applied": False,
            "message": "No hay revisiones usables; marca Autorizado, Desestimado o Escalado.",
            "reviews_read": 0,
            "labels_used": 0,
        }
    if "trace_row_id" not in rev.columns or "estado" not in rev.columns:
        raise ValueError("Las revisiones deben incluir trace_row_id y estado.")

    rev["decision"] = rev["estado"].map(adapt.ESTADO_DECISION)
    labels = rev.dropna(subset=["decision"])[["trace_row_id", "decision"]].copy()
    if "nota" in rev.columns:
        labels["nota"] = rev.loc[labels.index, "nota"].fillna("")
    else:
        labels["nota"] = ""
    labels["trace_row_id"] = labels["trace_row_id"].astype(str)
    if labels.empty:
        return {
            "applied": False,
            "message": "No hay revisiones usables; marca Autorizado, Desestimado o Escalado.",
            "reviews_read": int(len(rev)),
            "labels_used": 0,
        }

    engine = db.ensure_store()
    state = db.load_feedback_state(engine)
    current = db.load_window(engine)
    current["trace_row_id"] = current["tx_id"].astype(str)
    threshold_actual = (state or {}).get(
        "threshold_score_samples",
        scoring._load_bundle().get("threshold_score_samples"),  # noqa: SLF001
    )

    n_antes = int(current["is_anomaly"].sum())
    metrics = fb.feedback_metrics(current, labels)
    sug = fb.suggest_threshold(float(threshold_actual), metrics)
    weights = fb.suggest_rule_weights(current, labels)
    new_state = {
        "threshold_score_samples": float(sug["umbral_nuevo"]),
        "rule_weights": weights,
        "actualizado": pd.Timestamp.now("UTC").isoformat(),
    }
    db.save_feedback_state(engine, new_state)
    db.append_feedback_labels(engine, labels, revisor)
    db.update_review_states(engine, rev)

    rescored = scoring.score_window(db.load_window(engine), new_state)
    db.save_scored_window(engine, rescored)
    n_despues = int(rescored["is_anomaly"].sum())
    return {
        "applied": True,
        "message": "Correcciones aplicadas y transacciones re-puntuadas.",
        "reviews_read": int(len(rev)),
        "labels_used": int(len(labels)),
        "metrics": metrics,
        "threshold_before": float(threshold_actual),
        "threshold_after": float(sug["umbral_nuevo"]),
        "threshold_reason": sug["motivo"],
        "rule_weights": weights,
        "queue_before": n_antes,
        "queue_after": n_despues,
        "queue_delta": n_despues - n_antes,
    }


def main() -> None:
    _ensure_dashboard_data()
    db.ensure_store()
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
