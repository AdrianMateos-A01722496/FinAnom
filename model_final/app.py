"""Servidor Flask para FINANOM final.

Uso:
    uv run python model_final/app.py
    abrir http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_final import adapt  # noqa: E402
from model_final import build_dashboard  # noqa: E402
from model_final import model as M  # noqa: E402

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


@app.post("/api/apply-corrections")
def apply_corrections():
    payload = request.get_json(silent=True) or {}
    reviews = payload.get("reviews", [])
    if not isinstance(reviews, list):
        return jsonify({"applied": False, "message": "reviews debe ser una lista."}), 400
    try:
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
    state_data = M.load_state() or {}
    return jsonify({
        "feedback_state": state_data,
        "dashboard_records": json.loads(M.DASHBOARD_DATA.read_text(encoding="utf-8"))
        if M.DASHBOARD_DATA.exists() else [],
    })


def main() -> None:
    _ensure_dashboard_data()
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
