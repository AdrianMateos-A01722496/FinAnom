"""Cierra el lazo de aprendizaje: revisiones del auditor -> adapta -> regenera.

Recibe las revisiones del dashboard (columnas: trace_row_id, estado, [nota]), traduce las
decisiones, mide la precision de las alertas, ajusta el umbral del IF y los factores de
confianza por regla, guarda el estado aprendido y REGENERA anomalies.json.

Mapeo de estado del dashboard -> decision del modelo:
    Autorizado / Escalado -> anomalia_confirmada   (era una anomalia real)
    Desestimado           -> falso_positivo         (el modelo se equivoco)
    Pendiente             -> se ignora (sin revisar)

Uso CLI opcional:
    uv run python model_final/adapt.py <revisiones.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_final import feedback as fb  # noqa: E402
from model_final import build_dashboard  # noqa: E402
from model_final import model as M  # noqa: E402

ESTADO_DECISION = {
    "Autorizado": "anomalia_confirmada",
    "Escalado": "anomalia_confirmada",
    "Desestimado": "falso_positivo",
}


def adapt_reviews(reviews: pd.DataFrame, revisor: str = "dashboard") -> dict:
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
    rev["decision"] = rev["estado"].map(ESTADO_DECISION)
    label_cols = ["trace_row_id", "decision"]
    if "nota" in rev.columns:
        label_cols.append("nota")
    labels = rev.dropna(subset=["decision"])[label_cols].copy()
    labels["trace_row_id"] = labels["trace_row_id"].astype(int)
    if labels.empty:
        return {
            "applied": False,
            "message": "No hay revisiones usables; marca Autorizado, Desestimado o Escalado.",
            "reviews_read": int(len(rev)),
            "labels_used": 0,
        }

    report = M.load_base_report()
    state = M.load_state()
    threshold_actual = (state or {}).get("threshold_score_samples", M.base_threshold())

    # Cola actual (lo que vio el auditor) para medir precision.
    queue = M.compute_queue(report, state)
    n_antes = int(queue["is_anomaly"].sum())
    metrics = fb.feedback_metrics(queue, labels)

    # Aprender: umbral + factores de confianza por regla.
    sug = fb.suggest_threshold(threshold_actual, metrics)
    weights = fb.suggest_rule_weights(queue, labels)

    # Persistir y registrar las etiquetas en el almacen historico.
    M.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fb.save_feedback_state(M.STATE_FILE, sug["umbral_nuevo"], weights)
    store = M.STATE_FILE.parent / "feedback_labels.csv"
    fb.init_label_store(store)
    if "nota" not in labels.columns:
        labels["nota"] = ""
    labels["nota"] = labels["nota"].fillna("")
    labels.assign(
        revisor=revisor,
        timestamp_revision=pd.Timestamp.now("UTC").isoformat(),
    )[fb.FEEDBACK_COLUMNS].to_csv(store, mode="a", header=False, index=False)

    # Regenerar la cola con lo aprendido.
    build_dashboard.build()
    queue_nueva = M.compute_queue(report, M.load_state())
    n_despues = int(queue_nueva["is_anomaly"].sum())
    return {
        "applied": True,
        "message": "Correcciones aplicadas y dashboard regenerado.",
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


def adapt(revisiones_csv: Path | str) -> None:
    result = adapt_reviews(pd.read_csv(revisiones_csv))
    print(f"Revisiones leidas: {result['reviews_read']} | usables: {result['labels_used']}")
    if not result["applied"]:
        print(result["message"])
        return
    print(
        f"Precision global observada: {result['metrics'].get('precision_global')} "
        f"sobre {result['metrics'].get('alertas_revisadas')} alertas revisadas"
    )
    print(
        f"Umbral: {result['threshold_before']:.4f} -> {result['threshold_after']:.4f} "
        f"({result['threshold_reason']})"
    )
    print(f"Factores de confianza por regla aprendidos: {result['rule_weights'] or '(ninguno)'}")
    print(
        f"\nCola: {result['queue_before']:,} -> {result['queue_after']:,} alertas tras "
        f"aprender ({result['queue_delta']:+,}). Recarga el dashboard para verlo."
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: uv run python model_final/adapt.py <revisiones.csv>")
        sys.exit(1)
    adapt(sys.argv[1])


if __name__ == "__main__":
    main()
