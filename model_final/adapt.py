"""Cierra el lazo de aprendizaje: revisiones del auditor -> adapta -> regenera.

Lee el CSV que exporta el dashboard (columnas: trace_row_id, estado, [nota]), traduce las
decisiones, mide la precision de las alertas, ajusta el umbral del IF y los factores de
confianza por regla, guarda el estado aprendido y REGENERA anomalies.json.

Mapeo de estado del dashboard -> decision del modelo:
    Autorizado / Escalado -> anomalia_confirmada   (era una anomalia real)
    Desestimado           -> falso_positivo         (el modelo se equivoco)
    Pendiente             -> se ignora (sin revisar)

Uso:
    uv run python model_final/adapt.py <revisiones.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_Adrian import feedback as fb  # noqa: E402
from model_final import build_dashboard  # noqa: E402
from model_final import model as M  # noqa: E402

ESTADO_DECISION = {
    "Autorizado": "anomalia_confirmada",
    "Escalado": "anomalia_confirmada",
    "Desestimado": "falso_positivo",
}


def adapt(revisiones_csv: Path | str) -> None:
    rev = pd.read_csv(revisiones_csv)
    rev["decision"] = rev["estado"].map(ESTADO_DECISION)
    labels = rev.dropna(subset=["decision"])[["trace_row_id", "decision"]].copy()
    labels["trace_row_id"] = labels["trace_row_id"].astype(int)
    print(f"Revisiones leidas: {len(rev)} | usables (no Pendiente): {len(labels)}")
    if labels.empty:
        print("No hay revisiones usables; no se adapta nada.")
        return

    report = M.load_base_report()
    state = M.load_state()
    threshold_actual = (state or {}).get("threshold_score_samples", M.base_threshold())

    # Cola actual (lo que vio el auditor) para medir precision.
    queue = M.compute_queue(report, state)
    n_antes = int(queue["is_anomaly"].sum())
    metrics = fb.feedback_metrics(queue, labels)
    print(f"Precision global observada: {metrics.get('precision_global')}"
          f" sobre {metrics.get('alertas_revisadas')} alertas revisadas")

    # Aprender: umbral + factores de confianza por regla.
    sug = fb.suggest_threshold(threshold_actual, metrics)
    weights = fb.suggest_rule_weights(queue, labels)
    print(f"Umbral: {threshold_actual:.4f} -> {sug['umbral_nuevo']:.4f} ({sug['motivo']})")
    print(f"Factores de confianza por regla aprendidos: {weights or '(ninguno)'}")

    # Persistir y registrar las etiquetas en el almacen historico.
    M.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fb.save_feedback_state(M.STATE_FILE, sug["umbral_nuevo"], weights)
    store = M.STATE_FILE.parent / "feedback_labels.csv"
    fb.init_label_store(store)
    labels.assign(revisor="dashboard", timestamp_revision=pd.Timestamp.now("UTC").isoformat(),
                  nota="").to_csv(store, mode="a", header=False, index=False)

    # Regenerar la cola con lo aprendido.
    build_dashboard.build()
    queue_nueva = M.compute_queue(report, M.load_state())
    n_despues = int(queue_nueva["is_anomaly"].sum())
    print(f"\nCola: {n_antes:,} -> {n_despues:,} alertas tras aprender "
          f"({n_despues - n_antes:+,}). Recarga el dashboard para verlo.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: uv run python model_final/adapt.py <revisiones.csv>")
        sys.exit(1)
    adapt(sys.argv[1])


if __name__ == "__main__":
    main()
