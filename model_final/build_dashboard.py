"""Genera los datos del dashboard desde el modelo consolidado + lo aprendido.

Lee el reporte de `model_Adrian`, aplica el estado aprendido (si existe) y escribe
`model_final/dashboard/data/anomalies.json` (top-N de la cola de revision).

Uso:
    uv run python model_final/build_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_final import model as M  # noqa: E402


def build() -> None:
    report = M.load_base_report()
    clean = pd.read_parquet(M.CLEAN_FILE, columns=["t_monto", "t_carabo"])
    state = M.load_state()

    queue = M.compute_queue(report, state)
    records = M.to_dashboard_records(queue, clean)
    path = M.write_dashboard_data(records)

    n_cola = int(queue["is_anomaly"].sum())
    aprendido = "SI (umbral/pesos del auditor)" if state else "no (modelo base)"
    print(f"Estado aprendido aplicado: {aprendido}")
    print(f"Cola de revision: {n_cola:,} ({100 * n_cola / len(queue):.2f}%) | exportadas al dashboard: {len(records)}")
    print("Severidad en el dashboard:",
          pd.Series([r["severity"] for r in records]).value_counts().to_dict())
    print(f"Guardado: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
