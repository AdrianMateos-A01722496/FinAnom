"""Gestión de etiquetas manuales para validación del modelo.

Las etiquetas se persisten en un CSV local (`dashboard/feedback_manual.csv`).
Este archivo crece con cada revisión y se usará en el futuro como dataset
de entrenamiento supervisado.

Columnas guardadas:
    id_transaccion      : identificador de la transacción
    etiqueta_manual     : "Es anomalía" | "No es anomalía" | "Pendiente de revisar"
    comentario          : texto libre del revisor
    timestamp_revision  : ISO-8601 del momento en que se guardó
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

FEEDBACK_PATH = Path(__file__).resolve().parent / "feedback_manual.csv"

ETIQUETAS: list[str] = ["Es anomalía", "No es anomalía", "Pendiente de revisar"]

_COLS = ["id_transaccion", "etiqueta_manual", "comentario", "timestamp_revision"]


def load_feedback() -> pd.DataFrame:
    """Carga todas las etiquetas guardadas. Devuelve DataFrame vacío si no existe el archivo."""
    if FEEDBACK_PATH.exists():
        return pd.read_csv(FEEDBACK_PATH, dtype=str).fillna("")
    return pd.DataFrame(columns=_COLS)


def save_label(id_tx: str, etiqueta: str, comentario: str = "") -> None:
    """
    Guarda o actualiza la etiqueta para una transacción.

    Si ya existe una etiqueta para `id_tx`, la sobreescribe.
    Si no existe, agrega una fila nueva.
    """
    if etiqueta not in ETIQUETAS:
        raise ValueError(f"Etiqueta inválida: '{etiqueta}'. Opciones: {ETIQUETAS}")

    df = load_feedback()
    nueva = {
        "id_transaccion":     str(id_tx),
        "etiqueta_manual":    etiqueta,
        "comentario":         comentario,
        "timestamp_revision": datetime.now().isoformat(timespec="seconds"),
    }

    mask = df["id_transaccion"] == str(id_tx)
    if mask.any():
        for col, val in nueva.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([nueva])], ignore_index=True)

    df.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8")


def get_label(id_tx: str) -> str | None:
    """Devuelve la etiqueta actual para `id_tx`, o None si no tiene etiqueta."""
    df = load_feedback()
    row = df[df["id_transaccion"] == str(id_tx)]
    if not row.empty:
        return str(row.iloc[0]["etiqueta_manual"])
    return None


def get_labels_map() -> dict[str, str]:
    """Devuelve un dict {id_transaccion: etiqueta_manual} para uso en tablas."""
    df = load_feedback()
    if df.empty:
        return {}
    return df.set_index("id_transaccion")["etiqueta_manual"].to_dict()
