"""Bucle de feedback del auditor — "el modelo que aprende".

ANDAMIAJE listo para implementar. Mecanismo elegido: **adaptacion de umbral/pesos**.
Cuando el auditor nocturno revisa la cola y marca cada alerta como `anomalia_confirmada`
o `falso_positivo`, el modelo ajusta su sensibilidad:

- demasiados falsos positivos  -> el umbral del IF se endurece (menos alertas);
- pocas alertas y alta precision -> el umbral se afloja (mas cobertura);
- por tipo de regla, se ajustan los pesos de las que generan mas falsos positivos.

Qué ya funciona aquí (usable hoy):
  - Esquema de etiquetas + registrar/cargar feedback (`record_feedback`, `load_feedback`).
  - Métricas de precisión de las alertas, global / por severidad / por tipo (`feedback_metrics`).
  - Sugerencia de umbral por control proporcional (`suggest_threshold`).
  - Factor de confianza por regla desde su precisión (`suggest_rule_weights`).
  - Persistencia del estado aprendido en JSON (`save_feedback_state` / `load_feedback_state`).

El estado aprendido (`{threshold_score_samples, rule_weights}`) lo APLICA `model_final`
sobre el reporte consolidado para regenerar la cola, sin reentrenar el IF (adaptacion
instantanea). Ver `model_final/model.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Esquema de etiquetas
# --------------------------------------------------------------------------- #
FEEDBACK_COLUMNS = ["trace_row_id", "decision", "revisor", "timestamp_revision", "nota"]
DECISIONS = ("anomalia_confirmada", "falso_positivo")


@dataclass(frozen=True)
class FeedbackConfig:
    """Parametros del controlador de adaptacion."""

    target_precision: float = 0.5   # precision objetivo de las alertas
    min_labels: int = 30            # etiquetas minimas (global) antes de mover el umbral
    margin: float = 0.1             # banda muerta alrededor del objetivo
    step: float = 0.15              # tamano del nudge del umbral (fraccion)
    min_labels_por_regla: int = 10  # etiquetas minimas por cluster antes de ajustar su peso
    peso_min: float = 0.5           # piso del factor de confianza de una regla (gradual, no colapsa)


# --------------------------------------------------------------------------- #
# Almacen de etiquetas (FUNCIONA)
# --------------------------------------------------------------------------- #
def init_label_store(path: Path | str) -> Path:
    """Crea el CSV de etiquetas vacio si no existe."""
    path = Path(path)
    if not path.exists():
        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(path, index=False)
    return path


def record_feedback(path: Path | str, trace_row_id: int, decision: str,
                    revisor: str, nota: str = "") -> None:
    """Anexa una revision del auditor al almacen. Valida la decision."""
    if decision not in DECISIONS:
        raise ValueError(f"decision debe ser una de {DECISIONS}, no {decision!r}")
    path = init_label_store(path)
    row = {
        "trace_row_id": int(trace_row_id), "decision": decision, "revisor": revisor,
        "timestamp_revision": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nota": nota,
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)


def load_feedback(path: Path | str) -> pd.DataFrame:
    """Lee el almacen de etiquetas (DataFrame vacio con el esquema si no hay)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Metricas de precision de las alertas (FUNCIONA)
# --------------------------------------------------------------------------- #
def feedback_metrics(report: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Precision de las alertas segun el feedback: global, por severidad y por tipo."""
    if labels is None or labels.empty:
        return {"estado": "sin_etiquetas",
                "nota": "Aun no hay revisiones; el modelo opera con sus parametros base."}
    rev = report.merge(labels[["trace_row_id", "decision"]], on="trace_row_id", how="inner")
    rev = rev[rev["is_anomaly"]]
    if rev.empty:
        return {"estado": "sin_alertas_revisadas"}

    def precision(df: pd.DataFrame) -> float:
        return float((df["decision"] == "anomalia_confirmada").mean())

    por_sev = {s: round(precision(g), 3) for s, g in rev.groupby("severidad")}
    por_tipo = {}
    for tipo, g in rev.groupby("tipo_inconsistencia"):
        por_tipo[str(tipo)] = round(precision(g), 3)
    return {
        "estado": "evaluado",
        "alertas_revisadas": int(len(rev)),
        "precision_global": round(precision(rev), 3),
        "precision_por_severidad": por_sev,
        "precision_por_tipo": por_tipo,
    }


# --------------------------------------------------------------------------- #
# Adaptacion de umbral (FUNCIONA: sugerencia por control proporcional)
# --------------------------------------------------------------------------- #
def suggest_threshold(current_threshold: float, metrics: dict,
                      cfg: FeedbackConfig = FeedbackConfig()) -> dict:
    """Sugiere un nuevo umbral del IF (score_samples) a partir de la precision observada.

    Convencion: `is_if = score_samples <= threshold`. Un umbral MAS NEGATIVO endurece
    (menos alertas); MAS ALTO afloja (mas alertas). Control proporcional simple.
    """
    if metrics.get("estado") != "evaluado":
        return {"cambiar": False, "umbral_nuevo": current_threshold,
                "motivo": "sin metricas suficientes"}
    if metrics["alertas_revisadas"] < cfg.min_labels:
        return {"cambiar": False, "umbral_nuevo": current_threshold,
                "motivo": f"pocas etiquetas ({metrics['alertas_revisadas']} < {cfg.min_labels})"}

    p = metrics["precision_global"]
    if p < cfg.target_precision - cfg.margin:        # demasiados falsos positivos -> endurecer
        nuevo = current_threshold - cfg.step * abs(current_threshold)
        return {"cambiar": True, "umbral_nuevo": float(nuevo),
                "motivo": f"precision {p:.2f} < objetivo {cfg.target_precision:.2f}: endurecer"}
    if p > cfg.target_precision + cfg.margin:        # muy preciso -> ampliar cobertura
        nuevo = current_threshold + cfg.step * abs(current_threshold)
        return {"cambiar": True, "umbral_nuevo": float(nuevo),
                "motivo": f"precision {p:.2f} > objetivo {cfg.target_precision:.2f}: aflojar"}
    return {"cambiar": False, "umbral_nuevo": current_threshold,
            "motivo": f"precision {p:.2f} dentro de la banda objetivo"}


# --------------------------------------------------------------------------- #
# Persistencia del estado aprendido (FUNCIONA)
# --------------------------------------------------------------------------- #
def save_feedback_state(path: Path | str, threshold: float,
                        rule_weights: dict | None = None) -> None:
    """Guarda el estado aprendido (umbral del IF + factor de confianza por regla)."""
    state = {
        "threshold_score_samples": float(threshold),
        "rule_weights": rule_weights or {},
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feedback_state(path: Path | str) -> dict | None:
    """Carga el estado aprendido si existe."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Adaptacion de pesos por regla (FUNCIONA)
# --------------------------------------------------------------------------- #
def rule_type_precision(report: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Precision y n revisado por cada cluster de regla (un cargo cuenta en cada cluster que disparo)."""
    rev = report.merge(labels[["trace_row_id", "decision"]], on="trace_row_id", how="inner")
    rev = rev[rev["is_anomaly"]]
    out: dict[str, dict] = {}
    for _, row in rev.iterrows():
        confirmada = row["decision"] == "anomalia_confirmada"
        for cluster in str(row["tipo_inconsistencia"]).split(" | "):
            if not cluster:
                continue
            d = out.setdefault(cluster, {"n": 0, "confirmadas": 0})
            d["n"] += 1
            d["confirmadas"] += int(confirmada)
    for c, d in out.items():
        d["precision"] = round(d["confirmadas"] / d["n"], 3) if d["n"] else None
    return out


def suggest_rule_weights(report: pd.DataFrame, labels: pd.DataFrame,
                         cfg: FeedbackConfig = FeedbackConfig()) -> dict:
    """Factor de confianza por cluster de regla, en [peso_min, 1.0], desde su precision.

    Un cluster que el auditor rechaza seguido (baja precision) recibe un factor < 1 que
    luego MULTIPLICA su `rule_score` -> sus alertas bajan de severidad. Clusters con
    pocas etiquetas o buena precision se quedan en 1.0 (sin castigo). Simple y explicable.
    """
    prec = rule_type_precision(report, labels)
    weights: dict[str, float] = {}
    for cluster, d in prec.items():
        if d["n"] < cfg.min_labels_por_regla or d["precision"] is None:
            continue
        if d["precision"] < cfg.target_precision:
            # factor = precision observada, acotado por piso (p.ej. 30% precision -> factor 0.3)
            weights[cluster] = round(max(cfg.peso_min, d["precision"]), 3)
    return weights
