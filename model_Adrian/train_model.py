"""Entrenamiento del modelo FINANOM — pipeline HIBRIDO unificado (Adrian + Tony + Rogelio).

Apoyo al auditor nocturno: combina lo mejor del trabajo de los tres integrantes.

1. Isolation Forest no supervisado sobre las 63 features `feat_*`.
   - Muestreo ESTRATIFICADO y umbral ADAPTATIVO reutilizados de `model_Rogelio.train`.
2. Reglas de negocio TIPADAS, reutilizando el motor de `model_Tony.reglas`
   (8 detectores con catalogos de codigos, scoring y mensajes legibles) + la regla de
   metodo de pago (Visa<->Amex) que aporta Adrian.
3. Explicabilidad SHAP (TreeExplainer, reutilizada de `model_Rogelio.train`) sobre las
   filas que marca el IF: cada alerta del modelo trae las features que la aislaron.

Salida principal: `output/reporte_revision.parquet` (+ CSV de las marcadas). Reporte
fusionable con el informe del PMS: por transaccion trae score, severidad, tipo de
inconsistencia, motivo legible, evidencia SHAP, accion sugerida y bandera de aprobacion.

Cola de revision (rankeada por prioridad):
    is_anomaly = reglas operativas (Tony, score>=25)  OR  IF-flagged  OR  metodo_pago

Calidad evaluada SIN etiquetas: inyeccion sintetica (recall del IF), estabilidad
temporal y overlap IF/reglas (la evidencia de por que el modelo es hibrido). El bucle de
feedback del auditor queda DISENADO + stub (ver `apply_feedback`).

Uso:
    uv run python model_Adrian/train_model.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

# Reutilizacion del trabajo de los companeros (ROOT en path para imports cruzados).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_Tony.reglas import detectar_anomalias  # noqa: E402
from model_Rogelio.train import (  # noqa: E402
    compute_shap,
    pick_threshold,
    stratified_sample,
)


# --------------------------------------------------------------------------- #
# Rutas y parametros reproducibles
# --------------------------------------------------------------------------- #
TRAINING_DATA_DIR = ROOT / "training_data"
MODELED_FILE = TRAINING_DATA_DIR / "transacciones_modelado.parquet"
X_FILE = TRAINING_DATA_DIR / "X_modelo.parquet"
CLEAN_FILE = ROOT / "data_cleaning" / "output" / "transacciones_limpio.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_CARD_FILE = Path(__file__).resolve().parent / "modelo_card.md"

RANDOM_STATE = 42
N_ESTIMATORS = 200
FIT_SAMPLE_SIZE = 200_000          # muestra estratificada del periodo de entrenamiento
TIME_SPLIT_QUANTILE = 0.80         # 80% temporal entrena, 20% reciente valida estabilidad
IF_MIN_PCT = 1.0                   # piso del umbral adaptativo del IF (% de la base)
IF_MAX_PCT = 5.0                   # techo del umbral adaptativo del IF
TOP_FEATURES_K = 3
METODO_PAGO_SCORE = 30             # puntaje de la regla de metodo de pago (operativa)
CARD_CODES = ("AMEXCO", "TARCRE", "TARDEB")
INJECTION_N = 300

TRACE_COLS = [
    "trace_row_id", "trace_t_folio", "trace_t_folio_ext", "trace_t_referencia",
    "trace_t_transaccion", "trace_t_cve_res", "trace_t_cuarto", "trace_t_codigo",
    "trace_t_timestamp",
]

# Clusters de Tony + metodo_pago de Adrian -> mapeo a la inconsistencia de negocio.
CLUSTER_NEGOCIO = {
    "DUPLICADO": "Cargo duplicado / doble posteo",
    "SIGNO_CONTABLE": "Monto con signo contable inesperado",
    "FUERA_DE_ESTANCIA": "Cargo fuera de la ventana de estancia",
    "MONTO_ATIPICO": "Monto fuera de rango para su concepto",
    "CANCELACION_SOSPECHOSA": "Cancelacion/reposteo sospechoso",
    "CONTEXTO_RESERVACION": "Inconsistencia con el contexto de la reserva",
    "PAGO_PROVEEDOR_SOSPECHOSO": "Egreso/pago a proveedor sospechoso",
    "METODO_PAGO": "Metodo de pago mal usado (Visa<->Amex)",
}

# Etiquetas legibles de features para la evidencia SHAP.
FEATURE_LABELS = {
    "feat_monto_z_codigo_carabo": "monto atipico para su concepto",
    "feat_monto_abs_log": "magnitud del monto",
    "feat_impuesto_z_codigo_carabo": "impuesto atipico para su concepto",
    "feat_propina_z_codigo_carabo": "propina atipica para su concepto",
    "feat_impuesto_ratio_abs": "ratio impuesto/monto",
    "feat_propina_ratio_abs": "ratio propina/monto",
    "feat_monto_vs_tarifa_ratio": "monto vs tarifa diaria",
    "feat_monto_vs_tarifa_total_ratio": "monto vs tarifa total",
    "feat_dup_mismo_dia_log": "repeticion del cargo en el dia",
    "feat_folio_codigo_dia_count_log": "repeticion del concepto en el folio/dia",
    "feat_folio_dia_movimientos_log": "movimientos del folio en el dia",
    "feat_folio_total_movimientos_log": "movimientos historicos del folio",
    "feat_usuario_mod_distinto": "modificado por usuario distinto",
    "feat_cargo_fuera_estancia": "cargo fuera de la estancia",
    "feat_es_madrugada": "cargo en madrugada",
    "feat_t_usuario_freq": "frecuencia del usuario",
    "feat_t_codigo_freq": "frecuencia del concepto",
    "feat_h_tfa_scaled": "tarifa diaria de la reserva",
    "feat_h_tfa_total_scaled": "tarifa total de la reserva",
    "feat_h_tarifa_forzada_scaled": "tarifa forzada",
    "feat_h_dep_sol_scaled": "deposito solicitado",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("feat_", "").replace("_", " "))


@dataclass(frozen=True)
class TrainingPaths:
    """Rutas de artefactos generados por la fase de entrenamiento."""

    output_dir: Path
    model_file: Path
    scorer_file: Path
    report_parquet: Path
    report_csv: Path
    eval_report_file: Path
    eval_json_file: Path
    labels_store_file: Path
    score_plot_file: Path
    overlap_plot_file: Path
    shap_plot_file: Path
    model_card_file: Path

    @classmethod
    def from_dir(cls, output_dir: Path | str = OUTPUT_DIR,
                 model_card_file: Path | str = MODEL_CARD_FILE) -> "TrainingPaths":
        output_dir = Path(output_dir)
        return cls(
            output_dir=output_dir,
            model_file=output_dir / "modelo_iforest.joblib",
            scorer_file=output_dir / "scorer_bundle.joblib",
            report_parquet=output_dir / "reporte_revision.parquet",
            report_csv=output_dir / "reporte_revision_marcadas.csv",
            eval_report_file=output_dir / "reporte_evaluacion_modelo.md",
            eval_json_file=output_dir / "evaluacion_modelo.json",
            labels_store_file=output_dir / "feedback_labels.csv",
            score_plot_file=output_dir / "score_distribution.png",
            overlap_plot_file=output_dir / "overlap_if_reglas.png",
            shap_plot_file=output_dir / "shap_top_features.png",
            model_card_file=Path(model_card_file),
        )


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def to_numpy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte dtypes pyarrow/nullable a numpy para el motor de reglas de Tony.

    Tony desarrollo contra pandas numpy-backed; la base limpia usa dtypes pyarrow con
    <NA>. Rellena strings vacios para que sus comparaciones tipo `== '1'` no propaguen NA.
    """
    out = pd.DataFrame(index=range(len(df)))
    for c in df.columns:
        s = df[c]
        if "datetime" in str(s.dtype):
            out[c] = pd.to_datetime(s, errors="coerce")
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(bool)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[c] = s.astype("string").fillna("").astype(object)
    return out


def load_inputs(modeled_file: Path, x_file: Path, clean_file: Path):
    """Carga matriz de features, trazabilidad y la base limpia (para las reglas)."""
    X = pd.read_parquet(x_file)
    trace = pd.read_parquet(modeled_file, columns=TRACE_COLS)
    clean = pd.read_parquet(clean_file)
    if not (len(X) == len(trace) == len(clean)):
        raise ValueError("Longitudes desalineadas entre X, trazabilidad y base limpia.")
    if not bool((trace["trace_row_id"].to_numpy() == np.arange(len(trace))).all()):
        raise ValueError("trace_row_id no es posicional; no se puede alinear con la base limpia.")
    return X, trace, to_numpy_df(clean), list(X.columns)


def temporal_split(trace: pd.DataFrame, quantile: float = TIME_SPLIT_QUANTILE):
    """Mascara de entrenamiento = periodo base; el resto valida estabilidad."""
    ts = trace["trace_t_timestamp"]
    cutoff = ts.quantile(quantile)
    return (ts <= cutoff).to_numpy(), cutoff


# --------------------------------------------------------------------------- #
# Isolation Forest (muestreo estratificado + umbral adaptativo de Rogelio)
# --------------------------------------------------------------------------- #
def fit_isolation_forest(X: pd.DataFrame, train_mask: np.ndarray) -> IsolationForest:
    """Entrena el IF sobre una muestra ESTRATIFICADA del periodo de entrenamiento."""
    train = X.loc[train_mask]
    sample = stratified_sample(train, FIT_SAMPLE_SIZE)  # Rogelio: estratos por flags clave
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(sample.values)
    return model


def score_and_threshold(model: IsolationForest, X: pd.DataFrame, train_mask: np.ndarray):
    """score_samples (menor = mas anomalo) + umbral adaptativo de Rogelio.

    Devuelve: anomaly_score (mayor = mas anomalo), is_if, score_samples, umbral_ss.
    """
    ss = model.score_samples(X.values)
    threshold_ss, pct, rationale = pick_threshold(ss[train_mask], IF_MIN_PCT, IF_MAX_PCT)
    is_if = ss <= threshold_ss
    anomaly_score = -ss  # convencion del proyecto: mayor = mas anomalo
    return anomaly_score, is_if, ss, float(threshold_ss), rationale


# --------------------------------------------------------------------------- #
# Capa de reglas: motor de Tony + regla de metodo de pago (Adrian)
# --------------------------------------------------------------------------- #
def rule_metodo_pago(clean: pd.DataFrame) -> np.ndarray:
    """Amex (AMEXCO) vs tarjeta generica entre forma de pago de reserva y codigo del cargo."""
    if "h_for_pgo" not in clean.columns or "t_codigo" not in clean.columns:
        return np.zeros(len(clean), dtype=bool)
    res_pay = clean["h_for_pgo"].astype("string").str.strip()
    chg_pay = clean["t_codigo"].astype("string").str.strip()
    both = res_pay.isin(CARD_CODES) & chg_pay.isin(CARD_CODES)
    amex = both & ((res_pay == "AMEXCO") != (chg_pay == "AMEXCO"))
    return amex.fillna(False).to_numpy()


def run_rule_engine(clean: pd.DataFrame) -> dict:
    """Ejecuta el motor de Tony y agrega la regla de metodo de pago de Adrian."""
    alerts = detectar_anomalias(clean)          # motor de Tony (reutilizado)
    rule_score = alerts["score_riesgo"].to_numpy().astype(int)
    cluster = alerts["cluster_anomalia"].fillna("").astype(str).to_numpy()
    mensaje = alerts["mensaje_alerta"].fillna("").astype(str).to_numpy()
    operativa = alerts["es_alerta_operativa"].to_numpy()
    contexto = alerts["senales_contexto"].fillna("").astype(str).to_numpy()

    metodo_pago = rule_metodo_pago(clean)       # regla de Adrian (Tony no la tiene)
    rule_score = rule_score + metodo_pago.astype(int) * METODO_PAGO_SCORE
    operativa = operativa | metodo_pago
    cluster = np.array([
        (c + (" | " if c else "") + "METODO_PAGO") if m else c
        for c, m in zip(cluster, metodo_pago)
    ], dtype=object)

    return {
        "rule_score": rule_score, "cluster": cluster, "mensaje": mensaje,
        "operativa": operativa, "contexto": contexto, "metodo_pago": metodo_pago,
    }


# --------------------------------------------------------------------------- #
# Explicabilidad SHAP (sobre las filas que marca el IF) — reutiliza Rogelio
# --------------------------------------------------------------------------- #
def shap_top_features(model: IsolationForest, X: pd.DataFrame, is_if: np.ndarray,
                      feature_cols: list[str], k: int = TOP_FEATURES_K):
    """Top-k features por |SHAP| para cada fila marcada por el IF."""
    top = np.full(len(X), "", dtype=object)
    mean_abs = pd.Series(0.0, index=[f"shap_{c}" for c in feature_cols])
    if not is_if.any():
        return top, mean_abs
    shap_df = compute_shap(model, X, is_if)     # Rogelio: TreeExplainer
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    vals = shap_df.to_numpy()
    cols = [c.replace("shap_", "") for c in shap_df.columns]
    order = np.argsort(np.abs(vals), axis=1)[:, ::-1][:, :k]
    pos = np.where(is_if)[0]
    for row_i, p in enumerate(pos):
        top[p] = ", ".join(feature_label(cols[j]) for j in order[row_i])
    return top, mean_abs


# --------------------------------------------------------------------------- #
# Fusion -> reporte de revision
# --------------------------------------------------------------------------- #
def severidad(rule_score: np.ndarray, is_if: np.ndarray, contexto: np.ndarray,
              metodo_pago: np.ndarray) -> np.ndarray:
    """Combina puntaje de reglas + senal del IF en un nivel unico.

    En este hotel las cancelaciones y duplicados son COMUNES (~11% y ~8%), asi que el
    umbral operativo base de Tony (>=25) marcaria ~20%. Para una cola revisable por el
    auditor se sube el corte a rule_score>=60 (varias reglas o una fuerte + contexto).
    METODO_PAGO (raro y de alto valor) es siempre critico.
    """
    sev = np.full(len(rule_score), "BAJO", dtype=object)
    has_ctx = np.array([bool(c) for c in contexto])
    sev[(rule_score > 0) | has_ctx] = "MEDIO"
    sev[(rule_score >= 60) | is_if] = "ALTO"
    sev[(rule_score >= 90) | (is_if & (rule_score >= 50)) | metodo_pago] = "CRITICO"
    return sev


def build_report(trace, anomaly_score, is_if, rules, shap_top, ss):
    """Construye el reporte de revision (todas las filas; texto en las marcadas)."""
    n = len(trace)
    rule_score = rules["rule_score"]
    sev = severidad(rule_score, is_if, rules["contexto"], rules["metodo_pago"])
    is_anomaly = np.isin(sev, ["ALTO", "CRITICO"])

    report = trace.copy()
    report["anomaly_score"] = anomaly_score
    report["anomaly_pct"] = pd.Series(anomaly_score).rank(pct=True).to_numpy()
    report["score_samples"] = ss
    report["is_anomaly_if"] = is_if
    report["rule_score"] = rule_score
    report["severidad"] = sev
    report["is_anomaly"] = is_anomaly

    tipo = np.full(n, "", dtype=object)
    motivos = np.full(n, "", dtype=object)
    aprob = np.zeros(n, dtype=bool)
    qidx = np.where(is_anomaly)[0]

    for i in qidx:
        cl = rules["cluster"][i]
        if cl:
            tipo[i] = cl
            motivos[i] = rules["mensaje"][i] or "; ".join(
                CLUSTER_NEGOCIO.get(c, c) for c in cl.split(" | ")
            )
        else:  # marcada solo por IF, sin regla -> evidencia SHAP
            tipo[i] = "ATIPICO_IF"
            ev = shap_top[i]
            motivos[i] = ("Patron atipico detectado por Isolation Forest"
                          + (f" (evidencia: {ev})" if ev else "") + ".")
        aprob[i] = True

    report["tipo_inconsistencia"] = tipo
    report["motivos"] = motivos
    report["evidencia_shap"] = shap_top
    report["requiere_aprobacion"] = aprob
    return report, is_anomaly


# --------------------------------------------------------------------------- #
# Evaluacion recursiva (sin etiquetas)
# --------------------------------------------------------------------------- #
def evaluate_injection(model, X, ss, threshold_ss):
    """Inyecta anomalias de UNA dimension y mide recall del IF.

    Documenta la debilidad estructural del IF: una anomalia extrema en 1-2 de 63
    dimensiones se diluye (swamping) y el IF no la prioriza. Por eso el modelo es
    hibrido: las reglas tipadas cubren esos casos por construccion.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    normal_idx = np.where(ss > np.quantile(ss, 0.5))[0]  # ss alto = normal
    pick = rng.choice(normal_idx, size=min(INJECTION_N, len(normal_idx)), replace=False)
    base = X.iloc[pick].copy().reset_index(drop=True)

    def cmax(c):
        return float(X[c].max()) if c in X.columns else 0.0

    cases = {
        "monto_inflado": {"feat_monto_z_codigo_carabo": 12.0,
                          "feat_monto_abs_log": cmax("feat_monto_abs_log")},
        "duplicado_extremo": {"feat_dup_mismo_minuto_flag": 1, "feat_dup_mismo_dia_flag": 1,
                              "feat_dup_mismo_dia_log": 4.0},
        "cargo_fuera_estancia": {"feat_cargo_fuera_estancia": 1, "feat_cargo_antes_llegada": 1},
        "modificacion": {"feat_usuario_modificado": 1, "feat_usuario_mod_distinto": 1},
    }
    out = {}
    for name, changes in cases.items():
        pert = base.copy()
        for col, val in changes.items():
            if col in pert.columns:
                pert[col] = val
        s = model.score_samples(pert.values)
        out[name] = float((s <= threshold_ss).mean())
    out["promedio"] = float(np.mean(list(out.values())))
    return out


def evaluate_stability(ss, train_mask, threshold_ss):
    def summ(s):
        return {"p50": float(np.quantile(s, .50)), "p05": float(np.quantile(s, .05)),
                "p01": float(np.quantile(s, .01)),
                "tasa_marcado": float((s <= threshold_ss).mean())}
    return {"entrenamiento": summ(ss[train_mask]), "reciente": summ(ss[~train_mask])}


def evaluate_overlap(is_if, rules):
    """Overlap entre el IF y las reglas operativas (la evidencia del valor hibrido)."""
    operativa = rules["operativa"]
    return {
        "if_flagged": int(is_if.sum()),
        "reglas_operativas": int(operativa.sum()),
        "interseccion": int((is_if & operativa).sum()),
        "solo_reglas": int((operativa & ~is_if).sum()),
        "solo_if": int((is_if & ~operativa).sum()),
    }


# --------------------------------------------------------------------------- #
# Feedback loop (DISENO + STUB)
# --------------------------------------------------------------------------- #
FEEDBACK_COLUMNS = ["trace_row_id", "decision", "revisor", "timestamp_revision", "nota"]


def write_labels_template(path: Path) -> None:
    """Crea el almacen de etiquetas del auditor (vacio) si no existe.

    `decision` ∈ {anomalia_confirmada, falso_positivo}. Compatible con el etiquetado
    manual del dashboard (dashboard/state_manager.py) de Tony.
    """
    if not path.exists():
        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(path, index=False)


def apply_feedback(report: pd.DataFrame, labels: pd.DataFrame, threshold: float) -> dict:
    """STUB del bucle de aprendizaje (diseno listo, no se simula en esta fase).

    AHORA: con etiquetas del auditor mide la precision de las alertas y sugiere mover
    el umbral. DESPUES (TODO): re-ranker supervisado sobre `feat_*` con las etiquetas
    acumuladas y reentrenamiento del IF excluyendo falsos positivos confirmados.
    """
    if labels is None or labels.empty:
        return {"estado": "sin_etiquetas", "umbral_actual": threshold,
                "nota": "Aun no hay revisiones del auditor; el modelo opera con el umbral base."}
    rev = report.merge(labels[["trace_row_id", "decision"]], on="trace_row_id", how="inner")
    rev = rev[rev["is_anomaly"]]
    if rev.empty:
        return {"estado": "sin_alertas_revisadas", "umbral_actual": threshold}
    precision = float((rev["decision"] == "anomalia_confirmada").mean())
    return {"estado": "evaluado", "alertas_revisadas": int(len(rev)),
            "precision_observada": precision,
            "sugerencia": "subir umbral" if precision < 0.5 else "mantener/bajar umbral"}


# --------------------------------------------------------------------------- #
# Graficas
# --------------------------------------------------------------------------- #
def write_plots(anomaly_score, ss, threshold_ss, overlap, mean_abs_shap, paths: TrainingPaths) -> None:
    # 1. Distribucion de scores (estilo Rogelio: completo + zoom cola)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ss, bins=200, color="steelblue", alpha=0.8)
    axes[0].axvline(threshold_ss, color="crimson", ls="--", label="umbral IF")
    axes[0].set_title("score_samples — corpus completo")
    axes[0].legend()
    axes[0].set_xlabel("score_samples (menor = mas anomalo)")
    tail = ss[ss <= np.quantile(ss, 0.05)]
    axes[1].hist(tail, bins=120, color="coral", alpha=0.85)
    axes[1].axvline(threshold_ss, color="crimson", ls="--")
    axes[1].set_title("Zoom cola anomala (peor 5%)")
    axes[1].set_xlabel("score_samples")
    plt.tight_layout()
    fig.savefig(paths.score_plot_file, dpi=150)
    plt.close(fig)

    # 2. Overlap IF vs reglas
    plt.figure(figsize=(7, 4))
    labels = ["solo IF", "interseccion", "solo reglas"]
    vals = [overlap["solo_if"], overlap["interseccion"], overlap["solo_reglas"]]
    plt.bar(labels, vals, color=["#3b6ea8", "#7b5ea8", "#2f6f73"])
    plt.ylabel("transacciones")
    plt.title("Cobertura: Isolation Forest vs reglas")
    plt.tight_layout()
    plt.savefig(paths.overlap_plot_file, dpi=150)
    plt.close()

    # 3. SHAP top features (Rogelio)
    top = mean_abs_shap.head(15).iloc[::-1]
    if top.sum() > 0:
        plt.figure(figsize=(9, 6))
        plt.barh([feature_label(i.replace("shap_", "")) for i in top.index], top.values, color="steelblue")
        plt.xlabel("|SHAP| medio (filas marcadas por IF)")
        plt.title("Evidencia SHAP — top features")
        plt.tight_layout()
        plt.savefig(paths.shap_plot_file, dpi=150)
        plt.close()


# --------------------------------------------------------------------------- #
# Documentacion
# --------------------------------------------------------------------------- #
def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_eval_report(report, rules, injection, stability, overlap, threshold_ss,
                      cutoff, paths: TrainingPaths) -> None:
    n = len(report)
    n_anom = int(report["is_anomaly"].sum())
    span = (report["trace_t_timestamp"].max() - report["trace_t_timestamp"].min()).days or 1
    L = []
    L.append("# Reporte de evaluacion — Modelo FINANOM (hibrido consolidado)\n")
    L.append("Generado por `model_Adrian/train_model.py`. Consolida IF (muestreo "
             "estratificado + umbral adaptativo, Rogelio), reglas de negocio (Tony) y la "
             "regla de metodo de pago (Adrian), con explicabilidad SHAP (Rogelio).\n")
    L.append("## Resumen\n")
    L.append(f"- Transacciones evaluadas: {n:,}")
    L.append(f"- Marcadas para revision: {n_anom:,} ({100*n_anom/n:.2f}%)")
    L.append(f"- Aprox. alertas/dia: {n_anom/span:.1f}")
    L.append(f"- Umbral IF (score_samples): {threshold_ss:.4f} | corte temporal: {cutoff}\n")

    L.append("## Composicion por severidad\n")
    L.append("| Severidad | Transacciones | % |")
    L.append("| --------- | ------------- | - |")
    for s in ["CRITICO", "ALTO", "MEDIO", "BAJO"]:
        c = int((report["severidad"] == s).sum())
        L.append(f"| {s} | {c:,} | {100*c/n:.2f}% |")
    L.append("")

    L.append("## Composicion de la cola por tipo (cluster)\n")
    from collections import Counter
    cnt = Counter()
    for cl in report.loc[report["is_anomaly"], "tipo_inconsistencia"]:
        for p in str(cl).split(" | "):
            if p:
                cnt[p] += 1
    L.append("| Tipo | Marcadas | Inconsistencia de negocio |")
    L.append("| ---- | -------- | ------------------------- |")
    for k, v in cnt.most_common():
        L.append(f"| `{k}` | {v:,} | {CLUSTER_NEGOCIO.get(k, '-')} |")
    L.append("")

    L.append("## Iteracion 1 — Inyeccion sintetica (recall del IF)\n")
    L.append("Anomalias extremas en 1-2 dimensiones inyectadas en filas normales. El IF "
             "global las diluye (swamping entre 63 features); por eso NO basta solo. Las "
             "reglas tipadas las capturan por construccion -> arquitectura hibrida.\n")
    L.append("| Tipo inyectado | Recall IF |")
    L.append("| -------------- | --------- |")
    for k, v in injection.items():
        L.append(f"| {k} | {v:.2%} |")
    L.append("")

    L.append("## Iteracion 2 — Estabilidad temporal\n")
    L.append("| Periodo | p50 | p05 | p01 | tasa marcado IF |")
    L.append("| ------- | --- | --- | --- | --------------- |")
    for per, s in stability.items():
        L.append(f"| {per} | {s['p50']:.3f} | {s['p05']:.3f} | {s['p01']:.3f} | {s['tasa_marcado']:.2%} |")
    L.append("")

    L.append("## Iteracion 3 — Overlap IF vs reglas (por que es hibrido)\n")
    L.append(f"- IF-flagged: {overlap['if_flagged']:,}")
    L.append(f"- Reglas operativas: {overlap['reglas_operativas']:,}")
    L.append(f"- Interseccion: {overlap['interseccion']:,}")
    L.append(f"- Solo reglas (el IF NO las priorizaba): {overlap['solo_reglas']:,}")
    L.append(f"- Solo IF (atipicos sin regla): {overlap['solo_if']:,}")
    L.append("\nLas dos capas son mayormente disjuntas: cada una aporta alertas que la otra "
             "no ve. El IF da los atipicos multidimensionales desconocidos; las reglas, los "
             "tipos de negocio concretos y explicables.\n")
    paths.eval_report_file.write_text("\n".join(L), encoding="utf-8")


def write_model_card(report, feature_cols, threshold_ss, paths: TrainingPaths) -> None:
    n = len(report)
    n_anom = int(report["is_anomaly"].sum())
    L = []
    L.append("# Model card — FINANOM (modelo hibrido consolidado)\n")
    L.append("Apoyo al auditor nocturno. Consolida el trabajo de los tres integrantes:\n")
    L.append("- **Isolation Forest** no supervisado con **muestreo estratificado** y "
             "**umbral adaptativo** (reutilizado de `model_Rogelio/train.py`).")
    L.append("- **Reglas de negocio tipadas**: motor de `model_Tony/reglas.py` "
             "(8 detectores) + regla de **metodo de pago** Visa<->Amex (Adrian).")
    L.append("- **Explicabilidad SHAP** (TreeExplainer, `model_Rogelio`) sobre las filas del IF.\n")

    L.append("## 1. Entrada\n")
    L.append(f"- Matriz: `training_data/X_modelo.parquet` ({n:,} filas × {len(feature_cols)} `feat_*`).")
    L.append("- Reglas: base limpia `data_cleaning/output/transacciones_limpio.parquet` (columnas crudas).")
    L.append("- Trazabilidad: `trace_*` de `training_data/transacciones_modelado.parquet`.\n")

    L.append("## 2. Modelo\n")
    L.append(f"- IsolationForest: n_estimators={N_ESTIMATORS}, max_samples='auto', random_state={RANDOM_STATE}.")
    L.append(f"- Entrenado en el {TIME_SPLIT_QUANTILE:.0%} temporal mas antiguo (muestra estratificada de {FIT_SAMPLE_SIZE:,}).")
    L.append(f"- Umbral IF adaptativo (mayor salto en [{IF_MIN_PCT}%–{IF_MAX_PCT}%]) = {threshold_ss:.4f} sobre score_samples.")
    L.append(f"- Cola de revision: {n_anom:,} ({100*n_anom/n:.2f}%) = rule_score>=60 ∪ IF ∪ metodo_pago "
             "(severidad ALTO/CRITICO), rankeada por severidad y score.\n")

    L.append("## 3. Tipos de inconsistencia (mapeo al negocio)\n")
    L.append("| Cluster | Inconsistencia de negocio |")
    L.append("| ------- | ------------------------- |")
    for k, v in CLUSTER_NEGOCIO.items():
        L.append(f"| `{k}` | {v} |")
    L.append("")

    L.append("## 4. Esquema del reporte (`output/reporte_revision.parquet`)\n")
    schema = [
        ("trace_*", "Trazabilidad (folio, transaccion, referencia, cuarto, codigo, timestamp)."),
        ("anomaly_score", "Score del IF (mayor = mas anomalo)."),
        ("anomaly_pct", "Percentil del score (0-1)."),
        ("score_samples", "Score crudo del IF (menor = mas anomalo)."),
        ("is_anomaly_if", "Marcada por el IF."),
        ("rule_score", "Puntaje acumulado de las reglas (Tony + metodo_pago)."),
        ("severidad", "CRITICO / ALTO / MEDIO / BAJO."),
        ("is_anomaly", "Entra a la cola de revision (severidad ALTO/CRITICO)."),
        ("tipo_inconsistencia", "Clusters de regla (o ATIPICO_IF)."),
        ("motivos", "Razon legible para el auditor (mensaje de Tony o evidencia SHAP)."),
        ("evidencia_shap", "Top features SHAP que aislaron la transaccion (filas del IF)."),
        ("requiere_aprobacion", "Siempre True: la correccion exige aprobacion (human-in-the-loop)."),
    ]
    L.append("| Columna | Descripcion |")
    L.append("| ------- | ----------- |")
    for c, d in schema:
        L.append(f"| `{c}` | {d} |")
    L.append("")

    L.append("## 5. Bucle de feedback del auditor\n")
    L.append("Almacen `output/feedback_labels.csv` (compatible con el etiquetado del dashboard "
             "de Tony, `dashboard/state_manager.py`). `apply_feedback()` mide precision y sugiere "
             "umbral; el re-ranker supervisado queda como TODO (diseno listo).\n")

    L.append("## 6. Limitaciones\n")
    L.append("- No supervisado: calidad validada por inyeccion/overlap, no por etiquetas reales.")
    L.append("- Metodo de pago: Amex (`AMEXCO`) separable; Visa vs Mastercard no (ambos `TARCRE`).")
    L.append("- Sin datos de factura (vacios en MX): se aproxima via monto/duplicado/reposteo.")
    L.append("- El IF subpondera anomalias de pocas dimensiones; las reglas cubren ese hueco.")
    paths.model_card_file.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def run_training(
    modeled_file: Path | str = MODELED_FILE,
    x_file: Path | str = X_FILE,
    clean_file: Path | str = CLEAN_FILE,
    output_dir: Path | str = OUTPUT_DIR,
    model_card_file: Path | str = MODEL_CARD_FILE,
) -> dict[str, Path]:
    """Entrena el modelo hibrido consolidado y genera todos los artefactos."""
    warnings.filterwarnings("ignore")
    paths = TrainingPaths.from_dir(output_dir, model_card_file)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    X, trace, clean, feature_cols = load_inputs(Path(modeled_file), Path(x_file), Path(clean_file))
    print(f"  X: {X.shape} | base limpia: {clean.shape}")

    train_mask, cutoff = temporal_split(trace)
    print(f"  split temporal: train={int(train_mask.sum()):,} | reciente={int((~train_mask).sum()):,}")

    print("Entrenando Isolation Forest (muestreo estratificado, Rogelio)...")
    model = fit_isolation_forest(X, train_mask)
    anomaly_score, is_if, ss, threshold_ss, rationale = score_and_threshold(model, X, train_mask)
    print(f"  IF-flagged: {int(is_if.sum()):,} ({100*is_if.mean():.2f}%) | {rationale[:48]}")

    print("Aplicando reglas de negocio (motor de Tony + metodo de pago)...")
    rules = run_rule_engine(clean)
    print(f"  reglas operativas: {int(rules['operativa'].sum()):,} ({100*rules['operativa'].mean():.2f}%)")

    print("Calculando explicabilidad SHAP (Rogelio) sobre filas del IF...")
    shap_top, mean_abs_shap = shap_top_features(model, X, is_if, feature_cols)

    print("Construyendo reporte de revision...")
    report, is_anomaly = build_report(trace, anomaly_score, is_if, rules, shap_top, ss)
    print(f"  cola de revision: {int(is_anomaly.sum()):,} ({100*is_anomaly.mean():.2f}%)")

    print("Evaluando calidad (inyeccion, estabilidad, overlap)...")
    injection = evaluate_injection(model, X, ss, threshold_ss)
    stability = evaluate_stability(ss, train_mask, threshold_ss)
    overlap = evaluate_overlap(is_if, rules)
    print(f"  recall IF inyeccion (promedio): {injection['promedio']:.2%}")

    # Persistencia
    joblib.dump(model, paths.model_file)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "threshold_score_samples": threshold_ss,
         "score_convention": "score_samples (menor=mas anomalo); anomaly_score=-score_samples",
         "random_state": RANDOM_STATE},
        paths.scorer_file,
    )
    report.to_parquet(paths.report_parquet, index=False)
    marcadas = report[report["is_anomaly"]].sort_values(
        ["severidad", "anomaly_score"], ascending=[True, False])
    marcadas.to_csv(paths.report_csv, index=False)
    write_labels_template(paths.labels_store_file)
    write_plots(anomaly_score, ss, threshold_ss, overlap, mean_abs_shap, paths)

    import json
    diag = {
        "rows": len(report), "flagged": int(is_anomaly.sum()),
        "flagged_pct": float(is_anomaly.mean()), "threshold_score_samples": threshold_ss,
        "if_flagged": int(is_if.sum()), "rule_operativa": int(rules["operativa"].sum()),
        "injection_recall": injection, "stability": stability, "overlap": overlap,
    }
    paths.eval_json_file.write_text(json.dumps(diag, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_eval_report(report, rules, injection, stability, overlap, threshold_ss, cutoff, paths)
    write_model_card(report, feature_cols, threshold_ss, paths)

    print(f"\nGuardado modelo: {display_path(paths.model_file)}")
    print(f"Guardado reporte: {display_path(paths.report_parquet)} ({int(is_anomaly.sum()):,} marcadas)")
    print(f"Guardada model card: {display_path(paths.model_card_file)}")

    return {
        "model": paths.model_file, "scorer": paths.scorer_file,
        "report": paths.report_parquet, "report_csv": paths.report_csv,
        "eval_report": paths.eval_report_file, "eval_json": paths.eval_json_file,
        "labels_store": paths.labels_store_file, "model_card": paths.model_card_file,
        "score_plot": paths.score_plot_file, "overlap_plot": paths.overlap_plot_file,
        "shap_plot": paths.shap_plot_file,
    }


def main() -> None:
    artifacts = run_training()
    print("\nArtefactos generados:")
    for name, path in artifacts.items():
        print(f"  {name}: {display_path(Path(path))}")


if __name__ == "__main__":
    main()
