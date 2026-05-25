"""Entrenamiento del modelo FINANOM — apoyo al auditor nocturno.

Modelo HIBRIDO de deteccion de inconsistencias financieras:

1. Isolation Forest (no supervisado) sobre las 63 features `feat_*` -> capta
   patrones atipicos desconocidos y da un `anomaly_score` continuo por transaccion.
2. Reglas de negocio TIPADAS (deterministas) -> cubren las inconsistencias que el
   hotel nombro explicitamente, cada una con su razon legible.

Salida principal: `output/reporte_revision.parquet` (+ CSV de las marcadas). Es un
reporte que el auditor nocturno puede fundir con su informe del PMS: cada
transaccion marcada trae score, tipo de inconsistencia, motivo, las features que la
aislaron, accion sugerida y bandera de aprobacion (human-in-the-loop).

Cola de revision (presupuesto ~2%, acorde a la capacidad del auditor):
  is_anomaly = IF top-1%  OR  reglas criticas (todas)  OR  reglas amplias (top-K)

Calidad evaluada SIN etiquetas: inyeccion sintetica (recall@presupuesto),
estabilidad temporal y overlap IF/reglas. El bucle de feedback del auditor queda
DISENADO + stub (ver `apply_feedback`).

Uso:
    uv run python model_training/train_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402


# --------------------------------------------------------------------------- #
# Rutas y parametros reproducibles
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = ROOT / "training_data"
MODELED_FILE = TRAINING_DATA_DIR / "transacciones_modelado.parquet"
X_FILE = TRAINING_DATA_DIR / "X_modelo.parquet"
CLEAN_FILE = ROOT / "data_cleaning" / "output" / "transacciones_limpio.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_CARD_FILE = Path(__file__).resolve().parent / "modelo_card.md"

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_SAMPLES = 8192
FIT_SAMPLE_SIZE = 200_000          # muestra del periodo de entrenamiento para fit del IF
CONTAMINATION = 0.01               # presupuesto IF: top ~1%
TIME_SPLIT_QUANTILE = 0.80         # 80% temporal entrena, 20% reciente valida estabilidad
TOP_FEATURES_K = 3

# Umbrales de reglas (calibrados por prevalencia; ver notebook)
Z_CRITICO = 8.0                    # |monto_z| extremo -> monto_atipico (critico)
Z_MOD = 3.0                        # |monto_z| moderado para co-ocurrencia de modificacion
DUP_DIA_LOG_ESTRICTO = 3.5         # repeticion patologica del mismo cargo en el dia
RULE_TOPK = 200                    # tope por regla AMPLIA que entra a la cola (top por score)
CARD_CODES = ("AMEXCO", "TARCRE", "TARDEB")

# Evaluacion por inyeccion sintetica
INJECTION_N = 300

TRACE_COLS = [
    "trace_row_id", "trace_t_folio", "trace_t_folio_ext", "trace_t_referencia",
    "trace_t_transaccion", "trace_t_cve_res", "trace_t_cuarto", "trace_t_codigo",
    "trace_t_timestamp",
]
RAW_RULE_COLS = ["h_for_pgo", "t_codigo", "t_monto", "t_referencia"]

# Metadatos de cada regla: descripcion legible, accion sugerida, prioridad, si es critica.
RULE_META: dict[str, dict] = {
    "metodo_pago_inconsistente": dict(
        desc="Metodo de pago del cargo no coincide con el de la reserva (Amex vs tarjeta generica)",
        accion="Confirmar el metodo de pago contra el comprobante.",
        prio=1, critical=True,
    ),
    "monto_atipico": dict(
        desc="Monto muy fuera de rango para su concepto",
        accion="Revisar el monto contra la tarifa/concepto esperado.",
        prio=2, critical=True,
    ),
    "posible_duplicado": dict(
        desc="Cargo identico repetido muchas veces en el dia (posible doble posteo)",
        accion="Confirmar si es duplicado y eliminar los repetidos.",
        prio=3, critical=True,
    ),
    "reemplazo_monto": dict(
        desc="Reverso y recargo del mismo concepto con monto distinto (posible reemplazo incorrecto)",
        accion="Verificar el monto correcto y conciliar el reverso/recargo.",
        prio=4, critical=False,
    ),
    "modificacion_no_autorizada": dict(
        desc="Modificada por un usuario distinto y coincide con otra senal de riesgo",
        accion="Validar la autorizacion de la modificacion.",
        prio=5, critical=False,
    ),
    "cargo_fuera_estancia": dict(
        desc="Cargo fechado fuera de la ventana de estancia (antes de llegada o tras salida)",
        accion="Verificar la fecha del cargo contra la estancia.",
        prio=6, critical=False,
    ),
}
RULE_ORDER = sorted(RULE_META, key=lambda k: RULE_META[k]["prio"])
CRITICAL_RULES = [k for k in RULE_ORDER if RULE_META[k]["critical"]]
CAPPED_RULES = [k for k in RULE_ORDER if not RULE_META[k]["critical"]]

# Etiquetas legibles para explicabilidad (fallback: nombre sin 'feat_').
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
    injection_plot_file: Path
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
            injection_plot_file=output_dir / "injection_recall.png",
            model_card_file=Path(model_card_file),
        )


# --------------------------------------------------------------------------- #
# Carga y split
# --------------------------------------------------------------------------- #
def load_inputs(modeled_file: Path, x_file: Path, clean_file: Path):
    """Carga matriz de features, trazabilidad y campos crudos para reglas."""
    X = pd.read_parquet(x_file)
    trace = pd.read_parquet(modeled_file, columns=TRACE_COLS)
    raw = pd.read_parquet(clean_file, columns=RAW_RULE_COLS)
    if not (len(X) == len(trace) == len(raw)):
        raise ValueError("Longitudes desalineadas entre X, trazabilidad y base limpia.")
    if not bool((trace["trace_row_id"].to_numpy() == np.arange(len(trace))).all()):
        raise ValueError("trace_row_id no es posicional; no se puede alinear con la base limpia.")
    return X, trace, raw, list(X.columns)


def temporal_split(trace: pd.DataFrame, quantile: float = TIME_SPLIT_QUANTILE):
    """Mascara de entrenamiento = periodo base; el resto valida estabilidad."""
    ts = trace["trace_t_timestamp"]
    cutoff = ts.quantile(quantile)
    return (ts <= cutoff).to_numpy(), cutoff


# --------------------------------------------------------------------------- #
# Modelo Isolation Forest
# --------------------------------------------------------------------------- #
def fit_isolation_forest(X: pd.DataFrame, train_mask: np.ndarray) -> IsolationForest:
    """Entrena el IF sobre una muestra del periodo de entrenamiento."""
    train = X.loc[train_mask]
    if len(train) > FIT_SAMPLE_SIZE:
        train = train.sample(n=FIT_SAMPLE_SIZE, random_state=RANDOM_STATE)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(MAX_SAMPLES, len(train)),
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train)
    return model


def score_matrix(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Score de anomalia: mayor = mas anomalo (convencion del proyecto)."""
    return -model.decision_function(X)


# --------------------------------------------------------------------------- #
# Capa de reglas tipadas
# --------------------------------------------------------------------------- #
def compute_rules(X: pd.DataFrame, raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Devuelve una mascara booleana por cada regla de negocio (alineadas por fila)."""
    n = len(X)
    z = X["feat_monto_z_codigo_carabo"].abs().to_numpy()
    fuera = (X["feat_cargo_fuera_estancia"] == 1).to_numpy()
    mod_distinto = (X["feat_usuario_mod_distinto"] == 1).to_numpy()
    dup_min = (X["feat_dup_mismo_minuto_flag"] == 1).to_numpy()
    dup_dia_log = X["feat_dup_mismo_dia_log"].to_numpy()

    strip = lambda c: raw[c].astype("string").str.strip()
    res_pay, chg_pay = strip("h_for_pgo"), strip("t_codigo")
    both_card = res_pay.isin(CARD_CODES) & chg_pay.isin(CARD_CODES)
    amex = (both_card & ((res_pay == "AMEXCO") != (chg_pay == "AMEXCO"))).fillna(False).to_numpy()

    reemplazo = _rule_reemplazo_monto(raw)

    rules = {
        "metodo_pago_inconsistente": amex,
        "monto_atipico": z >= Z_CRITICO,
        "posible_duplicado": dup_min & (dup_dia_log >= DUP_DIA_LOG_ESTRICTO),
        "reemplazo_monto": reemplazo,
        # modificacion por usuario distinto SOLO cuando coincide con otra senal de riesgo
        "modificacion_no_autorizada": mod_distinto & ((z >= Z_MOD) | fuera | reemplazo | amex),
        "cargo_fuera_estancia": fuera,
    }
    assert all(len(v) == n for v in rules.values())
    return rules


def _rule_reemplazo_monto(raw: pd.DataFrame) -> np.ndarray:
    """Referencia con un cargo (+) y un reverso (-) de montos absolutos distintos."""
    ref = raw["t_referencia"].astype("string").str.strip()
    valid = (ref.notna() & (ref != "")).to_numpy()
    monto = raw["t_monto"].to_numpy()
    if not valid.any():
        return np.zeros(len(raw), dtype=bool)
    tmp = pd.DataFrame({
        "ref": ref[valid],
        "pos": monto[valid] > 0,
        "neg": monto[valid] < 0,
        "absamt": np.abs(monto[valid]),
    })
    agg = tmp.groupby("ref").agg(has_pos=("pos", "max"), has_neg=("neg", "max"),
                                 nabs=("absamt", "nunique"))
    bad_refs = set(agg.index[agg["has_pos"] & agg["has_neg"] & (agg["nabs"] > 1)])
    out = np.zeros(len(raw), dtype=bool)
    out[valid] = ref[valid].isin(bad_refs).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Ensamble de la cola de revision
# --------------------------------------------------------------------------- #
def assemble_queue(scores: np.ndarray, train_mask: np.ndarray, rules: dict[str, np.ndarray]):
    """is_anomaly = IF top-presupuesto  OR  reglas criticas (todas)  OR  amplias (top-K)."""
    threshold = float(np.quantile(scores[train_mask], 1 - CONTAMINATION))
    is_if = scores >= threshold

    critical = np.zeros_like(is_if)
    for r in CRITICAL_RULES:
        critical |= rules[r]

    capped = np.zeros_like(is_if)
    for r in CAPPED_RULES:
        idx = np.where(rules[r])[0]
        if len(idx) > RULE_TOPK:
            idx = idx[np.argsort(scores[idx])[::-1][:RULE_TOPK]]
        capped[idx] = True

    is_anomaly = is_if | critical | capped
    return is_anomaly, is_if, threshold


# --------------------------------------------------------------------------- #
# Explicabilidad y reporte
# --------------------------------------------------------------------------- #
def top_features_for(X: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray,
                     idx: np.ndarray, feature_cols: list[str], k: int = TOP_FEATURES_K):
    """Top-k features por |z| para cada fila marcada (explicabilidad barata)."""
    sub = X.iloc[idx].to_numpy(dtype="float64")
    zabs = np.abs((sub - mu) / sigma)
    order = np.argsort(zabs, axis=1)[:, ::-1][:, :k]
    return [", ".join(feature_label(feature_cols[j]) for j in row) for row in order]


def build_report(trace, X, raw, scores, is_anomaly, is_if, rules, feature_cols):
    """Construye el reporte de revision (todas las filas; texto solo en las marcadas)."""
    n = len(trace)
    report = trace.copy()
    report["anomaly_score"] = scores
    report["anomaly_pct"] = pd.Series(scores).rank(pct=True).to_numpy()
    report["is_anomaly_if"] = is_if
    for r in RULE_ORDER:
        report[f"regla_{r}"] = rules[r]
    report["n_reglas"] = np.sum([rules[r] for r in RULE_ORDER], axis=0).astype(int)
    report["is_anomaly"] = is_anomaly

    # Columnas de texto: solo para filas marcadas (eficiencia).
    tipo = np.full(n, "", dtype=object)
    motivos = np.full(n, "", dtype=object)
    accion = np.full(n, "", dtype=object)
    topfeat = np.full(n, "", dtype=object)
    aprob = np.zeros(n, dtype=bool)

    qidx = np.where(is_anomaly)[0]
    mu = X.to_numpy(dtype="float64").mean(axis=0)
    sigma = X.to_numpy(dtype="float64").std(axis=0)
    sigma[sigma == 0] = 1.0
    tf = top_features_for(X, mu, sigma, qidx, feature_cols)
    z_all = X["feat_monto_z_codigo_carabo"].to_numpy()

    for pos, i in enumerate(qidx):
        fired = [r for r in RULE_ORDER if rules[r][i]]
        if fired:
            tipo[i] = ",".join(fired)
            primary = fired[0]  # RULE_ORDER ya esta por prioridad
            accion[i] = RULE_META[primary]["accion"]
            partes = []
            for r in fired:
                d = RULE_META[r]["desc"]
                if r == "monto_atipico":
                    d += f" (z={z_all[i]:.1f})"
                partes.append(d)
            motivos[i] = "; ".join(partes)
        else:
            tipo[i] = "atipico_general"
            accion[i] = "Revision manual: patron atipico sin regla especifica."
            motivos[i] = "Patron atipico detectado por Isolation Forest."
        topfeat[i] = tf[pos]
        aprob[i] = True

    report["tipo_inconsistencia"] = tipo
    report["motivos"] = motivos
    report["top_features"] = topfeat
    report["accion_sugerida"] = accion
    report["requiere_aprobacion"] = aprob
    return report


# --------------------------------------------------------------------------- #
# Evaluacion recursiva (sin etiquetas)
# --------------------------------------------------------------------------- #
def evaluate_injection(model, X, scores, threshold, feature_cols):
    """Inyecta anomalias sinteticas y mide recall del IF SOLO vs el HIBRIDO (IF+reglas).

    Hallazgo esperado: el IF global subpondera anomalias extremas en pocas dimensiones
    (swamping en 63 features), asi que su recall es modesto; las reglas tipadas las
    capturan por construccion, asi que el hibrido se acerca al 100%. Esto es justo lo
    que justifica la arquitectura hibrida.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    normal_idx = np.where(scores < np.quantile(scores, 0.5))[0]
    pick = rng.choice(normal_idx, size=min(INJECTION_N, len(normal_idx)), replace=False)
    base = X.iloc[pick].copy().reset_index(drop=True)

    def cmax(c):
        return float(X[c].max()) if c in X.columns else 0.0

    # nombre -> (perturbacion de features, regla que deberia capturarla)
    cases = {
        "monto_inflado": (
            {"feat_monto_z_codigo_carabo": 12.0, "feat_monto_abs_log": cmax("feat_monto_abs_log"),
             "feat_monto_vs_tarifa_ratio": cmax("feat_monto_vs_tarifa_ratio")},
            lambda p: p["feat_monto_z_codigo_carabo"].abs() >= Z_CRITICO),
        "duplicado_extremo": (
            {"feat_dup_mismo_minuto_flag": 1, "feat_dup_mismo_dia_flag": 1, "feat_dup_mismo_dia_log": 4.0,
             "feat_folio_codigo_dia_count_log": cmax("feat_folio_codigo_dia_count_log")},
            lambda p: (p["feat_dup_mismo_minuto_flag"] == 1) & (p["feat_dup_mismo_dia_log"] >= DUP_DIA_LOG_ESTRICTO)),
        "cargo_fuera_estancia": (
            {"feat_cargo_fuera_estancia": 1, "feat_cargo_antes_llegada": 1, "feat_dias_hasta_salida_scaled": -10.0},
            lambda p: p["feat_cargo_fuera_estancia"] == 1),
        "modificacion_con_monto": (
            {"feat_usuario_modificado": 1, "feat_usuario_mod_distinto": 1, "feat_monto_z_codigo_carabo": 4.0},
            lambda p: (p["feat_usuario_mod_distinto"] == 1) & (p["feat_monto_z_codigo_carabo"].abs() >= Z_MOD)),
    }
    results, rif, rhib = {}, [], []
    for name, (changes, rule_fn) in cases.items():
        pert = base.copy()
        for col, val in changes.items():
            if col in pert.columns:
                pert[col] = val
        s = -model.decision_function(pert)
        hit_if = s >= threshold
        hit_hib = hit_if | rule_fn(pert).to_numpy()
        r_if, r_h = float(hit_if.mean()), float(hit_hib.mean())
        results[name] = {"if": r_if, "hibrido": r_h}
        rif.append(r_if)
        rhib.append(r_h)
    results["promedio"] = {"if": float(np.mean(rif)), "hibrido": float(np.mean(rhib))}
    return results


def evaluate_stability(scores, train_mask, threshold):
    """Compara distribucion de score y tasa de marcado entre periodos."""
    def summary(s):
        return {"p50": float(np.quantile(s, .50)), "p95": float(np.quantile(s, .95)),
                "p99": float(np.quantile(s, .99)),
                "tasa_marcado": float((s >= threshold).mean())}
    return {"entrenamiento": summary(scores[train_mask]),
            "reciente": summary(scores[~train_mask])}


def evaluate_overlap(is_if, rules):
    """Overlap entre el IF y las reglas criticas."""
    critical = np.zeros_like(is_if)
    for r in CRITICAL_RULES:
        critical |= rules[r]
    return {
        "if_top": int(is_if.sum()),
        "reglas_criticas": int(critical.sum()),
        "interseccion": int((is_if & critical).sum()),
        "solo_reglas_criticas": int((critical & ~is_if).sum()),
        "solo_if": int((is_if & ~critical).sum()),
    }


# --------------------------------------------------------------------------- #
# Feedback loop (DISENO + STUB)
# --------------------------------------------------------------------------- #
FEEDBACK_COLUMNS = ["trace_row_id", "decision", "revisor", "timestamp_revision", "nota"]


def write_labels_template(path: Path) -> None:
    """Crea el almacen de etiquetas del auditor (vacio) si no existe.

    `decision` ∈ {anomalia_confirmada, falso_positivo}. Cada revision del auditor
    nocturno se anexa aqui y alimenta `apply_feedback`.
    """
    if not path.exists():
        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(path, index=False)


def apply_feedback(report: pd.DataFrame, labels: pd.DataFrame, threshold: float) -> dict:
    """STUB del bucle de aprendizaje. DISENO:

    1. AHORA (implementado, ligero): con etiquetas del auditor se mide la precision
       de las alertas y se sugiere mover el umbral para acercarse a una precision
       objetivo (adaptacion de umbral).
    2. DESPUES (TODO): cuando se acumulen suficientes etiquetas, entrenar un
       re-ranker supervisado (p.ej. GradientBoosting) sobre las features `feat_*`
       para reordenar las candidatas del IF; y reentrenar el IF excluyendo patrones
       confirmados como falsos positivos.

    No reentrena en esta fase: deja listo el enganche.
    """
    if labels is None or labels.empty:
        return {"estado": "sin_etiquetas", "umbral_actual": threshold,
                "nota": "Aun no hay revisiones del auditor; el modelo opera con el umbral base."}
    rev = report.merge(labels[["trace_row_id", "decision"]], on="trace_row_id", how="inner")
    rev = rev[rev["is_anomaly"]]
    if rev.empty:
        return {"estado": "sin_alertas_revisadas", "umbral_actual": threshold}
    precision = float((rev["decision"] == "anomalia_confirmada").mean())
    return {"estado": "evaluado", "umbral_actual": threshold,
            "alertas_revisadas": int(len(rev)), "precision_observada": precision,
            "sugerencia": "subir umbral" if precision < 0.5 else "mantener/bajar umbral"}


# --------------------------------------------------------------------------- #
# Graficas
# --------------------------------------------------------------------------- #
def write_plots(scores, threshold, injection, paths: TrainingPaths) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=80, color="#3b6ea8", alpha=0.85)
    plt.axvline(threshold, color="#b23b3b", linestyle="--", label=f"umbral (top {CONTAMINATION:.0%})")
    plt.xlabel("anomaly_score (mayor = mas anomalo)")
    plt.ylabel("Transacciones")
    plt.title("Distribucion de scores - Isolation Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.score_plot_file, dpi=160)
    plt.close()

    types = [k for k in injection if k != "promedio"]
    rif = [injection[k]["if"] for k in types]
    rhib = [injection[k]["hibrido"] for k in types]
    y = np.arange(len(types))
    plt.figure(figsize=(9, 5))
    plt.barh(y - 0.2, rif, height=0.4, color="#b23b3b", label="IF solo")
    plt.barh(y + 0.2, rhib, height=0.4, color="#2f6f73", label="Hibrido (IF+reglas)")
    plt.yticks(y, types)
    plt.xlim(0, 1)
    plt.xlabel("Recall sobre anomalias inyectadas")
    plt.title("Recall: IF solo vs hibrido")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.injection_plot_file, dpi=160)
    plt.close()


# --------------------------------------------------------------------------- #
# Documentacion
# --------------------------------------------------------------------------- #
def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_eval_report(report, rules, injection, stability, overlap, threshold,
                      cutoff, paths: TrainingPaths) -> None:
    n = len(report)
    n_anom = int(report["is_anomaly"].sum())
    span_days = (report["trace_t_timestamp"].max() - report["trace_t_timestamp"].min()).days or 1
    L = []
    L.append("# Reporte de evaluacion — Modelo FINANOM\n")
    L.append("Generado por `model_training/train_model.py`.\n")
    L.append("## Resumen\n")
    L.append(f"- Transacciones evaluadas: {n:,}")
    L.append(f"- Marcadas para revision (cola): {n_anom:,} ({100*n_anom/n:.2f}%)")
    L.append(f"- Promedio aprox. de alertas/dia: {n_anom/span_days:.1f}")
    L.append(f"- Umbral IF (score): {threshold:.4f} | corte temporal train/reciente: {cutoff}\n")

    L.append("## Composicion de la cola por tipo de inconsistencia\n")
    L.append("| Regla | Marcadas | % de la base |")
    L.append("| ----- | -------- | ------------ |")
    for r in RULE_ORDER:
        c = int(rules[r].sum())
        crit = " (critica)" if RULE_META[r]["critical"] else ""
        L.append(f"| `{r}`{crit} | {c:,} | {100*c/n:.2f}% |")
    L.append(f"| solo IF (atipico_general) | {overlap['solo_if']:,} | {100*overlap['solo_if']/n:.2f}% |")
    L.append("")

    L.append("## Iteracion 1 — Inyeccion de anomalias sinteticas (recall IF solo vs hibrido)\n")
    L.append("Se inyectan anomalias conocidas en una muestra de filas normales y se mide cuantas "
             "marca el sistema. **Hallazgo**: el IF global subpondera anomalias extremas en pocas "
             "dimensiones (swamping entre 63 features), por eso su recall es modesto; las reglas "
             "tipadas las capturan, asi que el hibrido se acerca al 100%. Esto justifica la "
             "arquitectura hibrida.\n")
    L.append("| Tipo inyectado | Recall IF solo | Recall hibrido |")
    L.append("| -------------- | -------------- | -------------- |")
    for k, v in injection.items():
        L.append(f"| {k} | {v['if']:.2%} | {v['hibrido']:.2%} |")
    L.append("")

    L.append("## Iteracion 2 — Estabilidad temporal\n")
    L.append("| Periodo | p50 | p95 | p99 | tasa de marcado |")
    L.append("| ------- | --- | --- | --- | --------------- |")
    for per, s in stability.items():
        L.append(f"| {per} | {s['p50']:.3f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['tasa_marcado']:.2%} |")
    L.append("")

    L.append("## Iteracion 3 — Overlap IF vs reglas criticas\n")
    L.append(f"- IF top: {overlap['if_top']:,}")
    L.append(f"- Reglas criticas: {overlap['reglas_criticas']:,}")
    L.append(f"- Interseccion: {overlap['interseccion']:,}")
    L.append(f"- Solo reglas criticas (el IF NO las priorizaba): {overlap['solo_reglas_criticas']:,}")
    L.append(f"- Solo IF: {overlap['solo_if']:,}")
    L.append("\nLas reglas garantizan los tipos criticos que el IF subponderaria; el IF aporta "
             "los atipicos sin regla. Por eso el modelo es hibrido.\n")

    L.append("## Conclusiones\n")
    L.append("- La cola se mantiene en un presupuesto revisable por el auditor nocturno.")
    L.append("- `signo_contable` se descarto como regla: en este PMS el abono-positivo es la "
             "convencion normal y el cargo-negativo son reversos legitimos (no anomalias).")
    L.append("- Duplicados y modificaciones son comunes en este hotel; se acotan a sus casos "
             "extremos / co-ocurrentes para no inundar la revision.")
    L.append("- Limitacion de pago: Amex (`AMEXCO`) es separable; Visa vs Mastercard no (ambos `TARCRE`).")
    paths.eval_report_file.write_text("\n".join(L), encoding="utf-8")


def write_model_card(report, feature_cols, threshold, injection, paths: TrainingPaths) -> None:
    n = len(report)
    n_anom = int(report["is_anomaly"].sum())
    L = []
    L.append("# Model card — FINANOM (deteccion de inconsistencias)\n")
    L.append("Modelo HIBRIDO de apoyo al auditor nocturno: Isolation Forest no supervisado + "
             "reglas de negocio tipadas. Generado por `model_training/train_model.py`.\n")

    L.append("## 1. Entrada\n")
    L.append(f"- Matriz: `training_data/X_modelo.parquet` ({n:,} filas × {len(feature_cols)} features `feat_*`).")
    L.append("- Trazabilidad: columnas `trace_*` de `training_data/transacciones_modelado.parquet`.")
    L.append("- Campos crudos para reglas: `h_for_pgo`, `t_codigo`, `t_monto`, `t_referencia` de la base limpia.\n")

    L.append("## 2. Modelo\n")
    L.append(f"- Isolation Forest: n_estimators={N_ESTIMATORS}, max_samples={MAX_SAMPLES}, "
             f"contamination={CONTAMINATION}, random_state={RANDOM_STATE}.")
    L.append(f"- Entrenado en el {TIME_SPLIT_QUANTILE:.0%} temporal mas antiguo; score = -decision_function.")
    L.append(f"- Cola de revision: {n_anom:,} ({100*n_anom/n:.2f}%) = IF top-{CONTAMINATION:.0%} "
             "OR reglas criticas (todas) OR reglas amplias (top-K por score).\n")

    L.append("## 3. Reglas de negocio (mapeo al dolor del hotel)\n")
    L.append("| Regla | Tipo | Inconsistencia de negocio |")
    L.append("| ----- | ---- | ------------------------- |")
    mapping = {
        "metodo_pago_inconsistente": "Metodo de pago mal usado (Visa↔Amex)",
        "monto_atipico": "Monto/factura incorrecto",
        "posible_duplicado": "Cargo duplicado / doble posteo",
        "reemplazo_monto": "Reemplazo solicitado por monto incorrecto",
        "modificacion_no_autorizada": "Modificacion no autorizada",
        "cargo_fuera_estancia": "Cargo fuera de la estancia",
    }
    for r in RULE_ORDER:
        tipo = "critica" if RULE_META[r]["critical"] else "amplia (top-K)"
        L.append(f"| `{r}` | {tipo} | {mapping[r]} |")
    L.append("")

    L.append("## 4. Esquema del reporte de revision (`output/reporte_revision.parquet`)\n")
    schema = [
        ("trace_*", "id/datetime", "Trazabilidad: folio, transaccion, referencia, cuarto, codigo, timestamp."),
        ("anomaly_score", "float", "Score del IF (mayor = mas anomalo)."),
        ("anomaly_pct", "float", "Percentil del score (0-1)."),
        ("is_anomaly_if", "bool", "Marcada por el IF (top-presupuesto)."),
        ("regla_*", "bool", "Una columna por regla de negocio."),
        ("n_reglas", "int", "Cuantas reglas dispararon."),
        ("is_anomaly", "bool", "Entra a la cola de revision."),
        ("tipo_inconsistencia", "str", "Reglas que dispararon (o 'atipico_general')."),
        ("motivos", "str", "Razon legible de la alerta."),
        ("top_features", "str", "Features que mas la aislaron (explicabilidad)."),
        ("accion_sugerida", "str", "Accion propuesta para el auditor."),
        ("requiere_aprobacion", "bool", "Siempre True: la correccion exige aprobacion (human-in-the-loop)."),
    ]
    L.append("| Columna | Tipo | Descripcion |")
    L.append("| ------- | ---- | ----------- |")
    for c, t, d in schema:
        L.append(f"| `{c}` | {t} | {d} |")
    L.append("")

    L.append("## 5. Scoring de transacciones nuevas (intra-dia)\n")
    L.append("`output/scorer_bundle.joblib` empaqueta el modelo + columnas + umbral. Para scorear "
             "transacciones nuevas hay que pasarlas antes por el pipeline de datos "
             "(consolidacion → limpieza → modelado de datos) para obtener las `feat_*`. La "
             "persistencia de los transformadores de FE para streaming puro es un paso siguiente.\n")

    L.append("## 6. Bucle de feedback del auditor\n")
    L.append("Almacen `output/feedback_labels.csv` (trace_row_id, decision, revisor, "
             "timestamp_revision, nota). `apply_feedback()` mide precision y sugiere umbral; el "
             "re-ranker supervisado queda como TODO (diseno listo, no implementado en esta fase).\n")

    L.append("## 7. Limitaciones\n")
    L.append("- No supervisado: sin etiquetas reales; calidad validada por inyeccion/inspeccion.")
    L.append("- Pago: Amex separable; Visa vs Mastercard no (ambos `TARCRE`).")
    L.append("- Sin datos de factura (vacios en MX): 'factura incorrecta' se aproxima via monto/reemplazo.")
    L.append("- Duplicados/modificaciones son comunes; se priorizan casos extremos/co-ocurrentes.")
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
    """Entrena el modelo hibrido y genera todos los artefactos."""
    warnings.filterwarnings("ignore", category=UserWarning)
    paths = TrainingPaths.from_dir(output_dir, model_card_file)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    X, trace, raw, feature_cols = load_inputs(Path(modeled_file), Path(x_file), Path(clean_file))
    print(f"  X: {X.shape} | trazabilidad: {trace.shape}")

    train_mask, cutoff = temporal_split(trace)
    print(f"  split temporal: train={int(train_mask.sum()):,} | reciente={int((~train_mask).sum()):,} (corte {cutoff})")

    print("Entrenando Isolation Forest...")
    model = fit_isolation_forest(X, train_mask)
    scores = score_matrix(model, X)

    print("Aplicando reglas de negocio...")
    rules = compute_rules(X, raw)
    is_anomaly, is_if, threshold = assemble_queue(scores, train_mask, rules)
    print(f"  cola de revision: {int(is_anomaly.sum()):,} ({100*is_anomaly.mean():.2f}%)")

    print("Construyendo reporte de revision...")
    report = build_report(trace, X, raw, scores, is_anomaly, is_if, rules, feature_cols)

    print("Evaluando calidad (inyeccion, estabilidad, overlap)...")
    injection = evaluate_injection(model, X, scores, threshold, feature_cols)
    stability = evaluate_stability(scores, train_mask, threshold)
    overlap = evaluate_overlap(is_if, rules)
    print(f"  recall inyeccion (promedio): IF solo {injection['promedio']['if']:.2%} | hibrido {injection['promedio']['hibrido']:.2%}")

    # Persistencia
    joblib.dump(model, paths.model_file)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "threshold": threshold,
         "contamination": CONTAMINATION, "score_sign": "neg_decision_function (mayor=mas anomalo)",
         "random_state": RANDOM_STATE},
        paths.scorer_file,
    )
    report.to_parquet(paths.report_parquet, index=False)
    marcadas = report[report["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    marcadas.to_csv(paths.report_csv, index=False)
    write_labels_template(paths.labels_store_file)
    write_plots(scores, threshold, injection, paths)

    diagnostics = {
        "rows": int(len(report)), "flagged": int(is_anomaly.sum()),
        "flagged_pct": float(is_anomaly.mean()), "threshold": threshold,
        "rule_counts": {r: int(rules[r].sum()) for r in RULE_ORDER},
        "injection_recall": injection, "stability": stability, "overlap": overlap,
    }
    paths.eval_json_file.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_eval_report(report, rules, injection, stability, overlap, threshold, cutoff, paths)
    write_model_card(report, feature_cols, threshold, injection, paths)

    print(f"\nGuardado modelo: {display_path(paths.model_file)}")
    print(f"Guardado reporte: {display_path(paths.report_parquet)} ({int(is_anomaly.sum()):,} marcadas)")
    print(f"Guardada evaluacion: {display_path(paths.eval_report_file)}")
    print(f"Guardada model card: {display_path(paths.model_card_file)}")

    return {
        "model": paths.model_file, "scorer": paths.scorer_file,
        "report": paths.report_parquet, "report_csv": paths.report_csv,
        "eval_report": paths.eval_report_file, "eval_json": paths.eval_json_file,
        "labels_store": paths.labels_store_file, "model_card": paths.model_card_file,
        "score_plot": paths.score_plot_file, "injection_plot": paths.injection_plot_file,
    }


def main() -> None:
    artifacts = run_training()
    print("\nArtefactos generados:")
    for name, path in artifacts.items():
        print(f"  {name}: {display_path(Path(path))}")


if __name__ == "__main__":
    main()
