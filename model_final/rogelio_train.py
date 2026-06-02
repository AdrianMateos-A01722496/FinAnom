"""
Entrenamiento del modelo de deteccion de anomalias FINANOM.

Pipeline:
1. Carga X_modelo.parquet + columnas trace de trazabilidad
2. Muestra estratificada representativa para entrenamiento (~200k filas)
3. Entrena Isolation Forest
4. Puntua las 1.1M filas completas
5. Determina umbral desde distribucion de scores
6. Calcula SHAP para transacciones flaggeadas
7. Guarda modelo, parquets puntuados, plots y findings.md

Uso:
    uv run python model_final/rogelio_train.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
HERE       = Path(__file__).resolve().parent
X_FILE     = HERE / "training_data" / "X_modelo.parquet"
FULL_FILE  = HERE / "training_data" / "transacciones_modelado.parquet"

ARTIFACTS_DIR = HERE / "artifacts"
OUTPUT_DIR    = HERE / "output"
FINDINGS_FILE = HERE / "findings.md"

TRACE_COLS = [
    "trace_row_id", "trace_t_folio", "trace_t_folio_ext",
    "trace_t_referencia", "trace_t_transaccion", "trace_t_cve_res",
    "trace_t_cuarto", "trace_t_codigo", "trace_t_timestamp",
]

# ── Hiperparametros ────────────────────────────────────────────────────────
RANDOM_STATE      = 42
TRAIN_SAMPLE_SIZE = 200_000
N_ESTIMATORS      = 200     # mas arboles = scores mas estables que el proxy (100)
N_JOBS            = -1

STRAT_COLS = [
    "feat_es_abono",
    "feat_cargo_cancelado",
    "feat_dup_mismo_dia_flag",
    "feat_es_renta",
]


# ── Carga ──────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Cargando X_modelo.parquet ...")
    X_raw = pd.read_parquet(X_FILE)
    feat_cols = [c for c in X_raw.columns if c.startswith("feat_")]
    X = X_raw[feat_cols].copy()
    print(f"  {X.shape[0]:,} filas x {X.shape[1]} features")

    print("Cargando columnas de trazabilidad ...")
    trace = pd.read_parquet(FULL_FILE, columns=TRACE_COLS)
    return X, trace


# ── Muestra estratificada ──────────────────────────────────────────────────
def stratified_sample(X: pd.DataFrame, n: int = TRAIN_SAMPLE_SIZE) -> pd.DataFrame:
    """Muestra proporcional por combinacion de flags binarios clave."""
    rng = np.random.default_rng(RANDOM_STATE)
    strat_key = X[STRAT_COLS].astype(str).agg("_".join, axis=1)
    counts = strat_key.value_counts()

    frames: list[pd.DataFrame] = []
    for stratum, cnt in counts.items():
        idx = strat_key[strat_key == stratum].index
        n_take = max(1, round(n * cnt / len(X)))
        n_take = min(n_take, len(idx))
        chosen = rng.choice(idx, n_take, replace=False)
        frames.append(X.loc[chosen])

    sample = pd.concat(frames)
    if len(sample) > n:
        sample = sample.sample(n, random_state=RANDOM_STATE)

    print(f"  Muestra: {len(sample):,} filas en {len(counts)} estratos")
    return sample


# ── Entrenamiento ──────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame) -> IsolationForest:
    print(f"Entrenando IsolationForest (n_estimators={N_ESTIMATORS}) ...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples="auto",       # sklearn default: min(256, n_samples)
        contamination="auto",     # umbral determinado post-hoc desde distribucion
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train.values)
    print("  Modelo entrenado.")
    return model


# ── Puntuacion ─────────────────────────────────────────────────────────────
def score_all(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    print(f"Puntuando {len(X):,} filas ...")
    scores = model.score_samples(X.values)   # menor = mas anomalo
    print(
        f"  min={scores.min():.4f}  "
        f"p50={np.percentile(scores, 50):.4f}  "
        f"max={scores.max():.4f}"
    )
    return scores


# ── Umbral ─────────────────────────────────────────────────────────────────
def pick_threshold(
    scores: np.ndarray,
    min_pct: float = 0.5,
    max_pct: float = 5.0,
) -> tuple[float, float, str]:
    """
    Busca el mayor salto de score en el rango [min_pct, max_pct] de la cola.
    Requiere que el umbral flaggee entre min_pct y max_pct del corpus.
    Fallback: percentil 2.
    Returns (threshold, pct_flagged, rationale).
    """
    sorted_asc = np.sort(scores)
    n = len(sorted_asc)
    lo = int(min_pct / 100 * n)
    hi = int(max_pct / 100 * n)

    if hi - lo > 20:
        search = sorted_asc[:hi]
        diffs = np.diff(search)
        # Solo considerar posiciones que resultan en al menos min_pct flaggeados
        diffs_window = diffs[lo:hi]
        if len(diffs_window) > 0:
            local_idx = int(np.argmax(diffs_window))
            global_idx = local_idx + lo
            gap = float(diffs[global_idx])
            pos_diffs = diffs[diffs > 0]
            median_diff = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0

            if median_diff > 0 and gap > 2.5 * median_diff:
                threshold = float(search[global_idx + 1])
                pct = float(np.mean(scores <= threshold) * 100)
                rationale = (
                    f"Mayor salto en rango [{min_pct}%–{max_pct}%] "
                    f"(gap={gap:.4f} vs mediana_diff={median_diff:.4f})"
                )
                print(f"  Umbral por salto: {threshold:.4f} -> {pct:.2f}% anomalos")
                return threshold, pct, rationale

    threshold = float(np.percentile(scores, 2))
    pct = float(np.mean(scores <= threshold) * 100)
    rationale = "Percentil 2 (sin salto significativo en rango 0.5%–5%)"
    print(f"  Umbral por p2: {threshold:.4f} -> {pct:.2f}% anomalos")
    return threshold, pct, rationale


# ── SHAP ───────────────────────────────────────────────────────────────────
def compute_shap(
    model: IsolationForest,
    X: pd.DataFrame,
    flag: np.ndarray,
) -> pd.DataFrame:
    """SHAP (TreeExplainer) para las filas anomalas."""
    X_anom = X[flag]
    print(f"Calculando SHAP para {len(X_anom):,} anomalias ...")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_anom.values, check_additivity=False)

    shap_df = pd.DataFrame(
        raw,
        index=X_anom.index,
        columns=[f"shap_{c}" for c in X_anom.columns],
    )
    base = float(np.ravel(explainer.expected_value)[0])
    print(f"  SHAP listo. Base value: {base:.4f}")
    return shap_df


# ── Plots ──────────────────────────────────────────────────────────────────
def plot_score_distribution(scores: np.ndarray, threshold: float, pct: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(scores, bins=300, color="steelblue", alpha=0.7, edgecolor="none")
    axes[0].axvline(
        threshold, color="crimson", lw=2,
        label=f"Umbral = {threshold:.4f}\n({pct:.1f}% anomalos)",
    )
    axes[0].set_xlabel("Anomaly Score (score_samples)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribucion de scores — corpus completo")
    axes[0].legend()

    # Zoom cola anomala (peor 5%)
    p5 = np.percentile(scores, 5)
    tail_scores = scores[scores <= p5]
    axes[1].hist(tail_scores, bins=150, color="coral", alpha=0.8, edgecolor="none")
    axes[1].axvline(threshold, color="crimson", lw=2, label=f"Umbral = {threshold:.4f}")
    axes[1].set_xlabel("Anomaly Score (peor 5%)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Zoom cola anomala")
    axes[1].legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_shap_bar(shap_df: pd.DataFrame, top_n: int = 20) -> None:
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    labels = [c.replace("shap_feat_", "") for c in mean_abs.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), mean_abs.values, color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} features por importancia SHAP (anomalias)")
    plt.tight_layout()
    path = OUTPUT_DIR / "shap_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_shap_summary(
    shap_df: pd.DataFrame, X: pd.DataFrame, flag: np.ndarray, top_n: int = 20
) -> None:
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    top_feat = [c.replace("shap_", "") for c in mean_abs.index]
    top_shap = list(mean_abs.index)

    X_anom = X[flag][top_feat]
    # Sample para que el beeswarm no sea demasiado pesado
    max_pts = min(2_000, len(X_anom))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_anom), max_pts, replace=False)

    shap_vals = shap_df[top_shap].values[sample_idx]
    feat_vals = X_anom.values[sample_idx]

    shap.summary_plot(
        shap_vals,
        feat_vals,
        feature_names=[c.replace("feat_", "") for c in top_feat],
        show=False,
        max_display=top_n,
        plot_size=(10, 8),
    )
    plt.title("SHAP Summary — muestra de anomalias (beeswarm)")
    plt.tight_layout()
    path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path.name}")


def plot_anomaly_by_hour(trace: pd.DataFrame, flag: np.ndarray) -> None:
    ts = pd.to_datetime(trace["trace_t_timestamp"])
    hours = range(24)
    total_by_hour = ts.dt.hour.value_counts().reindex(hours, fill_value=0)
    anom_by_hour  = ts[flag].dt.hour.value_counts().reindex(hours, fill_value=0)
    rate = (anom_by_hour / total_by_hour.replace(0, np.nan) * 100).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(hours, total_by_hour.values, color="steelblue", alpha=0.6, label="Total")
    axes[0].bar(hours, anom_by_hour.values, color="crimson", alpha=0.8, label="Anomalias")
    axes[0].set_xlabel("Hora del dia")
    axes[0].set_ylabel("Transacciones")
    axes[0].set_title("Transacciones por hora")
    axes[0].set_xticks(hours)
    axes[0].legend()

    axes[1].bar(hours, rate.values, color="coral")
    axes[1].set_xlabel("Hora del dia")
    axes[1].set_ylabel("% anomalas")
    axes[1].set_title("Tasa de anomalia por hora")
    axes[1].set_xticks(hours)

    plt.tight_layout()
    path = OUTPUT_DIR / "anomaly_by_hour.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_anomaly_by_codigo(trace: pd.DataFrame, flag: np.ndarray, top_n: int = 20) -> None:
    total  = trace["trace_t_codigo"].value_counts()
    anom   = trace.loc[flag, "trace_t_codigo"].value_counts()
    common = total[total >= 100].index
    rate   = (anom / total * 100).fillna(0)[common].sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(rate)), rate.values, color="coral", alpha=0.85)
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels(rate.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% Anomalos")
    ax.set_title(f"Top {top_n} codigos por tasa de anomalia (min. 100 transacciones)")
    plt.tight_layout()
    path = OUTPUT_DIR / "anomaly_by_codigo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── Construccion de parquets de salida ─────────────────────────────────────
def build_scored_parquet(
    trace: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    flag = scores <= threshold
    # rank 1 = mas anomalo (score mas bajo)
    rank = pd.Series(scores).rank(method="first", ascending=True).astype("int32")
    pct  = pd.Series(scores).rank(pct=True).mul(100).astype("float32")

    result = trace.copy()
    result["anomaly_score"]            = scores.astype("float32")
    result["anomaly_score_percentile"] = pct.values
    result["anomaly_rank"]             = rank.values
    result["anomaly_flag"]             = flag
    return result


def save_artifacts(
    model: IsolationForest,
    scored: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "isolation_forest.joblib"
    joblib.dump(model, model_path)
    print(f"  Modelo: {model_path.name}")

    scored_path = ARTIFACTS_DIR / "scored_transactions.parquet"
    scored.to_parquet(scored_path, index=False)
    print(f"  Scored: {scored_path.name}  ({len(scored):,} filas)")

    # SHAP: join con columnas trace + score del subconjunto anomalo
    anom_scored = scored[scored["anomaly_flag"]].reset_index(drop=True)
    shap_reset  = shap_df.reset_index(drop=True)
    shap_full   = pd.concat([anom_scored, shap_reset], axis=1)
    shap_path   = ARTIFACTS_DIR / "shap_anomalies.parquet"
    shap_full.to_parquet(shap_path, index=False)
    print(f"  SHAP:   {shap_path.name}  ({len(shap_full):,} anomalias)")


# ── Findings ───────────────────────────────────────────────────────────────
def generate_findings(
    X: pd.DataFrame,
    trace: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    flag: np.ndarray,
    flagged_pct: float,
    rationale: str,
    shap_df: pd.DataFrame,
) -> None:
    import sklearn
    n_total = len(scores)
    n_anom  = int(flag.sum())
    X_anom  = X[flag]

    # Score percentiles
    pct_rows = "\n".join(
        f"| p{p} | {np.percentile(scores, p):.4f} |"
        for p in [0.5, 1, 2, 3, 5, 10, 50]
    )

    # SHAP top 15
    mean_abs  = shap_df.abs().mean().sort_values(ascending=False).head(15)
    shap_rows = "\n".join(
        f"| {i+1} | `{c.replace('shap_feat_', 'feat_')}` | {v:.4f} |"
        for i, (c, v) in enumerate(mean_abs.items())
    )

    # Temporales
    ts        = pd.to_datetime(trace["trace_t_timestamp"])
    hours     = range(24)
    total_h   = ts.dt.hour.value_counts().reindex(hours, fill_value=0)
    anom_h    = ts[flag].dt.hour.value_counts().reindex(hours, fill_value=0)
    rate_h    = (anom_h / total_h.replace(0, np.nan) * 100).fillna(0)
    peak_hour = int(rate_h.idxmax())
    peak_rate = float(rate_h.max())

    # Codigos top
    total_c  = trace["trace_t_codigo"].value_counts()
    anom_c   = trace.loc[flag, "trace_t_codigo"].value_counts()
    common_c = total_c[total_c >= 100].index
    rate_c   = (anom_c / total_c * 100).fillna(0)[common_c].sort_values(ascending=False).head(10)
    codigo_rows = "\n".join(
        f"| `{cod}` | {int(total_c[cod]):,} | {int(anom_c.get(cod, 0)):,} | {rate:.1f}% |"
        for cod, rate in rate_c.items()
    )

    # Indicadores operativos
    def rate(col: str, subset: pd.DataFrame = X) -> str:
        return f"{subset[col].mean() * 100:.1f}%"

    md = f"""\
# Hallazgos del Modelo de Anomalias — FINANOM

> Generado por `model_final/rogelio_train.py`

---

## 1. Metodologia

**Algoritmo:** Isolation Forest (scikit-learn {sklearn.__version__})

Isolation Forest construye arboles de aislamiento aleatorios; las muestras que se aíslan en pocas particiones tienen scores mas bajos (mas anomalos). Ventajas para este problema:

- Escala a millones de filas (complejidad O(n log n))
- No asume distribucion de los datos — adecuado para la mezcla de flags binarios, frecuencias y magnitudes escaladas
- Compatible con SHAP TreeExplainer para explicaciones por transaccion
- Alternativas descartadas: LOF (O(n²)), OCSVM (no escala >100k filas), autoencoders (requiere infraestructura adicional)

**Hiperparametros finales:**

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `n_estimators` | {N_ESTIMATORS} | Doble del proxy (100); scores mas estables |
| `max_samples` | auto (256) | Default sklearn; suficiente para capturar estructura local |
| `contamination` | auto | Umbral determinado post-hoc desde distribucion |

---

## 2. Muestra de entrenamiento

| Item | Valor |
|------|-------|
| Corpus total | {n_total:,} transacciones |
| Muestra de entrenamiento | {TRAIN_SAMPLE_SIZE:,} filas estratificadas |
| Scoring aplicado sobre | {n_total:,} filas (corpus completo) |

**Estratificacion:** proporcional por combinacion de `feat_es_abono`, `feat_cargo_cancelado`, `feat_dup_mismo_dia_flag`, `feat_es_renta`. Garantiza representacion de tipos de transaccion raros en la muestra de entrenamiento.

---

## 3. Distribucion de scores

> Ver: `output/score_distribution.png`

Scores de `score_samples` — valores mas negativos indican mayor anomalia.

| Percentil | Score |
|-----------|-------|
{pct_rows}

La mayoria de transacciones normales se concentra en el rango superior; la cola izquierda es larga y dispersa, tipica de datos financieros con eventos raros.

---

## 4. Umbral seleccionado

| Item | Valor |
|------|-------|
| Umbral | `{threshold:.4f}` |
| Criterio | {rationale} |
| Transacciones flaggeadas | {n_anom:,} ({flagged_pct:.2f}% del total) |

---

## 5. Anomalias detectadas: patrones clave

### Indicadores operativos — anomalias vs. total

| Indicador | Total | Solo anomalias |
|-----------|-------|----------------|
| Duplicados mismo dia | {rate("feat_dup_mismo_dia_flag")} | {rate("feat_dup_mismo_dia_flag", X_anom)} |
| Cargos cancelados | {rate("feat_cargo_cancelado")} | {rate("feat_cargo_cancelado", X_anom)} |
| Cargo fuera de estancia | {rate("feat_cargo_fuera_estancia")} | {rate("feat_cargo_fuera_estancia", X_anom)} |
| Transacciones de madrugada | {rate("feat_es_madrugada")} | {rate("feat_es_madrugada", X_anom)} |
| Monto negativo sin abono | {rate("feat_monto_negativo_sin_abono")} | {rate("feat_monto_negativo_sin_abono", X_anom)} |
| Modificado por usuario distinto | {rate("feat_usuario_mod_distinto")} | {rate("feat_usuario_mod_distinto", X_anom)} |

### Top 10 codigos por tasa de anomalia (minimo 100 transacciones)

> Ver: `output/anomaly_by_codigo.png`

| Codigo | Total | Anomalias | Tasa |
|--------|-------|-----------|------|
{codigo_rows}

### Patrones temporales

> Ver: `output/anomaly_by_hour.png`

- **Hora pico de anomalias:** {peak_hour:02d}:00 h — {peak_rate:.1f}% de transacciones en esa hora son anomalas

---

## 6. Importancia de features — SHAP

> Ver: `output/shap_bar.png` · `output/shap_summary.png`

**Interpretacion del valor SHAP:**
- **Negativo** → la feature empuja el score hacia abajo (mas anomalo)
- **Positivo** → la feature empuja el score hacia arriba (mas normal)
- **|SHAP| alto** → la feature tiene gran influencia en ese score, en cualquier direccion

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
{shap_rows}

---

## 7. Artefactos generados

| Archivo | Descripcion |
|---------|-------------|
| `artifacts/isolation_forest.joblib` | Modelo serializado (cargable con `joblib.load`) |
| `artifacts/scored_transactions.parquet` | {n_total:,} filas con `anomaly_score`, `anomaly_rank`, `anomaly_flag`, `anomaly_score_percentile` |
| `artifacts/shap_anomalies.parquet` | {n_anom:,} anomalias con columnas `shap_*` para cada feature |
| `output/score_distribution.png` | Histograma completo y zoom cola anomala con umbral marcado |
| `output/shap_bar.png` | Top 20 features por mean |SHAP| en anomalias |
| `output/shap_summary.png` | Beeswarm SHAP: distribucion de contribuciones por feature y valor |
| `output/anomaly_by_hour.png` | Conteo y tasa de anomalia por hora del dia |
| `output/anomaly_by_codigo.png` | Tasa de anomalia por codigo contable (top 20) |

---

## 8. Como explorar los resultados

```python
import pandas as pd

# Cargar todas las transacciones con score
scored = pd.read_parquet("model_final/artifacts/scored_transactions.parquet")

# Ver las 50 mas anomalas
top = scored.sort_values("anomaly_rank").head(50)

# Cargar anomalias con SHAP
shap_anom = pd.read_parquet("model_final/artifacts/shap_anomalies.parquet")

# Para una transaccion especifica, ver que features la hicieron anomala
tx = shap_anom[shap_anom["trace_t_transaccion"] == "TU_ID"]
shap_cols = [c for c in tx.columns if c.startswith("shap_")]
tx[shap_cols].T.sort_values(by=tx.index[0])
```

**Flujo recomendado para auditoria:**
1. Ordenar `scored_transactions.parquet` por `anomaly_rank` (rank 1 = mas anomalo)
2. Priorizar transacciones con `anomaly_flag=True` y `feat_dup_mismo_dia_flag=1` — candidatos directos a correccion
3. Para cada transaccion flaggeada, revisar columnas `shap_*` en `shap_anomalies.parquet` para entender la causa
4. Atender especialmente la hora pico ({peak_hour:02d}:00 h) y los codigos con mayor tasa de anomalia

---

## 9. Recomendaciones de ajuste

- **Si el umbral parece alto (demasiadas anomalias):** cambiar `TRAIN_SAMPLE_SIZE` o aplicar un filtro adicional por `anomaly_rank <= N`.
- **Para nuevas transacciones:** cargar `isolation_forest.joblib` con `joblib.load` y llamar `model.score_samples(X_nueva)`.
- **Etiquetado futuro:** si el equipo de auditoria valida anomalias, se puede entrenar un clasificador supervisado sobre `shap_anomalies.parquet` usando las columnas `feat_*` y las etiquetas manuales.
"""

    FINDINGS_FILE.write_text(md, encoding="utf-8")
    print(f"  {FINDINGS_FILE.name}")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FINANOM — Entrenamiento modelo anomalias")
    print("=" * 60)

    X, trace = load_data()

    print("\n[1/6] Muestra estratificada ...")
    X_train = stratified_sample(X)

    print("\n[2/6] Entrenamiento ...")
    model = train_model(X_train)

    print("\n[3/6] Puntuacion completa ...")
    scores = score_all(model, X)

    print("\n[4/6] Seleccion de umbral ...")
    threshold, flagged_pct, rationale = pick_threshold(scores)
    flag = scores <= threshold

    print("\n[5/6] SHAP ...")
    shap_df = compute_shap(model, X, flag)

    print("\n[6/6] Guardando artefactos ...")
    scored = build_scored_parquet(trace, scores, threshold)
    save_artifacts(model, scored, shap_df)

    print("\nGenerando plots ...")
    plot_score_distribution(scores, threshold, flagged_pct)
    plot_shap_bar(shap_df)
    plot_shap_summary(shap_df, X, flag)
    plot_anomaly_by_hour(trace, flag)
    plot_anomaly_by_codigo(trace, flag)

    print("\nGenerando findings.md ...")
    generate_findings(X, trace, scores, threshold, flag, flagged_pct, rationale, shap_df)

    print(f"\n{'='*60}")
    print(f"Listo. Anomalias detectadas: {flag.sum():,} ({flagged_pct:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
