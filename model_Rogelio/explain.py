"""
Pipeline de explicabilidad FINANOM.

Lee shap_anomalies.parquet y produce anomalies_explained.parquet con:
  - severity       : Alta / Media / Informativa
  - categories     : categorias de anomalia detectadas (separadas por ' | ')
  - explanation    : texto principal para el auditor (espanol, lenguaje contable)
  - reasons_json   : lista completa de razones ordenadas por impacto SHAP

Logica de explicabilidad:
  Para cada anomalia se evaluan ~20 generadores de razon, uno por grupo
  de features. Cada generador revisa si la feature contribuye a la anomalia
  (|SHAP| significativo) y si el valor de la feature soporta la interpretacion.
  Las razones resultantes se ordenan por |SHAP| y se deduplican por categoria.

Uso:
    uv run python model_Rogelio/explain.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FULL_FILE    = ROOT / "training_data" / "transacciones_modelado.parquet"
HERE         = Path(__file__).resolve().parent
SHAP_FILE    = HERE / "artifacts" / "shap_anomalies.parquet"
OUT_FILE     = HERE / "artifacts" / "anomalies_explained.parquet"
PREVIEW_FILE = HERE / "artifacts" / "anomalies_explained_preview.csv"

# Features a cargar de transacciones_modelado para generar las explicaciones
FEAT_NEEDED = [
    "feat_dup_mismo_dia_flag", "feat_dup_mismo_minuto_flag", "feat_dup_mismo_dia_log",
    "feat_es_abono", "feat_cargo_cancelado", "feat_cancelacion_sin_marca",
    "feat_es_split", "feat_es_renta", "feat_tiene_reservacion",
    "feat_monto_negativo_sin_abono", "feat_monto_positivo_en_abono",
    "feat_usuario_modificado", "feat_usuario_mod_distinto",
    "feat_es_madrugada", "feat_es_fin_semana",
    "feat_cargo_antes_llegada", "feat_cargo_fuera_estancia",
    "feat_monto_abs_log",
    "feat_monto_z_codigo_carabo", "feat_impuesto_z_codigo_carabo",
    "feat_impuesto_ratio_abs", "feat_propina_ratio_abs",
    "feat_monto_vs_tarifa_ratio", "feat_monto_vs_tarifa_total_ratio",
    "feat_folio_codigo_dia_count_log", "feat_folio_dia_movimientos_log",
    "feat_folio_total_movimientos_log",
    "feat_hora_sin", "feat_hora_cos",
    "feat_noches_delta_scaled",
    "feat_dias_desde_llegada_scaled", "feat_dias_hasta_salida_scaled",
    "feat_t_codigo_freq", "feat_t_usuario_freq", "feat_h_for_pgo_freq",
    "feat_obs_kw_ajuste", "feat_obs_kw_cancelacion", "feat_obs_kw_error",
    "feat_obs_kw_cortesia", "feat_obs_kw_reembolso",
    "feat_obs_missing", "feat_obs_len_log",
]

# Umbral minimo de |SHAP| para considerar una feature relevante en la explicacion
SHAP_MIN = 0.02

# Severidad basada en anomaly_rank global (rank 1 = mas anomalo de 1.1M transacciones).
# Las 6,126 anomalias ocupan los ranks 1–6,126.
# Alta   → rank <= 1,200  (~20% de anomalias, accion urgente)
# Media  → rank <= 4,000  (~46% de anomalias, revision en el turno)
# Informativa → resto      (~34% de anomalias, documentacion)
RANK_ALTA   = 1_200
RANK_MEDIA  = 4_000


# ── Tipo interno ───────────────────────────────────────────────────────────
Reason = tuple[str, str, float]  # (categoria, texto, shap_abs)


# ══════════════════════════════════════════════════════════════════════════════
# Carga de datos
# ══════════════════════════════════════════════════════════════════════════════

def load_concept_stats() -> dict[str, dict]:
    """
    Estadisticas de monto (log1p) por codigo de concepto.
    Necesarias para decir 'el monto es 3x el habitual del concepto X'.
    Solo carga 2 columnas del archivo completo.
    """
    print("  Cargando estadisticas de concepto ...")
    raw = pd.read_parquet(
        FULL_FILE,
        columns=["trace_t_codigo", "feat_monto_abs_log"],
    )
    grp = raw.groupby("trace_t_codigo")["feat_monto_abs_log"]
    stats_df = pd.DataFrame({
        "log_median": grp.median(),
        "log_q25":    grp.quantile(0.25),
        "log_q75":    grp.quantile(0.75),
        "count":      grp.count(),
    })
    stats_df["freq_pct"] = stats_df["count"] / stats_df["count"].sum() * 100
    return stats_df.to_dict("index")


def load_data(stats: dict) -> pd.DataFrame:
    """
    Carga shap_anomalies.parquet y le agrega feat_* para cada fila
    usando trace_row_id como indice posicional en transacciones_modelado.
    """
    print("  Cargando shap_anomalies.parquet ...")
    shap_df = pd.read_parquet(SHAP_FILE)

    print("  Cargando feat_* para las anomalias ...")
    full_feat = pd.read_parquet(FULL_FILE, columns=FEAT_NEEDED)
    row_ids = shap_df["trace_row_id"].values
    feat_subset = full_feat.iloc[row_ids].reset_index(drop=True)

    df = pd.concat([shap_df.reset_index(drop=True), feat_subset], axis=1)
    print(f"  Dataset listo: {len(df):,} anomalias x {df.shape[1]} columnas")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Generadores de razon
# Cada funcion recibe la fila (dict) y el stats dict.
# Retorna Reason o None.
# ══════════════════════════════════════════════════════════════════════════════

def _shap(row: dict, feat: str) -> float:
    return abs(row.get(f"shap_{feat}", 0.0))


def _r_dup_minuto(row: dict, stats: dict) -> Reason | None:
    feat = "feat_dup_mismo_minuto_flag"
    if row.get(feat, 0) != 1 or _shap(row, feat) < SHAP_MIN:
        return None
    return (
        "Posible duplicado",
        "Cargo identico (mismo concepto, monto y folio) registrado en el mismo minuto "
        "— posible doble posteo",
        _shap(row, feat),
    )


def _r_dup_dia(row: dict, stats: dict) -> Reason | None:
    feat = "feat_dup_mismo_dia_flag"
    if row.get(feat, 0) != 1 or _shap(row, feat) < SHAP_MIN:
        return None
    raw_log = row.get("feat_dup_mismo_dia_log", 0.0)
    n = max(2, round(np.expm1(raw_log) + 1))
    codigo = row.get("trace_t_codigo", "?")
    return (
        "Posible duplicado",
        f"Se detectaron {n} cargos con el mismo concepto ({codigo}) y monto "
        f"en este folio el mismo dia",
        _shap(row, feat),
    )


def _r_monto_atipico(row: dict, stats: dict) -> Reason | None:
    feat = "feat_monto_z_codigo_carabo"
    shap_abs = _shap(row, feat)
    z = row.get(feat, 0.0)
    if shap_abs < SHAP_MIN or abs(z) < 2.0:
        return None

    codigo = row.get("trace_t_codigo", "?")
    monto = np.expm1(row.get("feat_monto_abs_log", 0.0))
    cstats = stats.get(codigo, {})
    median_monto = np.expm1(cstats.get("log_median", 0.0)) if cstats else None

    if median_monto and median_monto > 0.01:
        mult = monto / median_monto
        if mult >= 1.5:
            texto = (
                f"El monto (${monto:,.0f}) es {mult:.1f}x el habitual "
                f"del concepto {codigo} (${median_monto:,.0f} en promedio)"
            )
        else:
            texto = (
                f"El monto (${monto:,.0f}) es notablemente menor "
                f"al habitual del concepto {codigo} (${median_monto:,.0f} en promedio)"
            )
    else:
        direccion = "superior" if z > 0 else "inferior"
        texto = (
            f"Monto significativamente {direccion} al habitual "
            f"para el concepto {codigo} ({z:+.1f} desviaciones del rango tipico)"
        )
    return ("Monto atipico", texto, shap_abs)


def _r_vs_tarifa(row: dict, stats: dict) -> Reason | None:
    # Usar el ratio vs tarifa diaria (mas interpretable que tarifa total)
    feat = "feat_monto_vs_tarifa_ratio"
    shap_abs = _shap(row, feat)
    ratio = row.get(feat, 0.0)
    if shap_abs < SHAP_MIN or ratio < 2.0:
        return None
    monto = np.expm1(row.get("feat_monto_abs_log", 0.0))
    return (
        "Cargo vs tarifa",
        f"El cargo (${monto:,.0f}) equivale a {ratio:.1f}x la tarifa diaria "
        f"pactada en la reservacion",
        shap_abs,
    )


def _r_noches_delta(row: dict, stats: dict) -> Reason | None:
    feat = "feat_noches_delta_scaled"
    shap_abs = _shap(row, feat)
    if shap_abs < SHAP_MIN or abs(row.get(feat, 0.0)) < 1.5:
        return None
    return (
        "Inconsistencia de reservacion",
        "Las noches del cargo no coinciden con las noches registradas "
        "en la reservacion enlazada",
        shap_abs,
    )


def _r_fuera_estancia(row: dict, stats: dict) -> Reason | None:
    feat = "feat_cargo_fuera_estancia"
    if row.get(feat, 0) != 1 or _shap(row, feat) < SHAP_MIN:
        return None
    if row.get("feat_cargo_antes_llegada", 0) == 1:
        texto = "Cargo fechado antes de la llegada del huesped a la reservacion enlazada"
    else:
        texto = "Cargo registrado despues de la salida del huesped segun la reservacion"
    return ("Fuera de estancia", texto, _shap(row, feat))


def _r_horario(row: dict, stats: dict) -> Reason | None:
    # Madrugada (flag directo)
    feat_mad = "feat_es_madrugada"
    if row.get(feat_mad, 0) == 1 and _shap(row, feat_mad) >= SHAP_MIN:
        ts = row.get("trace_t_timestamp")
        hora_str = pd.to_datetime(ts).strftime("%H:%M") if ts else "00:xx"
        return (
            "Horario inusual",
            f"Cargo registrado a las {hora_str}, durante la madrugada "
            f"(00:00–05:59), fuera del horario operativo habitual",
            _shap(row, feat_mad),
        )
    # Fallback: hora ciclica con SHAP alto, solo si la hora es realmente inusual (antes 6AM o despues 22PM)
    shap_hora = _shap(row, "feat_hora_sin") + _shap(row, "feat_hora_cos")
    if shap_hora >= SHAP_MIN * 4:
        ts = row.get("trace_t_timestamp")
        if ts:
            hora = pd.to_datetime(ts).hour
            if hora < 6 or hora >= 22:
                return (
                    "Horario inusual",
                    f"Cargo registrado a las {hora:02d}:xx h, "
                    f"en horario de baja operacion para este concepto",
                    shap_hora / 2,
                )
    return None


def _r_signo(row: dict, stats: dict) -> Reason | None:
    if row.get("feat_monto_negativo_sin_abono", 0) == 1:
        shap_abs = _shap(row, "feat_monto_negativo_sin_abono")
        if shap_abs >= SHAP_MIN:
            monto = np.expm1(row.get("feat_monto_abs_log", 0.0))
            return (
                "Inconsistencia de signo",
                f"Monto negativo (-${monto:,.0f}) registrado sin marca de abono "
                f"— revisar si es devolucion o error de posteo",
                shap_abs,
            )
    if row.get("feat_monto_positivo_en_abono", 0) == 1:
        shap_abs = _shap(row, "feat_monto_positivo_en_abono")
        if shap_abs >= SHAP_MIN:
            return (
                "Inconsistencia de signo",
                "Monto positivo en movimiento marcado como abono "
                "— verificar la naturaleza del cargo",
                shap_abs,
            )
    return None


def _r_modificacion(row: dict, stats: dict) -> Reason | None:
    feat = "feat_usuario_mod_distinto"
    if row.get(feat, 0) != 1 or _shap(row, feat) < SHAP_MIN:
        return None
    return (
        "Modificacion sospechosa",
        "La transaccion fue modificada por un usuario distinto al cajero original",
        _shap(row, feat),
    )


def _r_cancelacion(row: dict, stats: dict) -> Reason | None:
    # Solo flag de cancelacion explicita (feat_cargo_cancelado=1).
    # feat_cancelacion_sin_marca=1 es el estado normal de transacciones no canceladas
    # (campo NULL en el PMS), no es una senal de anomalia por si sola.
    if row.get("feat_cargo_cancelado", 0) != 1:
        return None
    shap_abs = _shap(row, "feat_cargo_cancelado")
    if shap_abs < SHAP_MIN:
        return None
    return (
        "Cancelacion irregular",
        "Transaccion cancelada con caracteristicas atipicas "
        "para el concepto y monto involucrados — verificar aplicacion del credito",
        shap_abs,
    )


def _r_densidad_concepto(row: dict, stats: dict) -> Reason | None:
    feat = "feat_folio_codigo_dia_count_log"
    shap_abs = _shap(row, feat)
    n = round(np.expm1(row.get(feat, 0.0)))
    if shap_abs < SHAP_MIN or n < 3:
        return None
    codigo = row.get("trace_t_codigo", "?")
    return (
        "Alta densidad de cargos",
        f"Se detectaron {n} cargos del concepto {codigo} en este folio el mismo dia",
        shap_abs,
    )


def _r_densidad_folio(row: dict, stats: dict) -> Reason | None:
    feat = "feat_folio_dia_movimientos_log"
    shap_abs = _shap(row, feat)
    n = round(np.expm1(row.get(feat, 0.0)))
    if shap_abs < SHAP_MIN or n < 20:
        return None
    return (
        "Alta densidad de cargos",
        f"El folio registra {n} transacciones en total este dia "
        f"— actividad inusualmente alta",
        shap_abs,
    )


def _r_impuesto(row: dict, stats: dict) -> Reason | None:
    feat = "feat_impuesto_ratio_abs"
    shap_abs = _shap(row, feat)
    ratio = row.get(feat, 0.0)
    monto = np.expm1(row.get("feat_monto_abs_log", 0.0))
    # Solo relevante si el monto es significativo y el ratio es inusual
    if shap_abs < SHAP_MIN or monto < 100 or abs(ratio - 0.16) < 0.05:
        return None
    pct = ratio * 100
    return (
        "Proporcion fiscal atipica",
        f"El IVA aplicado ({pct:.1f}%) difiere del patron habitual "
        f"para el concepto — revisar calculo fiscal",
        shap_abs,
    )


def _r_propina(row: dict, stats: dict) -> Reason | None:
    feat = "feat_propina_ratio_abs"
    shap_abs = _shap(row, feat)
    ratio = row.get(feat, 0.0)
    if shap_abs < SHAP_MIN or ratio < 0.15:
        return None
    monto = np.expm1(row.get("feat_monto_abs_log", 0.0))
    propina_est = monto * ratio
    return (
        "Propina atipica",
        f"La propina estimada (${propina_est:,.0f}, {ratio:.0%} del cargo) "
        f"supera el rango habitual para este tipo de transaccion",
        shap_abs,
    )


def _r_obs_keywords(row: dict, stats: dict) -> Reason | None:
    kw_map = {
        "feat_obs_kw_error":       "error",
        "feat_obs_kw_ajuste":      "ajuste",
        "feat_obs_kw_cancelacion": "cancelacion",
        "feat_obs_kw_cortesia":    "cortesia o compensacion",
        "feat_obs_kw_reembolso":   "reembolso o devolucion",
    }
    hits, best_shap = [], 0.0
    for feat, label in kw_map.items():
        if row.get(feat, 0) == 1:
            s = _shap(row, feat)
            if s >= SHAP_MIN:
                hits.append(label)
                best_shap = max(best_shap, s)
    if not hits:
        return None
    keywords = ", ".join(hits)
    return (
        "Observacion relevante",
        f"La descripcion de la transaccion menciona: {keywords}",
        best_shap,
    )


def _r_codigo_infrecuente(row: dict, stats: dict) -> Reason | None:
    feat = "feat_t_codigo_freq"
    shap_abs = _shap(row, feat)
    freq = row.get(feat, 1.0)
    if shap_abs < SHAP_MIN or freq > 0.001:
        return None
    codigo = row.get("trace_t_codigo", "?")
    pct = freq * 100
    return (
        "Codigo infrecuente",
        f"El concepto {codigo} representa solo el {pct:.3f}% de las transacciones "
        f"historicas del hotel — codigo poco utilizado",
        shap_abs,
    )


def _r_usuario_infrecuente(row: dict, stats: dict) -> Reason | None:
    feat = "feat_t_usuario_freq"
    shap_abs = _shap(row, feat)
    freq = row.get(feat, 1.0)
    if shap_abs < SHAP_MIN or freq > 0.002:
        return None
    pct = freq * 100
    return (
        "Usuario con baja actividad",
        f"El cajero que registro esta transaccion tiene baja actividad historica "
        f"({pct:.2f}% de transacciones totales)",
        shap_abs,
    )


def _r_es_split(row: dict, stats: dict) -> Reason | None:
    feat = "feat_es_split"
    if row.get(feat, 0) != 1 or _shap(row, feat) < SHAP_MIN:
        return None
    return (
        "Division de cuenta",
        "Cargo generado como division de cuenta (split) "
        "— verificar que la distribucion entre folios sea correcta",
        _shap(row, feat),
    )


def _r_sin_reservacion(row: dict, stats: dict) -> Reason | None:
    # Solo relevante si es un cargo de renta sin reservacion
    if row.get("feat_es_renta", 0) != 1:
        return None
    if row.get("feat_tiene_reservacion", 1) != 0:
        return None
    feat = "feat_tiene_reservacion"
    shap_abs = _shap(row, feat)
    if shap_abs < SHAP_MIN:
        return None
    return (
        "Sin reservacion enlazada",
        "Cargo de renta registrado sin reservacion enlazada en el sistema "
        "— verificar si la cuenta esta correctamente asociada",
        shap_abs,
    )


# Lista ordenada de generadores (el orden no afecta el resultado; se re-ordenan por SHAP)
REASON_GENERATORS = [
    _r_dup_minuto,
    _r_dup_dia,
    _r_monto_atipico,
    _r_vs_tarifa,
    _r_noches_delta,
    _r_fuera_estancia,
    _r_horario,
    _r_signo,
    _r_modificacion,
    _r_cancelacion,
    _r_densidad_concepto,
    _r_densidad_folio,
    _r_impuesto,
    _r_propina,
    _r_obs_keywords,
    _r_codigo_infrecuente,
    _r_usuario_infrecuente,
    _r_es_split,
    _r_sin_reservacion,
]


# ══════════════════════════════════════════════════════════════════════════════
# Logica de construccion de explicacion por fila
# ══════════════════════════════════════════════════════════════════════════════

def build_reasons(row: dict, stats: dict) -> list[Reason]:
    """
    Evalua todos los generadores y retorna lista de razones unicas por categoria,
    ordenadas por |SHAP| descendente (maximo 5).
    """
    raw: list[Reason] = []
    for gen in REASON_GENERATORS:
        result = gen(row, stats)
        if result is not None:
            raw.append(result)

    # Deduplicar por categoria: conservar la de mayor shap_abs
    best: dict[str, Reason] = {}
    for cat, text, shap_abs in raw:
        if cat not in best or shap_abs > best[cat][2]:
            best[cat] = (cat, text, shap_abs)

    return sorted(best.values(), key=lambda r: r[2], reverse=True)[:5]


def assign_severity(row: dict) -> str:
    rank = row.get("anomaly_rank", 9_999_999)

    # Criterio principal: posicion en ranking global de anomalia
    if rank <= RANK_ALTA:
        return "Alta"

    # El rank ya incorpora todas las features (incluyendo duplicados, horario, etc.)
    # No se usan overrides adicionales para evitar saturacion del nivel Alta.
    if rank <= RANK_MEDIA:
        return "Media"

    return "Informativa"


def compose_explanation(reasons: list[Reason]) -> str:
    if not reasons:
        return (
            "Transaccion con patron estadistico inusual detectado por el sistema. "
            "Se recomienda revision manual."
        )
    texts = [r[1] for r in reasons]
    if len(texts) == 1:
        return f"{texts[0]}."
    if len(texts) == 2:
        return f"{texts[0]}. Ademas, {texts[1][0].lower()}{texts[1][1:]}."
    return (
        f"{texts[0]}. Ademas, {texts[1][0].lower()}{texts[1][1:]}. "
        f"Tambien se detecto: {texts[2][0].lower()}{texts[2][1:]}."
    )


def process_row(row: dict, stats: dict) -> dict:
    reasons = build_reasons(row, stats)
    severity = assign_severity(row)
    categories = " | ".join(r[0] for r in reasons) if reasons else "Sin categoria especifica"
    explanation = compose_explanation(reasons)
    reasons_json = json.dumps(
        [{"categoria": r[0], "texto": r[1], "peso_shap": round(r[2], 4)} for r in reasons],
        ensure_ascii=False,
    )
    return {
        "severity":     severity,
        "n_reasons":    len(reasons),
        "categories":   categories,
        "explanation":  explanation,
        "reasons_json": reasons_json,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("FINANOM — Pipeline de explicabilidad")
    print("=" * 60)

    print("\n[1/4] Cargando datos ...")
    stats = load_concept_stats()
    df = load_data(stats)

    print("\n[2/4] Generando explicaciones ...")
    records = []
    for _, row in df.iterrows():
        records.append(process_row(row.to_dict(), stats))

    explain_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, explain_df], axis=1)

    print("\n[3/4] Guardando artefactos ...")
    HERE.joinpath("artifacts").mkdir(parents=True, exist_ok=True)

    # Columnas utiles para la interfaz de auditoria (sin shap crudos)
    ui_cols = (
        [c for c in result.columns if c.startswith("trace_")]
        + ["anomaly_score", "anomaly_score_percentile", "anomaly_rank"]
        + ["severity", "n_reasons", "categories", "explanation", "reasons_json"]
        + [c for c in result.columns if c.startswith("feat_")]
        + [c for c in result.columns if c.startswith("shap_")]
    )
    ui_cols = [c for c in ui_cols if c in result.columns]
    result[ui_cols].to_parquet(OUT_FILE, index=False)
    print(f"  Parquet completo: {OUT_FILE.name}  ({len(result):,} filas)")

    # Preview CSV: columnas esenciales para revision rapida
    preview_cols = [
        "trace_t_folio", "trace_t_cuarto", "trace_t_codigo", "trace_t_timestamp",
        "anomaly_rank", "severity", "categories", "explanation",
    ]
    result[preview_cols].sort_values("anomaly_rank").to_csv(PREVIEW_FILE, index=False, encoding="utf-8-sig")
    print(f"  Preview CSV:      {PREVIEW_FILE.name}")

    # Stats de distribucion
    print("\n[4/4] Resumen de resultados")
    print("\n  Distribucion de severidad:")
    for sev, cnt in result["severity"].value_counts().items():
        print(f"    {sev:15s}  {cnt:>5,}  ({cnt/len(result)*100:.1f}%)")

    print("\n  Categorias mas frecuentes:")
    all_cats: dict[str, int] = {}
    for cats_str in result["categories"]:
        for cat in cats_str.split(" | "):
            all_cats[cat] = all_cats.get(cat, 0) + 1
    for cat, cnt in sorted(all_cats.items(), key=lambda x: -x[1])[:8]:
        print(f"    {cat:35s}  {cnt:>5,}  ({cnt/len(result)*100:.1f}%)")

    print(f"\n  Promedio de razones por anomalia: {result['n_reasons'].mean():.1f}")
    print(f"  Anomalias sin ninguna razon detectada: {(result['n_reasons'] == 0).sum():,}")

    print(f"\n{'='*60}")
    print("Listo.")


if __name__ == "__main__":
    main()
