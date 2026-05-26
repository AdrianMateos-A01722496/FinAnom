"""Dashboard local de alertas financieras hoteleras — FinAnom / TCA Software Solutions.

Cómo correr:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite importar data_loader y state_manager desde esta misma carpeta
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (
    COLS_ALERTAS,
    COLS_TRANSACCIONES,
    compute_kpis,
    filter_by_date,
    get_clusters_disponibles,
    load_alertas,
    load_transacciones,
)
from state_manager import ETIQUETAS, get_label, get_labels_map, load_feedback, save_label

# ── Configuración de página (DEBE ir antes de cualquier otro st.*) ────────────
st.set_page_config(
    page_title="FinAnom — Alertas Financieras",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Encabezado de KPIs */
div[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #2d3250;
}
/* Badges de nivel de riesgo en markdown */
.badge-critico { background:#ff2b2b; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }
.badge-alto    { background:#ff8c00; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }
.badge-medio   { background:#ffd700; color:#1a1a1a; padding:2px 8px; border-radius:4px; font-weight:700; }
.badge-bajo    { background:#a0aab4; color:#1a1a1a; padding:2px 8px; border-radius:4px; font-weight:700; }
/* Tarjeta de detalle */
.detalle-card {
    border-radius: 6px;
    padding: 14px 20px;
    background: #1e2130;
    margin-bottom: 14px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Colores por nivel de riesgo ───────────────────────────────────────────────
_COLOR_BG = {
    "CRITICO": "#ff2b2b",
    "ALTO":    "#ff8c00",
    "MEDIO":   "#ffd700",
    "BAJO":    "#a0aab4",
}
_COLOR_FG = {
    "CRITICO": "white",
    "ALTO":    "white",
    "MEDIO":   "#1a1a1a",
    "BAJO":    "#1a1a1a",
}

# ── Helpers de estilo ─────────────────────────────────────────────────────────

def _highlight_fila(row: pd.Series) -> list[str]:
    """Aplica color de fondo + texto según nivel_riesgo en la fila completa."""
    nivel = str(row.get("nivel_riesgo", ""))
    bg = _COLOR_BG.get(nivel, "")
    fg = _COLOR_FG.get(nivel, "")
    style = f"background-color: {bg}; color: {fg};" if bg else ""
    return [style] * len(row)


def _style_tabla(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Aplica colores por nivel y formatea columnas numéricas."""
    fmt: dict[str, str] = {}
    if "monto"       in df.columns:
        fmt["monto"]       = "{:,.2f}"
    if "score_riesgo" in df.columns:
        fmt["score_riesgo"] = "{:d}"
    styler = df.style.apply(_highlight_fila, axis=1)
    if fmt:
        styler = styler.format(fmt, na_rep="—")
    return styler


def _badge(nivel: str) -> str:
    """Devuelve HTML de badge coloreado para usar en markdown."""
    cls = {"CRITICO": "critico", "ALTO": "alto", "MEDIO": "medio", "BAJO": "bajo"}.get(nivel, "bajo")
    return f'<span class="badge-{cls}">{nivel}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏨 FinAnom")
    st.caption("TCA Software Solutions")
    st.divider()

    # ── Fuente de datos ───────────────────────────────────────────────────────
    st.subheader("Fuente de datos")
    fuente_sel = st.radio(
        "Cargar desde:",
        ["Usar muestra de prueba", "Usar archivos completos si existen"],
        index=0,
        help=(
            "Muestra: dashboard/data/sample_alertas.csv (≤ 1,000 filas, rápido).\n"
            "Completo: anomaly_detection/output_alertas_operativas.csv."
        ),
    )
    use_sample = fuente_sel == "Usar muestra de prueba"

    st.divider()

    # ── Filtro de fecha ───────────────────────────────────────────────────────
    st.subheader("Período")
    preset_opts = ["Todas", "Hoy", "Esta semana", "Este mes", "Este año", "Rango personalizado"]
    preset = st.selectbox("Filtrar por:", preset_opts, index=0)

    fecha_inicio: date | None = None
    fecha_fin:    date | None = None
    if preset == "Rango personalizado":
        hoy = date.today()
        fecha_inicio = st.date_input("Desde", value=hoy - timedelta(days=30))
        fecha_fin    = st.date_input("Hasta", value=hoy)

    st.divider()

    # ── Filtros de tabla (se llena después de cargar datos) ───────────────────
    st.subheader("Filtros de alertas")

    niveles_sel = st.multiselect(
        "Nivel de riesgo",
        ["CRITICO", "ALTO", "MEDIO", "BAJO"],
        default=["CRITICO", "ALTO", "MEDIO", "BAJO"],
    )

    orden_opciones: dict[str, tuple[str, bool]] = {
        "Mayor score primero":   ("score_riesgo", False),
        "Más recientes primero": ("fecha",        False),
        "Más antiguas primero":  ("fecha",        True),
    }
    orden_sel = st.selectbox("Ordenar alertas por:", list(orden_opciones.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS  (cacheado 60 s para no recargar en cada interacción)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def _cargar_datos(use_sample: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    alertas, src = load_alertas(use_sample)
    todas        = load_transacciones(use_sample)
    return alertas, todas, src


alertas_raw, todas_raw, fuente_desc = _cargar_datos(use_sample)

# Error si no hay datos
if alertas_raw is None:
    st.error("⚠️  No se encontraron datos de anomalías.")
    st.markdown(fuente_desc)
    st.stop()

# Clusters disponibles (del dataset completo, antes de filtrar)
clusters_disponibles = get_clusters_disponibles(alertas_raw)

with st.sidebar:
    clusters_sel = st.multiselect(
        "Cluster de anomalía",
        clusters_disponibles,
        default=clusters_disponibles,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# APLICAR FILTROS
# ═══════════════════════════════════════════════════════════════════════════════

alertas = filter_by_date(alertas_raw.copy(), preset, fecha_inicio, fecha_fin)
todas   = filter_by_date(todas_raw.copy(), preset, fecha_inicio, fecha_fin) if todas_raw is not None else None

# Filtro por nivel de riesgo
if niveles_sel and "nivel_riesgo" in alertas.columns:
    alertas = alertas[alertas["nivel_riesgo"].isin(niveles_sel)]

# Filtro por cluster
if clusters_sel and "cluster_anomalia" in alertas.columns:
    alertas = alertas[
        alertas["cluster_anomalia"].apply(
            lambda v: any(c in str(v) for c in clusters_sel) if pd.notna(v) else False
        )
    ]

# Ordenar
col_ord, asc_ord = orden_opciones[orden_sel]
if col_ord in alertas.columns:
    alertas = alertas.sort_values(col_ord, ascending=asc_ord, na_position="last")


# ═══════════════════════════════════════════════════════════════════════════════
# ENCABEZADO
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏨 FinAnom — Alertas Financieras Hoteleras")
st.caption(f"📂 {fuente_desc}  ·  📅 Período: **{preset}**")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — KPIs
# ═══════════════════════════════════════════════════════════════════════════════
kpis = compute_kpis(alertas, todas)

# Fila 1: métricas globales
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("📊 Transacciones",      f"{kpis['total']:,}")
k2.metric("🚨 Alertas operativas", f"{kpis['alertas_operativas']:,}")
k3.metric("% con alerta",          f"{kpis['pct_alerta']:.1%}")
k4.metric("🔴 Críticas",           kpis["criticas"])
k5.metric("🟠 Altas",              kpis["altas"])
k6.metric("🟡 Medias",             kpis["medias"])

# Fila 2: breakdown por cluster
c1, c2, c3, c4, c5, _ = st.columns(6)
c1.metric("🔁 Duplicados",              kpis["duplicados"])
c2.metric("💸 Pagos sospechosos",       kpis["pagos_sospechosos"])
c3.metric("🗓️ Fuera de estancia",       kpis["fuera_estancia"])
c4.metric("📈 Montos atípicos",         kpis["montos_atipicos"])
c5.metric("❌ Cancelaciones sosp.",     kpis["cancelaciones"])

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Tabla de alertas operativas
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader(f"🚨 Alertas operativas ({len(alertas):,})")

if alertas.empty:
    st.info("No hay alertas para el período y filtros seleccionados.")
else:
    # Agregar etiqueta manual si existe
    etiquetas_map = get_labels_map()
    cols_show  = [c for c in COLS_ALERTAS if c in alertas.columns]
    df_display = alertas[cols_show].copy()

    if etiquetas_map:
        df_display["etiqueta_manual"] = (
            df_display["id_transaccion"].astype(str).map(etiquetas_map).fillna("—")
        )

    # Truncar mensajes largos para la tabla
    if "mensaje_alerta" in df_display.columns:
        df_display["mensaje_alerta"] = df_display["mensaje_alerta"].str[:120]

    st.dataframe(
        _style_tabla(df_display),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Detalle y etiquetado manual
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Detalle y etiquetado manual")

if alertas.empty:
    st.info("Selecciona un período o ajusta los filtros para ver alertas.")
else:
    ids_disponibles = alertas["id_transaccion"].astype(str).tolist()

    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        id_sel = st.selectbox("ID transacción:", ids_disponibles, label_visibility="collapsed")

    fila = alertas[alertas["id_transaccion"].astype(str) == id_sel]
    if not fila.empty:
        row    = fila.iloc[0]
        nivel  = str(row.get("nivel_riesgo", ""))
        borde  = _COLOR_BG.get(nivel, "#555555")

        # Tarjeta de detalle
        st.markdown(f"""
        <div class="detalle-card" style="border-left: 5px solid {borde};">
            <b>ID:</b> {row.get('id_transaccion', '—')} &nbsp;|&nbsp;
            <b>Folio:</b> {row.get('folio', '—')} &nbsp;|&nbsp;
            <b>Código:</b> {row.get('codigo', '—')} &nbsp;|&nbsp;
            <b>Fecha:</b> {row.get('fecha', '—')}<br>
            <b>Monto:</b> ${float(row.get('monto', 0) or 0):,.2f} &nbsp;|&nbsp;
            <b>Nivel:</b> {_badge(nivel)} &nbsp;|&nbsp;
            <b>Score:</b> <strong>{row.get('score_riesgo', '—')}</strong><br>
            <b>Cluster:</b> {row.get('cluster_anomalia', '—')}<br><br>
            <b>Mensaje:</b> {row.get('mensaje_alerta', '—')}
        </div>
        """, unsafe_allow_html=True)

        # Reglas activadas
        reglas_val = row.get("reglas_activadas", "")
        if pd.notna(reglas_val) and str(reglas_val).strip():
            with st.expander("📋 Reglas activadas"):
                for r in str(reglas_val).split(" | "):
                    if r.strip():
                        st.write(f"• {r.strip()}")

        # Señales de contexto (USUARIO_MODIFICACION, etc.)
        ctx_val = row.get("senales_contexto", "")
        if pd.notna(ctx_val) and str(ctx_val).strip():
            with st.expander("🔎 Señales de contexto"):
                st.write(str(ctx_val))

        # Formulario de etiquetado
        st.markdown("**Etiqueta manual:**")
        etiqueta_actual = get_label(id_sel)
        idx_default     = ETIQUETAS.index(etiqueta_actual) if etiqueta_actual in ETIQUETAS else 2

        with st.form(key=f"form_{id_sel}", clear_on_submit=False):
            etiqueta_nueva = st.radio(
                "Clasificación:",
                ETIQUETAS,
                index=idx_default,
                horizontal=True,
                label_visibility="collapsed",
            )
            comentario = st.text_input(
                "Comentario (opcional):",
                placeholder="Ej: revisado con contabilidad — parece OK",
            )
            guardado = st.form_submit_button("💾 Guardar etiqueta", type="primary")

        if guardado:
            save_label(id_sel, etiqueta_nueva, comentario)
            st.success(f"✅ Etiqueta guardada: **{etiqueta_nueva}** para tx `{id_sel}`")
            st.rerun()

    # Resumen de etiquetas guardadas
    fb_df = load_feedback()
    if not fb_df.empty:
        with st.expander(f"📝 Etiquetas guardadas ({len(fb_df)} revisiones)"):
            st.dataframe(fb_df, use_container_width=True, hide_index=True)
            csv_bytes = fb_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar feedback_manual.csv",
                data=csv_bytes,
                file_name="feedback_manual.csv",
                mime="text/csv",
            )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Todas las transacciones del período
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Todas las transacciones del período")

if todas is not None:
    df_tx = todas
    aviso = None
else:
    # Fallback: mostrar el dataset de alertas completo (sin filtro de cluster/nivel)
    df_tx = filter_by_date(alertas_raw.copy(), preset, fecha_inicio, fecha_fin)
    aviso = (
        "Mostrando solo alertas operativas. "
        "Para ver todas las transacciones genera `output_senales_contexto.csv` "
        "y vuelve a ejecutar `create_sample.py`."
    )

if aviso:
    st.info(aviso)

cols_tx = [c for c in COLS_TRANSACCIONES if c in df_tx.columns]
st.caption(f"{len(df_tx):,} transacciones")
st.dataframe(
    _style_tabla(df_tx[cols_tx]),
    use_container_width=True,
    height=380,
    hide_index=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("FinAnom v0.1 — TCA Software Solutions · Solo prototipo local · No usar en producción")
