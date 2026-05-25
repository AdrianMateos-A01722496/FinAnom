"""Demo del módulo de detección de anomalías por reglas de negocio.

Ejecuta el detector sobre:
  1. Un dataset sintético con anomalías plantadas a propósito.
  2. La base real (training_data/transacciones_modelado.parquet) si existe,
     usando solo una muestra de 5,000 filas para no tardar.

Uso:
    uv run python anomaly_detection/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anomaly_detection.reglas import detectar_anomalias, separar_alertas


# --------------------------------------------------------------------------- #
# Dataset sintético
# --------------------------------------------------------------------------- #
def crear_datos_prueba() -> pd.DataFrame:
    """
    Construye 20 transacciones con anomalías plantadas para validar cada regla.
    Los registros están comentados con el tipo de anomalía esperada.
    """
    base_ts = pd.Timestamp("2024-06-15 14:30:00")

    rows = [
        # ── NORMALES ──────────────────────────────────────────────────────────
        {
            "t_transaccion": "001", "t_folio": "F100", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 2000.0, "t_impuesto": 320.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "101",
            "t_usuario": "MCC", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF001", "t_noches": 2,
            "t_timestamp": base_ts,
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 2.0, "h_num_per": 2.0, "h_tfa": 2000.0, "h_tfa_total": 4000.0,
            "tiene_reservacion": True,
        },
        {
            "t_transaccion": "002", "t_folio": "F101", "t_folio_ext": "0",
            "t_codigo": "ALIMEN", "t_monto": 350.0, "t_impuesto": 56.0,
            "t_propina": 35.0, "t_carabo": "0", "t_cuarto": "205",
            "t_usuario": "IVL", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF002", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=1),
            "h_fec_lld": pd.Timestamp("2024-06-14"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 2.0, "h_num_per": 2.0, "h_tfa": 1800.0, "h_tfa_total": 3600.0,
            "tiene_reservacion": True,
        },

        # ── DUPLICADO EXACTO (001 repetido) ───────────────────────────────────
        {
            "t_transaccion": "003", "t_folio": "F100", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 2000.0, "t_impuesto": 320.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "101",
            "t_usuario": "MCC", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF001", "t_noches": 2,
            "t_timestamp": base_ts,  # mismo timestamp = mismo día
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 2.0, "h_num_per": 2.0, "h_tfa": 2000.0, "h_tfa_total": 4000.0,
            "tiene_reservacion": True,
        },

        # ── SIGNO CONTABLE: cargo con monto negativo ──────────────────────────
        {
            "t_transaccion": "004", "t_folio": "F102", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": -1500.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "303",
            "t_usuario": "ALM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF003", "t_noches": 1,
            "t_timestamp": base_ts + pd.Timedelta(hours=2),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 1.0, "h_num_per": 1.0, "h_tfa": 1500.0, "h_tfa_total": 1500.0,
            "tiene_reservacion": True,
        },

        # ── SIGNO CONTABLE: abono con monto positivo ──────────────────────────
        {
            "t_transaccion": "005", "t_folio": "F103", "t_folio_ext": "0",
            "t_codigo": "DEVOLU", "t_monto": 800.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "1", "t_cuarto": "410",
            "t_usuario": "RFR", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF004", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=3),
            "h_fec_lld": pd.Timestamp("2024-06-14"), "h_fec_sda": pd.Timestamp("2024-06-15"),
            "h_num_noc": 1.0, "h_num_per": 2.0, "h_tfa": 1800.0, "h_tfa_total": 1800.0,
            "tiene_reservacion": True,
        },

        # ── MONTO ATÍPICO: RENHAB por $95,000 ────────────────────────────────
        {
            "t_transaccion": "006", "t_folio": "F104", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 95000.0, "t_impuesto": 15200.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "512",
            "t_usuario": "MCC", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF005", "t_noches": 3,
            "t_timestamp": base_ts + pd.Timedelta(hours=4),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-18"),
            "h_num_noc": 3.0, "h_num_per": 2.0, "h_tfa": 2100.0, "h_tfa_total": 6300.0,
            "tiene_reservacion": True,
        },

        # ── CARGO DESPUÉS DE CHECK-OUT ────────────────────────────────────────
        {
            "t_transaccion": "007", "t_folio": "F105", "t_folio_ext": "0",
            "t_codigo": "MINBAR", "t_monto": 220.0, "t_impuesto": 35.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "601",
            "t_usuario": "IVL", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF006", "t_noches": 0,
            "t_timestamp": pd.Timestamp("2024-06-20 10:00:00"),  # 3 días después del checkout
            "h_fec_lld": pd.Timestamp("2024-06-14"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 3.0, "h_num_per": 1.0, "h_tfa": 1200.0, "h_tfa_total": 3600.0,
            "tiene_reservacion": True,
        },

        # ── CARGO ANTES DE CHECK-IN ───────────────────────────────────────────
        {
            "t_transaccion": "008", "t_folio": "F106", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 1800.0, "t_impuesto": 288.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "702",
            "t_usuario": "ALM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF007", "t_noches": 1,
            "t_timestamp": pd.Timestamp("2024-06-10 08:00:00"),  # 5 días antes del check-in
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 1.0, "h_num_per": 2.0, "h_tfa": 1800.0, "h_tfa_total": 1800.0,
            "tiene_reservacion": True,
        },

        # ── CANCELACIÓN CON MONTO ALTO ────────────────────────────────────────
        {
            "t_transaccion": "009", "t_folio": "F107", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 12000.0, "t_impuesto": 1920.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "801",
            "t_usuario": "MCC", "t_usuario_mod": None,
            "t_tra_cancelada": "1", "t_referencia": "REF008", "t_noches": 6,
            "t_timestamp": base_ts + pd.Timedelta(hours=5),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-21"),
            "h_num_noc": 6.0, "h_num_per": 3.0, "h_tfa": 2000.0, "h_tfa_total": 12000.0,
            "tiene_reservacion": True,
        },

        # ── FOLIO CON DEMASIADAS CANCELACIONES ───────────────────────────────
        {
            "t_transaccion": "010", "t_folio": "F108", "t_folio_ext": "0",
            "t_codigo": "ALIMEN", "t_monto": 300.0, "t_impuesto": 48.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "205",
            "t_usuario": "IVL", "t_usuario_mod": None,
            "t_tra_cancelada": "1", "t_referencia": "REF009", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=6),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 1.0, "h_num_per": 2.0, "h_tfa": 1600.0, "h_tfa_total": 1600.0,
            "tiene_reservacion": True,
        },
        {
            "t_transaccion": "011", "t_folio": "F108", "t_folio_ext": "0",
            "t_codigo": "ALIMEN", "t_monto": 310.0, "t_impuesto": 49.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "205",
            "t_usuario": "IVL", "t_usuario_mod": None,
            "t_tra_cancelada": "1", "t_referencia": "REF010", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=7),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 1.0, "h_num_per": 2.0, "h_tfa": 1600.0, "h_tfa_total": 1600.0,
            "tiene_reservacion": True,
        },
        {
            "t_transaccion": "012", "t_folio": "F108", "t_folio_ext": "0",
            "t_codigo": "ALIMEN", "t_monto": 295.0, "t_impuesto": 47.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "205",
            "t_usuario": "IVL", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF011", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=8),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-16"),
            "h_num_noc": 1.0, "h_num_per": 2.0, "h_tfa": 1600.0, "h_tfa_total": 1600.0,
            "tiene_reservacion": True,
        },

        # ── CONTEXTO: cargo sin reservación en código habitualmente con reserva ─
        {
            "t_transaccion": "013", "t_folio": "F109", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 1900.0, "t_impuesto": 304.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "305",
            "t_usuario": "ALM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF012", "t_noches": 1,
            "t_timestamp": base_ts + pd.Timedelta(hours=9),
            "h_fec_lld": None, "h_fec_sda": None,
            "h_num_noc": None, "h_num_per": None, "h_tfa": None, "h_tfa_total": None,
            "tiene_reservacion": False,
        },

        # ── CONTEXTO: noches inconsistentes ──────────────────────────────────
        {
            "t_transaccion": "014", "t_folio": "F110", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 10000.0, "t_impuesto": 1600.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "406",
            "t_usuario": "MCC", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF013", "t_noches": 5,
            "t_timestamp": base_ts + pd.Timedelta(hours=10),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 2.0, "h_num_per": 2.0, "h_tfa": 2000.0, "h_tfa_total": 4000.0,
            "tiene_reservacion": True,
        },

        # ── USUARIO MODIFICADOR DISTINTO ──────────────────────────────────────
        {
            "t_transaccion": "015", "t_folio": "F111", "t_folio_ext": "0",
            "t_codigo": "RENHAB", "t_monto": 5500.0, "t_impuesto": 880.0,
            "t_propina": 0.0, "t_carabo": "0", "t_cuarto": "510",
            "t_usuario": "IVL", "t_usuario_mod": "ALM",  # distinto
            "t_tra_cancelada": "0", "t_referencia": "REF014", "t_noches": 2,
            "t_timestamp": base_ts + pd.Timedelta(hours=11),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 2.0, "h_num_per": 3.0, "h_tfa": 2200.0, "h_tfa_total": 4400.0,
            "tiene_reservacion": True,
        },

        # ── PAGO PROVEEDOR DUPLICADO (abonos idénticos) ───────────────────────
        {
            "t_transaccion": "016", "t_folio": "F200", "t_folio_ext": "0",
            "t_codigo": "PAGPRO", "t_monto": 4500.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "1", "t_cuarto": "000",
            "t_usuario": "ADM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "PROV001", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=12),
            "h_fec_lld": None, "h_fec_sda": None,
            "h_num_noc": None, "h_num_per": None, "h_tfa": None, "h_tfa_total": None,
            "tiene_reservacion": False,
        },
        {
            "t_transaccion": "017", "t_folio": "F200", "t_folio_ext": "0",
            "t_codigo": "PAGPRO", "t_monto": 4500.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "1", "t_cuarto": "000",
            "t_usuario": "ADM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "PROV001",  # mismo proveedor
            "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=12, minutes=5),
            "h_fec_lld": None, "h_fec_sda": None,
            "h_num_noc": None, "h_num_per": None, "h_tfa": None, "h_tfa_total": None,
            "tiene_reservacion": False,
        },

        # ── PAGO PROVEEDOR SIN REFERENCIA ─────────────────────────────────────
        {
            "t_transaccion": "018", "t_folio": "F201", "t_folio_ext": "0",
            "t_codigo": "PAGPRO", "t_monto": 8000.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "1", "t_cuarto": "000",
            "t_usuario": "ADM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "",  # sin referencia
            "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=13),
            "h_fec_lld": None, "h_fec_sda": None,
            "h_num_noc": None, "h_num_per": None, "h_tfa": None, "h_tfa_total": None,
            "tiene_reservacion": False,
        },

        # ── PAGO EN HORARIO INUSUAL (madrugada) ───────────────────────────────
        {
            "t_transaccion": "019", "t_folio": "F202", "t_folio_ext": "0",
            "t_codigo": "PAGPRO", "t_monto": 3200.0, "t_impuesto": 0.0,
            "t_propina": 0.0, "t_carabo": "1", "t_cuarto": "000",
            "t_usuario": "ADM", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "PROV002", "t_noches": 0,
            "t_timestamp": pd.Timestamp("2024-06-15 03:15:00"),  # 3 AM
            "h_fec_lld": None, "h_fec_sda": None,
            "h_num_noc": None, "h_num_per": None, "h_tfa": None, "h_tfa_total": None,
            "tiene_reservacion": False,
        },

        # ── NORMAL (control de cierre) ────────────────────────────────────────
        {
            "t_transaccion": "020", "t_folio": "F112", "t_folio_ext": "0",
            "t_codigo": "ALIMEN", "t_monto": 420.0, "t_impuesto": 67.0,
            "t_propina": 42.0, "t_carabo": "0", "t_cuarto": "115",
            "t_usuario": "RFR", "t_usuario_mod": None,
            "t_tra_cancelada": "0", "t_referencia": "REF020", "t_noches": 0,
            "t_timestamp": base_ts + pd.Timedelta(hours=14),
            "h_fec_lld": pd.Timestamp("2024-06-15"), "h_fec_sda": pd.Timestamp("2024-06-17"),
            "h_num_noc": 2.0, "h_num_per": 2.0, "h_tfa": 1700.0, "h_tfa_total": 3400.0,
            "tiene_reservacion": True,
        },
    ]

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Helpers de visualización
# --------------------------------------------------------------------------- #
def _print_separador(titulo: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {titulo}")
    print(f"{'='*70}")


def _print_resumen(alertas: pd.DataFrame) -> None:
    total      = len(alertas)
    operativas, contexto = separar_alertas(alertas)
    n_op  = len(operativas)
    n_ctx = len(contexto)

    print(f"\nTotal transacciones analizadas : {total:,}")
    print(f"Alertas operativas (score≥25)  : {n_op:,} ({n_op/total:.1%})")
    print(f"Señales de contexto (débiles)  : {n_ctx:,} ({n_ctx/total:.1%})")
    print(f"Sin anomalía                   : {total - n_op - n_ctx:,}")

    if n_op == 0:
        print("\n[Sin alertas operativas]")
        return

    por_nivel = operativas["nivel_riesgo"].value_counts()
    print("\nAlertas operativas por nivel de riesgo:")
    for nivel in ["CRITICO", "ALTO", "MEDIO", "BAJO"]:
        n = por_nivel.get(nivel, 0)
        if n:
            print(f"  {nivel:10s}: {n:,}")

    por_cluster = (
        operativas["cluster_anomalia"]
        .str.split(" | ", regex=False)
        .explode()
        .str.strip()
        .loc[lambda s: s.str.len() > 0]
        .value_counts()
    )
    print("\nAlertas operativas por cluster:")
    for cluster, count in por_cluster.items():
        print(f"  {cluster:35s}: {count:,}")


def _print_alertas_detalle(alertas: pd.DataFrame, max_rows: int = 30) -> None:
    operativas, _ = separar_alertas(alertas)
    top = operativas.sort_values("score_riesgo", ascending=False).head(max_rows)
    print(f"\nTop {min(max_rows, len(top))} alertas operativas más críticas:\n")
    for _, row in top.iterrows():
        ctx = f" | ctx: {row['senales_contexto'][:80]}" if row.get("senales_contexto", "") else ""
        print(f"  [{row['nivel_riesgo']:8s}] score={row['score_riesgo']:3d} | {row['mensaje_alerta'][:110]}")
        if ctx:
            print(f"           {ctx}")
        print(f"           reglas: {row['reglas_activadas'][:110]}")
        print()


# --------------------------------------------------------------------------- #
# Ejecución demo
# --------------------------------------------------------------------------- #
def demo_sintetico() -> None:
    _print_separador("DEMO 1 — Datos sintéticos (20 transacciones con anomalías plantadas)")
    df = crear_datos_prueba()
    alertas = detectar_anomalias(df)
    _print_resumen(alertas)
    _print_alertas_detalle(alertas)

    salida = ROOT / "anomaly_detection" / "output_demo_sintetico.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    operativas, _ = separar_alertas(alertas)
    operativas.to_csv(salida, index=False, encoding="utf-8")
    print(f"Alertas operativas guardadas: {salida}")


def _construir_muestra_real(max_filas: int = 5_000, random_state: int = 42) -> pd.DataFrame | None:
    """
    Construye un DataFrame analítico desde hottra + hothsp crudos.
    Deriva t_timestamp y tiene_reservacion igual que el pipeline de consolidación.
    Devuelve None si los archivos no existen.
    """
    tra_path = ROOT / "tablas_parquet" / "hottra.parquet"
    hsp_path = ROOT / "tablas_parquet" / "hothsp.parquet"

    if not tra_path.exists():
        return None

    tra = pd.read_parquet(tra_path)
    for c in tra.select_dtypes("object"):
        tra[c] = tra[c].str.strip()

    # Derivar t_timestamp = fecha + hora + minuto
    tra["_fecha_dt"] = pd.to_datetime(tra["t_fecha"], format="%Y%m%d", errors="coerce")
    tra["_hra"]      = tra["t_tra_hra"].fillna("00").str.zfill(2)
    tra["_mto"]      = tra["t_tra_mto"].fillna("00").str.zfill(2)
    tra["t_timestamp"] = pd.to_datetime(
        tra["_fecha_dt"].dt.strftime("%Y-%m-%d") + " " + tra["_hra"] + ":" + tra["_mto"],
        errors="coerce",
    )
    tra["t_cve_res_clean"]  = tra["t_cve_res"].fillna("").str.strip()
    tra["tiene_reservacion"] = tra["t_cve_res_clean"].ne("")

    # Enriquecer con datos de reservación
    if hsp_path.exists():
        hsp = pd.read_parquet(hsp_path)
        for c in hsp.select_dtypes("object"):
            hsp[c] = hsp[c].str.strip()

        hsp_cols = [
            "h_res_cve", "h_fec_lld", "h_fec_sda",
            "h_num_per", "h_num_noc", "h_tfa", "h_tfa_total",
            "h_tpo_hab", "h_tpo_hsp", "h_seg_mer",
        ]
        hsp_sel = hsp[[c for c in hsp_cols if c in hsp.columns]].copy()
        for dc in ["h_fec_lld", "h_fec_sda"]:
            if dc in hsp_sel.columns:
                hsp_sel[dc] = pd.to_datetime(hsp_sel[dc], format="%Y%m%d", errors="coerce")
        for nc in ["h_num_per", "h_num_noc", "h_tfa", "h_tfa_total"]:
            if nc in hsp_sel.columns:
                hsp_sel[nc] = pd.to_numeric(hsp_sel[nc], errors="coerce")

        df = tra.merge(hsp_sel, left_on="t_cve_res_clean", right_on="h_res_cve", how="left")
    else:
        df = tra.copy()

    # Muestra reproducible
    return df.sample(n=min(max_filas, len(df)), random_state=random_state)


def demo_real(max_filas: int = 5_000) -> None:
    _print_separador(f"DEMO 2 — Base real TSA (muestra de {max_filas:,} filas de hottra + hothsp)")

    df = _construir_muestra_real(max_filas)
    if df is None:
        print("\n[INFO] No se encontró tablas_parquet/hottra.parquet.")
        print("       Coloca los archivos parquet en la carpeta tablas_parquet/ y vuelve a ejecutar.")
        return

    print(f"Filas en muestra: {len(df):,} | Columnas: {df.shape[1]}")
    print(f"Cargos (t_carabo=0): {(df['t_carabo']=='0').sum():,} | "
          f"Abonos (t_carabo=1): {(df['t_carabo']=='1').sum():,}")
    print(f"Con reservación: {df['tiene_reservacion'].sum():,} | "
          f"Sin reservación: {(~df['tiene_reservacion']).sum():,}\n")

    alertas = detectar_anomalias(df)
    _print_resumen(alertas)
    _print_alertas_detalle(alertas, max_rows=30)

    operativas, contexto = separar_alertas(alertas)
    salida_op  = ROOT / "anomaly_detection" / "output_alertas_operativas.csv"
    salida_ctx = ROOT / "anomaly_detection" / "output_senales_contexto.csv"
    operativas.to_csv(salida_op,  index=False, encoding="utf-8")
    contexto.to_csv(salida_ctx,   index=False, encoding="utf-8")
    print(f"\nAlertas operativas exportadas : {salida_op}")
    print(f"Señales de contexto exportadas: {salida_ctx}")


if __name__ == "__main__":
    demo_sintetico()
    demo_real()
