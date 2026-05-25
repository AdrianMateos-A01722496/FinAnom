"""Detección de anomalías financieras hoteleras por reglas de negocio.

Modelo inicial 100 % basado en lógica de negocio (sin ML).
Cada función detecta una categoría de anomalía, acumula un score de riesgo
y genera un mensaje de alerta legible para auditores.

Convención de signo en este PMS (TSA / HOTTRA):
    t_carabo = "0"  →  CARGO   (débito al folio del huésped; t_monto > 0 es normal)
    t_carabo = "1"  →  ABONO   (crédito al folio; t_monto > 0 también es normal —
                                 el signo contable lo da t_carabo, NO el signo de t_monto)
    t_monto < 0     →  Reversal / nota de crédito forzada (≈5 % de filas; anómalo)

Separación de dominios:
    - Cargos y abonos de huéspedes: todas las transacciones de hottra.
      Aplican: DUPLICADO, SIGNO_CONTABLE, MONTO_ATIPICO, FUERA_DE_ESTANCIA,
               CANCELACION_SOSPECHOSA, CONTEXTO_RESERVACION, USUARIO_MODIFICACION.
    - Egresos del hotel (devoluciones / pagos externos): subconjunto de abonos
      cuyo t_codigo NO pertenece a los medios de pago estándar de huéspedes.
      Aplica: PAGO_PROVEEDOR_SOSPECHOSO.
      Para pagos a proveedores externos (facturas, insumos) se requeriría una
      tabla adicional fuera de hottra; esta función los soportaría como un df
      separado en versiones futuras.

Uso rápido:
    from anomaly_detection.reglas import detectar_anomalias
    alertas = detectar_anomalias(df)

Demo ejecutable:
    uv run python anomaly_detection/run_demo.py
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Scores por tipo de regla
# --------------------------------------------------------------------------- #
SCORES: dict[str, int] = {
    "DUPLICADO":               40,
    "SIGNO_CONTABLE":          35,
    "FUERA_DE_ESTANCIA":       35,
    "MONTO_ATIPICO":           25,
    "CANCELACION_SOSPECHOSA":  25,
    "CONTEXTO_RESERVACION":    20,
    "USUARIO_MODIFICACION":    15,
    "PAGO_PROVEEDOR_SOSPECHOSO": 30,
}

# Umbrales de nivel de riesgo
NIVEL_RIESGO: list[tuple[int, str]] = [
    (80, "CRITICO"),
    (50, "ALTO"),
    (20, "MEDIO"),
    (0,  "BAJO"),
]

# ±Z_UMBRAL desviaciones estándar para considerar monto atípico
Z_UMBRAL = 2.0

# Folio con más de este % de movimientos cancelados = sospechoso
RATIO_CANCELACIONES_FOLIO = 0.5

# Proveedor que aparece menos de MIN_FRECUENCIA_PROVEEDOR veces = poco frecuente
MIN_FRECUENCIA_PROVEEDOR = 3

# Horario normal de operaciones hoteleras: 06:00–23:59
HORA_MIN_NORMAL = 6
HORA_MAX_NORMAL = 23

# Pagos similares al mismo proveedor en el mismo día para activar regla de división
MIN_PAGOS_DIVIDIDOS = 2
UMBRAL_PAGO_DIVIDIDO = 0.05  # 5 % de diferencia relativa

# --------------------------------------------------------------------------- #
# Catálogos de códigos conocidos (inferidos del análisis de hottra real)
# --------------------------------------------------------------------------- #
# Medios de pago y ajustes normales que los huéspedes generan como ABONOS.
# Un abono con estos códigos NO se considera egreso del hotel.
CODIGOS_PAGO_HUESPED: frozenset[str] = frozenset({
    # Formas de cobro directo al huésped
    "EFE", "TARCRE", "TARDEB", "AMEXCO", "TRANSF", "XFAC", "XFACTOT",
    # Depósitos de reservación y garantía
    "RESDEP", "DEPEFE", "DEPOS", "DEPRES", "DEPCHE", "DEPXBA",
    # Cupones, cuentas por cobrar y saldos
    "CUPON", "CXC", "CANCXC", "EMPCXC", "SALANT", "CUXAC",
    # Ajustes internos de renta / tarifa
    "RENAJU", "RENAJG", "SAAJU",
    # Ajustes misceláneos de huésped
    "AJMS", "AJPEXT", "AJPROP", "AJPAST", "AJPO", "AJDEEF", "AJCUPN",
    "AJUUPS", "AGALI", "MENAD",
    # Retenciones fiscales
    "RETIV2", "RETIS2",
})

# Códigos que representan devoluciones / reembolsos que el hotel paga al huésped.
# Estos SÍ se consideran egresos y se evalúan con reglas adicionales.
CODIGOS_EGRESO_HOTEL: frozenset[str] = frozenset({
    "DEVTRE",  # Devolución – transferencia
    "DEVTRJ",  # Devolución – tarjeta de crédito
    "DEVTRD",  # Devolución – tarjeta de débito
    "DEVEFF",  # Devolución – efectivo
    "DEVXBA",  # Devolución – banco
    "DEVCHE",  # Devolución – cheque
    "DEVTRB",  # Devolución – transferencia bancaria
})


# --------------------------------------------------------------------------- #
# Helpers internos
# --------------------------------------------------------------------------- #
def _col_presente(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        warnings.warn(f"[reglas] Columna '{col}' no encontrada — regla omitida.", stacklevel=3)
        return False
    return True


def _cols_presentes(df: pd.DataFrame, *cols: str) -> bool:
    return all(_col_presente(df, c) for c in cols)


def _asignar_nivel(score: int) -> str:
    for umbral, nivel in NIVEL_RIESGO:
        if score >= umbral:
            return nivel
    return "BAJO"


def _resultado_vacio(n: int) -> dict[str, list]:
    return {
        "cluster": [[] for _ in range(n)],
        "score":   [0] * n,
        "reglas":  [[] for _ in range(n)],
    }


def _merge(base: dict, nuevo: dict) -> None:
    for i in range(len(base["score"])):
        base["cluster"][i].extend(nuevo["cluster"][i])
        base["score"][i] += nuevo["score"][i]
        base["reglas"][i].extend(nuevo["reglas"][i])


def _leer_monto(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0)


def _leer_carabo_abono(df: pd.DataFrame) -> pd.Series:
    """Devuelve máscara booleana True = ABONO (t_carabo == '1')."""
    return df["t_carabo"].astype("string").str.strip() == "1"


# --------------------------------------------------------------------------- #
# 1. Duplicados
# --------------------------------------------------------------------------- #
def detectar_duplicados(df: pd.DataFrame) -> dict[str, list]:
    """
    Duplicado exacto: mismo folio, código, monto (centavos), habitación y fecha.
    Aplica tanto a cargos como a abonos.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_folio", "t_codigo", "t_monto", "t_cuarto"):
        return resultado

    monto = _leer_monto(df)
    work = pd.DataFrame({
        "folio":   df["t_folio"].astype("string").fillna("__NA__"),
        "codigo":  df["t_codigo"].astype("string").fillna("__NA__"),
        "cuarto":  df["t_cuarto"].astype("string").fillna("__NA__"),
        "monto_c": np.rint(monto * 100).astype("int64"),
    }, index=df.index)

    if "t_timestamp" in df.columns:
        work["fecha"] = pd.to_datetime(df["t_timestamp"]).dt.normalize()
    elif "t_fecha" in df.columns:
        work["fecha"] = pd.to_datetime(df["t_fecha"], errors="coerce").dt.normalize()
    else:
        work["fecha"] = pd.Timestamp("1970-01-01")

    conteos = work.groupby(
        ["folio", "codigo", "monto_c", "cuarto", "fecha"], dropna=False
    )["folio"].transform("size")
    mask_dup = conteos > 1

    for pos in range(n):
        if mask_dup.iloc[pos]:
            resultado["cluster"][pos].append("DUPLICADO")
            resultado["score"][pos]  += SCORES["DUPLICADO"]
            resultado["reglas"][pos].append(
                f"duplicado_exacto: {int(conteos.iloc[pos])} veces el mismo folio/código/monto/habitación/fecha"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 2. Inconsistencias de signo contable
# --------------------------------------------------------------------------- #
def detectar_signo_contable(df: pd.DataFrame) -> dict[str, list]:
    """
    Convención del PMS: t_monto SIEMPRE debería ser >= 0.
    El sentido contable (debe/haber) lo da t_carabo, no el signo del monto.

    Anomalías detectadas:
      - Monto negativo en CARGO (t_carabo=0): reversal sin cancelación marcada.
      - Monto negativo en ABONO (t_carabo=1): doble reversal, muy inusual.

    NO se marca como anomalía:
      - Abono con monto positivo → es lo normal en este PMS.
      - Cargo con monto positivo → es lo normal.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_carabo", "t_monto"):
        return resultado

    es_abono = _leer_carabo_abono(df)
    monto    = _leer_monto(df)

    mask_neg_cargo = (~es_abono) & (monto < 0)
    mask_neg_abono = es_abono    & (monto < 0)

    for pos in range(n):
        if mask_neg_cargo.iloc[pos]:
            resultado["cluster"][pos].append("SIGNO_CONTABLE")
            resultado["score"][pos]  += SCORES["SIGNO_CONTABLE"]
            resultado["reglas"][pos].append(
                f"cargo_monto_negativo: monto={monto.iloc[pos]:.2f} en CARGO (t_carabo=0); "
                "debería ser positivo o estar marcado como cancelado"
            )
        if mask_neg_abono.iloc[pos]:
            resultado["cluster"][pos].append("SIGNO_CONTABLE")
            resultado["score"][pos]  += SCORES["SIGNO_CONTABLE"]
            resultado["reglas"][pos].append(
                f"abono_monto_negativo: monto={monto.iloc[pos]:.2f} en ABONO (t_carabo=1); "
                "doble reversal — muy inusual"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 3. Montos atípicos (±Z_UMBRAL std por código de concepto)
# --------------------------------------------------------------------------- #
def detectar_montos_atipicos(df: pd.DataFrame, z_umbral: float = Z_UMBRAL) -> dict[str, list]:
    """
    Monto fuera de media ± z_umbral * std para su t_codigo.
    Usa el valor absoluto del monto para que cargos negativos legítimos
    no distorsionen la distribución de referencia.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_monto", "t_codigo"):
        return resultado

    monto  = _leer_monto(df).abs()
    codigo = df["t_codigo"].astype("string").str.strip().fillna("__NA__")

    stats = (
        df.assign(_m=monto, _c=codigo)
        .groupby("_c")["_m"]
        .agg(media="mean", std="std")
        .fillna({"std": 1.0})
        .replace({"std": {0.0: 1.0}})
    )

    medias = codigo.map(stats["media"]).fillna(float(monto.mean()))
    stds   = codigo.map(stats["std"]).fillna(float(monto.std())).replace(0, 1.0)
    z      = (monto - medias) / stds

    mask_atipico = z.abs() > z_umbral

    for pos in range(n):
        if mask_atipico.iloc[pos]:
            resultado["cluster"][pos].append("MONTO_ATIPICO")
            resultado["score"][pos]  += SCORES["MONTO_ATIPICO"]
            resultado["reglas"][pos].append(
                f"monto_atipico: monto_abs={monto.iloc[pos]:.2f}, z={float(z.iloc[pos]):.2f} "
                f"(umbral ±{z_umbral}) para código '{codigo.iloc[pos]}'"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 4. Cargos fuera de la estancia
# --------------------------------------------------------------------------- #
def detectar_fuera_estancia(df: pd.DataFrame) -> dict[str, list]:
    """
    Cargo antes del check-in (h_fec_lld) o después del check-out (h_fec_sda).
    Solo aplica a filas con reservación enlazada.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_timestamp", "h_fec_lld", "h_fec_sda"):
        return resultado

    ts      = pd.to_datetime(df["t_timestamp"], errors="coerce")
    llegada = pd.to_datetime(df["h_fec_lld"],   errors="coerce")
    salida  = pd.to_datetime(df["h_fec_sda"],   errors="coerce")
    fecha_cargo = ts.dt.normalize()

    tiene_res = (
        df["tiene_reservacion"].astype(bool)
        if "tiene_reservacion" in df.columns
        else llegada.notna()
    )

    antes_llegada  = tiene_res & (fecha_cargo < llegada)
    despues_salida = tiene_res & (fecha_cargo > salida)

    for pos in range(n):
        if antes_llegada.iloc[pos]:
            resultado["cluster"][pos].append("FUERA_DE_ESTANCIA")
            resultado["score"][pos]  += SCORES["FUERA_DE_ESTANCIA"]
            resultado["reglas"][pos].append(
                f"cargo_antes_llegada: fecha_cargo={fecha_cargo.iloc[pos].date()}, "
                f"check-in={llegada.iloc[pos].date()}"
            )
        if despues_salida.iloc[pos]:
            resultado["cluster"][pos].append("FUERA_DE_ESTANCIA")
            resultado["score"][pos]  += SCORES["FUERA_DE_ESTANCIA"]
            resultado["reglas"][pos].append(
                f"cargo_despues_checkout: fecha_cargo={fecha_cargo.iloc[pos].date()}, "
                f"check-out={salida.iloc[pos].date()}"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 5. Cancelaciones sospechosas
# --------------------------------------------------------------------------- #
def detectar_cancelaciones(df: pd.DataFrame) -> dict[str, list]:
    """
    Tres sub-reglas:
      a) Transacción cancelada con monto alto (> media+std de su código).
      b) Folio con ratio de cancelaciones > RATIO_CANCELACIONES_FOLIO.
      c) Cancelación seguida de re-posteo del mismo importe el mismo día.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_tra_cancelada", "t_monto"):
        return resultado

    cancelada = df["t_tra_cancelada"].astype("string").str.strip() == "1"
    monto_abs = _leer_monto(df).abs()

    # a) Cancelado con monto alto
    mask_cancel_alto = pd.Series(False, index=df.index)
    if _col_presente(df, "t_codigo"):
        codigo = df["t_codigo"].astype("string").str.strip().fillna("__NA__")
        stats  = df.assign(_m=monto_abs, _c=codigo).groupby("_c")["_m"].agg(media="mean", std="std").fillna(0)
        umbral = codigo.map(stats["media"]) + codigo.map(stats["std"].fillna(0))
        mask_cancel_alto = cancelada & (monto_abs > umbral)

    # b) Folio con ratio alto de cancelaciones
    mask_folio_cancel = pd.Series(False, index=df.index)
    ratio_cancel      = pd.Series(0.0, index=df.index)
    if _col_presente(df, "t_folio"):
        folio = df["t_folio"].astype("string").fillna("__NA__")
        total_folio  = folio.map(folio.value_counts())
        cancel_folio = folio.map(df.assign(_c=cancelada).groupby(folio)["_c"].sum())
        ratio_cancel = cancel_folio / total_folio.replace(0, np.nan)
        mask_folio_cancel = (ratio_cancel > RATIO_CANCELACIONES_FOLIO) & (total_folio > 2)

    # c) Cancelación + re-posteo
    mask_reposteo = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_folio", "t_codigo"):
        folio  = df["t_folio"].astype("string").fillna("__NA__")
        codigo = df["t_codigo"].astype("string").str.strip().fillna("__NA__")
        work   = pd.DataFrame({
            "folio":   folio,
            "codigo":  codigo,
            "monto_c": np.rint(monto_abs * 100).astype("int64"),
            "cancel":  cancelada.astype(int),
        })
        if "t_timestamp" in df.columns:
            work["fecha"] = pd.to_datetime(df["t_timestamp"]).dt.normalize()
        else:
            work["fecha"] = pd.Timestamp("1970-01-01")

        grp              = work.groupby(["folio", "codigo", "monto_c", "fecha"], dropna=False)
        tiene_cancelada  = grp["cancel"].transform("max") == 1
        tiene_activa     = grp["cancel"].transform(lambda s: (s == 0).any())
        mask_reposteo    = tiene_cancelada & tiene_activa

    for pos in range(n):
        cluster_agregado = False
        if mask_cancel_alto.iloc[pos]:
            resultado["cluster"][pos].append("CANCELACION_SOSPECHOSA")
            resultado["score"][pos]  += SCORES["CANCELACION_SOSPECHOSA"]
            resultado["reglas"][pos].append(
                f"cancelacion_monto_alto: monto cancelado={monto_abs.iloc[pos]:.2f}"
            )
            cluster_agregado = True
        if mask_folio_cancel.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CANCELACION_SOSPECHOSA")
                cluster_agregado = True
            resultado["score"][pos]  += SCORES["CANCELACION_SOSPECHOSA"]
            r = float(ratio_cancel.iloc[pos]) if pd.notna(ratio_cancel.iloc[pos]) else 0.0
            resultado["reglas"][pos].append(
                f"folio_alto_ratio_cancelaciones: {r:.0%} de movimientos cancelados en folio"
            )
        if mask_reposteo.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CANCELACION_SOSPECHOSA")
            resultado["score"][pos]  += SCORES["CANCELACION_SOSPECHOSA"]
            resultado["reglas"][pos].append(
                "cancelacion_con_reposteo: mismo folio/código/monto tiene versión cancelada y activa el mismo día"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 6. Inconsistencias de contexto de reservación
# --------------------------------------------------------------------------- #
def detectar_contexto_reservacion(df: pd.DataFrame) -> dict[str, list]:
    """
    Sub-reglas:
      a) Cargo sin reservación en código que normalmente sí la tiene (>90 % de casos).
      b) Noches del cargo muy diferentes a las noches de la reservación (delta > 2).
      c) Número de personas en el cargo diferente al de la reservación.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    # a) Código que casi siempre tiene reserva pero este cargo no la tiene
    mask_sin_res_inusual = pd.Series(False, index=df.index)
    if _cols_presentes(df, "tiene_reservacion", "t_codigo"):
        tiene_res = df["tiene_reservacion"].astype(bool)
        codigo    = df["t_codigo"].astype("string").str.strip().fillna("__NA__")
        pct_con_res = tiene_res.groupby(codigo).transform("mean")
        mask_sin_res_inusual = (~tiene_res) & (pct_con_res > 0.90)

    # b) Delta de noches mayor a 2
    mask_noches_delta = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_noches", "h_num_noc"):
        nc = pd.to_numeric(df["t_noches"],  errors="coerce")
        nr = pd.to_numeric(df["h_num_noc"], errors="coerce")
        mask_noches_delta = ((nc - nr).abs() > 2) & nc.notna() & nr.notna()

    # c) Número de personas inconsistente (t_num_per es string en hottra raw)
    mask_personas = pd.Series(False, index=df.index)
    if _cols_presentes(df, "h_num_per") and "t_num_per" in df.columns:
        pc = pd.to_numeric(df["t_num_per"], errors="coerce")
        pr = pd.to_numeric(df["h_num_per"], errors="coerce")
        mask_personas = ((pc - pr).abs() > 1) & pc.notna() & pr.notna()

    for pos in range(n):
        cluster_agregado = False
        if mask_sin_res_inusual.iloc[pos]:
            resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"sin_reservacion_en_codigo_habitual: código '{df['t_codigo'].iloc[pos]}' "
                "normalmente tiene reservación pero este cargo no"
            )
            cluster_agregado = True
        if mask_noches_delta.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
                cluster_agregado = True
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"noches_inconsistentes: cargo={df['t_noches'].iloc[pos]}, "
                f"reservación={df['h_num_noc'].iloc[pos]}"
            )
        if mask_personas.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"personas_inconsistentes: cargo={df['t_num_per'].iloc[pos]}, "
                f"reservación={df['h_num_per'].iloc[pos]}"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 7. Modificaciones sospechosas de usuario
# --------------------------------------------------------------------------- #
def detectar_modificaciones_usuario(df: pd.DataFrame) -> dict[str, list]:
    """
    Usuario modificador distinto al creador. Peso adicional si el monto es alto
    (> percentil 75 global de montos absolutos).
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_usuario", "t_usuario_mod", "t_monto"):
        return resultado

    usuario     = df["t_usuario"].astype("string").fillna("__NA__")
    usuario_mod = df["t_usuario_mod"].astype("string")
    monto_abs   = _leer_monto(df).abs()

    mod_distinto = usuario_mod.notna() & (usuario_mod.str.strip() != usuario.str.strip())
    p75         = float(monto_abs.quantile(0.75))
    monto_alto  = monto_abs > p75

    for pos in range(n):
        if mod_distinto.iloc[pos]:
            resultado["cluster"][pos].append("USUARIO_MODIFICACION")
            resultado["score"][pos]  += SCORES["USUARIO_MODIFICACION"]
            extra = " (monto alto)" if monto_alto.iloc[pos] else ""
            resultado["reglas"][pos].append(
                f"usuario_mod_distinto: creado por '{usuario.iloc[pos]}', "
                f"modificado por '{usuario_mod.iloc[pos]}'{extra}"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 8. Egresos / pagos sospechosos
# --------------------------------------------------------------------------- #
def detectar_pagos_sospechosos(df: pd.DataFrame) -> dict[str, list]:
    """
    Aplica ÚNICAMENTE a transacciones que representan egresos reales del hotel:
      - t_carabo = "1" (abono) Y t_codigo IN CODIGOS_EGRESO_HOTEL, O
      - t_carabo = "1" Y sin reservación vinculada Y código NO conocido como
        medio de pago de huésped (CODIGOS_PAGO_HUESPED).

    Para pagos a proveedores externos (facturas, insumos) se requeriría una
    tabla adicional que no forma parte de hottra; este detector los aceptaría
    como un DataFrame separado en versiones futuras.

    Sub-reglas dentro del subconjunto de egresos:
      a) Duplicado: mismo proveedor/referencia, monto y fecha.
      b) Monto fuera del rango histórico del proveedor (±2 std).
      c) Proveedor poco frecuente (< MIN_FRECUENCIA_PROVEEDOR transacciones).
      d) Pago en horario inusual (fuera de HORA_MIN_NORMAL–HORA_MAX_NORMAL).
      e) Pagos divididos: varios montos similares al mismo proveedor el mismo día.
      f) Pago sin referencia.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _col_presente(df, "t_carabo"):
        return resultado

    es_abono = _leer_carabo_abono(df)
    codigo   = df["t_codigo"].astype("string").str.strip().fillna("__NA__") if "t_codigo" in df.columns else pd.Series("__NA__", index=df.index, dtype="string")
    tiene_res = df["tiene_reservacion"].astype(bool) if "tiene_reservacion" in df.columns else pd.Series(False, index=df.index)

    # Identificar egresos: abono con código de devolución, o abono sin reserva y sin
    # código estándar de pago de huésped.
    es_egreso = es_abono & (
        codigo.isin(CODIGOS_EGRESO_HOTEL) |
        (~tiene_res & ~codigo.isin(CODIGOS_PAGO_HUESPED))
    )

    if not es_egreso.any():
        return resultado

    monto_abs = _leer_monto(df).abs()

    # Proveedor identificado por t_referencia (código más descriptivo disponible)
    if "t_referencia" in df.columns:
        proveedor = df["t_referencia"].astype("string").str.strip().fillna("__SIN_REF__")
    else:
        proveedor = codigo  # fallback al código de concepto

    if "t_timestamp" in df.columns:
        fecha = pd.to_datetime(df["t_timestamp"]).dt.normalize()
        hora  = pd.to_datetime(df["t_timestamp"]).dt.hour
    elif "t_fecha" in df.columns:
        fecha = pd.to_datetime(df["t_fecha"], errors="coerce").dt.normalize()
        hora  = pd.Series(12, index=df.index)
    else:
        fecha = pd.Series(pd.Timestamp("1970-01-01"), index=df.index)
        hora  = pd.Series(12, index=df.index)

    monto_c = np.rint(monto_abs * 100).astype("int64")

    # a) Duplicado dentro de egresos
    dup_work = pd.DataFrame({"prov": proveedor, "monto_c": monto_c, "fecha": fecha, "egreso": es_egreso})
    dup_count = dup_work[es_egreso].groupby(["prov", "monto_c", "fecha"], dropna=False)["prov"].transform("size")
    dup_mask  = pd.Series(False, index=df.index)
    dup_mask[es_egreso] = dup_count > 1

    # b) Monto fuera del rango histórico del proveedor
    fuera_rango_mask = pd.Series(False, index=df.index)
    z_prov = pd.Series(0.0, index=df.index)
    egreso_idx = es_egreso[es_egreso].index
    if len(egreso_idx) > 0:
        prov_stats = (
            df.loc[egreso_idx]
            .assign(_m=monto_abs[egreso_idx], _p=proveedor[egreso_idx])
            .groupby("_p")["_m"]
            .agg(media="mean", std="std")
            .fillna({"std": 0})
        )
        if not prov_stats.empty:
            prov_media = proveedor.map(prov_stats["media"])
            prov_std   = proveedor.map(prov_stats["std"]).fillna(0).replace(0, 1.0)
            z_prov     = (monto_abs - prov_media) / prov_std
            fuera_rango_mask = es_egreso & (z_prov.abs() > Z_UMBRAL) & prov_media.notna()

    # c) Proveedor poco frecuente
    frec_prov      = proveedor[es_egreso].value_counts()
    prov_poco_frec = es_egreso & (proveedor.map(frec_prov).fillna(0) < MIN_FRECUENCIA_PROVEEDOR)

    # d) Horario inusual
    hora_inusual = es_egreso & ((hora < HORA_MIN_NORMAL) | (hora > HORA_MAX_NORMAL))

    # e) Pagos divididos
    div_mask = pd.Series(False, index=df.index)
    if len(egreso_idx) > 0:
        div_work = pd.DataFrame({
            "_p": proveedor[egreso_idx],
            "_f": fecha[egreso_idx],
            "_m_round": (monto_abs[egreso_idx] / (1 + UMBRAL_PAGO_DIVIDIDO)).round(-1),
        })
        conteo_div = div_work.groupby(["_p", "_f", "_m_round"], dropna=False)["_p"].transform("size")
        div_mask[div_work.index[conteo_div >= MIN_PAGOS_DIVIDIDOS]] = True

    # f) Sin referencia
    sin_ref_mask = pd.Series(False, index=df.index)
    if "t_referencia" in df.columns:
        ref = df["t_referencia"].astype("string").str.strip()
        sin_ref_mask = es_egreso & (ref.isna() | ref.isin(["", "nan", "None", "0", "__SIN_REF__"]))

    for pos in range(n):
        if not es_egreso.iloc[pos]:
            continue
        mensajes: list[str] = []
        if dup_mask.iloc[pos]:
            mensajes.append(
                f"egreso_duplicado: misma referencia/monto/fecha "
                f"({proveedor.iloc[pos]}, {monto_abs.iloc[pos]:.2f})"
            )
        if fuera_rango_mask.iloc[pos]:
            mensajes.append(
                f"egreso_fuera_rango: monto={monto_abs.iloc[pos]:.2f}, "
                f"z={float(z_prov.iloc[pos]):.2f} para referencia '{proveedor.iloc[pos]}'"
            )
        if prov_poco_frec.iloc[pos]:
            frec = int(frec_prov.get(proveedor.iloc[pos], 0))
            mensajes.append(
                f"referencia_poco_frecuente: '{proveedor.iloc[pos]}' aparece {frec} veces en egresos"
            )
        if hora_inusual.iloc[pos]:
            mensajes.append(
                f"egreso_hora_inusual: hora={hora.iloc[pos]}:00 (fuera de {HORA_MIN_NORMAL}:00–{HORA_MAX_NORMAL}:00)"
            )
        if div_mask.iloc[pos]:
            mensajes.append(
                "egreso_dividido: varios montos similares a la misma referencia el mismo día"
            )
        if sin_ref_mask.iloc[pos]:
            mensajes.append("egreso_sin_referencia: abono sin número de referencia")

        if mensajes:
            resultado["cluster"][pos].append("PAGO_PROVEEDOR_SOSPECHOSO")
            resultado["score"][pos]  += SCORES["PAGO_PROVEEDOR_SOSPECHOSO"]
            resultado["reglas"][pos].extend(mensajes)

    return resultado


# --------------------------------------------------------------------------- #
# Score, nivel y mensaje
# --------------------------------------------------------------------------- #
def calcular_score_riesgo(scores_raw: list[int]) -> pd.Series:
    return pd.Series(scores_raw, dtype="int32")


def asignar_cluster_anomalia(clusters: list[list[str]]) -> pd.Series:
    def _fmt(c: list[str]) -> str:
        return " | ".join(dict.fromkeys(c)) if c else ""
    return pd.Series([_fmt(c) for c in clusters], dtype="string")


def generar_mensaje_alerta(
    row_id: Any,
    folio: Any,
    codigo: Any,
    monto: float,
    clusters: list[str],
    reglas: list[str],
) -> str:
    if not clusters:
        return ""
    cluster_str = " | ".join(dict.fromkeys(clusters))
    reglas_str  = "; ".join(reglas[:3])
    return (
        f"Alerta en transacción {row_id} (folio {folio}, código {codigo}, "
        f"monto {monto:.2f}): [{cluster_str}] — {reglas_str}"
    )


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def detectar_anomalias(
    df: pd.DataFrame,
    id_col: str = "t_transaccion",
) -> pd.DataFrame:
    """
    Ejecuta todas las reglas y devuelve una tabla de alertas.

    Parámetros
    ----------
    df      : DataFrame con columnas de hottra (raw o consolidado).
              Columnas mínimas requeridas: t_folio, t_codigo, t_monto, t_carabo.
              Columnas opcionales: todas las demás enriquecen las reglas.
    id_col  : Columna que se usa como id_transaccion en el resultado.

    Retorna
    -------
    DataFrame con columnas:
        id_transaccion, folio, codigo, monto, fecha,
        es_anomalia, cluster_anomalia, nivel_riesgo,
        mensaje_alerta, reglas_activadas, score_riesgo
    """
    n = len(df)
    base = _resultado_vacio(n)

    for detector in [
        detectar_duplicados,
        detectar_signo_contable,
        detectar_montos_atipicos,
        detectar_fuera_estancia,
        detectar_cancelaciones,
        detectar_contexto_reservacion,
        detectar_modificaciones_usuario,
        detectar_pagos_sospechosos,
    ]:
        _merge(base, detector(df))

    # Columnas de salida
    id_tx  = df[id_col].astype("string")  if id_col  in df.columns else pd.Series(range(n), dtype="string")
    folio  = df["t_folio"].astype("string")  if "t_folio"  in df.columns else pd.Series("", index=df.index, dtype="string")
    codigo = df["t_codigo"].astype("string").str.strip() if "t_codigo" in df.columns else pd.Series("", index=df.index, dtype="string")
    monto  = _leer_monto(df) if "t_monto" in df.columns else pd.Series(0.0, index=df.index)

    if "t_timestamp" in df.columns:
        fecha = pd.to_datetime(df["t_timestamp"], errors="coerce").dt.date.astype("string")
    elif "t_fecha" in df.columns:
        fecha = pd.to_datetime(df["t_fecha"], errors="coerce").dt.date.astype("string")
    else:
        fecha = pd.Series("", index=df.index, dtype="string")

    scores   = calcular_score_riesgo(base["score"])
    clusters = asignar_cluster_anomalia(base["cluster"])
    es_anom  = scores > 0
    niveles  = pd.Series([_asignar_nivel(s) for s in base["score"]], dtype="string")

    mensajes = pd.Series([
        generar_mensaje_alerta(
            id_tx.iloc[i], folio.iloc[i], codigo.iloc[i],
            float(monto.iloc[i]), base["cluster"][i], base["reglas"][i],
        )
        for i in range(n)
    ], dtype="string")

    reglas_str = pd.Series(
        [" | ".join(r) if r else "" for r in base["reglas"]],
        dtype="string",
    )

    return pd.DataFrame({
        "id_transaccion":   id_tx.values,
        "folio":            folio.values,
        "codigo":           codigo.values,
        "monto":            monto.values,
        "fecha":            fecha.values,
        "es_anomalia":      es_anom.values,
        "cluster_anomalia": clusters.values,
        "nivel_riesgo":     niveles.values,
        "mensaje_alerta":   mensajes.values,
        "reglas_activadas": reglas_str.values,
        "score_riesgo":     scores.values,
    })
