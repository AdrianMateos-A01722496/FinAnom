"""Detección de anomalías financieras hoteleras por reglas de negocio.

Modelo basado 100 % en lógica de negocio (sin ML).

Convención de signo en este PMS (TSA / HOTTRA):
    t_carabo = "0"  → CARGO   (débito al folio; t_monto ≥ 0 es normal)
    t_carabo = "1"  → ABONO   (crédito al folio; t_monto ≥ 0 también es normal)
    t_monto < 0     → Reversal forzado — solo anómalo cuando el código NO es un
                      reversal/ajuste esperado Y la transacción NO está cancelada.

Dos niveles de salida:
    alertas_operativas  : score ≥ SCORE_MIN_OPERATIVO — para revisión inmediata.
    senales_contexto    : 0 < score < SCORE_MIN_OPERATIVO — señal débil, loggear
                          pero no exponer en el dashboard todavía.

USUARIO_MODIFICACION es siempre contextual: aparece en el mensaje de alerta
si hay otra regla activa, pero NO suma al score por sí sola.

Uso rápido:
    from anomaly_detection.reglas import detectar_anomalias, separar_alertas
    alertas = detectar_anomalias(df)
    operativas, contexto = separar_alertas(alertas)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Parámetros y umbrales
# --------------------------------------------------------------------------- #
SCORES: dict[str, int] = {
    "DUPLICADO":               40,
    "SIGNO_CONTABLE":          35,
    "FUERA_DE_ESTANCIA":       35,
    "MONTO_ATIPICO":           25,
    "CANCELACION_SOSPECHOSA":  25,
    "CONTEXTO_RESERVACION":    20,
    "PAGO_PROVEEDOR_SOSPECHOSO": 30,
    # USUARIO_MODIFICACION no suma al score: es contextual
}

# Umbral mínimo para considerar una alerta "operativa" (revisión inmediata)
SCORE_MIN_OPERATIVO: int = 25

NIVEL_RIESGO: list[tuple[int, str]] = [
    (80, "CRITICO"),
    (50, "ALTO"),
    (20, "MEDIO"),
    (0,  "BAJO"),
]

Z_UMBRAL                 = 2.0
RATIO_CANCELACIONES_FOLIO = 0.5
MIN_FRECUENCIA_PROVEEDOR  = 3
HORA_MIN_NORMAL           = 6
HORA_MAX_NORMAL           = 23
MIN_PAGOS_DIVIDIDOS       = 2
UMBRAL_PAGO_DIVIDIDO      = 0.05


# --------------------------------------------------------------------------- #
# Catálogos de códigos (inferidos del análisis de hottra real)
# --------------------------------------------------------------------------- #

# Códigos donde monto negativo ES ESPERADO (reversales, ajustes, cancelaciones).
# Un monto negativo en cualquiera de estos NO se marca como SIGNO_CONTABLE.
CODIGOS_REVERSAL_NORMAL: frozenset[str] = frozenset({
    # Cancelaciones explícitas de movimientos previos
    "CANCXC",   # Cancelación de CXC (cuenta por cobrar)
    "CANXFA",   # Cancelación de XFAC (factura electrónica)
    # Ajustes de cualquier tipo (AJ*)
    "AJCUPN", "AJDEEF", "AJMS", "AJPEXT", "AJPROP",
    "AJPAST", "AJPO", "AJUUPS", "AJPROP",
    # Ajustes de renta / saldo
    "RENAJU", "RENAJG", "SAAJU", "SALANT",
    # Devoluciones explícitas al huésped (DEV*)
    "DEVTRE", "DEVTRJ", "DEVTRD", "DEVEFF", "DEVXBA", "DEVCHE", "DEVTRB",
    # Depósitos (su reversal negativo = devolución de depósito)
    "DEPHAB", "DEPRES", "DEPEFE", "DEPOS", "DEPCHE", "DEPXBA",
    # Medios de pago y créditos cuyo negativo = refund autorizado
    "EFE", "TARCRE", "TARDEB", "XFAC", "TRANSF", "AMEXCO", "XFACTOT",
    "RESDEP", "CUPON", "CXC", "EMPCXC", "CUXAC",
    # Retenciones fiscales
    "RETIV2", "RETIS2",
})

# Medios de pago estándar de huéspedes (abonos normales, no son egresos del hotel)
CODIGOS_PAGO_HUESPED: frozenset[str] = frozenset({
    "EFE", "TARCRE", "TARDEB", "AMEXCO", "TRANSF", "XFAC", "XFACTOT",
    "RESDEP", "DEPEFE", "DEPOS", "DEPRES", "DEPCHE", "DEPXBA",
    "CUPON", "CXC", "CANCXC", "EMPCXC", "SALANT", "CUXAC",
    "RENAJU", "RENAJG", "SAAJU",
    "AJMS", "AJPEXT", "AJPROP", "AJPAST", "AJPO", "AJDEEF", "AJCUPN",
    "AJUUPS", "AGALI", "MENAD",
    "RETIV2", "RETIS2",
})

# Devoluciones que sí representan egresos del hotel hacia el huésped
CODIGOS_EGRESO_HOTEL: frozenset[str] = frozenset({
    "DEVTRE", "DEVTRJ", "DEVTRD", "DEVEFF", "DEVXBA", "DEVCHE", "DEVTRB",
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
        "cluster":  [[] for _ in range(n)],
        "contexto": [[] for _ in range(n)],  # señales sin score
        "score":    [0] * n,
        "reglas":   [[] for _ in range(n)],
    }


def _merge(base: dict, nuevo: dict) -> None:
    for i in range(len(base["score"])):
        base["cluster"][i].extend(nuevo["cluster"][i])
        base["contexto"][i].extend(nuevo["contexto"][i])
        base["score"][i]    += nuevo["score"][i]
        base["reglas"][i].extend(nuevo["reglas"][i])


def _leer_monto(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0)


def _leer_carabo_abono(df: pd.DataFrame) -> pd.Series:
    return df["t_carabo"].astype("string").str.strip() == "1"


def _leer_cancelada(df: pd.DataFrame) -> pd.Series:
    """True si la transacción está marcada como cancelada."""
    return df["t_tra_cancelada"].astype("string").str.strip() == "1"


def _leer_codigo(df: pd.DataFrame) -> pd.Series:
    return df["t_codigo"].astype("string").str.strip().fillna("__NA__")


def _leer_usuario_mod_distinto(df: pd.DataFrame) -> pd.Series:
    """
    True si t_usuario_mod existe (non-empty), fue modificado, y es DISTINTO al creador.
    Maneja el caso frecuente donde t_usuario_mod es string vacío '' (≠ null),
    lo cual en este PMS significa 'sin modificador'.
    """
    usr     = df["t_usuario"].astype("string").str.strip().fillna("__NA__")
    usr_mod = df["t_usuario_mod"].astype("string").str.strip()
    # String vacío = sin modificador (igual que null en este contexto)
    tiene_mod = usr_mod.notna() & usr_mod.ne("")
    return tiene_mod & usr_mod.ne(usr)


# --------------------------------------------------------------------------- #
# 1. Duplicados
# --------------------------------------------------------------------------- #
def detectar_duplicados(df: pd.DataFrame) -> dict[str, list]:
    """
    Duplicado de alta confianza: mismo folio, subfolio, código, naturaleza contable,
    monto, habitación y minuto.

    La coincidencia por día tiene alto recall, pero suele mezclar cargos legitimos
    repetidos. Por eso queda como contexto si no hay evidencia fuerte.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_folio", "t_codigo", "t_monto", "t_cuarto"):
        return resultado

    monto = _leer_monto(df)
    work = pd.DataFrame({
        "folio":   df["t_folio"].astype("string").fillna("__NA__"),
        "folio_ext": df["t_folio_ext"].astype("string").fillna("__NA__") if "t_folio_ext" in df.columns else "__NA__",
        "codigo":  _leer_codigo(df),
        "carabo": df["t_carabo"].astype("string").str.strip().fillna("__NA__") if "t_carabo" in df.columns else "__NA__",
        "cuarto":  df["t_cuarto"].astype("string").fillna("__NA__"),
        "monto_c": np.rint(monto * 100).astype("int64"),
    }, index=df.index)

    if "t_timestamp" in df.columns:
        ts = pd.to_datetime(df["t_timestamp"], errors="coerce")
        work["fecha"] = ts.dt.normalize()
        work["minuto"] = ts.dt.floor("min")
    elif "t_fecha" in df.columns:
        ts = pd.to_datetime(df["t_fecha"], errors="coerce")
        work["fecha"] = ts.dt.normalize()
        work["minuto"] = ts.dt.floor("min")
    else:
        work["fecha"] = pd.Timestamp("1970-01-01")
        work["minuto"] = pd.Timestamp("1970-01-01")

    key_dia = ["folio", "folio_ext", "codigo", "monto_c", "cuarto"]
    key_minuto = ["folio", "folio_ext", "cuarto", "codigo", "carabo", "monto_c", "minuto"]
    conteos_dia = work.groupby(
        key_dia + ["fecha"], dropna=False
    )["folio"].transform("size")
    conteos_minuto = work.groupby(
        key_minuto, dropna=False
    )["folio"].transform("size")
    mask_dup_dia = conteos_dia > 1
    mask_dup_fuerte = mask_dup_dia & (conteos_minuto > 1)

    for pos in range(n):
        if mask_dup_fuerte.iloc[pos]:
            resultado["cluster"][pos].append("DUPLICADO")
            resultado["score"][pos]  += SCORES["DUPLICADO"]
            resultado["reglas"][pos].append(
                "duplicado_alta_confianza: "
                f"{int(conteos_minuto.iloc[pos])} veces el mismo folio/subfolio/codigo/"
                "naturaleza/monto/habitacion/minuto"
            )
        elif mask_dup_dia.iloc[pos]:
            resultado["contexto"][pos].append(
                "DUPLICADO_DIA: repeticion por folio/codigo/monto/habitacion/dia; "
                "no se suma score porque no coincide mismo minuto ni referencia"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 2. Inconsistencias de signo contable
# --------------------------------------------------------------------------- #
def detectar_signo_contable(df: pd.DataFrame) -> dict[str, list]:
    """
    Monto negativo en una transacción NO cancelada cuyo código NO pertenece
    al catálogo CODIGOS_REVERSAL_NORMAL.

    Lógica:
      - Si t_tra_cancelada == '1' → el sistema ya lo marcó; no es anomalía.
      - Si t_codigo ∈ CODIGOS_REVERSAL_NORMAL → el negativo es esperado (ajuste/reversal).
      - En cualquier otro caso: monto negativo = signo contable incorrecto.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_monto"):
        return resultado

    monto     = _leer_monto(df)
    cancelada = _leer_cancelada(df) if "t_tra_cancelada" in df.columns else pd.Series(False, index=df.index)
    codigo    = _leer_codigo(df)    if "t_codigo"        in df.columns else pd.Series("__NA__", index=df.index, dtype="string")

    es_reversal_normal = codigo.isin(CODIGOS_REVERSAL_NORMAL)
    mask_anomalo       = (monto < 0) & ~cancelada & ~es_reversal_normal

    es_abono = _leer_carabo_abono(df) if "t_carabo" in df.columns else pd.Series(False, index=df.index)

    for pos in range(n):
        if mask_anomalo.iloc[pos]:
            resultado["cluster"][pos].append("SIGNO_CONTABLE")
            resultado["score"][pos]  += SCORES["SIGNO_CONTABLE"]
            tipo = "ABONO" if es_abono.iloc[pos] else "CARGO"
            resultado["reglas"][pos].append(
                f"monto_negativo_inesperado: monto={monto.iloc[pos]:.2f} en {tipo} "
                f"código '{codigo.iloc[pos]}' — no cancelado y no es código de reversal"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 3. Montos atípicos
# --------------------------------------------------------------------------- #
def detectar_montos_atipicos(df: pd.DataFrame, z_umbral: float = Z_UMBRAL) -> dict[str, list]:
    """
    Monto (valor absoluto) fuera de media ± z_umbral * std para su t_codigo.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_monto", "t_codigo"):
        return resultado

    monto  = _leer_monto(df).abs()
    codigo = _leer_codigo(df)

    stats  = (
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
    """Cargo antes del check-in o después del check-out (solo con reserva enlazada)."""
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_timestamp", "h_fec_lld", "h_fec_sda"):
        return resultado

    ts          = pd.to_datetime(df["t_timestamp"], errors="coerce")
    llegada     = pd.to_datetime(df["h_fec_lld"],   errors="coerce")
    salida      = pd.to_datetime(df["h_fec_sda"],   errors="coerce")
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
    a) Transacción cancelada con monto alto (> media+std de su código).
    b) Folio con ratio de cancelaciones > RATIO_CANCELACIONES_FOLIO.
    c) Cancelación + re-posteo del mismo importe el mismo día.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_tra_cancelada", "t_monto"):
        return resultado

    cancelada = _leer_cancelada(df)
    monto_abs = _leer_monto(df).abs()

    # a) Cancelado con monto alto
    mask_cancel_alto = pd.Series(False, index=df.index)
    if _col_presente(df, "t_codigo"):
        codigo = _leer_codigo(df)
        stats  = df.assign(_m=monto_abs, _c=codigo).groupby("_c")["_m"].agg(media="mean", std="std").fillna(0)
        umbral = codigo.map(stats["media"]) + codigo.map(stats["std"].fillna(0))
        mask_cancel_alto = cancelada & (monto_abs > umbral)

    # b) Folio con ratio alto de cancelaciones (mínimo 3 movimientos en el folio)
    mask_folio_cancel = pd.Series(False, index=df.index)
    ratio_cancel      = pd.Series(0.0, index=df.index)
    if _col_presente(df, "t_folio"):
        folio        = df["t_folio"].astype("string").fillna("__NA__")
        total_folio  = folio.map(folio.value_counts())
        cancel_folio = folio.map(df.assign(_c=cancelada).groupby(folio)["_c"].sum())
        ratio_cancel  = cancel_folio / total_folio.replace(0, np.nan)
        mask_folio_cancel = (ratio_cancel > RATIO_CANCELACIONES_FOLIO) & (total_folio > 2)

    # c) Cancelación + re-posteo
    mask_reposteo = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_folio", "t_codigo"):
        folio  = df["t_folio"].astype("string").fillna("__NA__")
        codigo = _leer_codigo(df)
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
        grp             = work.groupby(["folio", "codigo", "monto_c", "fecha"], dropna=False)
        tiene_cancelada = grp["cancel"].transform("max") == 1
        tiene_activa    = grp["cancel"].transform(lambda s: (s == 0).any())
        mask_reposteo   = tiene_cancelada & tiene_activa

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
                f"folio_alto_ratio_cancelaciones: {r:.0%} de movimientos cancelados"
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
    a) Código que normalmente tiene reserva, pero este cargo no la tiene (>90 %).
    b) Noches del cargo vs noches de la reservación (delta > 2).
    c) Número de personas inconsistente (delta > 1).
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    mask_sin_res = pd.Series(False, index=df.index)
    if _cols_presentes(df, "tiene_reservacion", "t_codigo"):
        tiene_res   = df["tiene_reservacion"].astype(bool)
        codigo      = _leer_codigo(df)
        pct_con_res = tiene_res.groupby(codigo).transform("mean")
        mask_sin_res = (~tiene_res) & (pct_con_res > 0.90)

    mask_noches = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_noches", "h_num_noc"):
        nc = pd.to_numeric(df["t_noches"],  errors="coerce")
        nr = pd.to_numeric(df["h_num_noc"], errors="coerce")
        mask_noches = ((nc - nr).abs() > 2) & nc.notna() & nr.notna()

    mask_personas = pd.Series(False, index=df.index)
    if _cols_presentes(df, "h_num_per") and "t_num_per" in df.columns:
        pc = pd.to_numeric(df["t_num_per"], errors="coerce")
        pr = pd.to_numeric(df["h_num_per"], errors="coerce")
        mask_personas = ((pc - pr).abs() > 1) & pc.notna() & pr.notna()

    for pos in range(n):
        cluster_agregado = False
        if mask_sin_res.iloc[pos]:
            resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"sin_reservacion_en_codigo_habitual: código '{df['t_codigo'].iloc[pos]}'"
            )
            cluster_agregado = True
        if mask_noches.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
                cluster_agregado = True
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"noches_inconsistentes: cargo={df['t_noches'].iloc[pos]}, reserva={df['h_num_noc'].iloc[pos]}"
            )
        if mask_personas.iloc[pos]:
            if not cluster_agregado:
                resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"personas_inconsistentes: cargo={df['t_num_per'].iloc[pos]}, reserva={df['h_num_per'].iloc[pos]}"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 7. Modificaciones de usuario — CONTEXTUAL (sin score propio)
# --------------------------------------------------------------------------- #
def detectar_modificaciones_usuario(df: pd.DataFrame) -> dict[str, list]:
    """
    Detecta cuando t_usuario_mod es distinto al creador (t_usuario).

    IMPORTANTE: Esta señal NO suma al score de anomalía por sí sola.
    Aparece en las reglas de contexto y enriquece el mensaje de alerta
    solo si otra regla ya activó un score ≥ SCORE_MIN_OPERATIVO.

    Nota técnica: t_usuario_mod viene como string vacío '' cuando no hay
    modificador (NO como null). El check filtra explícitamente los vacíos.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _cols_presentes(df, "t_usuario", "t_usuario_mod"):
        return resultado

    mod_distinto = _leer_usuario_mod_distinto(df)
    monto_abs    = _leer_monto(df).abs() if "t_monto" in df.columns else pd.Series(0.0, index=df.index)
    p75          = float(monto_abs.quantile(0.75))
    usuario      = df["t_usuario"].astype("string").str.strip().fillna("__NA__")
    usuario_mod  = df["t_usuario_mod"].astype("string").str.strip()

    for pos in range(n):
        if mod_distinto.iloc[pos]:
            extra = " (monto alto)" if monto_abs.iloc[pos] > p75 else ""
            # Solo se agrega a "contexto", NO a "cluster" ni al score
            resultado["contexto"][pos].append(
                f"USUARIO_MODIFICACION: creado por '{usuario.iloc[pos]}', "
                f"modificado por '{usuario_mod.iloc[pos]}'{extra}"
            )
    return resultado


# --------------------------------------------------------------------------- #
# 8. Egresos / pagos sospechosos
# --------------------------------------------------------------------------- #
def detectar_pagos_sospechosos(df: pd.DataFrame) -> dict[str, list]:
    """
    Solo aplica a abonos que representan egresos del hotel (códigos DEV* o
    abonos sin reserva y fuera del catálogo CODIGOS_PAGO_HUESPED).
    Para pagos a proveedores externos se requiere una tabla adicional.
    """
    n = len(df)
    resultado = _resultado_vacio(n)
    if not _col_presente(df, "t_carabo"):
        return resultado

    es_abono  = _leer_carabo_abono(df)
    codigo    = _leer_codigo(df) if "t_codigo" in df.columns else pd.Series("__NA__", index=df.index, dtype="string")
    tiene_res = df["tiene_reservacion"].astype(bool) if "tiene_reservacion" in df.columns else pd.Series(False, index=df.index)

    es_egreso = es_abono & (
        codigo.isin(CODIGOS_EGRESO_HOTEL) |
        (~tiene_res & ~codigo.isin(CODIGOS_PAGO_HUESPED))
    )
    if not es_egreso.any():
        return resultado

    monto_abs   = _leer_monto(df).abs()
    proveedor   = df["t_referencia"].astype("string").str.strip().fillna("__SIN_REF__") if "t_referencia" in df.columns else codigo

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

    # a) Duplicado
    dup_work  = pd.DataFrame({"prov": proveedor, "monto_c": monto_c, "fecha": fecha})
    dup_count = dup_work[es_egreso].groupby(["prov", "monto_c", "fecha"], dropna=False)["prov"].transform("size")
    dup_mask  = pd.Series(False, index=df.index)
    dup_mask[es_egreso] = dup_count > 1

    # b) Monto fuera del rango histórico del proveedor
    fuera_rango_mask = pd.Series(False, index=df.index)
    z_prov           = pd.Series(0.0, index=df.index)
    egreso_idx       = es_egreso[es_egreso].index
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
            mensajes.append(f"egreso_duplicado: ({proveedor.iloc[pos]}, {monto_abs.iloc[pos]:.2f})")
        if fuera_rango_mask.iloc[pos]:
            mensajes.append(f"egreso_fuera_rango: monto={monto_abs.iloc[pos]:.2f}, z={float(z_prov.iloc[pos]):.2f}")
        if prov_poco_frec.iloc[pos]:
            mensajes.append(f"referencia_poco_frecuente: '{proveedor.iloc[pos]}' ({int(frec_prov.get(proveedor.iloc[pos], 0))} veces)")
        if hora_inusual.iloc[pos]:
            mensajes.append(f"egreso_hora_inusual: {hora.iloc[pos]}:00")
        if div_mask.iloc[pos]:
            mensajes.append("egreso_dividido: montos similares al mismo destino mismo día")
        if sin_ref_mask.iloc[pos]:
            mensajes.append("egreso_sin_referencia")
        if mensajes:
            resultado["cluster"][pos].append("PAGO_PROVEEDOR_SOSPECHOSO")
            resultado["score"][pos]  += SCORES["PAGO_PROVEEDOR_SOSPECHOSO"]
            resultado["reglas"][pos].extend(mensajes)
    return resultado


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def _asignar_cluster_str(clusters: list[list[str]]) -> pd.Series:
    def _fmt(c: list[str]) -> str:
        return " | ".join(dict.fromkeys(c)) if c else ""
    return pd.Series([_fmt(c) for c in clusters], dtype="string")


def _generar_mensaje(
    row_id: Any, folio: Any, codigo: Any, monto: float,
    clusters: list[str], contextos: list[str], reglas: list[str],
) -> str:
    if not clusters and not contextos:
        return ""
    partes = []
    if clusters:
        partes.append(f"[{' | '.join(dict.fromkeys(clusters))}]")
    reglas_str = "; ".join(reglas[:3])
    if reglas_str:
        partes.append(reglas_str)
    if contextos:
        partes.append("| ctx: " + " / ".join(contextos[:2]))
    return (
        f"Alerta tx {row_id} (folio {folio}, cód {codigo}, monto {monto:.2f}): "
        + " — ".join(partes)
    )


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def detectar_anomalias(df: pd.DataFrame, id_col: str = "t_transaccion") -> pd.DataFrame:
    """
    Ejecuta todas las reglas y devuelve una tabla unificada con:

    Columnas de identidad:
        id_transaccion, folio, codigo, monto, fecha

    Columnas de anomalía:
        es_anomalia           : cualquier score > 0 o contexto detectado
        es_alerta_operativa   : score ≥ SCORE_MIN_OPERATIVO (auditoría inmediata)
        es_senal_contexto     : 0 < score < SCORE_MIN_OPERATIVO (señal débil)
        cluster_anomalia      : códigos de regla activos (solo los que suman score)
        señales_contexto_str  : señales sin score (ej. USUARIO_MODIFICACION)
        nivel_riesgo          : BAJO/MEDIO/ALTO/CRITICO por score
        score_riesgo          : entero acumulado
        reglas_activadas      : descripción detallada de cada regla disparada
        mensaje_alerta        : texto legible para auditor
    """
    n   = len(df)
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

    id_tx  = df[id_col].astype("string")     if id_col  in df.columns else pd.Series(range(n), dtype="string")
    folio  = df["t_folio"].astype("string")  if "t_folio"  in df.columns else pd.Series("", index=df.index, dtype="string")
    codigo = _leer_codigo(df)                if "t_codigo" in df.columns else pd.Series("", index=df.index, dtype="string")
    monto  = _leer_monto(df)                 if "t_monto"  in df.columns else pd.Series(0.0, index=df.index)

    if "t_timestamp" in df.columns:
        fecha = pd.to_datetime(df["t_timestamp"], errors="coerce").dt.date.astype("string")
    elif "t_fecha" in df.columns:
        fecha = pd.to_datetime(df["t_fecha"], errors="coerce").dt.date.astype("string")
    else:
        fecha = pd.Series("", index=df.index, dtype="string")

    scores   = pd.Series(base["score"], dtype="int32")
    clusters = _asignar_cluster_str(base["cluster"])
    contexto_str = pd.Series(
        [" / ".join(c) if c else "" for c in base["contexto"]], dtype="string"
    )

    hay_anomalia   = (scores > 0) | contexto_str.str.len().gt(0)
    es_operativa   = scores >= SCORE_MIN_OPERATIVO
    es_contexto    = (scores > 0) & ~es_operativa

    niveles  = pd.Series([_asignar_nivel(s) for s in base["score"]], dtype="string")
    mensajes = pd.Series([
        _generar_mensaje(
            id_tx.iloc[i], folio.iloc[i], codigo.iloc[i], float(monto.iloc[i]),
            base["cluster"][i], base["contexto"][i], base["reglas"][i],
        )
        for i in range(n)
    ], dtype="string")
    reglas_str = pd.Series(
        [" | ".join(r) if r else "" for r in base["reglas"]], dtype="string"
    )

    return pd.DataFrame({
        "id_transaccion":       id_tx.values,
        "folio":                folio.values,
        "codigo":               codigo.values,
        "monto":                monto.values,
        "fecha":                fecha.values,
        "es_anomalia":          hay_anomalia.values,
        "es_alerta_operativa":  es_operativa.values,
        "es_senal_contexto":    es_contexto.values,
        "cluster_anomalia":     clusters.values,
        "senales_contexto":     contexto_str.values,
        "nivel_riesgo":         niveles.values,
        "score_riesgo":         scores.values,
        "reglas_activadas":     reglas_str.values,
        "mensaje_alerta":       mensajes.values,
    })


def separar_alertas(alertas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el resultado de detectar_anomalias() en dos DataFrames:

    Returns
    -------
    alertas_operativas : score ≥ SCORE_MIN_OPERATIVO — para dashboard y revisión inmediata.
    senales_contexto   : señales débiles (solo CONTEXTO_RESERVACION o señales sin score)
                         — para logging o revisión cuando hay capacidad.
    """
    operativas = alertas[alertas["es_alerta_operativa"]].copy()
    contexto   = alertas[alertas["es_anomalia"] & ~alertas["es_alerta_operativa"]].copy()
    return operativas, contexto
