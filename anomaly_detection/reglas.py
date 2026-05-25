"""Detección de anomalías financieras hoteleras por reglas de negocio.

Modelo inicial basado 100% en lógica de negocio (sin ML).
Cada función detecta una categoría de anomalía, acumula un score de riesgo
y genera un mensaje de alerta legible para auditores.

Uso rápido:
    from anomaly_detection.reglas import detectar_anomalias
    alertas = detectar_anomalias(df)

Uso completo con datos de prueba:
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
    "DUPLICADO": 40,
    "SIGNO_CONTABLE": 35,
    "FUERA_DE_ESTANCIA": 35,
    "MONTO_ATIPICO": 25,
    "CANCELACION_SOSPECHOSA": 25,
    "CONTEXTO_RESERVACION": 20,
    "USUARIO_MODIFICACION": 15,
    "PAGO_PROVEEDOR_SOSPECHOSO": 30,
}

# Umbrales para nivel de riesgo
NIVEL_RIESGO: list[tuple[int, str]] = [
    (80, "CRITICO"),
    (50, "ALTO"),
    (20, "MEDIO"),
    (0,  "BAJO"),
]

# Cuántas desviaciones estándar para considerar monto atípico
Z_UMBRAL = 2.0

# Un folio con más de este % de transacciones canceladas es sospechoso
RATIO_CANCELACIONES_FOLIO = 0.5

# Un proveedor que aparece menos de esta cantidad de veces se considera poco frecuente
MIN_FRECUENCIA_PROVEEDOR = 3

# Horario normal de operaciones: 06:00 – 23:59
HORA_MIN_NORMAL = 6
HORA_MAX_NORMAL = 23

# Umbral para pago dividido: pagos al mismo proveedor con montos muy similares (% diferencia)
UMBRAL_PAGO_DIVIDIDO = 0.05

# Número de pagos similares en mismo día para activar regla de división
MIN_PAGOS_DIVIDIDOS = 2


# --------------------------------------------------------------------------- #
# Helpers internos
# --------------------------------------------------------------------------- #
def _col_presente(df: pd.DataFrame, col: str) -> bool:
    """Devuelve True si la columna existe; avisa si falta."""
    if col not in df.columns:
        warnings.warn(f"[reglas] Columna '{col}' no encontrada — regla omitida.", stacklevel=3)
        return False
    return True


def _cols_presentes(df: pd.DataFrame, *cols: str) -> bool:
    """Devuelve True solo si TODAS las columnas existen."""
    return all(_col_presente(df, c) for c in cols)


def _asignar_nivel(score: int) -> str:
    for umbral, nivel in NIVEL_RIESGO:
        if score >= umbral:
            return nivel
    return "BAJO"


def _safe_mean(s: pd.Series) -> float:
    m = s.mean()
    return float(m) if pd.notna(m) else 0.0


def _safe_std(s: pd.Series) -> float:
    s_val = s.std()
    return float(s_val) if pd.notna(s_val) and s_val > 0 else 1.0


# --------------------------------------------------------------------------- #
# Resultado intermedio por fila
# --------------------------------------------------------------------------- #
def _resultado_vacio(n: int) -> dict[str, list]:
    return {
        "cluster": [[] for _ in range(n)],
        "score":   [0] * n,
        "reglas":  [[] for _ in range(n)],
    }


def _merge(base: dict, nuevo: dict) -> None:
    """Fusiona resultados en base, acumulando listas y sumando scores."""
    for i in range(len(base["score"])):
        base["cluster"][i].extend(nuevo["cluster"][i])
        base["score"][i] += nuevo["score"][i]
        base["reglas"][i].extend(nuevo["reglas"][i])


# --------------------------------------------------------------------------- #
# 1. Duplicados
# --------------------------------------------------------------------------- #
def detectar_duplicados(df: pd.DataFrame) -> dict[str, list]:
    """
    Duplicado exacto: mismo folio, código, monto (redondeado a centavos),
    habitación y fecha dentro de la misma base.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    cols_req = ["t_folio", "t_codigo", "t_monto", "t_cuarto"]
    if not _cols_presentes(df, *cols_req):
        return resultado

    work = pd.DataFrame(index=df.index)
    work["folio"]   = df["t_folio"].astype("string").fillna("__NA__")
    work["codigo"]  = df["t_codigo"].astype("string").fillna("__NA__")
    work["cuarto"]  = df["t_cuarto"].astype("string").fillna("__NA__")
    work["monto_c"] = np.rint(pd.to_numeric(df["t_monto"], errors="coerce").fillna(0) * 100).astype("int64")

    if "t_timestamp" in df.columns:
        work["fecha"] = pd.to_datetime(df["t_timestamp"]).dt.normalize()
    elif "t_fecha" in df.columns:
        work["fecha"] = pd.to_datetime(df["t_fecha"], errors="coerce").dt.normalize()
    else:
        work["fecha"] = pd.Timestamp("1970-01-01")

    group_cols = ["folio", "codigo", "monto_c", "cuarto", "fecha"]
    conteos = work.groupby(group_cols, dropna=False)["folio"].transform("size")
    mask_dup = conteos > 1

    idx_list = df.index.tolist()
    for pos, idx in enumerate(idx_list):
        if mask_dup.iloc[pos]:
            n_dup = int(conteos.iloc[pos])
            resultado["cluster"][pos].append("DUPLICADO")
            resultado["score"][pos]  += SCORES["DUPLICADO"]
            resultado["reglas"][pos].append(
                f"duplicado_exacto: {n_dup} transacciones con mismo folio/código/monto/habitación/fecha"
            )

    return resultado


# --------------------------------------------------------------------------- #
# 2. Inconsistencias de signo contable
# --------------------------------------------------------------------------- #
def detectar_signo_contable(df: pd.DataFrame) -> dict[str, list]:
    """
    Cargo (t_carabo=0) con monto negativo, o abono (t_carabo=1) con monto positivo.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    if not _cols_presentes(df, "t_carabo", "t_monto"):
        return resultado

    es_abono = df["t_carabo"].astype("string").str.strip() == "1"
    monto    = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0)

    mask_neg_cargo  = (~es_abono) & (monto < 0)
    mask_pos_abono  = es_abono    & (monto > 0)

    for pos in range(n):
        if mask_neg_cargo.iloc[pos]:
            resultado["cluster"][pos].append("SIGNO_CONTABLE")
            resultado["score"][pos]  += SCORES["SIGNO_CONTABLE"]
            resultado["reglas"][pos].append(
                f"cargo_monto_negativo: monto={monto.iloc[pos]:.2f} en transacción marcada como CARGO"
            )
        if mask_pos_abono.iloc[pos]:
            resultado["cluster"][pos].append("SIGNO_CONTABLE")
            resultado["score"][pos]  += SCORES["SIGNO_CONTABLE"]
            resultado["reglas"][pos].append(
                f"abono_monto_positivo: monto={monto.iloc[pos]:.2f} en transacción marcada como ABONO"
            )

    return resultado


# --------------------------------------------------------------------------- #
# 3. Montos atípicos (±Z_UMBRAL desviaciones estándar por código de concepto)
# --------------------------------------------------------------------------- #
def detectar_montos_atipicos(df: pd.DataFrame, z_umbral: float = Z_UMBRAL) -> dict[str, list]:
    """
    Monto fuera de media ± z_umbral * std para su código de transacción.
    Usa solo montos positivos (cargos normales) para calcular la distribución.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    if not _cols_presentes(df, "t_monto", "t_codigo"):
        return resultado

    monto  = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0)
    codigo = df["t_codigo"].astype("string").fillna("__NA__")

    stats = (
        df.assign(_monto=monto, _codigo=codigo)
        .groupby("_codigo")["_monto"]
        .agg(media="mean", std="std")
        .fillna({"std": 1.0})
        .replace({"std": {0.0: 1.0}})
    )

    medias = codigo.map(stats["media"]).fillna(monto.mean())
    stds   = codigo.map(stats["std"]).fillna(monto.std()).replace(0, 1.0)
    z_scores = (monto - medias) / stds

    mask_atipico = z_scores.abs() > z_umbral

    for pos in range(n):
        if mask_atipico.iloc[pos]:
            resultado["cluster"][pos].append("MONTO_ATIPICO")
            resultado["score"][pos]  += SCORES["MONTO_ATIPICO"]
            z_val = float(z_scores.iloc[pos])
            resultado["reglas"][pos].append(
                f"monto_atipico: monto={monto.iloc[pos]:.2f}, z={z_val:.2f} "
                f"(umbral ±{z_umbral}) para código {codigo.iloc[pos]}"
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

    cols_req = ["t_timestamp", "h_fec_lld", "h_fec_sda"]
    tiene_res_col = "tiene_reservacion" in df.columns

    if not _cols_presentes(df, *cols_req):
        return resultado

    ts      = pd.to_datetime(df["t_timestamp"], errors="coerce")
    llegada = pd.to_datetime(df["h_fec_lld"],   errors="coerce")
    salida  = pd.to_datetime(df["h_fec_sda"],   errors="coerce")
    fecha_cargo = ts.dt.normalize()

    if tiene_res_col:
        tiene_res = df["tiene_reservacion"].astype(bool)
    else:
        tiene_res = llegada.notna()

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
      a) Cargo cancelado con monto alto (>media+std del concepto).
      b) Folio con ratio de cancelaciones > RATIO_CANCELACIONES_FOLIO.
      c) Cancelación seguida de re-posteo (mismo folio/código/monto en el mismo día,
         con al menos una cancelada y otra activa).
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    if not _cols_presentes(df, "t_tra_cancelada", "t_monto"):
        return resultado

    cancelada = df["t_tra_cancelada"].astype("string").str.strip() == "1"
    monto      = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0).abs()

    # a) Cancelado con monto alto
    if _col_presente(df, "t_codigo"):
        codigo = df["t_codigo"].astype("string").fillna("__NA__")
        stats  = df.assign(_m=monto, _c=codigo).groupby("_c")["_m"].agg(media="mean", std="std").fillna(0)
        umbral_alto = codigo.map(stats["media"]) + codigo.map(stats["std"].fillna(0))
        mask_cancel_alto = cancelada & (monto > umbral_alto)
    else:
        mask_cancel_alto = pd.Series(False, index=df.index)

    # b) Folio con demasiadas cancelaciones
    if _col_presente(df, "t_folio"):
        folio = df["t_folio"].astype("string").fillna("__NA__")
        total_folio   = folio.map(folio.value_counts())
        cancel_folio  = folio.map(df.assign(_c=cancelada).groupby(folio)["_c"].sum())
        ratio_cancel  = cancel_folio / total_folio.replace(0, np.nan)
        mask_folio_cancel = (ratio_cancel > RATIO_CANCELACIONES_FOLIO) & (total_folio > 2)
    else:
        mask_folio_cancel = pd.Series(False, index=df.index)

    # c) Cancelación + reposteo (mismo folio/código/monto/día, mix activa+cancelada)
    mask_reposteo = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_folio", "t_codigo"):
        work = pd.DataFrame({
            "folio":   df["t_folio"].astype("string").fillna("__NA__"),
            "codigo":  df["t_codigo"].astype("string").fillna("__NA__"),
            "monto_c": np.rint(monto * 100).astype("int64"),
            "cancel":  cancelada.astype(int),
        })
        if "t_timestamp" in df.columns:
            work["fecha"] = pd.to_datetime(df["t_timestamp"]).dt.normalize()
        else:
            work["fecha"] = pd.Timestamp("1970-01-01")

        grp = work.groupby(["folio", "codigo", "monto_c", "fecha"], dropna=False)
        tiene_cancelada = grp["cancel"].transform("max") == 1
        tiene_activa    = grp["cancel"].transform(lambda s: (s == 0).any())
        mask_reposteo   = tiene_cancelada & tiene_activa

    for pos in range(n):
        if mask_cancel_alto.iloc[pos]:
            resultado["cluster"][pos].append("CANCELACION_SOSPECHOSA")
            resultado["score"][pos]  += SCORES["CANCELACION_SOSPECHOSA"]
            resultado["reglas"][pos].append(
                f"cancelacion_monto_alto: monto cancelado={monto.iloc[pos]:.2f}"
            )
        if mask_folio_cancel.iloc[pos]:
            if "CANCELACION_SOSPECHOSA" not in resultado["cluster"][pos]:
                resultado["cluster"][pos].append("CANCELACION_SOSPECHOSA")
            resultado["score"][pos]  += SCORES["CANCELACION_SOSPECHOSA"]
            r = float(ratio_cancel.iloc[pos]) if not mask_folio_cancel.isna().iloc[pos] else 0.0
            resultado["reglas"][pos].append(
                f"folio_alto_ratio_cancelaciones: {r:.0%} de movimientos cancelados"
            )
        if mask_reposteo.iloc[pos]:
            if "CANCELACION_SOSPECHOSA" not in resultado["cluster"][pos]:
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
      a) Cargo sin reservación vinculada en código que casi siempre la tiene (>90%).
      b) Noches del cargo muy diferentes a las noches de la reservación.
      c) Número de personas en el cargo diferente al de la reservación.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    # a) Cargo sin reservación en código que normalmente la tiene
    if _cols_presentes(df, "tiene_reservacion", "t_codigo"):
        tiene_res = df["tiene_reservacion"].astype(bool)
        codigo    = df["t_codigo"].astype("string").fillna("__NA__")
        pct_con_res = tiene_res.groupby(codigo).transform("mean")
        mask_sin_res_inusual = (~tiene_res) & (pct_con_res > 0.90)
    else:
        mask_sin_res_inusual = pd.Series(False, index=df.index)

    # b) Delta de noches muy grande
    mask_noches_delta = pd.Series(False, index=df.index)
    if _cols_presentes(df, "t_noches", "h_num_noc"):
        noches_cargo = pd.to_numeric(df["t_noches"],  errors="coerce")
        noches_res   = pd.to_numeric(df["h_num_noc"], errors="coerce")
        delta_noches = (noches_cargo - noches_res).abs()
        mask_noches_delta = delta_noches > 2

    # c) Inconsistencia en número de personas
    mask_personas_delta = pd.Series(False, index=df.index)
    if _cols_presentes(df, "h_num_per") and "t_num_per" in df.columns:
        per_cargo = pd.to_numeric(df["t_num_per"],  errors="coerce")
        per_res   = pd.to_numeric(df["h_num_per"],  errors="coerce")
        delta_per = (per_cargo - per_res).abs()
        mask_personas_delta = (delta_per > 1) & per_cargo.notna() & per_res.notna()

    for pos in range(n):
        if mask_sin_res_inusual.iloc[pos]:
            resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            resultado["reglas"][pos].append(
                f"sin_reservacion_en_codigo_habitual: código {df['t_codigo'].iloc[pos]} "
                "normalmente tiene reservación pero este cargo no tiene"
            )
        if mask_noches_delta.iloc[pos]:
            resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            nc = df["t_noches"].iloc[pos]
            nr = df["h_num_noc"].iloc[pos]
            resultado["reglas"][pos].append(
                f"noches_inconsistentes: cargo dice {nc} noches, reservación dice {nr}"
            )
        if mask_personas_delta.iloc[pos]:
            resultado["cluster"][pos].append("CONTEXTO_RESERVACION")
            resultado["score"][pos]  += SCORES["CONTEXTO_RESERVACION"]
            pc = df["t_num_per"].iloc[pos] if "t_num_per" in df.columns else "?"
            pr = df["h_num_per"].iloc[pos]
            resultado["reglas"][pos].append(
                f"personas_inconsistentes: cargo dice {pc} personas, reservación dice {pr}"
            )

    return resultado


# --------------------------------------------------------------------------- #
# 7. Modificaciones sospechosas de usuario
# --------------------------------------------------------------------------- #
def detectar_modificaciones_usuario(df: pd.DataFrame) -> dict[str, list]:
    """
    Usuario modificador distinto al creador, especialmente si el monto es alto.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    if not _cols_presentes(df, "t_usuario", "t_usuario_mod", "t_monto"):
        return resultado

    usuario     = df["t_usuario"].astype("string").fillna("__NA__")
    usuario_mod = df["t_usuario_mod"].astype("string")
    monto       = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0).abs()

    mod_distinto = usuario_mod.notna() & (usuario_mod != usuario)

    # Monto alto: por encima del percentil 75 global
    p75 = float(monto.quantile(0.75))
    monto_alto = monto > p75

    for pos in range(n):
        if mod_distinto.iloc[pos]:
            resultado["cluster"][pos].append("USUARIO_MODIFICACION")
            resultado["score"][pos]  += SCORES["USUARIO_MODIFICACION"]
            extra = " (monto alto)" if monto_alto.iloc[pos] else ""
            resultado["reglas"][pos].append(
                f"usuario_mod_distinto: creado por {usuario.iloc[pos]}, "
                f"modificado por {usuario_mod.iloc[pos]}{extra}"
            )

    return resultado


# --------------------------------------------------------------------------- #
# 8. Pagos a proveedores sospechosos
# --------------------------------------------------------------------------- #
def detectar_pagos_sospechosos(df: pd.DataFrame) -> dict[str, list]:
    """
    Aplica SOLO a filas donde t_carabo == 1 (abonos = egresos del hotel).
    Sub-reglas:
      a) Pago duplicado: mismo proveedor/referencia/monto/fecha.
      b) Monto fuera del rango histórico del proveedor (±2 std).
      c) Proveedor poco frecuente (< MIN_FRECUENCIA_PROVEEDOR transacciones).
      d) Pago en horario inusual (fuera de HORA_MIN_NORMAL – HORA_MAX_NORMAL).
      e) Pago dividido: varios pagos al mismo proveedor con montos similares el mismo día.
      f) Pago sin referencia.
    """
    n = len(df)
    resultado = _resultado_vacio(n)

    if not _col_presente(df, "t_carabo"):
        return resultado

    es_abono = df["t_carabo"].astype("string").str.strip() == "1"
    if not es_abono.any():
        return resultado

    monto = pd.to_numeric(df["t_monto"], errors="coerce").fillna(0.0).abs()

    proveedor_col = None
    for candidato in ["t_referencia", "h_cod_age", "t_cuarto"]:
        if candidato in df.columns:
            proveedor_col = candidato
            break

    if proveedor_col:
        proveedor = df[proveedor_col].astype("string").fillna("__SIN_PROVEEDOR__")
    else:
        proveedor = pd.Series("__SIN_PROVEEDOR__", index=df.index, dtype="string")

    if "t_timestamp" in df.columns:
        fecha = pd.to_datetime(df["t_timestamp"]).dt.normalize()
        hora  = pd.to_datetime(df["t_timestamp"]).dt.hour
    elif "t_fecha" in df.columns:
        fecha = pd.to_datetime(df["t_fecha"], errors="coerce").dt.normalize()
        hora  = pd.Series(12, index=df.index)
    else:
        fecha = pd.Series(pd.Timestamp("1970-01-01"), index=df.index)
        hora  = pd.Series(12, index=df.index)

    referencia_col = "t_referencia" if "t_referencia" in df.columns else None

    # a) Duplicado proveedor/monto/fecha
    monto_c = np.rint(monto * 100).astype("int64")
    dup_work = pd.DataFrame({
        "prov": proveedor, "monto_c": monto_c, "fecha": fecha, "es_abono": es_abono
    })
    dup_count = dup_work[es_abono].groupby(["prov", "monto_c", "fecha"], dropna=False)["prov"].transform("size")
    dup_mask  = pd.Series(False, index=df.index)
    dup_mask[es_abono] = dup_count > 1

    # b) Monto fuera de rango histórico del proveedor
    fuera_rango_mask = pd.Series(False, index=df.index)
    prov_stats = (
        df[es_abono].assign(_m=monto[es_abono], _p=proveedor[es_abono])
        .groupby("_p")["_m"]
        .agg(media="mean", std="std")
        .fillna({"std": 0})
    )
    if not prov_stats.empty:
        prov_media = proveedor.map(prov_stats["media"])
        prov_std   = proveedor.map(prov_stats["std"]).fillna(0).replace(0, 1.0)
        z_prov     = (monto - prov_media) / prov_std
        fuera_rango_mask = es_abono & (z_prov.abs() > Z_UMBRAL) & prov_media.notna()

    # c) Proveedor poco frecuente
    frec_proveedor  = proveedor[es_abono].value_counts()
    prov_poco_frec  = es_abono & (proveedor.map(frec_proveedor).fillna(0) < MIN_FRECUENCIA_PROVEEDOR)

    # d) Horario inusual
    hora_inusual = es_abono & ((hora < HORA_MIN_NORMAL) | (hora > HORA_MAX_NORMAL))

    # e) Pago dividido (varios pagos similares al mismo proveedor el mismo día)
    div_mask = pd.Series(False, index=df.index)
    div_work = df[es_abono].assign(_p=proveedor[es_abono], _f=fecha[es_abono], _m=monto[es_abono]).copy()
    if len(div_work) > 0:
        div_work["_m_round"] = (div_work["_m"] / (1 + UMBRAL_PAGO_DIVIDIDO)).round(-1)
        conteo_div = div_work.groupby(["_p", "_f", "_m_round"], dropna=False)["_p"].transform("size")
        div_mask_inner = conteo_div >= MIN_PAGOS_DIVIDIDOS
        div_mask[div_work.index[div_mask_inner]] = True

    # f) Sin referencia
    sin_ref_mask = pd.Series(False, index=df.index)
    if referencia_col:
        ref = df[referencia_col].astype("string").str.strip()
        sin_ref_mask = es_abono & (ref.isna() | ref.isin(["", "nan", "None", "0", "__NA__"]))

    for pos in range(n):
        mensajes = []
        if dup_mask.iloc[pos]:
            mensajes.append(f"pago_duplicado: mismo proveedor/monto/fecha ({proveedor.iloc[pos]}, {monto.iloc[pos]:.2f})")
        if fuera_rango_mask.iloc[pos]:
            z_val = float(z_prov.iloc[pos]) if not prov_stats.empty else 0.0
            mensajes.append(f"pago_fuera_rango_historico: monto={monto.iloc[pos]:.2f}, z={z_val:.2f} para proveedor {proveedor.iloc[pos]}")
        if prov_poco_frec.iloc[pos]:
            frec = int(frec_proveedor.get(proveedor.iloc[pos], 0))
            mensajes.append(f"proveedor_poco_frecuente: {proveedor.iloc[pos]} aparece solo {frec} veces")
        if hora_inusual.iloc[pos]:
            mensajes.append(f"pago_hora_inusual: hora={hora.iloc[pos]}:00 (fuera de {HORA_MIN_NORMAL}:00–{HORA_MAX_NORMAL}:00)")
        if div_mask.iloc[pos]:
            mensajes.append(f"pago_dividido: múltiples pagos similares al mismo proveedor el mismo día")
        if sin_ref_mask.iloc[pos]:
            mensajes.append("pago_sin_referencia: abono sin número de referencia")

        if mensajes:
            resultado["cluster"][pos].append("PAGO_PROVEEDOR_SOSPECHOSO")
            resultado["score"][pos]  += SCORES["PAGO_PROVEEDOR_SOSPECHOSO"]
            resultado["reglas"][pos].extend(mensajes)

    return resultado


# --------------------------------------------------------------------------- #
# Score, nivel de riesgo y mensaje de alerta
# --------------------------------------------------------------------------- #
def calcular_score_riesgo(scores_raw: list[int]) -> pd.Series:
    return pd.Series(scores_raw, dtype="int32")


def asignar_cluster_anomalia(clusters: list[list[str]]) -> pd.Series:
    """Devuelve el cluster de mayor puntaje (primero en la lista). Si hay varios, los une."""
    def _fmt(c: list[str]) -> str:
        unicos = list(dict.fromkeys(c))  # deduplica preservando orden
        return " | ".join(unicos) if unicos else ""
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
    reglas_str  = "; ".join(reglas[:3])  # máximo 3 reglas en el mensaje principal
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
    Ejecuta todas las reglas sobre el DataFrame y devuelve una tabla de alertas.

    Parámetros
    ----------
    df      : DataFrame con columnas de hottra / base consolidada.
    id_col  : Nombre de la columna que se usa como id_transaccion en el resultado.

    Retorna
    -------
    DataFrame con columnas:
        id_transaccion, folio, codigo, monto, fecha,
        es_anomalia, cluster_anomalia, nivel_riesgo,
        mensaje_alerta, reglas_activadas, score_riesgo
    """
    n = len(df)
    base = _resultado_vacio(n)

    detectores = [
        detectar_duplicados,
        detectar_signo_contable,
        detectar_montos_atipicos,
        detectar_fuera_estancia,
        detectar_cancelaciones,
        detectar_contexto_reservacion,
        detectar_modificaciones_usuario,
        detectar_pagos_sospechosos,
    ]

    for detector in detectores:
        _merge(base, detector(df))

    # Columnas de trazabilidad
    id_transaccion = df[id_col].astype("string") if id_col in df.columns else pd.Series(range(n), dtype="string")
    folio  = df["t_folio"].astype("string")  if "t_folio"  in df.columns else pd.Series("", index=df.index, dtype="string")
    codigo = df["t_codigo"].astype("string") if "t_codigo" in df.columns else pd.Series("", index=df.index, dtype="string")
    monto  = pd.to_numeric(df["t_monto"],    errors="coerce").fillna(0.0) if "t_monto" in df.columns else pd.Series(0.0, index=df.index)

    if "t_timestamp" in df.columns:
        fecha = pd.to_datetime(df["t_timestamp"], errors="coerce").dt.date.astype("string")
    elif "t_fecha" in df.columns:
        fecha = pd.to_datetime(df["t_fecha"], errors="coerce").dt.date.astype("string")
    else:
        fecha = pd.Series("", index=df.index, dtype="string")

    scores  = calcular_score_riesgo(base["score"])
    clusters = asignar_cluster_anomalia(base["cluster"])

    es_anomalia = scores > 0

    niveles = pd.Series([_asignar_nivel(s) for s in base["score"]], dtype="string")

    mensajes = pd.Series([
        generar_mensaje_alerta(
            id_transaccion.iloc[i],
            folio.iloc[i],
            codigo.iloc[i],
            float(monto.iloc[i]),
            base["cluster"][i],
            base["reglas"][i],
        )
        for i in range(n)
    ], dtype="string")

    reglas_str = pd.Series(
        [" | ".join(r) if r else "" for r in base["reglas"]],
        dtype="string",
    )

    result = pd.DataFrame({
        "id_transaccion":  id_transaccion.values,
        "folio":           folio.values,
        "codigo":          codigo.values,
        "monto":           monto.values,
        "fecha":           fecha.values,
        "es_anomalia":     es_anomalia.values,
        "cluster_anomalia": clusters.values,
        "nivel_riesgo":    niveles.values,
        "mensaje_alerta":  mensajes.values,
        "reglas_activadas": reglas_str.values,
        "score_riesgo":    scores.values,
    })

    return result
