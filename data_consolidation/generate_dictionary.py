"""Genera el diccionario en markdown de la base consolidada.

Lee `output/transacciones_consolidado.parquet`, calcula estadísticas reales por
columna (tipo, % nulos, valores únicos, ejemplos, rango) y las combina con las
descripciones de negocio para escribir `diccionario_base_consolidada.md`.

Uso:
    python data_consolidation/generate_dictionary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "output" / "transacciones_consolidado.parquet"
OUT_MD = HERE / "diccionario_base_consolidada.md"

# Origen (hottra=transacción, hothsp=reservación, derivada=calculada aquí).
ORIGEN = {**{c: "hottra" for c in [
    "t_folio", "t_folio_ext", "t_referencia", "t_transaccion", "t_codigo",
    "t_cve_res", "t_cuarto", "t_centro_consumo", "t_fecha",
    "t_tra_hra", "t_tra_mto", "t_monto", "t_impuesto", "t_impuesto2", "t_propina",
    "t_carabo", "t_tra_cancelada", "t_can_dia", "t_can_mes", "t_can_hra",
    "t_can_mto", "t_usuario", "t_usuario_mod", "t_num_adu", "t_num_per",
    "t_noches", "t_inc_tfa", "t_observaciones"]},
    "t_timestamp": "derivada",
    "es_split": "derivada",
    "es_renta": "derivada",
    "tiene_reservacion": "derivada",
    **{c: "hothsp" for c in [
        "h_status", "h_tpo_hab", "h_tpo_hsp", "h_seg_mer", "h_cod_age",
        "h_tpo_plan", "h_for_pgo", "h_tpo_mon", "h_num_per", "h_num_adu",
        "h_num_men", "h_num_noc", "h_tfa", "h_tfa_total",
        "h_tfa_renta", "h_tfa_impuestos", "h_tfa_extras", "h_tarifa_forzada",
        "h_dep_sol", "h_lim_cre", "h_fec_lld", "h_fec_sda", "h_fec_reg",
        "h_res_usr", "h_rec_usr"]},
}

# Descripción de negocio + utilidad para detección de anomalías.
DESC = {
    "t_folio": "Folio de la cuenta del huésped (cuenta donde se agrupan los cargos). Trazabilidad y agrupación de cargos por cuenta.",
    "t_folio_ext": "Extensión del folio (sub-cuenta). Junto con t_folio identifica la cuenta exacta.",
    "t_referencia": "Referencia cruzada del movimiento. Ayuda a emparejar cargos con sus reversos/abonos.",
    "t_transaccion": "Número correlativo de la transacción. Identificador secuencial del movimiento.",
    "t_codigo": "Código del concepto del cargo (RENHAB=renta, PROPTI=propina, DEPOS=depósito, AJ*=ajustes, DEV*=devoluciones, etc.). Eje para z-score de monto por concepto. SIN catálogo oficial entregado.",
    "t_cve_res": "Llave de la reservación a la que pertenece el cargo (formato tpo+num+mbo). Llave de join con la reservación (hothsp). Vacío = walk-in / depósito / cargo sin reservación.",
    "t_cuarto": "Número de habitación donde se generó el cargo.",
    "t_centro_consumo": "Centro de consumo (00=recepción domina; 01 marginal). Casi siempre recepción: señal débil en este hotel.",
    "t_fecha": "Fecha del cargo (parseada). Eje temporal de la transacción.",
    "t_tra_hra": "Hora del cargo (HH, 00–23). Cargos en madrugada fuera del night audit son sospechosos.",
    "t_tra_mto": "Minuto del cargo (MM, 00–59). Componente del timestamp; útil para detectar duplicados en segundos/minutos.",
    "t_monto": "Monto del movimiento (MXN). Eje del modelo. ~5% negativos (reversos/notas de crédito legítimos): el signo se modela como feature, NO se filtra.",
    "t_impuesto": "Impuesto principal del cargo. Un ratio impuesto/monto fuera de ~16% es señal de anomalía.",
    "t_impuesto2": "Impuesto secundario. Generalmente 0; valores grandes ameritan revisión.",
    "t_propina": "Propina del cargo. Propina anormal respecto al monto es una regla candidata.",
    "t_carabo": "Naturaleza del movimiento: 0=CARGO, 1=ABONO. Define el signo contable; clave para conciliar cargos vs abonos.",
    "t_tra_cancelada": "Estado de cancelación: 0=activa, 1=cancelada, <NA>=no aplica/sin marca. Weak label: las canceladas son candidatas a anomalía.",
    "t_can_dia": "Día de la cancelación (DD, sin año). Componente del momento de cancelación.",
    "t_can_mes": "Mes de la cancelación (MM, sin año). Componente del momento de cancelación.",
    "t_can_hra": "Hora de la cancelación (HH). Cancelaciones en horario atípico son señal.",
    "t_can_mto": "Minuto de la cancelación (MM).",
    "t_usuario": "Usuario que generó el cargo (clave de 3 letras). Detección de patrones por operador (fraude/error sistemático).",
    "t_usuario_mod": "Usuario que modificó el cargo. Modificaciones por usuarios distintos al que creó pueden ser señal.",
    "t_num_adu": "Adultos asociados al cargo. Inconsistencia vs ocupación de la reservación es señal.",
    "t_num_per": "Personas totales asociadas al cargo.",
    "t_noches": "Noches asociadas al cargo (0–134). Estancias muy largas son outliers.",
    "t_inc_tfa": "Concepto incluido en tarifa: S / N / <NA>. Distingue cargos que forman parte de la tarifa.",
    "t_observaciones": "Texto libre del cargo (p.ej. 'RENTA: HABITACION...'). Útil para reasons del auditor y detección de keywords (ajuste, cortesía, error). Puede contener nombres: tratar como sensible.",
    "t_timestamp": "DERIVADA: fecha+hora+minuto del cargo en un solo datetime. Base para features temporales y detección de duplicados por ventana.",
    "es_split": "DERIVADA: True si el cargo es partido (split). Colapsa las 3 columnas de split originales (77 cargos). Los splits pueden generar doble conteo.",
    "es_renta": "DERIVADA: True si es cargo de renta/hospedaje (flag original t_renta='S', night audit). Distingue renta de extras/ajustes.",
    "tiene_reservacion": "DERIVADA: True si el cargo cruzó con una reservación en hothsp. Separa cargos con/sin contexto de reservación.",
    # --- contexto de reservación (hothsp) ---
    "h_status": "Estado de la reservación (00=registro, 01=cancelada, 02=no show, 10=en casa, 50=salida, ...). Contexto del ciclo de vida; cargos sobre reservas canceladas/no-show son sospechosos.",
    "h_tpo_hab": "Tipo de habitación de la reservación. Baseline para comparar tarifas por categoría.",
    "h_tpo_hsp": "Tipo de huésped (NOR, VIP, VP2...). Las tarifas VIP/cortesía explican algunos outliers.",
    "h_seg_mer": "Segmento de mercado (ONL, MAY, GPO...). Baseline tarifario por segmento.",
    "h_cod_age": "Agencia / cuenta que originó la reservación. Anomalías por agencia (tarifas 0, descuadres).",
    "h_tpo_plan": "Plan tarifario contratado (B2C, MAYN, GSHO...).",
    "h_for_pgo": "Forma de pago de la reservación (EFE, TARCRE, XFAC...).",
    "h_tpo_mon": "Moneda de la reservación (NAL=MXN domina, DLS marginal). Confirma que casi todo es MXN.",
    "h_num_per": "Personas esperadas en la reservación. Comparar vs ocupación del cargo.",
    "h_num_adu": "Adultos esperados en la reservación.",
    "h_num_men": "Menores esperados en la reservación.",
    "h_num_noc": "Noches de la reservación (máx 527). Estancias extremas son outliers.",
    "h_tfa": "Tarifa diaria de la reservación. Comparar contra cargos de renta.",
    "h_tfa_total": "Tarifa total esperada de la reservación. Descuadre vs suma de cargos (t_monto) del folio = regla de anomalía.",
    "h_tfa_renta": "Componente de renta de la tarifa.",
    "h_tfa_impuestos": "Componente de impuestos de la tarifa (semántica a confirmar; mean alto).",
    "h_tfa_extras": "Componente de extras de la tarifa.",
    "h_tarifa_forzada": "Tarifa forzada manualmente (override). Forzar tarifa fuera de rango es señal de autorización indebida.",
    "h_dep_sol": "Depósito solicitado en la reservación. Conciliación contra cargos de depósito (DEP*).",
    "h_lim_cre": "Límite de crédito (contiene valores centinela 999999999). Tratar centinelas como faltante.",
    "h_fec_lld": "Fecha de llegada (check-in) de la reservación.",
    "h_fec_sda": "Fecha de salida (check-out) de la reservación. Cargos posteriores a la salida son sospechosos.",
    "h_fec_reg": "Fecha de registro de la reservación.",
    "h_res_usr": "Usuario que creó la reservación.",
    "h_rec_usr": "Usuario que registró (check-in) la reservación.",
}


def fmt_sample(s: pd.Series, k: int = 3) -> str:
    is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    vals = s.dropna().unique()[:k]
    out = []
    for v in vals:
        if is_numeric:
            out.append(str(round(float(v), 2)))
        else:
            out.append(str(v)[:22])
    return ", ".join(out)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n = len(df)
    lines: list[str] = []

    lines.append("# Diccionario — Base consolidada FINANOM\n")
    lines.append(
        "Base única para la detección no supervisada de anomalías/inconsistencias "
        "financieras. Generada por `data_consolidation/consolidate.py` a partir de "
        "`tablas_parquet/`.\n"
    )
    lines.append("## 1. Resumen\n")
    lines.append("- **Archivo**: `data_consolidation/output/transacciones_consolidado.parquet`")
    lines.append(f"- **Filas**: {n:,} (una por transacción de `hottra`)")
    lines.append(f"- **Columnas**: {df.shape[1]}")
    lines.append("- **Grano**: una transacción = una predicción del modelo")
    rng = (df["t_timestamp"].min(), df["t_timestamp"].max())
    lines.append(f"- **Rango temporal**: {rng[0]:%Y-%m-%d} → {rng[1]:%Y-%m-%d}")
    cob = 100 * df["tiene_reservacion"].mean()
    lines.append(f"- **Cobertura de reservación** (LEFT JOIN con `hothsp`): {cob:.1f}%")
    lines.append("- **Tablas fuente**: `hottra` (transacciones) + `hothsp` (contexto de reservación)\n")

    lines.append("## 2. Cómo se construyó\n")
    lines.append(
        "1. Se carga `hottra` y se conservan solo columnas relevantes para anomalías "
        "financieras (montos, impuestos, propina, control/cancelación, autoría, "
        "ocupación, código de concepto, timestamp).\n"
        "2. Se limpia: `strip` de strings, blancos → `<NA>`, fechas `YYYYMMDD` → "
        "datetime (`00000000` → NaT), ocupación string → entero nullable, y se arma "
        "`t_timestamp` (fecha+hora+minuto).\n"
        "3. Se une por LEFT JOIN `t_cve_res == h_res_cve` con un subconjunto de "
        "`hothsp` (estado, tarifas, ocupación esperada, agencia/segmento, fechas de "
        "estancia, usuarios). Se conservan TODAS las transacciones; las que no tienen "
        "reservación quedan con contexto nulo y `tiene_reservacion = False`.\n"
    )

    lines.append("## 3. Columnas\n")
    lines.append("Prefijo `t_` = transacción, `h_`/`Num_` = reservación, sin prefijo = derivada.\n")
    lines.append("| Columna | Origen | Tipo | % nulos | Únicos | Ejemplos | Descripción / utilidad |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for c in df.columns:
        s = df[c]
        nulls = 100 * s.isna().mean()
        nuniq = s.nunique(dropna=True)
        sample = fmt_sample(s)
        desc = DESC.get(c, "—")
        origen = ORIGEN.get(c, "—")
        dt = str(s.dtype)
        lines.append(
            f"| `{c}` | {origen} | {dt} | {nulls:.1f} | {nuniq:,} | {sample} | {desc} |"
        )

    lines.append("\n## 4. Qué se descartó y por qué\n")
    lines.append(
        "- **Columnas muertas de `hottra`** (un solo valor): `ibuff`, `t_ya_facturada`, "
        "`t_tipo_trans`, `t_folio_origen`, `t_tra_origen`, `t_ref_origen`, "
        "`t_autorizacion`, `t_claveorigen`, `t_tipo_cambio`. Las cuatro de "
        "trazabilidad de origen y `t_ya_facturada` fueron confirmadas como NO "
        "aplicables a México / vacías en estos datos.\n"
        "- **Campos de facturación de `hothsp`** (100% vacíos): `h_ya_fact`, "
        "`h_num_fac`, `h_fec_fac`, `h_status_pago`. Confirmado: facturación "
        "adelantada y WebCheckIn no aplican / no se exportaron.\n"
        "- **PII** (`h_nom`, `h_nombre`, `h_apellido_*`, `h_tar_cre`, correos, etc.): "
        "no aportan a la detección financiera y son sensibles.\n"
        "- **Columnas casi vacías o constantes al grano de transacción**: "
        "`t_numctapago` (99% vacío), `t_num_men` (99.9% vacío, redundante con "
        "`t_num_per`), `Num_cancelacion` (las reservas canceladas casi no generan "
        "cargos → 0.04% lleno) y `h_tot_hab` (constante = 1 en los cargos enlazados). "
        "Las 3 columnas de split se colapsaron en el booleano `es_split`.\n"
        "- **`hotcag`**: NO es el catálogo de `t_codigo` (solo 43 de 1.14M filas "
        "cruzan; 107 de 108 códigos quedan huérfanos). No enriquece.\n"
        "- **`hotvta`** (ventas agregadas), **`hotcvt`** (tipo de cambio) y catálogos "
        "menores: distinto grano o no necesarios; casi todo es MXN.\n"
        "- **Tablas operativas** (bloqueos, status de cuarto, requerimientos, "
        "eventos): no financieras.\n"
    )

    lines.append("## 5. Limitaciones conocidas\n")
    lines.append(
        "- Los 108 `t_codigo` no tienen catálogo oficial; su semántica se infiere por "
        "prefijo (REN=renta, DEP=depósito, AJ=ajuste, DEV=devolución, PROP=propina...).\n"
        "- La cancelación trae día/mes/hora/minuto pero **no año**: no se puede armar "
        "un datetime completo de cancelación.\n"
        "- `h_lim_cre` contiene valores centinela (999999999); tratarlos como faltante.\n"
        "- ~21.5% de cargos no tienen reservación enlazada (walk-in / depósito / centro "
        "de consumo); su contexto `h_*` es nulo por diseño.\n"
        "- No hay conversión de moneda: se asume MXN (los datos son de un hotel en México).\n"
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Diccionario escrito: {OUT_MD}  ({len(lines)} líneas)")


if __name__ == "__main__":
    main()
