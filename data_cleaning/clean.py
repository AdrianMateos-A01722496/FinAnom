"""Limpieza de la base FINANOM para Isolation Forest.

Toma la base consolidada (`data_consolidation/output/transacciones_consolidado.parquet`)
y produce una base limpia eliminando columnas que no aportan a la detección de
anomalías. NO toca filas: se conservan las 1,145,526 transacciones, incluidas las
raras (p.ej. PUNTAZ). En detección NO supervisada los casos raros pueden ser anomalías
legítimas, así que no se borran.

Eliminación en DOS pases (23 columnas; ver DROP_REDUNDANTE y DROP_BAJO_VALOR):
  - Pase 1 (estructural): redundantes / constantes / colineales / duplicadas.
  - Pase 2 (bajo valor): casi-constantes sin sentido de regla, redundancias de
    ocupación (h_num_per = adu + men), usuarios/fechas de la reserva poco ligados al
    cargo, y la cancelación troceada sin año (ya resumida por t_tra_cancelada).
Cada razón está verificada en initial_exploration.ipynb y se re-chequea al ejecutar.

Lo que este script NO hace (es fase de modelado / feature engineering, NO limpieza):
  - Encoding de categóricas (recomendado: frecuencia, para no enmascarar códigos raros).
  - Imputación de nulos (Isolation Forest de sklearn NO acepta NaN).
  - Escalado de monto por concepto (z-score robusto por t_codigo) ← paso de mayor impacto.
  - Features derivados como `n_duplicados` (conteo de cargos idénticos por folio/código/
    monto/día): el "cargo duplicado" del user story NO se detecta con dedup, porque
    t_transaccion es un ID secuencial (0 filas 100% idénticas).

Uso:
    uv run python data_cleaning/clean.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data_consolidation" / "output" / "transacciones_consolidado.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "transacciones_limpio.parquet"
DICT_FILE = Path(__file__).resolve().parent / "diccionario_limpio.md"

# --------------------------------------------------------------------------- #
# Columnas a eliminar — DOS pases
# --------------------------------------------------------------------------- #
# Pase 1 (estructural): redundantes, constantes, colineales o duplicadas.
DROP_REDUNDANTE: dict[str, str] = {
    "t_centro_consumo": "Redundante con t_codigo (centro '01' ⟺ PUNTAZ); 24.4% nulos; 99.97% '00'.",
    "h_tpo_mon":        "Constante (NAL=MXN 99.99%); sin varianza.",
    "h_lim_cre":        "99.74% en 0 + centinelas (999999999); sin señal.",
    "h_tfa_impuestos":  "Mal nombrada; == h_tfa_total en 97.3% de filas (duplicada).",
    "h_tfa_renta":      "Colineal con h_tfa_total (r=0.992).",
    "t_impuesto2":      "Colineal con t_impuesto (r=0.92) y 89.7% en 0.",
    "t_num_per":        "Duplica h_num_per (r=0.997); h_num_per cubre 49,474 filas más.",
    "t_num_adu":        "Duplica h_num_adu (r=0.991); 69.6% nulos vs 21.5%.",
    "t_tra_hra":        "Componente de t_timestamp (coincide 100%).",
    "t_tra_mto":        "Componente de t_timestamp (coincide 100%).",
}

# Pase 2 (bajo valor para detección de anomalías): casi-constantes sin sentido de
# regla, redundancias de ocupación, usuarios/fechas de reserva poco ligados al cargo
# y la cancelación troceada sin año (ya resumida por el flag t_tra_cancelada).
DROP_BAJO_VALOR: dict[str, str] = {
    "t_fecha":      "Redundante con t_timestamp (normalize == t_fecha al 100%).",
    "t_inc_tfa":    "Casi-constante (97.5% 'S') + 25.8% nulos; solapa con es_renta.",
    "h_status":     "Sin señal: solo 50=salida (99.98%) y 10=en casa; sin cancelación/no-show.",
    "h_tfa_extras": "Casi-constante (97.4% en 0); el extra real ya está en t_monto.",
    "h_num_adu":    "Redundante: h_num_per = h_num_adu + h_num_men en 99.9% de filas.",
    "h_num_men":    "Redundante: h_num_per = h_num_adu + h_num_men en 99.9% de filas.",
    "h_res_usr":    "Usuario que creó la reserva; débil para anomalía a nivel cargo.",
    "h_rec_usr":    "Usuario de check-in; débil para anomalía a nivel cargo.",
    "h_fec_reg":    "Fecha de registro de la reserva; baja relevancia al cargo.",
    "t_can_dia":    "85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada.",
    "t_can_mes":    "85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada.",
    "t_can_hra":    "85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada.",
    "t_can_mto":    "85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada.",
}

DROP_COLS: dict[str, str] = {**DROP_REDUNDANTE, **DROP_BAJO_VALOR}

# --------------------------------------------------------------------------- #
# Roles de las columnas que SE CONSERVAN (guía para la fase de modelado)
# --------------------------------------------------------------------------- #
# id       : identificadores. Para agrupar/derivar/trazar; NO entrar como feature crudo.
# num      : numéricas. Feature directo (luego escalar; idealmente por t_codigo).
# cat      : categóricas. Feature tras encoding (recomendado: por frecuencia).
# flag     : banderas / weak labels. Útiles para reglas; señal débil para IF.
# datetime : temporales. Base para features (hora, día de semana, mes, deltas); NO crudas.
# text     : texto libre. Para keywords / 'reason' del auditor; NO feature numérico.
ROLES: dict[str, list[str]] = {
    "id": [
        "t_folio", "t_folio_ext", "t_referencia", "t_transaccion",
        "t_cve_res", "t_cuarto",
    ],
    "cat": [
        "t_codigo", "t_carabo", "t_usuario", "t_usuario_mod",
        "h_tpo_hab", "h_tpo_hsp", "h_seg_mer", "h_cod_age", "h_tpo_plan",
        "h_for_pgo",
    ],
    "num": [
        "t_monto", "t_impuesto", "t_propina", "t_noches",
        "h_num_per", "h_num_noc",
        "h_tfa", "h_tfa_total", "h_tarifa_forzada", "h_dep_sol",
    ],
    "flag": [
        "t_tra_cancelada", "es_split", "es_renta", "tiene_reservacion",
    ],
    "datetime": [
        "t_timestamp", "h_fec_lld", "h_fec_sda",
    ],
    "text": [
        "t_observaciones",
    ],
}

ROLE_DESC = {
    "id": "Identificador (agrupar/derivar/trazar; NO feature crudo)",
    "num": "Numérica (feature; escalar, idealmente por t_codigo)",
    "cat": "Categórica (feature tras encoding por frecuencia)",
    "flag": "Bandera / weak label (regla; señal débil para IF)",
    "datetime": "Temporal (base de features; NO cruda)",
    "text": "Texto libre (keywords / reason; NO feature numérico)",
}


# --------------------------------------------------------------------------- #
# Verificación defensiva (avisa si la base regenerada ya no cumple los supuestos)
# --------------------------------------------------------------------------- #
def verify_assumptions(df: pd.DataFrame) -> None:
    """Reimprime las redundancias clave para confirmar que los drops siguen válidos."""
    print("Verificando supuestos de redundancia...")

    def eq_pct(a: str, b: str) -> float:
        sub = df[[a, b]].dropna()
        return 100 * (abs(sub[a] - sub[b]) <= 1).mean() if len(sub) else float("nan")

    checks = {
        "h_tfa_impuestos == h_tfa_total (esperado ~97%)": eq_pct("h_tfa_impuestos", "h_tfa_total"),
        "centro '01' ⟺ PUNTAZ":
            100 * (df.loc[df["t_centro_consumo"] == "01", "t_codigo"] == "PUNTAZ").mean(),
    }
    ts = df["t_timestamp"]
    checks["hora(t_timestamp) == t_tra_hra"] = 100 * (ts.dt.hour == df["t_tra_hra"].astype("Int64")).mean()
    checks["min(t_timestamp)  == t_tra_mto"] = 100 * (ts.dt.minute == df["t_tra_mto"].astype("Int64")).mean()
    # Pase 2: redundancias de bajo valor
    checks["normalize(t_timestamp) == t_fecha"] = 100 * (ts.dt.normalize() == df["t_fecha"]).mean()
    occ = df.dropna(subset=["h_num_per", "h_num_adu", "h_num_men"])
    checks["h_num_per == h_num_adu + h_num_men"] = (
        100 * ((occ["h_num_per"] - (occ["h_num_adu"] + occ["h_num_men"])).abs() <= 0).mean()
    )
    nn_status = df["h_status"].dropna().nunique()
    checks["h_status sin cancelación (<=2 valores: 50/10)"] = 100.0 if nn_status <= 2 else 0.0
    for name, pct in checks.items():
        flag = "OK " if pct >= 95 else "!! "
        print(f"  {flag}{name}: {pct:.1f}%")


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas estructuralmente inútiles y reordena por rol. NO toca filas."""
    verify_assumptions(df)

    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas a eliminar que no existen en la base: {missing}")

    out = df.drop(columns=list(DROP_COLS))

    # Reordena por rol (legibilidad). Cualquier columna no clasificada va al final.
    ordered = [c for cols in ROLES.values() for c in cols if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    if rest:
        print(f"  AVISO: columnas sin rol asignado (van al final): {rest}")
    return out[ordered + rest]


# --------------------------------------------------------------------------- #
# Diccionario de la base limpia (drop log + roles + stats)
# --------------------------------------------------------------------------- #
def write_dictionary(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    dictionary_file: Path | str = DICT_FILE,
    output_file: Path | str = OUTPUT_FILE,
) -> None:
    """Genera diccionario_limpio.md con el log de eliminaciones y los roles."""
    dictionary_file = Path(dictionary_file)
    output_file = Path(output_file)
    n = len(df_out)
    lines: list[str] = []
    lines.append("# Diccionario — Base LIMPIA FINANOM\n")
    lines.append(
        "Generado por `data_cleaning/clean.py` a partir de la base consolidada. "
        "Limpieza en dos pases (sin encoding/imputación/escalado, que son fase de modelado).\n"
    )
    lines.append("## 1. Resumen\n")
    lines.append(f"- **Archivo**: `data_cleaning/output/{output_file.name}`")
    lines.append(f"- **Filas**: {n:,} (todas; no se eliminó ninguna)")
    lines.append(f"- **Columnas**: {df_out.shape[1]} (antes {df_in.shape[1]}; se eliminaron {len(DROP_COLS)})\n")

    lines.append("## 2. Columnas eliminadas y por qué\n")
    lines.append(f"### Pase 1 — estructural ({len(DROP_REDUNDANTE)}): redundantes / constantes / colineales\n")
    lines.append("| Columna | Razón |")
    lines.append("| ------- | ----- |")
    for c, r in DROP_REDUNDANTE.items():
        lines.append(f"| `{c}` | {r} |")
    lines.append("")
    lines.append(f"### Pase 2 — bajo valor para detección de anomalías ({len(DROP_BAJO_VALOR)})\n")
    lines.append("| Columna | Razón |")
    lines.append("| ------- | ----- |")
    for c, r in DROP_BAJO_VALOR.items():
        lines.append(f"| `{c}` | {r} |")
    lines.append("")

    lines.append("## 3. Columnas conservadas (por rol)\n")
    for role, cols in ROLES.items():
        cols = [c for c in cols if c in df_out.columns]
        if not cols:
            continue
        lines.append(f"### {role} — {ROLE_DESC[role]}\n")
        lines.append("| Columna | Tipo | % nulos | Únicos |")
        lines.append("| ------- | ---- | ------- | ------ |")
        for c in cols:
            s = df_out[c]
            null_pct = 100 * s.isna().mean()
            lines.append(f"| `{c}` | {s.dtype} | {null_pct:.1f} | {s.nunique():,} |")
        lines.append("")

    lines.append("## 4. Siguientes pasos (fase de modelado, NO limpieza)\n")
    lines.append("1. **Imputar nulos** (sklearn IsolationForest no acepta NaN): centinela + flag de faltante; `tiene_reservacion` ya marca el bloque `h_`.")
    lines.append("2. **Encoding por frecuencia** de las categóricas (no enmascara códigos raros, a diferencia de un cubo 'otros').")
    lines.append("3. **Escalar `t_monto` por `t_codigo`** (z-score robusto por concepto): el paso de mayor impacto para que el IF encuentre inconsistencias y no solo 'números grandes'.")
    lines.append("4. **Feature `n_duplicados`**: conteo de cargos idénticos por (folio, código, monto, día) — el 'cargo duplicado' del user story.")
    lines.append("")

    dictionary_file.parent.mkdir(parents=True, exist_ok=True)
    dictionary_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guardado diccionario: {dictionary_file}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_cleaning(
    input_file: Path | str = INPUT_FILE,
    output_file: Path | str = OUTPUT_FILE,
    dictionary_file: Path | str = DICT_FILE,
) -> Path:
    """Ejecuta la fase de limpieza y devuelve el parquet generado."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    print(f"Leyendo base consolidada: {input_file}")
    df_in = pd.read_parquet(input_file)
    print(f"  entrada: {df_in.shape[0]:,} filas, {df_in.shape[1]} columnas\n")

    df_out = clean(df_in)
    print(f"\n  salida:  {df_out.shape[0]:,} filas, {df_out.shape[1]} columnas")
    print(f"  eliminadas: {len(DROP_COLS)} columnas | filas: sin cambios\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_file, index=False)
    size_mb = output_file.stat().st_size / 1e6
    print(f"Guardado: {output_file}  ({size_mb:.1f} MB)")

    write_dictionary(df_in, df_out, dictionary_file, output_file)
    return output_file


def main() -> None:
    run_cleaning()


if __name__ == "__main__":
    main()
