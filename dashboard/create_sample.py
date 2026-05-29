"""Genera muestras representativas para el dashboard local FinAnom.

Selecciona hasta MAX_SAMPLE filas con representación balanceada:
  - Todos los CRITICO y ALTO (suelen ser pocos)
  - Representación de MEDIO y BAJO
  - Al menos un ejemplo de cada cluster

Uso:
    uv run python dashboard/create_sample.py

Salida:
    dashboard/data/sample_alertas.csv        — alertas operativas (≤ MAX_SAMPLE)
    dashboard/data/sample_transacciones.csv  — alertas + señales de contexto (≤ MAX_SAMPLE)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
DASHBD = Path(__file__).resolve().parent
DATA   = DASHBD / "data"

PATHS: dict[str, Path] = {
    "operativas": ROOT / "model_Tony" / "output_alertas_operativas.csv",
    "contexto":   ROOT / "model_Tony" / "output_senales_contexto.csv",
    "demo":       ROOT / "model_Tony" / "output_demo_sintetico.csv",
}

MAX_SAMPLE = 1_000


def _cargar_alertas() -> pd.DataFrame | None:
    """Carga la mejor fuente disponible de alertas operativas."""
    for key in ("operativas", "demo"):
        p = PATHS[key]
        if p.exists():
            print(f"  ✓ Cargando {p.name} ({p.stat().st_size // 1024} KB)")
            return pd.read_csv(p)
    return None


def _muestra_estratificada(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """
    Construye una muestra balanceada por nivel de riesgo y cluster.

    Prioridad:
        1. Todos los CRITICO (raramente más de 10–20)
        2. Todos los ALTO
        3. Hasta 50 % del cupo restante en MEDIO
        4. Hasta 20 % del cupo restante en BAJO/otros
        5. Al menos 1 ejemplo de cada cluster (si cabe)
    """
    if "nivel_riesgo" not in df.columns or len(df) <= n:
        return df.sample(min(n, len(df)), random_state=random_state)

    criticos = df[df["nivel_riesgo"] == "CRITICO"]
    altos    = df[df["nivel_riesgo"] == "ALTO"]
    medios   = df[df["nivel_riesgo"] == "MEDIO"]
    resto    = df[~df["nivel_riesgo"].isin(["CRITICO", "ALTO", "MEDIO"])]

    partes = [criticos, altos]
    cupo   = n - len(criticos) - len(altos)

    if cupo > 0:
        n_medios = min(len(medios), int(cupo * 0.70))
        n_resto  = min(len(resto),  cupo - n_medios)
        if n_medios > 0:
            partes.append(medios.sample(n_medios, random_state=random_state))
        if n_resto > 0:
            partes.append(resto.sample(n_resto, random_state=random_state))

    resultado = pd.concat(partes, ignore_index=True).head(n)

    # Garantizar al menos 1 fila por cluster
    if "cluster_anomalia" in df.columns:
        clusters_presentes = set(
            c.strip()
            for v in resultado["cluster_anomalia"].dropna()
            for c in str(v).split(" | ")
            if c.strip()
        )
        for v in df["cluster_anomalia"].dropna().unique():
            for c in str(v).split(" | "):
                c = c.strip()
                if c and c not in clusters_presentes:
                    filas_cluster = df[df["cluster_anomalia"].str.contains(c, na=False, regex=False)]
                    if not filas_cluster.empty and len(resultado) < n:
                        resultado = pd.concat(
                            [resultado, filas_cluster.head(1)], ignore_index=True
                        )
                        clusters_presentes.add(c)

    return resultado.head(n)


def _print_distribucion(df: pd.DataFrame, titulo: str) -> None:
    print(f"\n  {titulo}:")
    if "nivel_riesgo" in df.columns:
        for nivel, cnt in df["nivel_riesgo"].value_counts().items():
            bar = "█" * min(cnt, 40)
            print(f"    {nivel:10s} {cnt:4d}  {bar}")
    if "cluster_anomalia" in df.columns:
        clusters: dict[str, int] = {}
        for v in df["cluster_anomalia"].dropna():
            for c in str(v).split(" | "):
                c = c.strip()
                if c:
                    clusters[c] = clusters.get(c, 0) + 1
        print("  Clusters:")
        for c, n in sorted(clusters.items(), key=lambda x: -x[1]):
            print(f"    {c:35s}: {n}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  FinAnom — Generación de muestras para el dashboard")
    print("=" * 60)

    # ── Alertas operativas ────────────────────────────────────────────────────
    df_alertas = _cargar_alertas()
    if df_alertas is None:
        print("\n  ERROR: No se encontró ningún archivo de salida del modelo.")
        print("  Genera los datos con:\n    uv run python anomaly_detection/run_demo.py")
        sys.exit(1)

    print(f"\n  Total alertas disponibles : {len(df_alertas):,}")
    sample_alertas = _muestra_estratificada(df_alertas, MAX_SAMPLE)
    _print_distribucion(sample_alertas, f"sample_alertas.csv ({len(sample_alertas)} filas)")

    out_alertas = DATA / "sample_alertas.csv"
    sample_alertas.to_csv(out_alertas, index=False, encoding="utf-8")
    print(f"\n  ✓ Guardado: {out_alertas}")

    # ── Transacciones (alertas + señales de contexto) ─────────────────────────
    partes = [df_alertas]
    ctx_path = PATHS["contexto"]
    if ctx_path.exists():
        df_ctx = pd.read_csv(ctx_path)
        partes.append(df_ctx)
        print(f"\n  + Señales de contexto: {len(df_ctx):,} filas")

    df_todas = pd.concat(partes, ignore_index=True)
    if "id_transaccion" in df_todas.columns:
        df_todas = df_todas.drop_duplicates(subset="id_transaccion")

    sample_tx = _muestra_estratificada(df_todas, MAX_SAMPLE)
    _print_distribucion(sample_tx, f"sample_transacciones.csv ({len(sample_tx)} filas)")

    out_tx = DATA / "sample_transacciones.csv"
    sample_tx.to_csv(out_tx, index=False, encoding="utf-8")
    print(f"\n  ✓ Guardado: {out_tx}")

    print("\n  Muestras generadas. Corre el dashboard con:")
    print("    uv run streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
