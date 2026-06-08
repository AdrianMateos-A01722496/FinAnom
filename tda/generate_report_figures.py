from __future__ import annotations

from pathlib import Path

import kmapper as km
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "model_final" / "output" / "reporte_revision.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "anomaly_score",
    "rule_score",
    "hora",
    "dia_semana",
    "mes",
    "severidad_num",
]

CATEGORY_COLORS = {
    "DUPLICADO": "#2f6fbb",
    "MONTO_ATIPICO": "#c43d3d",
    "CANCELACION_SOSPECHOSA": "#6a4c93",
    "CONTEXTO_RESERVACION": "#2a9d8f",
    "FUERA_DE_ESTANCIA": "#e89f3d",
    "ATIPICO_IF": "#5c677d",
    "SIGNO_CONTABLE": "#8a5a44",
    "METODO_PAGO": "#4d908e",
    "Sin categoria": "#777777",
}


def savefig(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def prepare_data() -> tuple[pd.DataFrame, np.ndarray]:
    df_full = pd.read_parquet(REPORT_PATH)
    if "severidad" in df_full.columns and "severity" not in df_full.columns:
        df_full = df_full.rename(columns={"severidad": "severity"})

    anomaly_col = "is_anomaly" if "is_anomaly" in df_full.columns else "is_anomaly_if"
    df = df_full[df_full[anomaly_col].astype(bool)].copy().reset_index(drop=True)

    ts = pd.to_datetime(df["trace_t_timestamp"], errors="coerce")
    df["hora"] = ts.dt.hour.fillna(0)
    df["dia_semana"] = ts.dt.dayofweek.fillna(0)
    df["mes"] = ts.dt.month.fillna(0)

    severity_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2, "CRITICO": 3}
    df["severidad_num"] = df["severity"].map(severity_map).fillna(0)
    df["cat_principal"] = (
        df["tipo_inconsistencia"]
        .fillna("Sin categoria")
        .str.split(" | ", regex=False)
        .str[0]
    )

    x_raw = df[FEATURES].fillna(0).to_numpy()
    x = RobustScaler().fit_transform(x_raw)
    return df, x


def build_mapper(df: pd.DataFrame, x: np.ndarray) -> tuple[np.ndarray, dict, pd.DataFrame, nx.Graph]:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    lens = reducer.fit_transform(x)

    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens,
        x,
        clusterer=DBSCAN(eps=0.6, min_samples=3),
        cover=km.Cover(n_cubes=12, perc_overlap=0.5),
    )

    g = nx.Graph()
    for node_id, member_idx in graph["nodes"].items():
        members = df.iloc[list(member_idx)]
        category_counts = members["cat_principal"].value_counts()
        cat_top = category_counts.index[0] if not category_counts.empty else "Sin categoria"
        g.add_node(
            node_id,
            n=len(members),
            score_mean=members["anomaly_score"].mean(),
            rule_score_mean=members["rule_score"].mean(),
            severity_mean=members["severidad_num"].mean(),
            cat_top=cat_top,
            cats_unicas=members["cat_principal"].nunique(),
        )

    for source, targets in graph["links"].items():
        for target in targets:
            if source != target:
                g.add_edge(source, target)

    profiles = pd.DataFrame(
        [
            {
                "nodo": node,
                "n": attrs["n"],
                "score_mean": attrs["score_mean"],
                "rule_score_mean": attrs["rule_score_mean"],
                "severidad_mean": attrs["severity_mean"],
                "cat_top": attrs["cat_top"],
                "cats_unicas": attrs["cats_unicas"],
                "degree": g.degree(node),
            }
            for node, attrs in g.nodes(data=True)
        ]
    )
    profiles = profiles.sort_values(["score_mean", "rule_score_mean"], ascending=False)
    return lens, graph, profiles, g


def plot_lens_by_category(df: pd.DataFrame, lens: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for cat, group in df.groupby("cat_principal", sort=False):
        idx = group.index.to_numpy()
        ax.scatter(
            lens[idx, 0],
            lens[idx, 1],
            s=5,
            alpha=0.38,
            linewidths=0,
            color=CATEGORY_COLORS.get(cat, "#777777"),
            label=cat,
            rasterized=True,
        )
    ax.set_title("Proyección UMAP de anomalías por categoría")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    savefig(fig, "tda_umap_categorias.png")


def plot_lens_by_score(df: pd.DataFrame, lens: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    scatter = ax.scatter(
        lens[:, 0],
        lens[:, 1],
        c=df["anomaly_score"],
        cmap="magma",
        s=5,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("Proyección UMAP coloreada por anomaly score")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.18)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("anomaly_score")
    savefig(fig, "tda_umap_score.png")


def plot_mapper_graph(profiles: pd.DataFrame, g: nx.Graph) -> None:
    pos = nx.spring_layout(g, seed=42, iterations=150, k=0.24)
    node_order = list(g.nodes())
    sizes = np.array([g.nodes[n]["n"] for n in node_order])
    scores = np.array([g.nodes[n]["score_mean"] for n in node_order])
    scaled_sizes = 18 + 260 * np.sqrt(sizes / max(sizes.max(), 1))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.45, alpha=0.18, edge_color="#4b5563")
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=scaled_sizes,
        node_color=scores,
        cmap="magma",
        linewidths=0.25,
        edgecolors="#ffffff",
        alpha=0.92,
    )
    ax.set_title("Grafo Mapper de anomalías")
    ax.set_axis_off()
    cbar = fig.colorbar(nodes, ax=ax, shrink=0.82)
    cbar.set_label("anomaly_score promedio del nodo")

    legend_sizes = [5, 25, 75]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#737373",
            markeredgecolor="white",
            markersize=np.sqrt(18 + 260 * np.sqrt(size / max(sizes.max(), 1))) / 1.4,
            label=f"{size} transacciones",
        )
        for size in legend_sizes
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, title="Tamaño del nodo")
    savefig(fig, "tda_mapper_grafo_estatico.png")


def plot_node_risk_profile(profiles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    for cat, group in profiles.groupby("cat_top", sort=False):
        ax.scatter(
            group["rule_score_mean"],
            group["score_mean"],
            s=20 + 8 * np.sqrt(group["n"]),
            alpha=0.68,
            color=CATEGORY_COLORS.get(cat, "#777777"),
            linewidths=0.35,
            edgecolors="white",
            label=cat,
        )

    ax.set_title("Perfil de riesgo por nodo Mapper")
    ax.set_xlabel("rule_score promedio")
    ax.set_ylabel("anomaly_score promedio")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    savefig(fig, "tda_perfil_riesgo_nodos.png")


def plot_category_summary(df: pd.DataFrame, profiles: pd.DataFrame) -> None:
    anomaly_counts = df["cat_principal"].value_counts().sort_values()
    node_counts = profiles["cat_top"].value_counts().reindex(anomaly_counts.index).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    axes[0].barh(
        anomaly_counts.index,
        anomaly_counts.values,
        color=[CATEGORY_COLORS.get(cat, "#777777") for cat in anomaly_counts.index],
    )
    axes[0].set_title("Transacciones anómalas por categoría")
    axes[0].set_xlabel("Transacciones")
    axes[0].grid(axis="x", alpha=0.18)

    axes[1].barh(
        node_counts.index,
        node_counts.values,
        color=[CATEGORY_COLORS.get(cat, "#777777") for cat in node_counts.index],
    )
    axes[1].set_title("Nodos Mapper por categoría dominante")
    axes[1].set_xlabel("Nodos")
    axes[1].grid(axis="x", alpha=0.18)
    savefig(fig, "tda_resumen_categorias.png")


def plot_top_nodes_table(profiles: pd.DataFrame) -> None:
    table_data = profiles.head(10).copy()
    table_data["score_mean"] = table_data["score_mean"].map(lambda x: f"{x:.3f}")
    table_data["rule_score_mean"] = table_data["rule_score_mean"].map(lambda x: f"{x:.1f}")
    table_data["severidad_mean"] = table_data["severidad_mean"].map(lambda x: f"{x:.2f}")
    table_data = table_data[
        ["nodo", "n", "score_mean", "rule_score_mean", "severidad_mean", "cat_top", "degree"]
    ]
    table_data.columns = [
        "Nodo",
        "n",
        "score",
        "rule score",
        "sev.",
        "categoria",
        "grado",
    ]

    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
        colColours=["#e5e7eb"] * len(table_data.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_text_props(weight="bold")
        elif col == 5:
            cat = table_data.iloc[row - 1, col]
            cell.set_facecolor(CATEGORY_COLORS.get(cat, "#ffffff"))
            cell.set_text_props(color="white")
    ax.set_title("Top 10 nodos Mapper por anomaly_score promedio", pad=14)
    savefig(fig, "tda_top_nodos_riesgo.png")


def write_summary(df: pd.DataFrame, graph: dict, profiles: pd.DataFrame, g: nx.Graph) -> None:
    components = list(nx.connected_components(g))
    largest_component = max((len(c) for c in components), default=0)
    summary = {
        "n_anomalias": len(df),
        "n_nodes": len(graph["nodes"]),
        "n_edges_unique": g.number_of_edges(),
        "n_components": len(components),
        "largest_component_nodes": largest_component,
        "top_category_transactions": df["cat_principal"].value_counts().idxmax(),
        "top_category_nodes": profiles["cat_top"].value_counts().idxmax(),
        "max_node_score": profiles["score_mean"].max(),
        "max_node_rule_score": profiles["rule_score_mean"].max(),
    }
    lines = [f"{key}: {value}" for key, value in summary.items()]
    (OUTPUT_DIR / "tda_summary.txt").write_text("\n".join(lines) + "\n")
    profiles.to_csv(OUTPUT_DIR / "tda_node_profiles.csv", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    df, x = prepare_data()
    lens, graph, profiles, g = build_mapper(df, x)

    plot_lens_by_category(df, lens)
    plot_lens_by_score(df, lens)
    plot_mapper_graph(profiles, g)
    plot_node_risk_profile(profiles)
    plot_category_summary(df, profiles)
    plot_top_nodes_table(profiles)
    write_summary(df, graph, profiles, g)

    print(f"Generated figures in {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
