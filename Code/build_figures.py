#!/usr/bin/env python
# make_all_figures.py
# ------------------------------------------------------------
import os, argparse, math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ALGS_ORDER = ["AIRL", "BC", "BCO", "GAIFO", "GAIL", "SQIL"]
TRAJ_ORDER = [5, 10, 20, 50, 100]
COLORS     = {5:"#9467bd", 10:"#d62728", 20:"#2ca02c",
              50:"#ff7f0e", 100:"#1f77b4"}
HEAT_CMAP  = "RdYlGn"
FIGSIZE    = (12, 6)

# ------------------------------------------------------------------
def read_summary(path):
    """Lee el CSV/XLSX y normaliza nombres, números y coma decimal."""
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in {".xlsx", ".xls"} else pd.read_csv(path)

    df.columns = [c.lower().strip() for c in df.columns]

    if "trayectoria" in df.columns and "trayectorias" not in df.columns:
        df = df.rename(columns={"trayectoria": "trayectorias"})

    df["trayectorias"] = pd.to_numeric(df["trayectorias"], errors="coerce")
    df["algoritmo"] = df["algoritmo"].astype(str).str.strip().str.upper()

    for col in ["media", "std"]:
        df[col] = (df[col].astype(str)
                          .str.replace(",", ".", regex=False)
                          .astype(float))
    return df

# ------------------------------------------------------------------
def figure_et(df, episodes, outdir):
    """Figura 1 – curvas media ± ET (error típico)."""
    df = df.dropna(subset=["trayectorias"]).copy()
    df["et"] = df["std"] / math.sqrt(episodes)

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, alg in enumerate(ALGS_ORDER):
        ax  = axes[idx]
        sub = (df.query("algoritmo == @alg")
                 .set_index("trayectorias")
                 .reindex(TRAJ_ORDER))

        if sub["media"].isna().all():
            ax.set_visible(False); continue

        xs, ys, yerr = sub.index.values, sub["media"].values, sub["et"].values
        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="-",
                    color="black", ecolor="gray",
                    elinewidth=2.0, capsize=5)
        for x, yv in zip(xs, ys):
            if not pd.isna(yv):
                ax.scatter(x, yv, s=40, color=COLORS[x])

        ax.set_title(alg, fontsize=10)
        ax.grid(alpha=0.3)
        if idx // 3 == 1: ax.set_xlabel("Nº trayectorias")
        if idx % 3 == 0:  ax.set_ylabel("Recompensa media")
        ax.set_xticks(TRAJ_ORDER)

    plt.suptitle("Figura 1 – Rendimiento medio ± ET vs. nº trayectorias", y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "figure1_sample_efficiency")
    fig.savefig(fname + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------
def heatmap_percent(df, outdir):
    mask_exp  = df["algoritmo"].str.contains("EXPERT")
    expert_mu = df.loc[mask_exp].set_index("env")["media"].to_dict()

    df2 = df.loc[~mask_exp].copy()
    df2["% expert"] = df2.apply(lambda r: 100*r["media"]/expert_mu[r["env"]], axis=1)

    pivot = (df2.pivot(index="algoritmo", columns="trayectorias", values="% expert")
               .reindex(index=ALGS_ORDER, columns=TRAJ_ORDER))

    plt.figure(figsize=FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=HEAT_CMAP,
                vmin=50, vmax=120, cbar_kws={"label": "% experto"})
    plt.title("Heat-map – Recompensa media normalizada al experto")
    plt.ylabel(""); plt.xlabel("Nº trayectorias")

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "heatmap_percent_expert.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------
def errorbars_std(df, outdir):
    df = df[~df["algoritmo"].str.contains("EXPERT")].copy()

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
    axes = axes.ravel()

    for idx, alg in enumerate(ALGS_ORDER):
        ax = axes[idx]
        sub = (df.query("algoritmo == @alg")
                 .set_index("trayectorias")
                 .reindex(TRAJ_ORDER))
        xs, ys, yerr = sub.index.values, sub["media"].values, sub["std"].values

        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="-", color="black",
                    ecolor="lightgray", elinewidth=1.5, capsize=4)
        ax.set_title(alg, fontsize=10)
        ax.set_xticks(TRAJ_ORDER)
        ax.grid(alpha=0.3)
        if idx // 3 == 1: ax.set_xlabel("Trayectorias")
        if idx % 3 == 0:  ax.set_ylabel("Recompensa media")

    plt.suptitle("Anexo B – Media ± std (100 episodios)", y=0.94)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "errorbars_media_std.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera Figura 1, Heat-map y Error-bars.")
    ap.add_argument("--summary", required=True,
                    help="CSV/XLSX con columnas algoritmo, trayectorias, media, std, incluye EXPERT.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Episodios usados para el cálculo del ET.")
    ap.add_argument("--outdir", default="figures",
                    help="Carpeta raíz donde guardar las figuras.")
    args = ap.parse_args()

    df = read_summary(args.summary)

    figure_et(df, args.episodes, args.outdir)           # Figura 1
    heatmap_percent(df, os.path.join(args.outdir, "annex"))
    errorbars_std(df,   os.path.join(args.outdir, "annex"))

    print("Figuras guardadas en:", os.path.abspath(args.outdir))
