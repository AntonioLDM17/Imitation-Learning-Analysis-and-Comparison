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
EXPERT_Y   = 6042.55          # recompensa media del experto

# ------------------------------------------------------------------
def read_summary(path):
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
def _ensure_expert_visible(axes):
    """Eleva el límite superior si es necesario para mostrar EXPERT_Y."""
    for ax in axes:
        if not ax.get_visible():
            continue
        ymin, ymax = ax.get_ylim()
        if ymax < EXPERT_Y * 1.02:          # deja ~2 % de margen
            ax.set_ylim(top=EXPERT_Y * 1.02)

# ------------------------------------------------------------------
def figure_et(df, episodes, outdir):
    df = df.dropna(subset=["trayectorias"]).copy()
    df["et"] = df["std"] / math.sqrt(episodes)

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, alg in enumerate(ALGS_ORDER):
        ax = axes[idx]
        sub = (df.query("algoritmo == @alg")
                 .set_index("trayectorias")
                 .reindex(TRAJ_ORDER))

        # línea del experto
        label_exp = "Experto" if idx == 0 else "_nolegend_"
        ax.axhline(EXPERT_Y, ls="--", lw=1, color="gold", label=label_exp)

        if sub["media"].isna().all():
            ax.set_visible(False)
            continue

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

    _ensure_expert_visible(axes)

    plt.suptitle("Figura 1 – Rendimiento medio ± ET vs. nº trayectorias", y=0.97)
    fig.legend(loc="lower center", ncol=6, fontsize=7)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "figure1_sample_efficiency")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------
def heatmap_percent(df, outdir):
    mask_exp  = df["algoritmo"].str.contains("EXPERT")
    expert_mu = df.loc[mask_exp].set_index("env")["media"].to_dict()

    df2 = df.loc[~mask_exp].copy()
    df2["% expert"] = df2.apply(
        lambda r: 100 * r["media"] / expert_mu[r["env"]], axis=1)

    pivot = (df2.pivot(index="algoritmo", columns="trayectorias", values="% expert")
               .reindex(index=ALGS_ORDER, columns=TRAJ_ORDER))

    plt.figure(figsize=FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=HEAT_CMAP,
                vmin=40, vmax=110, cbar_kws={"label": "% experto"})
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

        # línea del experto
        label_exp = "Experto" if idx == 0 else "_nolegend_"
        ax.axhline(EXPERT_Y, ls="--", lw=1, color="gold", label=label_exp)

        xs, ys, yerr = sub.index.values, sub["media"].values, sub["std"].values
        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="-", color="black",
                    ecolor="gray", elinewidth=2, capsize=4)
        ax.set_title(alg, fontsize=10)
        ax.set_xticks(TRAJ_ORDER)
        ax.grid(alpha=0.3)
        if idx // 3 == 1: ax.set_xlabel("Trayectorias")
        if idx % 3 == 0:  ax.set_ylabel("Recompensa media")

    _ensure_expert_visible(axes)

    plt.suptitle("Recompensa Media ± std (100 episodios)", y=0.94)
    fig.legend(loc="lower center", ncol=6, fontsize=7)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "errorbars_media_std.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    
def combined_heatmap_errorbars(df, outdir):
    """
    Dibuja en una sola figura, a la izquierda el heatmap de % experto
    y a la derecha el grid de Error-bars ± std, uno al lado del otro.
    """
    # --- Datos para el heatmap ---
    mask_exp  = df["algoritmo"].str.contains("EXPERT")
    expert_mu = df.loc[mask_exp].set_index("env")["media"].to_dict()
    df2 = df.loc[~mask_exp].copy()
    df2["% expert"] = df2.apply(
        lambda r: 100 * r["media"] / expert_mu[r["env"]], axis=1)
    pivot = (df2
             .pivot(index="algoritmo", columns="trayectorias", values="% expert")
             .reindex(index=ALGS_ORDER, columns=TRAJ_ORDER))

    # --- Datos para los errorbars ± std ---
    df_err = df.loc[~df["algoritmo"].str.contains("EXPERT")].copy()

    # --- Figura combinada ---
    fig = plt.figure(figsize=FIGSIZE)
    # reducimos la proporción de la columna del heatmap de 3 a 2
    gs = fig.add_gridspec(2, 4,
                          width_ratios=[2, 1, 1, 1],
                          wspace=0.25, hspace=0.3)

    # Heatmap en la columna 0
    ax0 = fig.add_subplot(gs[:, 0])
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=HEAT_CMAP,
                vmin=40, vmax=110,
                cbar_kws={
                  "label": "% experto",
                  "shrink": 0.7,     # barra más pequeña
                  "pad": 0.02       # menos espacio entre heatmap y cbar
                },
                ax=ax0)
    ax0.set_title("Heat-map – Recompensa media normalizada al experto")
    ax0.set_xlabel("Nº trayectorias")
    ax0.set_ylabel("")
    # rotamos y centramos las etiquetas
    ax0.set_xticklabels(TRAJ_ORDER, rotation=0, ha="center")

    # Error-bars ± std en grid 2×3
    axes = []
    for idx, alg in enumerate(ALGS_ORDER):
        row = idx // 3
        col = (idx % 3) + 1
        ax = fig.add_subplot(gs[row, col])

        sub = (df_err.query("algoritmo == @alg")
                   .set_index("trayectorias")
                   .reindex(TRAJ_ORDER))

        label_exp = "Experto" if idx == 0 else "_nolegend_"
        ax.axhline(EXPERT_Y, ls="--", lw=1, color="gold", label=label_exp)

        xs, ys, yerr = sub.index.values, sub["media"].values, sub["std"].values
        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="-",
                    color="black", ecolor="gray",
                    elinewidth=2, capsize=4)

        ax.set_title(alg, fontsize=10)
        ax.set_xticks(TRAJ_ORDER)
        ax.grid(alpha=0.3)
        if row == 1: ax.set_xlabel("Trayectorias")
        if col == 1: ax.set_ylabel("Recompensa media")
        axes.append(ax)

        # Fijamos una escala uniforme de 0 a ~6 163 (2 000 pasos), y ticks en 0,2000,4000,6000
        y_max = EXPERT_Y * 1.02
        for ax in axes:
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, 2000, 4000, 6000])


    plt.suptitle("Heat-map y Recompensa Media ± std", y=0.98)

    # **Ajustamos márgenes** para eliminar espacios muertos
    fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.10)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "combined_heatmap_errorbars")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Genera Figura 1, Heat-map y Error-bars.")
    ap.add_argument("--summary", required=True,
                    help="CSV/XLSX con columnas algoritmo, trayectorias, media, std, incluye EXPERT.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Episodios usados para el cálculo del ET.")
    ap.add_argument("--outdir", default="figures",
                    help="Carpeta raíz donde guardar las figuras.")
    args = ap.parse_args()

    df = read_summary(args.summary)
    figure_et(df, args.episodes, args.outdir)
    heatmap_percent(df, os.path.join(args.outdir, "annex"))
    errorbars_std(df, os.path.join(args.outdir, "annex"))
    combined_heatmap_errorbars(df, os.path.join(args.outdir, "annex"))
    print("Figuras guardadas en:", os.path.abspath(args.outdir))
