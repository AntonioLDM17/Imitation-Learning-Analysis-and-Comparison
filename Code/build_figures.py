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
EXPERT_Y   = 6042.55          # Mean reward of the expert in HalfCheetah-v4

def read_summary(path):
    """ Reads a summary file and returns a DataFrame. """
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

def _ensure_expert_visible(axes):
    """Ensures that the expert line is visible in all axes."""
    for ax in axes:
        if not ax.get_visible():
            continue
        ymin, ymax = ax.get_ylim()
        if ymax < EXPERT_Y * 1.02:          # leave some space above
            ax.set_ylim(top=EXPERT_Y * 1.02)

def figure_et(df, episodes, outdir):
    """ 
    Generates Figure 1: Mean reward ± ET vs. number of trajectories.
    The ET is calculated as std / sqrt(episodes).
    """
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

def heatmap_percent(df, outdir):
    """
    Generates a heatmap showing the mean reward as a percentage of the expert's reward.
    The expert's reward is calculated from the rows containing "EXPERT" in the "algoritmo" column.
    """
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

def errorbars_std(df, outdir):
    """
    Generates a grid of error bars showing the mean reward ± std for each algorithm.
    Excludes rows containing "EXPERT" in the "algoritmo" column.
    """
    df = df[~df["algoritmo"].str.contains("EXPERT")].copy()

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
    axes = axes.ravel()

    for idx, alg in enumerate(ALGS_ORDER):
        ax = axes[idx]
        sub = (df.query("algoritmo == @alg")
                 .set_index("trayectorias")
                 .reindex(TRAJ_ORDER))

        # expert line
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
    Generates a combined figure with a heatmap and error bars ± std.
    The heatmap shows the mean reward as a percentage of the expert's reward.
    """
    # data preparation for heatmap
    mask_exp  = df["algoritmo"].str.contains("EXPERT")
    expert_mu = df.loc[mask_exp].set_index("env")["media"].to_dict()
    df2 = df.loc[~mask_exp].copy()
    df2["% expert"] = df2.apply(
        lambda r: 100 * r["media"] / expert_mu[r["env"]], axis=1)
    pivot = (df2
             .pivot(index="algoritmo", columns="trayectorias", values="% expert")
             .reindex(index=ALGS_ORDER, columns=TRAJ_ORDER))

    # Data preparation for error bars
    df_err = df.loc[~df["algoritmo"].str.contains("EXPERT")].copy()

    # Combine the heatmap and error bars in a single figure
    fig = plt.figure(figsize=FIGSIZE)
    # Gridspec for layout
    gs = fig.add_gridspec(2, 4,
                          width_ratios=[2, 1, 1, 1],
                          wspace=0.25, hspace=0.3)

    # Heatmap in the first column
    ax0 = fig.add_subplot(gs[:, 0])
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=HEAT_CMAP,
                vmin=40, vmax=110,
                cbar_kws={
                  "label": "% experto",
                  "shrink": 0.7,     # shrink the colorbar
                  "pad": 0.02       # pad between colorbar and heatmap
                },
                ax=ax0)
    ax0.set_title("Heat-map – Recompensa media normalizada al experto")
    ax0.set_xlabel("Nº trayectorias")
    ax0.set_ylabel("")
    # rotate x-ticks for better visibility
    ax0.set_xticklabels(TRAJ_ORDER, rotation=0, ha="center")

    # Error-bars ± std in grid 2×3
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

        # Set y-limits to ensure expert line is visible
        y_max = EXPERT_Y * 1.02
        for ax in axes:
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, 2000, 4000, 6000])


    plt.suptitle("Heat-map y Recompensa Media ± std", y=0.98)

    # Addjust layout
    fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.10)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "combined_heatmap_errorbars")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate Figura 1, Heat-map y Error-bars.")
    ap.add_argument("--summary", required=True,
                    help="CSV/XLSX with columnns algoritmo, trayectorias, media, std, includes EXPERT.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Episodes used to calculate the ET.")
    ap.add_argument("--outdir", default="figures",
                    help="Output folder to save the figures.")
    args = ap.parse_args()

    df = read_summary(args.summary)
    figure_et(df, args.episodes, args.outdir)
    heatmap_percent(df, os.path.join(args.outdir, "annex"))
    errorbars_std(df, os.path.join(args.outdir, "annex"))
    combined_heatmap_errorbars(df, os.path.join(args.outdir, "annex"))
    print("Figures saved on:", os.path.abspath(args.outdir))
