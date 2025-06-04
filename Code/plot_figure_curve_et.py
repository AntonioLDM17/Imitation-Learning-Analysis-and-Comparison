#!/usr/bin/env python
import os, argparse, math
import pandas as pd
import matplotlib.pyplot as plt

ALGS_ORDER = ["AIRL", "BC", "BCO", "GAIFO", "GAIL", "SQIL"]
TRAJ_ORDER = [5, 10, 20, 50, 100]
COLORS     = {5:"#9467bd", 10:"#d62728", 20:"#2ca02c", 50:"#ff7f0e", 100:"#1f77b4"}

# ---------------- lectura robusta -----------------
def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, sep=None, engine="python", decimal=".")
        except ValueError:
            df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    else:
        raise ValueError("Formato no soportado.")

    # normaliza columnas
    df.columns = [c.lower().strip() for c in df.columns]

    # --- fragmento corregido dentro de read_table() -----------------
    # unifica nombre de la columna trayectorias
    if "trayectoria" in df.columns and "trayectorias" not in df.columns:
        df = df.rename(columns={"trayectoria": "trayectorias"})

    # convierte a número; '-' u otros valores no numéricos → NaN
    df["trayectorias"] = pd.to_numeric(df["trayectorias"], errors="coerce")

    # elimina filas sin número de trayectorias (p. ej. EXPERT)
    df = df.dropna(subset=["trayectorias"])
    df["trayectorias"] = df["trayectorias"].astype(int)
# ----------------------------------------------------------------


    df["trayectorias"] = df["trayectorias"].astype(int)
    df["algoritmo"] = df["algoritmo"].astype(str).str.strip().str.upper()

    for col in ["media", "std"]:
        df[col] = (df[col].astype(str)
                          .str.replace(",", ".", regex=False)
                          .astype(float))
    return df

# ---------------- figura --------------------------
def main(infile, outfile, episodes):
    df = read_table(infile)
    df["et"] = df["std"] / math.sqrt(episodes)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, alg in enumerate(ALGS_ORDER):
        ax = axes[idx]
        sub = (df.query("algoritmo == @alg")
                 .set_index("trayectorias")
                 .reindex(TRAJ_ORDER))

        if sub["media"].isna().all():        # nada que mostrar
            ax.set_visible(False)
            continue

        xs  = sub.index.values
        ys  = sub["media"].values
        yerr = sub["et"].values

        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="-",
                    color="black", ecolor="gray",
                    elinewidth=2.0, capsize=5)

        for x, yv in zip(xs, ys):
            if not pd.isna(yv):
                ax.scatter(x, yv, s=40, color=COLORS[x])

        ax.set_title(alg, fontsize=10)
        ax.grid(alpha=0.3)
        if idx // 3 == 1:
            ax.set_xlabel("Nº trayectorias")
        if idx % 3 == 0:
            ax.set_ylabel("Recompensa media")
        ax.set_xticks(TRAJ_ORDER)

    fig.suptitle("Figura 1 – Rendimiento medio ± ET vs. número de trayectorias",
                 fontsize=12, y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    base = outfile or "figure1_sample_efficiency"
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print("Figura guardada en:", base + ".png / .pdf")

# ---------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True)
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    main(args.infile, args.outfile, args.episodes)
