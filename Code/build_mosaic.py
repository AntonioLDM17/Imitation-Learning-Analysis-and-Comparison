import os, glob, numpy as np, matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator as ea

LOG_ROOT = "logs_finales"

TAGS = {
    "bco":  "eval/mean_reward",
    "sqil": "Reward/Steps",
    "gaifo":"Reward/Evaluation",
    "gail": "evaluation/mean_reward_steps",
    "airl": "evaluation/mean_reward_steps",
    "bc":   "evaluation/mean_reward",
}

algorithms  = ["bc", "bco", "gail", "gaifo", "airl", "sqil"]
traj_order  = [100, 50, 20, 10, 5]
colors      = {100:"#1f77b4", 50:"#ff7f0e", 20:"#2ca02c", 10:"#d62728", 5:"#9467bd"}
HORIZON     = 2_000_000
SPANS       = [50_000, 75_000, 100_000, 125_000, 150_000, 200_000]           # spans a probar
EXPERT_REW  = 6_000                              # nivel experto

# ---------------- lectura de TensorBoard ----------------
def events(run):
    return glob.glob(os.path.join(run, "**", "events.*"), recursive=True)

def read(ev_file, tag):
    acc = ea.EventAccumulator(ev_file, size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None, None
    data = acc.Scalars(tag)
    s = np.fromiter((d.step for d in data), dtype=float)
    v = np.fromiter((d.value for d in data), dtype=float)
    return s, v

def merge_bco(run, tag):
    series = [read(ev, tag) for ev in events(run)]
    series = [(s, v) for s, v in series if s is not None]
    if not series:
        return None, None
    s = np.concatenate([s for s, _ in series])
    v = np.concatenate([v for _, v in series])
    order = np.argsort(s);  s, v = s[order], v[order]
    _, idx = np.unique(s, return_index=True)
    return s[idx], v[idx]

def best(run, tag):
    best, ln = None, 0
    for ev in events(run):
        s, _ = read(ev, tag)
        if s is not None and len(s) > ln:
            best, ln = ev, len(s)
    return read(best, tag) if best else (None, None)

def load(run, algo, tag):
    return merge_bco(run, tag) if algo == "bco" else best(run, tag)

# ---------------- suavizado ----------------
def smooth_by_steps(steps, values, span):
    if len(values) < 3:
        return steps, values
    step_size = np.median(np.diff(steps))
    win = max(int(span / step_size), 3)
    kernel = np.ones(win) / win
    pad_L, pad_R = win // 2, win - win // 2 - 1
    padded = np.pad(values, (pad_L, pad_R), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return steps, smooth

# ---------------- generación de mosaico ----------------
def build_mosaic(span):
    fig, axs = plt.subplots(1, len(algorithms),
                            figsize=(16, 3), sharex=True, sharey=True)

    for col, algo in enumerate(algorithms):
        ax, tag = axs[col], TAGS[algo]

        # línea del experto (una sola etiqueta en el primer subplot)
        label_exp = "Experto" if col == 0 else "_nolegend_"
        ax.axhline(EXPERT_REW, color="gold", ls="--", lw=1.0, label=label_exp)

        for n in traj_order:
            run_dir = os.path.join(LOG_ROOT, f"{algo}_halfcheetah_{n}")
            steps, rew = load(run_dir, algo, tag)
            if steps is None:
                continue

            steps = steps * (HORIZON / steps[-1])          # escala
            steps, rew = smooth_by_steps(steps, rew, span) # suaviza
            ax.plot(steps, rew, color=colors[n], label=str(n))

        ax.set_title(algo.upper(), fontsize=9)
        ax.tick_params(labelsize=7)

    fig.text(0.5, -0.03, "Steps (0 – 2 000 000)", ha="center")
    fig.text(-0.03, 0.5, "Mean episode reward", va="center", rotation="vertical")

    # leyenda deduplicada
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)

    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)

    fig.legend(uniq.values(), uniq.keys(),
               title="Trayectorias", ncol=6, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.25))

    fig.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()

    out_dir = os.path.join("figures", "mosaics"); os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"mosaic_span{span//1000}k_expert.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print("Guardado:", fname)

# ---------------- ejecución ----------------
if __name__ == "__main__":
    for sp in SPANS:
        print(f"\n=== Generando mosaico con span = {sp:,} pasos ===")
        build_mosaic(sp)
