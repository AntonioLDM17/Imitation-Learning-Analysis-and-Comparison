import os, glob, numpy as np, matplotlib.pyplot as plt
from collections import OrderedDict
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
colors      = {100:"#1f77b4", 50:"#ff7f0e", 20:"#2ca02c",
               10:"#d62728",  5:"#9467bd"}
HORIZON     = 2_000_000
EXPERT_REW  = 6_000            # expert reward for HalfCheetah

# TensorBoard event files are stored in a nested directory structure.
def events(run):
    return glob.glob(os.path.join(run, "**", "events.*"), recursive=True)

def read(ev_file, tag):
    acc = ea.EventAccumulator(ev_file, size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None, None
    data = acc.Scalars(tag)
    s = np.fromiter((d.step  for d in data), dtype=float)
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

# Smoothing function for EMA
def smooth_ema(values, beta):
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for t in range(1, len(values)):
        smoothed[t] = beta * smoothed[t-1] + (1.0 - beta) * values[t]
    return smoothed


def build_mosaic():
    """ Builds a mosaic plot comparing different algorithms on HalfCheetah with EMA smoothing."""
    fig, axs = plt.subplots(2, 3, figsize=(16, 6), sharex=True, sharey=True)
    axs = axs.ravel()

    for idx, algo in enumerate(algorithms):
        ax, tag = axs[idx], TAGS[algo]

        # expert line
        ax.axhline(EXPERT_REW, color="gold", ls="--", lw=1.0,
                   label="Experto" if idx == 0 else "_nolegend_")

        # parameters of smoothing
        beta      = 0.9 if algo == "sqil" else 0.6
        alpha_raw = 0.2 if algo == "sqil" else 0.4
        lw_raw    = 0.4 if algo == "sqil" else 0.5

        for n in traj_order:
            run_dir = os.path.join(LOG_ROOT, f"{algo}_halfcheetah_{n}")
            steps, rew_raw = load(run_dir, algo, tag)
            if steps is None:
                continue

            steps = steps * (HORIZON / steps[-1])      # Normalize to 2M steps
            rew   = smooth_ema(rew_raw, beta)

            ax.plot(steps, rew_raw, color=colors[n], alpha=alpha_raw, lw=lw_raw)
            ax.plot(steps, rew,      color=colors[n], label=str(n))

        ax.set_title(algo.upper(), fontsize=9)
        ax.tick_params(labelsize=7)

    # Global settings
    fig.text(0.5, 0, "Steps (0 – 2 000 000)", ha="center")
    fig.text(-0.01, 0.5, "Mean episode reward", va="center", rotation="vertical")

    # Global legend
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)

    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)

    fig.legend(uniq.values(), uniq.keys(),
               title="Trayectorias", ncol=6, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.1))

    fig.subplots_adjust(bottom=0.12)
    plt.tight_layout()
    plt.show()

    out_dir = os.path.join("figures", "mosaics")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "mosaic_ema_2x3.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print("Guardado:", fname)

if __name__ == "__main__":
    build_mosaic()
