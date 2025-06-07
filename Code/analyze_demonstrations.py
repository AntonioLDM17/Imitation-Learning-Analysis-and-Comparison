import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────
def _reshape_actions(actions: np.ndarray) -> np.ndarray:
    """
    Ensures the action array has shape (n_samples, n_dims).
    Discrete policies often store actions as 1-D int arrays; we convert them
    to column vectors for unified downstream processing.
    """
    if actions.ndim == 1:              # (n_samples,)  →  (n_samples, 1)
        actions = actions.reshape(-1, 1)
    return actions

def extract_transitions_from_trajectory(traj):
    """
    Given a TrajectoryWithRew object, extract (s, a, s', done) tuples.
    Provided here for completeness; not used in the current analytics flow.
    """
    transitions = []
    obs, acts = traj.obs, traj.acts
    n = len(obs)
    for i in range(n - 1):
        done = (i == n - 2) and getattr(traj, "terminal", False)
        transitions.append((obs[i], acts[i], obs[i + 1], done))
    return transitions

# ──────────────────────────────────────────────────────────────
# Main analysis routine
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze expert demonstrations for richness and diversity"
    )
    parser.add_argument(
        "--demo_path", type=str, required=True,
        help="Path to the .npy file containing the demonstrations"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=60,
        help="Number of episodes expected / logged (for directory naming only)"
    )
    parser.add_argument(
        "--env", type=str, default="halfcheetah",
        choices=["cartpole", "halfcheetah"],
        help="Environment name: 'cartpole' (discrete) or 'halfcheetah' (continuous)"
    )
    args = parser.parse_args()

    # 1. Load demonstrations ----------------------------------------------------
    demos = np.load(args.demo_path, allow_pickle=True)
    print(f"Loaded {len(demos)} demonstrations from {args.demo_path}")

    # 2. Aggregate state and action arrays across all trajectories --------------
    all_states, all_actions = [], []
    for traj in demos:
        all_states.append(traj.obs)
        all_actions.append(traj.acts)

    all_states  = np.concatenate(all_states,  axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_actions = _reshape_actions(all_actions)  # key fix ✔

    print(f"Total states: {all_states.shape}, Total actions: {all_actions.shape}")

    # 3. Detect action-space type ----------------------------------------------
    # Heuristic: if dtype is integer **and** env is known discrete → discrete.
    action_space_type = (
        "Discrete" if (np.issubdtype(all_actions.dtype, np.integer) or args.env == "cartpole")
        else "Continuous"
    )

    # 4. Compute statistics & PCA ----------------------------------------------
    state_mean, state_std = np.mean(all_states, axis=0), np.std(all_states, axis=0)

    pca = PCA(n_components=2, random_state=0)
    states_pca = pca.fit_transform(all_states)

    # 5. TensorBoard setup ------------------------------------------------------
    log_dir = os.path.join("logs/analyze_demos", args.env, str(args.num_episodes))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # 6. Log basic stats --------------------------------------------------------
    stats_text = (
        f"Total demonstrations: {len(demos)}\n"
        f"Total transitions (states): {all_states.shape[0]}\n"
        f"State mean: {state_mean}\n"
        f"State std : {state_std}\n"
        f"Action space type: {action_space_type}\n"
    )
    writer.add_text("Demo_Stats", stats_text, global_step=0)

    # 7. PCA scatter plot -------------------------------------------------------
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    ax_pca.scatter(states_pca[:, 0], states_pca[:, 1], s=1, alpha=0.5)
    ax_pca.set_title("PCA of States from Demonstrations")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    writer.add_figure("PCA/States", fig_pca, global_step=0)
    plt.close(fig_pca)

    # 8. Action distribution visualisation -------------------------------------
    if action_space_type == "Discrete":
        # Bar plot of counts per discrete action
        unique_acts, counts = np.unique(all_actions[:, 0], return_counts=True)
        fig_act, ax_act = plt.subplots(figsize=(8, 6))
        ax_act.bar(unique_acts, counts, width=0.6)
        ax_act.set_title("Action Counts (Discrete)")
        ax_act.set_xlabel("Action")
        ax_act.set_ylabel("Frequency")
        writer.add_figure("Actions/Discrete_Counts", fig_act, global_step=0)
        plt.close(fig_act)
    else:
        # Histograms per action dimension
        num_action_dims = all_actions.shape[1]
        for dim in range(num_action_dims):
            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
            ax_hist.hist(all_actions[:, dim], bins=50, alpha=0.7)
            ax_hist.set_title(f"Histogram of Action Dimension {dim}")
            ax_hist.set_xlabel("Action value")
            ax_hist.set_ylabel("Frequency")
            writer.add_figure(f"Actions/Histogram_dim_{dim}", fig_hist, global_step=0)
            plt.close(fig_hist)

    writer.close()
    print(
        "Analysis logged to TensorBoard. "
        f"Run `tensorboard --logdir {log_dir}` to review the visuals."
    )

# ──────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Example usage:")
    print(
        "python analyze_demonstrations.py --demo_path data/demonstrations/100/halfcheetah_demonstrations.npy --num_episodes 100 --env halfcheetah"
    )
    print(
        "python analyze_demonstrations.py --demo_path data/demonstrations/100/cartpole_demonstrations_100.npy --num_episodes 100 --env cartpole"
    )
    main()
