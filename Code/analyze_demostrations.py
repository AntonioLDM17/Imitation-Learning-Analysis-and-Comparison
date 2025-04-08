import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

def extract_transitions_from_trajectory(traj):
    """ 
    Given a TrajectoryWithRew object, extract transitions in the form of tuples:
    (state, action, next_state, done).
    It is assumed that:
        - 'obs' is an array of observations,
        - 'acts' is an array of actions,
        - 'terminal' indicates if the trajectory ends.
    Transitions are generated for each consecutive pair of observations.
    The final transition is marked as done if traj.terminal is True.
    """
    transitions = []
    obs = traj.obs
    acts = traj.acts
    n = len(obs)
    for i in range(n - 1):
        done = (i == n - 2) and getattr(traj, "terminal", False)
        transitions.append((obs[i], acts[i], obs[i+1], done))
    return transitions

def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert demonstrations for richness and diversity"
    )
    parser.add_argument("--demo_path", type=str, required=True,
                        help="Path to the demonstration file (.npy)")
    parser.add_argument("--log_dir", type=str, default="logs/analyze_demos",
                        help="Directory for TensorBoard logs")
    args = parser.parse_args()
    
    # Load demonstrations
    demos = np.load(args.demo_path, allow_pickle=True)
    print(f"Loaded {len(demos)} demonstrations from {args.demo_path}")

    # Add all states and actions from each trajectory
    all_states = []
    all_actions = []
    for traj in demos:
        # We suppose that each traj has attributes 'obs' and 'acts'
        all_states.append(traj.obs)
        all_actions.append(traj.acts)
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"Total states: {all_states.shape}, Total actions: {all_actions.shape}")
    
    # Calculate mean and std of states
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    
    # Calculate PCA for states (reduce to 2 dimensions)
    pca = PCA(n_components=2)
    states_pca = pca.fit_transform(all_states)
    
    # Create TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Log statistics to TensorBoard
    stats_text = (
        f"Total demonstrations: {len(demos)}\n"
        f"Total transitions (states): {all_states.shape[0]}\n"
        f"State Mean: {state_mean}\n"
        f"State Std: {state_std}\n"
    )
    writer.add_text("Demo_Stats", stats_text, global_step=0)
    
    # Generate and log PCA plot of states
    fig_pca, ax_pca = plt.subplots(figsize=(8,6))
    ax_pca.scatter(states_pca[:,0], states_pca[:,1], s=1, alpha=0.5)
    ax_pca.set_title("PCA of States from Demonstrations")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    writer.add_figure("PCA/States", fig_pca, global_step=0)
    plt.close(fig_pca)
    
    # Generate and log histograms of actions (for each dimension)
    num_action_dims = all_actions.shape[1]
    for i in range(num_action_dims):
        fig_hist, ax_hist = plt.subplots(figsize=(8,6))
        ax_hist.hist(all_actions[:, i], bins=50, alpha=0.7)
        ax_hist.set_title(f"Histogram of Action Dimension {i}")
        ax_hist.set_xlabel("Action value")
        ax_hist.set_ylabel("Frequency")
        writer.add_figure(f"Histogram/Action_dim_{i}", fig_hist, global_step=0)
        plt.close(fig_hist)
    
    writer.close()
    print("Analysis logged to TensorBoard. Ejecuta 'tensorboard --logdir {}' para ver las gráficas.".format(args.log_dir))

if __name__ == "__main__":
    print("Example usage:")
    print("python analyze_demostrations.py --demo_path data/demonstrations/halfcheetah_demonstrations.npy --log_dir logs/analyze_demos")
    main()
