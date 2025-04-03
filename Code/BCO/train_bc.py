import os
import argparse, types, sys
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from bco import set_seed, PolicyNetwork, train_policy

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def extract_demo_data(demo_obj):
    """
    Extracts states and actions from demonstration data.
    The demonstration file is expected to contain expert demonstrations
    with attributes 'obs' and 'acts'. It can be either a single object or a list.
    """
    # If the demo object is a list (of trajectories)
    if isinstance(demo_obj, list) and hasattr(demo_obj[0], 'obs') and hasattr(demo_obj[0], 'acts'):
        states_list = [np.array(traj.obs).astype(np.float32) for traj in demo_obj]
        actions_list = [np.array(traj.acts) for traj in demo_obj]
        if len(states_list) == 1:
            states = states_list[0]
            actions = actions_list[0]
        else:
            states = np.concatenate(states_list, axis=0)
            actions = np.concatenate(actions_list, axis=0)
    # If the demo object is a single object with 'obs' and 'acts'
    elif hasattr(demo_obj, 'obs') and hasattr(demo_obj, 'acts'):
        states = np.array(demo_obj.obs).astype(np.float32)
        actions = np.array(demo_obj.acts)
    else:
        # Fallback: assume demo_obj is already a tuple (states, actions)
        try:
            states, actions = demo_obj
            states = np.array(states).astype(np.float32)
            actions = np.array(actions)
        except Exception as e:
            raise ValueError("Could not extract demonstration data. Ensure the file contains expert states and actions.") from e

    # Check for size mismatch between states and actions
    if states.shape[0] != actions.shape[0]:
        min_samples = min(states.shape[0], actions.shape[0])
        print(f"Size mismatch: states {states.shape[0]} vs actions {actions.shape[0]}, truncating to {min_samples}.")
        states = states[:min_samples]
        actions = actions[:min_samples]
    return states, actions

def main():
    parser = argparse.ArgumentParser(description="Train a Behavioral Cloning (BC) model using expert demonstrations.")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' (discrete) or 'halfcheetah' (continuous)")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="Path to demonstration file (npy) in data/demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Hyperparameters for policy training
    parser.add_argument("--policy_epochs", type=int, default=20, help="Number of epochs for policy training")
    parser.add_argument("--policy_lr", type=float, default=1e-3, help="Learning rate for policy training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for policy training")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.env == "cartpole":
        env_name = "CartPole-v1"
        discrete = True
    else:
        env_name = "HalfCheetah-v4"
        discrete = False

    # Setup TensorBoard logging
    log_dir = os.path.join("logs", f"bc_{args.env}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Load demonstration data
    if args.demo_file is None:
        demo_dir = os.path.join("..", "data", "demonstrations")
        demo_filename = f"{args.env}_demonstrations.npy"
        demo_path = os.path.join(demo_dir, demo_filename)
    else:
        demo_path = args.demo_file
    print(f"Loading demonstrations from {demo_path} ...")
    demo_data = np.load(demo_path, allow_pickle=True)
    if isinstance(demo_data, np.ndarray):
        if demo_data.dtype == object:
            if demo_data.size == 1:
                demo_obj = demo_data.item()
            else:
                demo_obj = demo_data.tolist()
        else:
            demo_obj = demo_data
    else:
        demo_obj = demo_data

    states, actions = extract_demo_data(demo_obj)
    print(f"Demonstration contains {states.shape[0]} samples.")

    # Initialize policy network
    obs_dim = states.shape[1]
    # For discrete actions, we determine the number of actions by the maximum label + 1
    action_dim = actions.shape[1] if not discrete else int(actions.max() + 1)
    policy_net = PolicyNetwork(obs_dim, action_dim, discrete=discrete)

    print("Training policy via Behavioral Cloning...")
    policy_net = train_policy(policy_net, states, actions,
                              discrete=discrete, epochs=args.policy_epochs,
                              lr=args.policy_lr, batch_size=args.batch_size,
                              writer=writer)

    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    model_name = f"bc_{args.env}.pt"
    save_path = os.path.join(models_dir, model_name)
    torch.save(policy_net.state_dict(), save_path)
    print(f"Behavioral Cloning model saved at {save_path}")

    writer.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_bc.py --env halfcheetah --demo_file ../data/demonstrations/halfcheetah_demonstrations.npy --seed 42")
    main()
