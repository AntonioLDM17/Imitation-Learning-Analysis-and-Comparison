import os
import argparse, types, sys
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from bco import set_seed, InverseDynamicsModel, PolicyNetwork, collect_exploration_data, create_dataloader, train_inverse_model, infer_expert_actions, train_policy

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def main():
    parser = argparse.ArgumentParser(description="Train a BCO model.")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' (discrete) or 'halfcheetah' (continuous)")
    parser.add_argument("--pre_interactions", type=int, default=2000,
                        help="Number of pre-demonstration interactions to train the inverse model")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="Path to the demonstrations (npy) in data/demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Set environment name and flag for discrete/continuous
    if args.env == "cartpole":
        env_name = "CartPole-v1"
        discrete = True
    else:
        env_name = "HalfCheetah-v4"
        discrete = False

    # Create a TensorBoard log directory and SummaryWriter
    log_dir = os.path.join("logs", f"bco_{args.env}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    env = gym.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0]
    action_dim = act_space.n if discrete else act_space.shape[0]

    print("Collecting exploration data...")
    s_exp, s_next_exp, a_exp = collect_exploration_data(env, args.pre_interactions)
    print(f"Collected {s_exp.shape[0]} transitions.")

    loader = create_dataloader(s_exp, s_next_exp, a_exp, batch_size=64)
    inv_model = InverseDynamicsModel(obs_dim, action_dim, discrete=discrete)
    print("Training the inverse dynamics model...")
    inv_model = train_inverse_model(inv_model, loader, discrete=discrete, epochs=10, lr=1e-3, writer=writer)

    if args.demo_file is None:
        demo_dir = os.path.join("..", "data", "demonstrations")
        demo_filename = f"{args.env}_demonstrations.npy"
        demo_path = os.path.join(demo_dir, demo_filename)
    else:
        demo_path = args.demo_file

    print(f"Loading demonstrations from {demo_path} ...")
    demo_data = np.load(demo_path, allow_pickle=True)
    # Handle numpy array of objects
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

    # Extract observations:
    if isinstance(demo_obj, list) and hasattr(demo_obj[0], 'obs'):
        obs_list = [np.array(traj.obs).astype(np.float32) for traj in demo_obj]
        if len(obs_list) == 1:
            demo_traj = obs_list[0]
        else:
            demo_traj = np.concatenate(obs_list, axis=0)
    elif hasattr(demo_obj, 'obs'):
        if isinstance(demo_obj.obs, list):
            demo_traj = np.stack([np.array(s).astype(np.float32) for s in demo_obj.obs])
        else:
            demo_traj = np.array(demo_obj.obs).astype(np.float32)
    else:
        print("No 'obs' attribute found in demonstration. Check the content:")
        print(demo_obj)
        demo_traj = np.array(demo_obj).astype(np.float32)

    print(f"Demonstration has {demo_traj.shape[0]} states.")

    print("Inferring expert actions using the inverse dynamics model...")
    states_bc, inferred_actions = infer_expert_actions(inv_model, demo_traj, discrete=discrete)
    print(f"Inferred actions for {states_bc.shape[0]} state pairs.")

    policy_net = PolicyNetwork(obs_dim, action_dim, discrete=discrete)
    print("Training the policy (Behavioral Cloning) with inferred actions...")
    policy_net = train_policy(policy_net, states_bc, inferred_actions,
                              discrete=discrete, epochs=20, lr=1e-3, batch_size=64, writer=writer)

    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    model_name = f"bco_{args.env}.pt"
    save_path = os.path.join(models_dir, model_name)
    torch.save(policy_net.state_dict(), save_path)
    print(f"BCO model saved at {save_path}")

    writer.close()
    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_bco.py --env halfcheetah --pre_interactions 2000 --seed 42")
    main()
