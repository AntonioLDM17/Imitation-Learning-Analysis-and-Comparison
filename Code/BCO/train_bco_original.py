""" 
Credits to:
@inproceedings{torabi2018bco,
  author = {Faraz Torabi and Garrett Warnell and Peter Stone}, 
  title = {{Behavioral Cloning from Observation}}, 
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)}, 
  year = {2018} 
}
Where the code is based on the original BCO implementation by Faraz Torabi.
"""

import os
import argparse, types, sys
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from bco import (
    set_seed,
    InverseDynamicsModel,
    PolicyNetwork,
    collect_exploration_data,
    create_dataloader,
    train_inverse_model,
    infer_expert_actions,
    train_policy,
    collect_policy_data
)

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def main():
    parser = argparse.ArgumentParser(description="Train a BCO model (or BCO(alpha)) based on the alpha value.")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' (discrete) or 'halfcheetah' (continuous)")
    parser.add_argument("--pre_interactions", type=int, default=2000,
                        help="Number of pre-demonstration interactions to train the inverse model")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Alpha value. If 0, runs BCO(0) (normal); if >0, runs iterative BCO(alpha)")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterative improvement iterations (used only if alpha > 0)")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="Path to demonstrations (npy) in data/demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo_episodes", type=int, default=50, help="Number of expert episodes for training (used only if demo_file is None)")
    # Hyperparameters for inverse model training
    parser.add_argument("--inv_epochs", type=int, default=10, help="Epochs for pre-demonstration inverse model training")
    parser.add_argument("--inv_lr", type=float, default=1e-3, help="Learning rate for inverse model training")
    # Hyperparameters for policy training (initial)
    parser.add_argument("--policy_epochs", type=int, default=20, help="Epochs for initial policy training")
    parser.add_argument("--policy_lr", type=float, default=1e-3, help="Learning rate for initial policy training")
    # Hyperparameters for iterative retraining (if alpha > 0)
    parser.add_argument("--iter_inv_epochs", type=int, default=5, help="Epochs for inverse model retraining in each iteration")
    parser.add_argument("--iter_policy_epochs", type=int, default=10, help="Epochs for policy retraining in each iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training both models")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.env == "cartpole":
        env_name = "CartPole-v1"
        discrete = True
    else:
        env_name = "HalfCheetah-v4"
        discrete = False

    # Setup TensorBoard logging
    log_dir = os.path.join("logs", f"bco_{args.env}_{args.alpha}_{args.demo_episodes}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    env = gym.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0]
    action_dim = act_space.n if discrete else act_space.shape[0]

    # Pre-demonstration phase: collect exploration data (I_pre)
    print("Collecting pre-demonstration exploration data...")
    s_pre, s_next_pre, a_pre = collect_exploration_data(env, args.pre_interactions)
    print(f"Collected {s_pre.shape[0]} pre-demonstration transitions.")

    loader = create_dataloader(s_pre, s_next_pre, a_pre, batch_size=args.batch_size)
    inv_model = InverseDynamicsModel(obs_dim, action_dim, discrete=discrete)
    print("Training inverse dynamics model on pre-demonstration data...")
    inv_model = train_inverse_model(inv_model, loader, discrete=discrete,
                                    epochs=args.inv_epochs, lr=args.inv_lr, writer=writer)

    # Load demonstration data
    if args.demo_file is None:
        demo_dir = os.path.join("..", "data", "demonstrations", str(args.demo_episodes))
        demo_filename = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
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

    # Extract demonstration observations
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

    if args.alpha == 0.0:
        # BCO(0): Non-iterative (normal) BCO
        print("Running BCO(0) (non-iterative behavioral cloning from observation).")
        print("Inferring expert actions using the inverse dynamics model...")
        states_bc, inferred_actions = infer_expert_actions(inv_model, demo_traj, discrete=discrete)
        print(f"Inferred actions for {states_bc.shape[0]} state pairs.")

        policy_net = PolicyNetwork(obs_dim, action_dim, discrete=discrete)
        print("Training policy (Behavioral Cloning) with inferred actions...")
        policy_net = train_policy(policy_net, states_bc, inferred_actions,
                                  discrete=discrete, epochs=args.policy_epochs, lr=args.policy_lr,
                                  batch_size=args.batch_size, writer=writer)
    else:
        # BCO(alpha): Iterative improvement
        print("Running BCO(alpha) iterative improvement.")
        # Initial inference and policy training (as in BCO(0))
        print("Inferring expert actions using the inverse dynamics model...")
        states_bc, inferred_actions = infer_expert_actions(inv_model, demo_traj, discrete=discrete)
        print(f"Inferred actions for {states_bc.shape[0]} state pairs.")

        policy_net = PolicyNetwork(obs_dim, action_dim, discrete=discrete)
        print("Initial training of policy (Behavioral Cloning) with inferred actions...")
        policy_net = train_policy(policy_net, states_bc, inferred_actions,
                                  discrete=discrete, epochs=args.policy_epochs, lr=args.policy_lr,
                                  batch_size=args.batch_size, writer=writer)

        # Calculate number of post-demonstration interactions per iteration
        post_interactions = int(args.alpha * args.pre_interactions)
        print(f"Starting iterative improvement: {args.num_iterations} iterations, each with {post_interactions} post-demonstration interactions.")
        for itr in range(args.num_iterations):
            print(f"--- Iteration {itr+1}/{args.num_iterations} ---")
            # Collect post-demonstration data using current policy
            s_post, s_next_post, a_post = collect_policy_data(policy_net, env, post_interactions, discrete)
            print(f"Collected {s_post.shape[0]} post-demonstration transitions.")
            # Combine pre-demonstration and post-demonstration data
            s_combined = np.concatenate([s_pre, s_post], axis=0)
            s_next_combined = np.concatenate([s_next_pre, s_next_post], axis=0)
            a_combined = np.concatenate([a_pre, a_post], axis=0)
            combined_loader = create_dataloader(s_combined, s_next_combined, a_combined, batch_size=args.batch_size)
            # Update inverse model with combined data
            print("Re-training inverse dynamics model on combined data...")
            inv_model = train_inverse_model(inv_model, combined_loader, discrete=discrete,
                                            epochs=args.iter_inv_epochs, lr=args.inv_lr, writer=writer)
            # Re-infer expert actions using the updated inverse model
            print("Re-inferring expert actions using updated inverse dynamics model...")
            states_bc, inferred_actions = infer_expert_actions(inv_model, demo_traj, discrete=discrete)
            # Re-train policy with updated inferred actions
            print("Re-training policy (Behavioral Cloning) with updated inferred actions...")
            policy_net = train_policy(policy_net, states_bc, inferred_actions,
                                      discrete=discrete, epochs=args.iter_policy_epochs, lr=args.policy_lr,
                                      batch_size=args.batch_size, writer=writer)

    models_dir = os.path.join("models", f"bco_{args.env}_{args.alpha}_{args.demo_episodes}")
    os.makedirs(models_dir, exist_ok=True)
    if args.alpha == 0.0:
        model_name = f"bco_{args.env}.pt"
    else:
        model_name = f"bco_alpha_{args.env}.pt"
    save_path = os.path.join(models_dir, model_name)
    torch.save(policy_net.state_dict(), save_path)
    print(f"BCO model saved at {save_path}")

    writer.close()
    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_bco.py --env halfcheetah --pre_interactions 2000 --alpha 0.0 --seed 42 --demo_episodes 50  (for BCO(0))")
    print("python train_bco.py --env halfcheetah --pre_interactions 2000 --alpha 0.01 --num_iterations 5 --seed 42 --demo_episodes 50  (for BCO(alpha))")
    main()
