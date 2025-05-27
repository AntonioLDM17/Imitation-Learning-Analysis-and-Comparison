import os
import sys
import types
import math
import argparse
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Imitation BC imports
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env

# Evaluation
from stable_baselines3.common.evaluation import evaluate_policy

# Create dummy modules for "mujoco_py" to avoid compiling its extensions
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def main():
    parser = argparse.ArgumentParser(
        description="Train a Behavioral Cloning (BC) model with TensorBoard logging and step-based metrics."
    )
    parser.add_argument(
        "--env", choices=["cartpole", "halfcheetah"], default="cartpole",
        help="Environment: 'cartpole' or 'halfcheetah'"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total timesteps for BC training"
    )
    parser.add_argument(
        "--seed", type=int, default=44,
        help="Random seed"
    )
    parser.add_argument(
        "--demo_episodes",
        type=int,
        default=50,
        help="Number of expert episodes for training"
    )
    args = parser.parse_args()

    # Environment and seed
    ENV_NAME = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"
    SEED = args.seed

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_DIR = os.path.join(BASE_DIR, os.pardir, "data", "demonstrations", str(args.demo_episodes))
    DEMO_FILE = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODEL_DIR = os.path.join(BASE_DIR, "models", f"bc_{args.env}_{args.demo_episodes}_2M_SEED_{SEED}")
    MODEL_NAME = f"bc_{args.env}"
    LOG_DIR = os.path.join(BASE_DIR, "logs", f"bc_{args.env}_{args.demo_episodes}_2M_SEED_{SEED}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # TensorBoard writer for custom metrics
    writer = SummaryWriter(LOG_DIR)

    # Create vectorized environment
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
    )

    # Load and flatten expert demonstrations
    demo_path = os.path.join(DEMO_DIR, DEMO_FILE)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()
    transitions = rollout.flatten_trajectories(demonstrations)

    # Compute total demonstration steps (D) (8000 for HalfCheetah)
    try:
        D = transitions.obs.shape[0] 
    except Exception:
        D = sum(len(traj["obs"]) for traj in demonstrations)
    print(f"Total demonstration steps (D): {D}")
    EPOCHS = math.ceil(args.timesteps / D)
    print(f"Total epochs to train: {EPOCHS}")
    # Initialize Behavioral Cloning trainer without default logger
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(SEED),
        custom_logger=None,
    )

    # Pre-training evaluation
    pre_rewards, _ = evaluate_policy(
        bc_trainer.policy, env, 10, return_episode_rewards=True
    )
    mean_pre = float(np.mean(pre_rewards))
    print(f"Mean reward before training: {mean_pre}")
    writer.add_scalar('evaluation/pre_training_reward', mean_pre, 0)
    writer.add_scalar('evaluation/mean_reward', mean_pre, 0)
    writer.add_scalar('bc/mean_nll', 0, 0)
    writer.add_scalar('bc/epoch_duration_s', 0, 0)
    writer.flush()
    # Training loop: epoch-level with step mapping and computed NLL loss
    for epoch in tqdm(range(1, EPOCHS + 1), desc="BC epochs"):
        start = time.time()
        # Train one epoch
        bc_trainer.train(n_epochs=1)
        duration = time.time() - start

        # Compute NLL loss on full demonstration set
        device = next(bc_trainer.policy.parameters()).device
        obs = torch.tensor(transitions.obs, dtype=torch.float32).to(device)
        acts = torch.tensor(transitions.acts, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, _, log_prob = bc_trainer.policy.evaluate_actions(obs, acts)
            mean_nll = -log_prob.mean().item()

        # Evaluate policy
        eval_rewards, _ = evaluate_policy(
            bc_trainer.policy, env, 10, return_episode_rewards=True
        )
        mean_reward = float(np.mean(eval_rewards))

        # Map epoch to environment steps
        step = epoch * D

        # Log to TensorBoard at correct step
        writer.add_scalar('bc/mean_nll', mean_nll, step)
        writer.add_scalar('evaluation/mean_reward', mean_reward, step)
        writer.add_scalar('bc/epoch_duration_s', duration, step)
        writer.flush()

        # Console output
        print(f"Step {step}: NLL Loss={mean_nll:.4f}, Reward={mean_reward:.2f}, Time={duration:.2f}s")

    # Post-training evaluation
    final_rewards, _ = evaluate_policy(
        bc_trainer.policy, env, 10, return_episode_rewards=True
    )
    mean_final = float(np.mean(final_rewards))
    print(f"Mean reward after training: {mean_final}")
    writer.add_scalar('evaluation/post_training_reward', mean_final, EPOCHS * D)

    # Save model
    MODEL_NAME = MODEL_NAME+f"_{EPOCHS*D}.pt"
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    torch.save(bc_trainer.policy.state_dict(), model_path)
    print(f"Behavioral Cloning model saved at {model_path}")

    # Clean up
    writer.close()
    env.close()


if __name__ == "__main__":
    print("Example: python train_bc.py --env halfcheetah --timesteps 2000000 --seed 44 --demo_episodes 50")
    print("Monitor with: tensorboard --logdir logs")
    main()
        
