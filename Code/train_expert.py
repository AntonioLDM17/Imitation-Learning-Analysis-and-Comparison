import os
import sys
import types
import argparse

# Create dummy modules for "mujoco_py" (useful for HalfCheetah)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper

def main():
    parser = argparse.ArgumentParser(
        description="Train an expert using a selectable policy (ppo, trpo, sac)"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole", "halfcheetah", "mountaincar", "acrobot"],
        default="halfcheetah",
        help="Environment to use: 'cartpole' or 'halfcheetah or 'mountaincar' or 'acrobot'"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["ppo", "trpo", "sac"],
        default="ppo",
        help="Policy algorithm to use: ppo, trpo, or sac"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    SEED = args.seed

    # Determine environment name based on the argument
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    elif args.env == "halfcheetah":
        ENV_NAME = "HalfCheetah-v4"
    elif args.env == "mountaincar":
        ENV_NAME = "MountainCar-v0"
    elif args.env == "acrobot":
        ENV_NAME = "Acrobot-v1"
    else:
        raise ValueError("The --env parameter must be 'cartpole' or 'halfcheetah'.")

    # Ensure SAC is only used for continuous action environments
    if args.policy == "sac" and args.env == "cartpole":
        raise ValueError("SAC is not compatible with discrete action spaces. Please select 'halfcheetah' or use PPO/TRPO for CartPole.")

    TOTAL_TIMESTEPS = args.timesteps

    # Define directories and names using the centralized "data/experts" folder
    EXPERT_MODEL_DIR = os.path.join("data", "experts")
    os.makedirs(EXPERT_MODEL_DIR, exist_ok=True)
    EXPERT_MODEL_NAME = f"{args.env}_expert_{args.policy}_{TOTAL_TIMESTEPS}"
    log_dir = os.path.join("logs", f"expert_{args.env}_{args.policy}_{TOTAL_TIMESTEPS}")

    # Create a vectorized environment with RolloutInfoWrapper
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
    )

    # Configure the logger for TensorBoard
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Select the algorithm based on the provided argument
    if args.policy == "ppo":
        from stable_baselines3 import PPO
        expert = PPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=log_dir)
    elif args.policy == "trpo":
        from sb3_contrib import TRPO
        expert = TRPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=log_dir)
    elif args.policy == "sac":
        from stable_baselines3 import SAC
        expert = SAC("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported policy algorithm.")

    expert.set_logger(new_logger)
    expert.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save the expert model to the centralized folder (the save function adds .zip automatically)
    save_path = os.path.join(EXPERT_MODEL_DIR, EXPERT_MODEL_NAME)
    expert.save(save_path)
    print(f"Expert model ({args.policy.upper()}) saved at {save_path}.zip")
    env.close()

if __name__ == "__main__":
    print("Usage example:")
    print("python train_expert.py --env halfcheetah --policy sac --timesteps 1000000 --seed 44")
    main()
