import os
import sys
import types
import argparse

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained expert model"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole", "halfcheetah"],
        default="halfcheetah",
        help="Environment name: 'cartpole' or 'halfcheetah'"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["ppo", "trpo", "sac"],
        default="sac",
        help="Policy algorithm used to train the expert: ppo, trpo, or sac"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total timesteps for training the expert"
    )
    args = parser.parse_args()

    # Determine environment name based on the argument
    if args.env == "cartpole":
        env_name = "CartPole-v1"
    elif args.env == "halfcheetah":
        env_name = "HalfCheetah-v4"
    else:
        raise ValueError("The --env parameter must be 'cartpole' or 'halfcheetah'.")

    # Build expert model path using the centralized folder "data/experts"
    # Do not include the ".zip" extension (SB3.load adds it automatically)
    model_path = os.path.join("data", "experts", f"{args.env}_expert_{args.policy}_{args.timesteps}")

    # Create the environment
    env = gym.make(env_name)

    # Load the expert model with the appropriate algorithm
    if args.policy.lower() == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
    elif args.policy.lower() == "trpo":
        from sb3_contrib import TRPO
        model = TRPO.load(model_path, env=env)
    elif args.policy.lower() == "sac":
        from stable_baselines3 import SAC
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError("Unsupported policy algorithm.")

    # Evaluate the expert model
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=False
    )

    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Standard deviation: {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_expert.py --env halfcheetah --policy sac --n_episodes 100")
    main()
