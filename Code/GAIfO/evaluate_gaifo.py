import os, sys, types, argparse

# Create dummy modules for "mujoco_py" to avoid compiling its extensions
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO

def main():
    parser = argparse.ArgumentParser(description="Evaluate a GAIfO model trained with TRPO.")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained GAIfO model (.zip file)")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    args = parser.parse_args()

    # Select the environment based on the provided argument
    if args.env == "cartpole":
        env_name = "CartPole-v1"
    else:
        env_name = "HalfCheetah-v4"

    env = gym.make(env_name)
    env.reset(seed=args.seed)

    # If no model path is provided, use a default filename in the models folder
    if args.model_path is None:
        model_filename = f"gaifo_{args.env}.zip"
        args.model_path = os.path.join("models", model_filename)

    # Load the trained GAIfO model (using TRPO)
    model = TRPO.load(args.model_path)

    # Evaluate the model over the specified number of episodes
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=args.n_episodes, 
        deterministic=True
    )

    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Standard deviation: {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_gaifo.py --env halfcheetah --model_path models/gaifo_halfcheetah.zip --n_episodes 20 --seed 44")
    main()
