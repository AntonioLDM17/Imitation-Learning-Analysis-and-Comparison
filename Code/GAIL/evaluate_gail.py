import os, sys, types, argparse

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
# You can add minimal attributes if needed; for now we leave them empty.
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gymnasium as gym
import numpy as np
from sb3_contrib import TRPO  # We use TRPO because it is the algorithm used in our experiments
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GAIL model using TRPO.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Name of the environment (e.g., 'HalfCheetah-v4' or 'CartPole-v1').")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join("models", "gail_halfcheetah.zip"),
                        help="Path to the trained GAIL model (e.g., 'models/gail_halfcheetah.zip').")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env)

    # Load the trained model using TRPO
    model = TRPO.load(args.model_path)

    # Evaluate the model
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
    print("Example execution:")
    print("python evaluate_gail.py --env HalfCheetah-v4 --model_path models/gail_halfcheetah.zip --n_episodes 100")
    main()
