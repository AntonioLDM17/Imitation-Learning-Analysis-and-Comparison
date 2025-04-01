""" 
ESTE CÓDIGO NO FUNCIONA.
"""
import os, types, sys
import argparse
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO  # BC policy was trained using PPO
# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BC model.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Environment name (e.g., 'HalfCheetah-v4' or 'CartPole-v1').")
    parser.add_argument("--model_path", type=str, default=os.path.join("models", "bc_halfcheetah.zip"),
                        help="Path to the trained BC model (e.g., 'models/bc_halfcheetah.zip').")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    args = parser.parse_args()

    env = gym.make(args.env)
    # Convert the model path to an absolute path and load the model.
    model_path = os.path.abspath(args.model_path)
    print("Loading model from:", model_path)
    model = PPO.load(model_path)
    
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.n_episodes, deterministic=True
    )
    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Standard deviation: {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_bc.py --env HalfCheetah-v4 --model_path models/bc_halfcheetah.zip --n_episodes 100")
    main()
