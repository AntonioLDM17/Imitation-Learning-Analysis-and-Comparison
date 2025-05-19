import os, sys, types, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create dummy modules for "mujoco_py" to avoid compiling its extensions
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AIRL PPO policy."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gym environment ID (e.g., 'CartPole-v1' or 'HalfCheetah-v4')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("models", "airl_cartpole.zip"),
        help="Path to the trained AIRL PPO model (e.g., 'models/airl_cartpole.zip')."
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes."
    )
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env)

    # Load the trained PPO model
    model = PPO.load(args.model_path)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        deterministic=True,
        return_episode_rewards=False
    )

    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Std reward: {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    print("Example execution:")
    print("python evaluate_airl.py --env HalfCheetah-v4 --model_path models/airl_halfcheetah.zip --n_episodes 100")
    print("python evaluate_airl.py --env CartPole-v1 --model_path models/airl_cartpole.zip --n_episodes 100")
    main()
