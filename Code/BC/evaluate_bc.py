print("NOT WORKING")
print("This code is not working. It is a placeholder for the original code.")


import os
import sys
import types
import argparse
import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

class PolicyWrapper:
    """
    A minimal wrapper that provides a .predict(obs, ...) method,
    allowing evaluate_policy() to interact with an imitation BC policy.
    """
    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=True):
        # Ensure obs is 2D
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_th = th.as_tensor(obs, dtype=th.float32)
        with th.no_grad():
            output = self.policy.forward(obs_th, deterministic=deterministic)
            if isinstance(output, (tuple, list)):
                actions_th = output[0]
            else:
                actions_th = output
        actions_np = actions_th.cpu().numpy()
        if actions_np.shape[0] == 1:
            actions_np = actions_np[0]
        return actions_np, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a BC model saved as a zip file.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Gym environment ID (e.g., 'HalfCheetah-v4' or 'CartPole-v1').")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join("models", "bc_halfcheetah.zip"),
                        help="Path to the saved BC model zip file from train_bc.py.")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of evaluation episodes.")
    args = parser.parse_args()

    # Create evaluation environment
    env = gym.make(args.env)

    # Create a dummy BC instance (with a single dummy demonstration) to extract the default policy class.
    dummy_demo = [{
        "obs": env.observation_space.sample(),
        "acts": env.action_space.sample(),
        "dones": False,
        "next_obs": env.observation_space.sample(),
        "infos": {}
    }]
    dummy_bc = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=dummy_demo,
        rng=np.random.default_rng(42)
    )
    PolicyClass = dummy_bc.policy.__class__

    model_path = os.path.abspath(args.model_path)
    print("Loading BC policy from:", model_path)
    try:
        # Load the policy using the class's load() method without an env argument.
        # Provide a dummy lr_schedule via constructor_params if needed.
        policy = PolicyClass.load(model_path)
    except Exception as e:
        print("Error loading policy:", e)
        sys.exit(1)

    # If the loaded policy does not have set_env, we skip it.
    if hasattr(policy, "set_env"):
        policy.set_env(env)

    # Wrap the loaded policy so that it exposes a .predict() method.
    wrapped_policy = PolicyWrapper(policy)

    mean_reward, std_reward = evaluate_policy(
        wrapped_policy, env, n_eval_episodes=args.n_episodes, deterministic=True
    )
    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Std reward:  {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_bc.py --env HalfCheetah-v4 --model_path models/bc_halfcheetah.zip --n_episodes 10")
    main()
