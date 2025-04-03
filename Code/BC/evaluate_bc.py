import os
import sys
import types
import argparse
import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC

# Create dummy modules for mujoco_py (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

class PolicyWrapper:
    """
    Minimal wrapper providing a .predict() method so that evaluate_policy()
    can interact with the loaded BC policy.
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
        return actions_np, None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a BC policy saved as a zip file or .pt file from train_bc.py."
    )
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gym environment ID (e.g., 'CartPole-v1' or 'HalfCheetah-v4').")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join("models", "bc_cartpole.pt"),
                        help="Path to the saved BC model file (zip or .pt) from train_bc.py.")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of evaluation episodes.")
    args = parser.parse_args()

    # Create evaluation environment
    env = gym.make(args.env)

    # Create a dummy BC instance to extract the default policy class.
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
        if model_path.endswith(".pt"):
            # Assume file contains only state_dict.
            ckpt = th.load(model_path, map_location="cpu")
            # Provide necessary constructor parameters (using env info)
            constructor_params = {
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "lr_schedule": lambda _: 1e-3,  # dummy constant schedule
            }
            policy = PolicyClass(**constructor_params)
            policy.load_state_dict(ckpt)
        else:
            # Otherwise assume the file is a zip saved with .save()
            policy = PolicyClass.load(model_path)
    except Exception as e:
        print("Error loading policy:", e)
        sys.exit(1)

    # Wrap the loaded policy to provide a .predict() interface.
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
    print("python evaluate_bc.py --env CartPole-v1 --model_path models/bc_cartpole.pt --n_episodes 10")
    print("python evaluate_bc.py --env HalfCheetah-v4 --model_path models/bc_halfcheetah.pt --n_episodes 20")
    main()
