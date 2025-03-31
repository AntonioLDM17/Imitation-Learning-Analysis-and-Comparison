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
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ----------------------------------------------------------------
# Actor definitions for discrete vs. continuous
# ----------------------------------------------------------------
class DiscreteActor(nn.Module):
    """
    Outputs logits of shape [batch_size, n_actions].
    We'll sample from Categorical for evaluation.
    """
    def __init__(self, obs_dim, n_actions):
        super(DiscreteActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)  # raw logits
        )

    def forward(self, x):
        return self.net(x)


class ContinuousActor(nn.Module):
    """
    Outputs a continuous action vector of shape [batch_size, act_dim].
    """
    def __init__(self, obs_dim, act_dim):
        super(ContinuousActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------
# Evaluate the actor policy
# ----------------------------------------------------------------
def evaluate_policy(actor, env, n_episodes=10, device="cpu", is_discrete=False):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if is_discrete:
                    # Discrete: sample from logits
                    logits = actor(obs_tensor)
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())
                else:
                    # Continuous: output a vector
                    action = actor(obs_tensor).cpu().numpy()[0]

            if is_discrete:
                step_action = action  # integer
            else:
                step_action = np.array([action])  # shape (act_dim,)

            result = env.step(step_action)
            if len(result) == 5:
                next_obs, reward_val, terminated, truncated, _ = result
                done = terminated or truncated
            elif len(result) == 4:
                next_obs, reward_val, done, info = result
            else:
                raise ValueError("Unexpected env.step() return format")

            obs = next_obs
            ep_reward += float(reward_val)
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DAC actor model.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Environment name (e.g., HalfCheetah-v4 or CartPole-v1)")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join("models", "dac_halfcheetah.pt"),
                        help="Path to the saved DAC actor model (e.g., models/dac_halfcheetah.pt)")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine if the environment is discrete or continuous
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        n_actions = env.action_space.n
        obs_dim = env.observation_space.shape[0]
        # Use the DiscreteActor
        actor = DiscreteActor(obs_dim, n_actions).to(device)
    else:
        is_discrete = False
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # Use the ContinuousActor
        actor = ContinuousActor(obs_dim, act_dim).to(device)

    # Load the model parameters
    actor.load_state_dict(torch.load(args.model_path, map_location=device))
    actor.eval()

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        actor, env, n_episodes=args.n_episodes, device=device, is_discrete=is_discrete
    )
    print(f"Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Standard deviation: {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_dac.py --env HalfCheetah-v4 --model_path models/dac_halfcheetah.pt --n_episodes 100")
    main()
