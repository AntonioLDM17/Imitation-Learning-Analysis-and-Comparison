import os
import argparse, types, sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from bco import PolicyNetwork

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def evaluate_policy(policy_net, env, discrete=True, n_episodes=10):
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = policy_net(s)
                if discrete:
                    action = torch.argmax(output, dim=1).item()
                else:
                    action = output.squeeze().cpu().numpy()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BCO policy.")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' o 'halfcheetah'")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Ruta al modelo entrenado (pt) de BCO")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Número de episodios para evaluación")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    args = parser.parse_args()

    if args.env == "cartpole":
        env_name = "CartPole-v1"
        discrete = True
    else:
        env_name = "HalfCheetah-v4"
        discrete = False

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    policy_net = PolicyNetwork(obs_dim, action_dim, discrete=discrete)
    if args.model_path is None:
        model_name = f"bco_{args.env}.pt"
        args.model_path = os.path.join("models", model_name)
    policy_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    policy_net.eval()

    mean_reward, std_reward = evaluate_policy(policy_net, env, discrete=discrete, n_episodes=args.n_episodes)
    print(f" Evaluation over {args.n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f" Standard deviation: {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_bco.py --env cartpole --model_path models/bco_cartpole.pt --n_episodes 20 --seed 44")
    main()
