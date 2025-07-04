import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import sys, types
from sqil_agent import SQILAgent

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def select_action_discrete(agent, state, action_dim, evaluate=False):
    """
    For discrete action spaces, use the actor network's shared layers to produce logits,
    and select the action with the highest probability.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.actor.net[0].weight.device)
    features = agent.actor.net(state_tensor)
    logits = agent.actor.mean_linear(features)  # Use mean_linear as logits
    probabilities = torch.softmax(logits, dim=-1)
    action_index = torch.argmax(probabilities, dim=-1).item()
    return action_index

def main():
    parser = argparse.ArgumentParser(description="Evaluate a SQIL trained model")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Gym environment name (e.g., HalfCheetah-v4 or CartPole-v1)")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory where the models are stored (e.g., models/halfcheetah or models/cartpole)")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]

    # Check if the action space is continuous or discrete
    if isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    elif isinstance(env.action_space, gym.spaces.Discrete):
        discrete = True
        action_dim = env.action_space.n
        action_range = 1.0  # Not used in discrete mode
    else:
        raise NotImplementedError("This evaluation supports only continuous or discrete environments.")

    # Determine the model subdirectory based on the environment
    if args.env == "HalfCheetah-v4":
        model_subdir = "halfcheetah"
    elif args.env == "CartPole-v1":
        model_subdir = "cartpole"
    else:
        raise ValueError("Unsupported environment. Use 'HalfCheetah-v4' or 'CartPole-v1'.")

    # Instantiate the SQIL agent (other parameters can be adjusted as needed)
    agent = SQILAgent(state_dim, action_dim, action_range=action_range, batch_size=256)

    # Load trained model weights from the appropriate subdirectory
    actor_path = os.path.join(args.model_dir, model_subdir, "sqil_actor.pth")
    critic_path = os.path.join(args.model_dir, model_subdir, "sqil_critic.pth")
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        raise FileNotFoundError("Model files not found in the specified directory.")
    
    agent.actor.load_state_dict(torch.load(actor_path, map_location=torch.device("cpu")))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=torch.device("cpu")))
    print(f"Models loaded from {os.path.join(args.model_dir, model_subdir)}")

    # Evaluation: run the agent for a number of episodes and measure accumulated reward
    rewards = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)  # Vary seed per episode for more variability
        done = False
        ep_reward = 0
        while not done:
            if not discrete:
                action = agent.select_action(state, evaluate=True)
            else:
                action = select_action_discrete(agent, state, action_dim, evaluate=True)
            # For discrete environments, pass the integer action to env.step(); for continuous, pass the action vector.
            next_state, reward, done, truncated, _ = env.step(action if discrete else action)
            ep_reward += reward
            state = next_state
            if done or truncated:
                break
        rewards.append(ep_reward)
        print(f"Episode {ep}: Accumulated Reward = {ep_reward:.2f}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Standard Deviation: {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python evaluate_sqil.py --env HalfCheetah-v4 --seed 44 --episodes 50 --model_dir models")
    print("python evaluate_sqil.py --env CartPole-v1 --seed 44 --episodes 50 --model_dir models")
    main()
