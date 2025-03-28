import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import trange
import matplotlib.pyplot as plt

# SQIL Parameters
GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 100000
LEARNING_RATE = 3e-4
ALPHA = 0.2

# Environments
ENV_NAME = 'HalfCheetah-v4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

# Policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        return y_t, normal.log_prob(x_t).sum(dim=-1)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = self.sample(state)
        return action.squeeze().detach().cpu().numpy()

# Replay buffer
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        transitions = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s_, d = zip(*transitions)
        return map(lambda x: torch.tensor(x, dtype=torch.float32, device=device), (s, a, r, s_, d))

    def __len__(self):
        return len(self.buffer)

# Compatibilidad con distintas versiones de Gym
def safe_reset(env):
    reset_out = env.reset()
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out

def safe_step(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
    else:
        next_state, reward, done, info = step_result
        terminated = done
        truncated = False
    return next_state, reward, terminated, truncated, info

# Expert demos
def load_expert_demos(env, policy_fn, num_episodes=10):
    demos = []
    for _ in range(num_episodes):
        state = safe_reset(env)
        done = False
        while not done:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = safe_step(env, action)
            done = terminated or truncated
            demos.append((state, action, 1.0, next_state, float(done)))
            state = next_state
    return demos

# Evaluate policy
def evaluate_policy(env, policy, episodes=5):
    rewards = []
    for _ in range(episodes):
        state = safe_reset(env)
        done = False
        total_reward = 0
        while not done:
            action = policy.act(state)
            next_state, reward, terminated, truncated, _ = safe_step(env, action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# Save and load utilities
def save_policy(policy, path="sqil_policy.pt"):
    torch.save(policy.state_dict(), path)

def load_policy(policy_class, state_dim, action_dim, path="sqil_policy.pt"):
    policy = policy_class(state_dim, action_dim)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.to(device)
    return policy

def evaluate_saved_policy(policy_path="sqil_policy.pt", env_name="HalfCheetah-v4", episodes=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = load_policy(PolicyNetwork, state_dim, action_dim, path=policy_path)
    avg_reward, std_reward = evaluate_policy(env, policy, episodes=episodes)
    print(f"[Evaluation] Average return over {episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")

def run_saved_policy(policy_path="sqil_policy.pt", env_name="HalfCheetah-v4", render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = load_policy(PolicyNetwork, state_dim, action_dim, path=policy_path)

    state = safe_reset(env)
    done = False
    total_reward = 0
    while not done:
        action = policy.act(state)
        state, reward, terminated, truncated, _ = safe_step(env, action)
        done = terminated or truncated
        total_reward += reward
        if render:
            env.render()
    env.close()
    print(f"[Run] Episode reward: {total_reward:.2f}")

# Train SQIL
def train_sqil(env, expert_demos, num_steps=100000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    q_net1 = QNetwork(state_dim, action_dim).to(device)
    q_net2 = QNetwork(state_dim, action_dim).to(device)
    q_optimizer1 = optim.Adam(q_net1.parameters(), lr=LEARNING_RATE)
    q_optimizer2 = optim.Adam(q_net2.parameters(), lr=LEARNING_RATE)

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    buffer = ReplayBuffer()
    for demo in expert_demos:
        buffer.push(*demo)

    state = safe_reset(env)
    rewards_log = []

    for step in trange(num_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = policy.sample(state_tensor)
            action = action.squeeze().cpu().numpy()

        next_state, _, terminated, truncated, _ = safe_step(env, action)
        done = terminated or truncated
        buffer.push(state, action, 0.0, next_state, float(done))
        state = next_state if not done else safe_reset(env)

        if len(buffer) < BATCH_SIZE:
            continue

        s, a, r, s_, d = buffer.sample()

        with torch.no_grad():
            next_a, logp = policy.sample(s_)
            q1_target = q_net1(s_, next_a)
            q2_target = q_net2(s_, next_a)
            q_target = torch.min(q1_target, q2_target) - ALPHA * logp.unsqueeze(1)
            y = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * GAMMA * q_target

        q1_loss = nn.MSELoss()(q_net1(s, a), y)
        q2_loss = nn.MSELoss()(q_net2(s, a), y)

        q_optimizer1.zero_grad()
        q1_loss.backward()
        q_optimizer1.step()

        q_optimizer2.zero_grad()
        q2_loss.backward()
        q_optimizer2.step()

        sampled_a, logp = policy.sample(s)
        q1_val = q_net1(s, sampled_a)
        q2_val = q_net2(s, sampled_a)
        q_val = torch.min(q1_val, q2_val)
        policy_loss = (ALPHA * logp.unsqueeze(1) - q_val).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if step % 1000 == 0:
            avg_reward, _ = evaluate_policy(env, policy)
            rewards_log.append(avg_reward)
            print(f"Step {step}, Avg Reward: {avg_reward:.2f}")

    return policy, rewards_log

# Main script
if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    class ExpertPolicy:
        def __init__(self, env):
            self.env = env
        def act(self, state):
            return self.env.action_space.sample()

    expert = ExpertPolicy(env)
    demos = load_expert_demos(env, expert.act, num_episodes=10)
    sqil_policy, reward_history = train_sqil(env, demos, num_steps=50000)

    # Save policy
    save_policy(sqil_policy, "sqil_policy.pt")

    # Compare expert and SQIL
    expert_return, _ = evaluate_policy(env, expert)
    sqil_return, _ = evaluate_policy(env, sqil_policy)

    print(f"Expert Policy Average Return: {expert_return:.2f}")
    print(f"SQIL Policy Average Return: {sqil_return:.2f}")

    # Plot
    plt.plot(np.arange(0, len(reward_history)) * 1000, reward_history, label="SQIL")
    plt.axhline(expert_return, color='r', linestyle='--', label="Expert")
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.title("SQIL vs Expert Performance")
    plt.legend()
    plt.grid()
    plt.show()
