# SQIL HalfCheetah con correcciones, normalización, targets y logging detallado
import sys, types, os, copy
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# Dummy mujoco_py
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gym
import d4rl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import trange
import matplotlib.pyplot as plt

# Parámetros
GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 100000
LEARNING_RATE = 3e-4
ALPHA = 0.2
ENV_NAME = 'bullet-halfcheetah-expert-v0'
TAU = 0.005  # Para soft updates
CLIP_VALUE = 1000.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
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
        y_t = torch.tanh(x_t)  # Salida en [-1, 1]
        return y_t, normal.log_prob(x_t).sum(dim=-1)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = self.sample(state)
        return action.squeeze().detach().cpu().numpy()

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*transitions)
        return map(lambda x: torch.tensor(x, dtype=torch.float32, device=device), (s, a, r, s_, d))

    def __len__(self):
        return len(self.buffer)

def load_d4rl_expert_demos(env, limit=100000):
    dataset = env.get_dataset()
    obs_shape = np.array(dataset['observations'][0]).shape
    demos = []
    for i in range(min(limit, len(dataset['observations']))):
        state = np.array(dataset['observations'][i]).reshape(obs_shape)
        action = dataset['actions'][i]
        next_state = np.array(dataset['next_observations'][i]).reshape(obs_shape)
        done = float(dataset['terminals'][i])
        demos.append((state, action, 1.0, next_state, done))
    return demos

def evaluate_policy(env, policy, episodes=5):
    rewards = []
    demo_shape = env.observation_space.shape
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        state = np.array(state).reshape(demo_shape)
        done = False
        total_reward = 0
        while not done:
            action = policy.act(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            step = env.step(action)
            if len(step) == 5:
                next_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step
            state = np.array(next_state).reshape(demo_shape)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def compute_expert_return_from_dataset(env):
    dataset = env.get_dataset()
    rewards, terminals = dataset['rewards'], dataset['terminals']
    timeouts = dataset.get('timeouts', np.zeros_like(terminals))
    episode_returns, ep_return = [], 0.0
    for r, done, timeout in zip(rewards, terminals, timeouts):
        ep_return += r
        if done or timeout:
            episode_returns.append(ep_return)
            ep_return = 0.0
    return np.mean(episode_returns) if episode_returns else float('nan')

def safe_reset(env, expected_shape):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs)
    if obs.shape != expected_shape:
        raise ValueError(f"Reset returned observation with shape {obs.shape}, expected {expected_shape}")
    return obs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def train_sqil(env, expert_demos, num_steps=100000):
    demo_shape = expert_demos[0][0].shape
    state_dim = demo_shape[0]
    action_dim = env.action_space.shape[0]

    q_net1 = QNetwork(state_dim, action_dim).to(device)
    q_net2 = QNetwork(state_dim, action_dim).to(device)
    q_net1_target = copy.deepcopy(q_net1).to(device)
    q_net2_target = copy.deepcopy(q_net2).to(device)
    q_optimizer1 = optim.Adam(q_net1.parameters(), lr=LEARNING_RATE)
    q_optimizer2 = optim.Adam(q_net2.parameters(), lr=LEARNING_RATE)

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    expert_buffer = ReplayBuffer()
    online_buffer = ReplayBuffer()
    for demo in expert_demos:
        expert_buffer.push(*demo)

    state = safe_reset(env, demo_shape)
    rewards_log, q1_losses, q2_losses, policy_losses, reward_means = [], [], [], [], []

    for step in trange(num_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = policy.sample(state_tensor)
            action = action.squeeze().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, _, done, _ = step_out

        next_state = np.array(next_state).reshape(demo_shape)
        online_buffer.push(state, action, 0.0, next_state, float(done))
        state = safe_reset(env, demo_shape) if done else next_state

        if len(online_buffer) < BATCH_SIZE // 2: continue

        exp_batch = expert_buffer.sample(BATCH_SIZE // 2)
        onl_batch = online_buffer.sample(BATCH_SIZE // 2)
        s_e, a_e, r_e, s_e_, d_e = exp_batch
        s_o, a_o, r_o, s_o_, d_o = onl_batch

        s = torch.cat([s_e, s_o], dim=0)
        a = torch.cat([a_e, a_o], dim=0)
        r = torch.cat([r_e, r_o], dim=0)
        s_ = torch.cat([s_e_, s_o_], dim=0)
        d = torch.cat([d_e, d_o], dim=0)

        with torch.no_grad():
            next_a, logp = policy.sample(s_)
            q1_target = q_net1_target(s_, next_a)
            q2_target = q_net2_target(s_, next_a)
            q_target = torch.min(q1_target, q2_target) - ALPHA * logp.unsqueeze(1)
            y = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * GAMMA * q_target
            y = torch.clamp(y, -CLIP_VALUE, CLIP_VALUE)

        q1_loss = nn.MSELoss()(q_net1(s, a), y)
        q2_loss = nn.MSELoss()(q_net2(s, a), y)
        q_optimizer1.zero_grad(); q1_loss.backward(); q_optimizer1.step()
        q_optimizer2.zero_grad(); q2_loss.backward(); q_optimizer2.step()

        sampled_a, logp = policy.sample(s)
        q1_val, q2_val = q_net1(s, sampled_a), q_net2(s, sampled_a)
        q_val = torch.min(q1_val, q2_val)
        policy_loss = (ALPHA * logp.unsqueeze(1) - q_val).mean()
        policy_optimizer.zero_grad(); policy_loss.backward(); policy_optimizer.step()

        soft_update(q_net1_target, q_net1, TAU)
        soft_update(q_net2_target, q_net2, TAU)

        if step % 1000 == 0:
            avg_reward, _ = evaluate_policy(env, policy)
            avg_batch_reward = r.mean().item()
            rewards_log.append(avg_reward)
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())
            policy_losses.append(policy_loss.item())
            reward_means.append(avg_batch_reward)
            print(f"Step {step}, Avg Reward: {avg_reward:.2f}, Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Avg Batch Reward: {avg_batch_reward:.4f}")

    plt.figure()
    plt.plot(np.arange(0, len(rewards_log)) * 1000, rewards_log, label="SQIL Return")
    plt.xlabel("Steps"); plt.ylabel("Return"); plt.title("SQIL Evaluation Return"); plt.legend(); plt.grid()

    plt.figure()
    plt.plot(np.arange(0, len(q1_losses)) * 1000, q1_losses, label="Q1 Loss")
    plt.plot(np.arange(0, len(q2_losses)) * 1000, q2_losses, label="Q2 Loss")
    plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("Q-Network Losses"); plt.legend(); plt.grid()

    plt.figure()
    plt.plot(np.arange(0, len(policy_losses)) * 1000, policy_losses, label="Policy Loss")
    plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("Policy Loss"); plt.legend(); plt.grid()

    plt.figure()
    plt.plot(np.arange(0, len(reward_means)) * 1000, reward_means, label="Avg Batch Reward")
    plt.xlabel("Steps"); plt.ylabel("Reward"); plt.title("Recompensa Media por Batch"); plt.legend(); plt.grid()

    plt.show()
    return policy, rewards_log

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    demos = load_d4rl_expert_demos(env)
    print(f"Cargadas {len(demos)} transiciones de demostración experta desde D4RL.")
    sqil_policy, reward_history = train_sqil(env, demos, num_steps=100000)
    torch.save(sqil_policy.state_dict(), "sqil_policy.pt")
    expert_return = compute_expert_return_from_dataset(env)
    sqil_return, _ = evaluate_policy(env, sqil_policy)
    print(f"Expert Policy Estimated Return (D4RL): {expert_return:.2f}")
    print(f"SQIL Policy Average Return: {sqil_return:.2f}")
