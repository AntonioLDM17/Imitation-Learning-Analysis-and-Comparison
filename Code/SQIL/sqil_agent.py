import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor Network (Gaussian Policy) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Limits the action to [-1, 1]
        # Adjust log_prob considering the tanh transformation
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

# --- Critic Network (Double Q Network) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # First Q network
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Second Q network
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        q1 = self.q1_net(xu)
        q2 = self.q2_net(xu)
        return q1, q2

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# --- SQIL Agent based on SAC ---
class SQILAgent:
    def __init__(self, state_dim, action_dim, action_range=1.0, 
                 actor_hidden=256, critic_hidden=256, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4,
                 gamma=0.99, tau=0.005, target_entropy=None,
                 demo_buffer_capacity=100000, agent_buffer_capacity=100000,
                 batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Actor network and its optimizer
        self.actor = Actor(state_dim, action_dim, hidden_dim=actor_hidden).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network and its optimizer
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Target critic network
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Automatic entropy coefficient (alpha)
        if target_entropy is None:
            self.target_entropy = -action_dim  # Suggested value in SAC
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Replay buffers for demonstrations and agent transitions
        self.demo_buffer = ReplayBuffer(demo_buffer_capacity)
        self.agent_buffer = ReplayBuffer(agent_buffer_capacity)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0] * self.action_range
    
    # Store demonstration transition (reward forced to 1)
    def store_demo(self, state, action, next_state, done):
        self.demo_buffer.push(state, action, 1.0, next_state, done)
        
    # Store agent transition (reward forced to 0)
    def store_agent(self, state, action, next_state, done):
        self.agent_buffer.push(state, action, 0.0, next_state, done)
    
    def update(self):
        # Ensure both buffers have enough samples
        if len(self.demo_buffer) < self.batch_size // 2 or len(self.agent_buffer) < self.batch_size // 2:
            return
        
        demo_batch_size = self.batch_size // 2
        agent_batch_size = self.batch_size - demo_batch_size
        
        state_d, action_d, reward_d, next_state_d, done_d = self.demo_buffer.sample(demo_batch_size)
        state_a, action_a, reward_a, next_state_a, done_a = self.agent_buffer.sample(agent_batch_size)
        
        # Combine both batches
        state = np.concatenate([state_d, state_a], axis=0)
        action = np.concatenate([action_d, action_a], axis=0)
        reward = np.concatenate([reward_d, reward_a], axis=0).reshape(-1, 1)
        next_state = np.concatenate([next_state_d, next_state_a], axis=0)
        done = np.concatenate([done_d, done_a], axis=0).reshape(-1, 1)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Critic update (soft Bellman equation)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q
            
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        action_new, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Automatic entropy coefficient (alpha) update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update of the target critic network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
