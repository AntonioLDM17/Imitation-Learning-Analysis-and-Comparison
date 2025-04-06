import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Detecta si se dispone de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Red Actor (Política Gaussian) ---
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
        x_t = normal.rsample()  # Truco de reparametrización
        action = torch.tanh(x_t)  # Limita la acción a [-1, 1]
        # Ajuste del log_prob considerando tanh
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

# --- Red Critic (Red Q doble) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Primera Q
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Segunda Q
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

# --- Agente SQIL basado en SAC ---
class SQILAgent:
    def __init__(self, state_dim, action_dim, action_range=1.0, 
                 actor_hidden=256, critic_hidden=256, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None,
                 demo_buffer_capacity=100000, agent_buffer_capacity=100000,
                 batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Red actor y optimizador
        self.actor = Actor(state_dim, action_dim, hidden_dim=actor_hidden).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Red critic y optimizador
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Red target critic
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Coeficiente de entropía (alpha) automático
        if target_entropy is None:
            self.target_entropy = -action_dim  # Valor sugerido en SAC
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Buffers de replay para demostraciones y para transiciones del agente
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
    
    # Almacenar transición de demostración (recompensa forzada a 1)
    def store_demo(self, state, action, next_state, done):
        self.demo_buffer.push(state, action, 1.0, next_state, done)
        
    # Almacenar transición del agente (recompensa forzada a 0)
    def store_agent(self, state, action, next_state, done):
        self.agent_buffer.push(state, action, 0.0, next_state, done)
    
    def update(self):
        # Se requiere que ambos buffers tengan suficientes muestras
        if len(self.demo_buffer) < self.batch_size // 2 or len(self.agent_buffer) < self.batch_size // 2:
            return
        
        demo_batch_size = self.batch_size // 2
        agent_batch_size = self.batch_size - demo_batch_size
        
        state_d, action_d, reward_d, next_state_d, done_d = self.demo_buffer.sample(demo_batch_size)
        state_a, action_a, reward_a, next_state_a, done_a = self.agent_buffer.sample(agent_batch_size)
        
        # Combinar ambos batches
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
        
        # Actualización del Critic (ecuación de Bellman suave)
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
        
        # Actualización del Actor
        action_new, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Actualización automática del coeficiente de entropía
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Actualización suave de la red target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
