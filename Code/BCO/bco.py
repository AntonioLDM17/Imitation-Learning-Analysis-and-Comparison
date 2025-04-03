""" 
Credits to:
@inproceedings{torabi2018bco,
  author = {Faraz Torabi and Garrett Warnell and Peter Stone}, 
  title = {{Behavioral Cloning from Observation}}, 
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)}, 
  year = {2018} 
}
Where the code is based on the original BCO implementation by Faraz Torabi.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

# Function to set the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Inverse Dynamics Model: receives s and s_next and predicts the action.
# For discrete environments, outputs logits; for continuous, direct prediction.
class InverseDynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, discrete=True):
        super(InverseDynamicsModel, self).__init__()
        self.discrete = discrete
        hidden_size = 64
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, s, s_next):
        x = torch.cat([s, s_next], dim=1)
        out = self.net(x)
        return out  # For discrete: logits; for continuous: direct prediction

# Policy Network (Behavioral Cloning): maps state to action
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, discrete=True):
        super(PolicyNetwork, self).__init__()
        hidden_size = 64
        self.discrete = discrete
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, s):
        out = self.net(s)
        return out  # For discrete: logits; for continuous: action

# Function to collect exploration data (pre-demonstration) using a random policy
def collect_exploration_data(env, num_interactions):
    s_list, s_next_list, a_list = [], [], []
    obs, _ = env.reset()
    total_steps = 0

    while total_steps < num_interactions:
        action = env.action_space.sample()
        s = np.array(obs)
        obs, reward, done, truncated, _ = env.step(action)
        s_next = np.array(obs)
        s_list.append(s)
        s_next_list.append(s_next)
        a_list.append(action)
        total_steps += 1
        if done or truncated:
            obs, _ = env.reset()
    return np.array(s_list), np.array(s_next_list), np.array(a_list)

# Function to create a DataLoader from data
def create_dataloader(s, s_next, a, batch_size=64):
    dataset = TensorDataset(torch.tensor(s, dtype=torch.float32),
                            torch.tensor(s_next, dtype=torch.float32),
                            torch.tensor(a))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the inverse dynamics model
def train_inverse_model(model, dataloader, discrete=True, epochs=10, lr=1e-3, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for s, s_next, a in dataloader:
            optimizer.zero_grad()
            outputs = model(s, s_next)
            if discrete:
                loss = criterion(outputs, a.long().squeeze())
            else:
                loss = criterion(outputs, a.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Inverse Model Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if writer is not None:
            writer.add_scalar("InverseModel/Loss", avg_loss, epoch+1)
    return model

# Infer expert actions from a demonstration (state trajectory)
def infer_expert_actions(model, demo_states, discrete=True):
    # Form consecutive pairs (s, s_next)
    s_demo = demo_states[:-1]
    s_next_demo = demo_states[1:]
    model.eval()
    with torch.no_grad():
        s_tensor = torch.tensor(s_demo, dtype=torch.float32)
        s_next_tensor = torch.tensor(s_next_demo, dtype=torch.float32)
        outputs = model(s_tensor, s_next_tensor)
        if discrete:
            inferred_actions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            inferred_actions = outputs.cpu().numpy()
    return s_demo, inferred_actions

# Train the policy from (state, inferred action) pairs
def train_policy(policy_net, states, actions, discrete=True, epochs=20, lr=1e-3, batch_size=64, writer=None):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()

    policy_net.train()
    dataset = TensorDataset(torch.tensor(states, dtype=torch.float32),
                            torch.tensor(actions))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        losses = []
        for s, a in dataloader:
            optimizer.zero_grad()
            outputs = policy_net(s)
            if discrete:
                loss = criterion(outputs, a.long().squeeze())
            else:
                loss = criterion(outputs, a.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Policy Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if writer is not None:
            writer.add_scalar("Policy/Loss", avg_loss, epoch+1)
    return policy_net

# Function to collect data using the current policy (for post-demonstration interactions)
def collect_policy_data(policy_net, env, num_interactions, discrete):
    s_list, s_next_list, a_list = [], [], []
    obs, _ = env.reset()
    total_steps = 0
    while total_steps < num_interactions:
        state = np.array(obs)
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = policy_net(s_tensor)
            if discrete:
                action = torch.argmax(output, dim=1).item()
            else:
                action = output.squeeze().cpu().numpy()
        next_obs, reward, done, truncated, _ = env.step(action)
        s_list.append(state)
        s_next_list.append(np.array(next_obs))
        a_list.append(action)
        total_steps += 1
        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return np.array(s_list), np.array(s_next_list), np.array(a_list)
