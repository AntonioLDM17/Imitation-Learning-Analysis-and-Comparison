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

class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model (IDM) for Behavioral Cloning from Observation (BCO).
    This model predicts the action taken to transition from state `s` to state `s_next`.
    Args:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        discrete (bool): Whether the action space is discrete (default: True).
    Returns:
        out (torch.Tensor): Predicted action logits (for discrete) or direct action (for continuous).
    
    """
    def __init__(self, obs_dim, action_dim, discrete=True):
        """ 
        Initializes the Inverse Dynamics Model.
        """
        # Initialize the parent class
        super(InverseDynamicsModel, self).__init__()
        self.discrete = discrete
        hidden_size = 256          # before 64
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(obs_dim*2, hidden_size),
            nn.ReLU(), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_dim)
        )

        
    def forward(self, s, s_next):
        x = torch.cat([s, s_next], dim=1)
        out = self.net(x)
        return out  # For discrete: logits; for continuous: direct prediction

class PolicyNetwork(nn.Module):
    """ 
    Policy Network for Behavioral Cloning.
    This network takes the state as input and outputs the action to be taken.
    Args:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        discrete (bool): Whether the action space is discrete (default: True).
    Returns:
        out (torch.Tensor): Action logits (for discrete) or direct action (for continuous).
    Note:
        For discrete action spaces, the output is logits for each action.
        For continuous action spaces, the output is the action itself.
    """
    def __init__(self, obs_dim, action_dim, discrete=True):
        super(PolicyNetwork, self).__init__()
        hidden_size = 64
        self.discrete = discrete
        # Define the network architecture
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

def collect_exploration_data(env, num_interactions):
    """ 
    Collects exploration data from the environment using a random policy.
    Args:
        env (gym.Env): The environment to collect data from.
        num_interactions (int): Number of interactions to collect.
    Returns:
        s_list (np.ndarray): List of states observed.
        s_next_list (np.ndarray): List of next states observed.
        a_list (np.ndarray): List of actions taken.
    """
    s_list, s_next_list, a_list = [], [], []
    obs, _ = env.reset()
    total_steps = 0
    # Collect data until we reach the specified number of interactions
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
    # Ensure the lists are numpy arrays
    return np.array(s_list), np.array(s_next_list), np.array(a_list)

def create_dataloader(s, s_next, a, batch_size=64):
    """
    Creates a DataLoader from the collected data.
    Args:
        s (np.ndarray): List of states observed.
        s_next (np.ndarray): List of next states observed.
        a (np.ndarray): List of actions taken.
        batch_size (int): Size of each batch for training.
    Returns:
        DataLoader: A DataLoader object for batching the data.
    """
    dataset = TensorDataset(torch.tensor(s, dtype=torch.float32),
                            torch.tensor(s_next, dtype=torch.float32),
                            torch.tensor(a))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_inverse_model(model, dataloader, discrete=True, epochs=10, lr=1e-3, writer=None):
    """
    Trains the Inverse Dynamics Model (IDM) using the provided data.
    Args:
        model (InverseDynamicsModel): The IDM to be trained.
        dataloader (DataLoader): DataLoader containing (s, s_next, a) pairs.
        discrete (bool): Whether the action space is discrete (default: True).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        writer (SummaryWriter, optional): TensorBoard writer for logging.
    Returns:
        model (InverseDynamicsModel): The trained IDM.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()
    # Ensure model is in training mode
    model.train()
    # Iterate over epochs
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

def infer_expert_actions(model, demo_states, discrete=True):
    """
    Infers expert actions from a demonstration using the trained Inverse Dynamics Model.
    Args:
        model (InverseDynamicsModel): The trained IDM.
        demo_states (np.ndarray): Array of states from the demonstration.
        discrete (bool): Whether the action space is discrete (default: True).
    Returns:
        s_demo (np.ndarray): States from the demonstration excluding the last state.
        inferred_actions (np.ndarray): Inferred actions corresponding to the states.
    """
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

def train_policy(policy_net, states, actions, discrete=True, epochs=20, lr=1e-3, batch_size=64, writer=None):
    """
    Trains the policy network using the inferred actions from the IDM.
    Args:
        policy_net (PolicyNetwork): The policy network to be trained.
        states (np.ndarray): Array of states from the demonstration.
        actions (np.ndarray): Inferred actions corresponding to the states.
        discrete (bool): Whether the action space is discrete (default: True).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Size of each batch for training.
        writer (SummaryWriter, optional): TensorBoard writer for logging.
    Returns:
        policy_net (PolicyNetwork): The trained policy network.
    """
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()

    policy_net.train()
    dataset = TensorDataset(torch.tensor(states, dtype=torch.float32),
                            torch.tensor(actions))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Iterate over epochs
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

def collect_policy_data(policy_net, env, num_interactions, discrete):
    """
    Collects data from the environment using the current policy.
    Args:
        policy_net (PolicyNetwork): The trained policy network.
        env (gym.Env): The environment to collect data from.
        num_interactions (int): Number of interactions to collect.
        discrete (bool): Whether the action space is discrete (default: True).
    Returns:
        s_list (np.ndarray): List of states observed.
        s_next_list (np.ndarray): List of next states observed.
        a_list (np.ndarray): List of actions taken.
    """
    s_list, s_next_list, a_list = [], [], []
    obs, _ = env.reset()
    total_steps = 0
    while total_steps < num_interactions:
        state = np.array(obs)
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
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
