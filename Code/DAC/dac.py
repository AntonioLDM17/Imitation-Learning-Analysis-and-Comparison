"""
DAC Implementation (Discriminator Actor Critic)
Based on ChanB's implementation:
    https://github.com/chanb/rl_sandbox_public/blob/master/rl_sandbox/algorithms/dac/dac.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class DAC_Discriminator(nn.Module):
    """
    Discriminator for DAC.
    It receives the concatenation of an observation and an action, and outputs raw probability values (after a Sigmoid).
    The network dimensions are computed from the provided observation_space and action_space.
    """
    def __init__(self, observation_space, action_space):
        super(DAC_Discriminator, self).__init__()
        if not hasattr(observation_space, "shape") or not hasattr(action_space, "shape"):
            raise ValueError("Both observation_space and action_space must have a 'shape' attribute")
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- Networks for Continuous DAC ---

class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ContinuousActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

class ContinuousCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ContinuousCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- DAC (Discriminator-Actor-Critic) ---
class DAC:
    """
    Discriminator-Actor-Critic implementation.
    """
    def __init__(
        self,
        actor,
        critic,
        discriminator,
        actor_optimizer,
        critic_optimizer,
        disc_optimizer,
        device="cpu",
        gamma=0.99,
        is_discrete=False,
        n_actions=None
    ):
        self.actor = actor
        self.critic = critic
        self.discriminator = discriminator
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.gamma = gamma

        self.is_discrete = is_discrete
        self.n_actions = n_actions

        # Simple replay buffer
        self.replay_buffer = []

    def update_discriminator(self, expert_batch, policy_batch):
        """
        Update the discriminator using BCEWithLogitsLoss.
        """
        expert_obs, expert_act = expert_batch
        policy_obs, policy_act = policy_batch

        expert_obs = expert_obs.to(self.device)
        expert_act = expert_act.to(self.device)
        policy_obs = policy_obs.to(self.device)
        policy_act = policy_act.to(self.device)

        # Labels: 1 for expert, 0 for policy
        expert_labels = torch.ones((expert_obs.size(0), 1), device=self.device)
        policy_labels = torch.zeros((policy_obs.size(0), 1), device=self.device)

        # Raw logits from the discriminator
        pred_expert = self.discriminator(expert_obs, expert_act)
        pred_policy = self.discriminator(policy_obs, policy_act)

        # Loss using logits-based loss
        loss_expert = F.binary_cross_entropy_with_logits(pred_expert, expert_labels)
        loss_policy = F.binary_cross_entropy_with_logits(pred_policy, policy_labels)
        loss = loss_expert + loss_policy

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss.item()

    def update_actor_critic(self, batch):
        obs, act, reward, next_obs, done = batch
        # If the inputs are not already torch.Tensors, convert them and send them to self.device;
        # otherwise, just move them.
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float32, device=self.device)
        else:
            act = act.to(self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            reward = reward.to(self.device)
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        else:
            next_obs = next_obs.to(self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            done = done.to(self.device)

        with torch.no_grad():
            if hasattr(self.actor, "sample"):
                next_action, next_log_prob, _ = self.actor.sample(next_obs)
            else:
                # In case sample() is not available (for discrete actor, for example)
                next_action = self.actor(next_obs)
                next_log_prob = torch.zeros((next_obs.size(0), 1), device=self.device)
            target_q = self.critic(next_obs, next_action)
            target = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if hasattr(self.actor, "sample"):
            action_new, log_prob, _ = self.actor.sample(obs)
        else:
            action_new = self.actor(obs)
            log_prob = torch.zeros((obs.size(0), 1), device=self.device)
        q_new = self.critic(obs, action_new)
        actor_loss = (-q_new).mean()  # Maximizar Q(s,pi(s)) equivale a minimizar -Q(s,pi(s))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()
