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
    Now we remove the final Sigmoid. We'll output raw logits,
    and rely on BCEWithLogitsLoss for stable training.
    """
    def __init__(self, obs_dim, act_dim):
        super(DAC_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # No Sigmoid here
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)  # raw logits (no sigmoid)


class ContinuousActor(nn.Module):
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


class DiscreteActor(nn.Module):
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
        # Returns logits [batch_size, n_actions]
        return self.net(x)


class DiscreteCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(DiscreteCritic, self).__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        # act is [batch_size] of discrete indices
        if act.dim() > 1:
            act = act.squeeze(-1)
        # Convert to one-hot
        act_onehot = F.one_hot(act.long(), num_classes=self.n_actions).float()
        x = torch.cat([obs, act_onehot], dim=1)
        return self.net(x)


class DAC:
    """
    Discriminator-Actor-Critic
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
        We'll use BCEWithLogitsLoss to handle raw logits from the Discriminator.
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

        # Raw logits
        pred_expert = self.discriminator(expert_obs, expert_act)
        pred_policy = self.discriminator(policy_obs, policy_act)

        # Use logits-based loss
        loss_expert = F.binary_cross_entropy_with_logits(pred_expert, expert_labels)
        loss_policy = F.binary_cross_entropy_with_logits(pred_policy, policy_labels)
        loss = loss_expert + loss_policy

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss.item()

    def update_actor_critic(self, batch):
        obs, act, reward, next_obs, done = batch
        obs = obs.to(self.device)
        act = act.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # ---------- Critic update ----------
        with torch.no_grad():
            # Next action
            if self.is_discrete:
                logits_next = self.actor(next_obs)
                dist_next = Categorical(logits=logits_next)
                next_act = dist_next.sample()  # [batch_size]
            else:
                next_act = self.actor(next_obs)

            target_q = self.critic(next_obs, next_act)
            target = reward + (1.0 - done) * self.gamma * target_q

        current_q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Actor update ----------
        # Maximize Q(s, pi(s)) => minimize -Q(s, pi(s))
        if self.is_discrete:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            sampled_act = dist.sample()
            actor_loss = -self.critic(obs, sampled_act).mean()
        else:
            actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()
