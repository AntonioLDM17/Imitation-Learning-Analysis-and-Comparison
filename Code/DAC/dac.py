"""
DAC Implementation (Discriminator Actor Critic)
Based on ChanB's implementation:
    https://github.com/chanb/rl_sandbox_public/blob/master/rl_sandbox/algorithms/dac/dac.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAC_Discriminator(nn.Module):
    """
    Discriminator network that receives the concatenation of an observation and an action,
    and estimates the probability that the pair comes from an expert.
    Dimensions are extracted from the provided spaces.
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
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

class ContinuousActor(nn.Module):
    """
    Actor network for continuous actions.
    Outputs actions in the range [-1, 1] using a tanh activation.
    """
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
    """
    Critic network for continuous actions.
    """
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

class DAC:
    """
    Discriminator-Actor-Critic (DAC) class.
    It manages the actor, critic, discriminator, and their respective optimizers,
    as well as a simple replay buffer.
    """
    def __init__(self, actor, critic, discriminator,
                 actor_optimizer, critic_optimizer, disc_optimizer,
                 device="cpu", gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.discriminator = discriminator
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.gamma = gamma
        self.replay_buffer = []  # Simple replay buffer

    def update_actor_critic(self, batch):
        obs, act, reward, next_obs, done = batch

        with torch.no_grad():
            target_q = self.critic(next_obs, self.actor(next_obs))
            target = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def update_discriminator(self, expert_batch, policy_batch):
        expert_obs, expert_act = expert_batch
        policy_obs, policy_act = policy_batch

        expert_obs = expert_obs.to(self.device)
        expert_act = expert_act.to(self.device)
        policy_obs = policy_obs.to(self.device)
        policy_act = policy_act.to(self.device)

        # Labels: 1 for expert, 0 for policy
        expert_labels = torch.ones((expert_obs.size(0), 1), device=self.device)
        policy_labels = torch.zeros((policy_obs.size(0), 1), device=self.device)

        pred_expert = self.discriminator(expert_obs, expert_act)
        pred_policy = self.discriminator(policy_obs, policy_act)

        loss_expert = nn.functional.binary_cross_entropy_with_logits(pred_expert, expert_labels)
        loss_policy = nn.functional.binary_cross_entropy_with_logits(pred_policy, policy_labels)
        loss = loss_expert + loss_policy

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss.item()
