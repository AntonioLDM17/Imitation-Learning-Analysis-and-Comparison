import os, sys, types, argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# Create dummy modules for mujoco_py (to avoid compilation issues)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

# Set device for computation (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy callback to be used with collect_rollouts
class DummyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DummyCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        return True
    def _on_rollout_start(self) -> None:
        pass
    def _on_rollout_end(self) -> None:
        pass

# Discriminator that expects flattened observations
class GAIfODiscriminator(nn.Module):
    def __init__(self, flat_obs_dim, hidden_dim=64):
        super(GAIfODiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in (0,1)
        )
    def forward(self, s, s_next):
        # Assume s and s_next are of shape (batch_size, flat_obs_dim)
        x = torch.cat([s, s_next], dim=1)
        return self.net(x)

# Function to compute the gradient penalty for the discriminator
def compute_gradient_penalty(discriminator, s_expert, s_policy, s_expert_next, s_policy_next, device):
    batch_size = s_expert.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    # Interpolate between expert and policy samples
    s_hat = alpha * s_expert + (1 - alpha) * s_policy
    s_hat_next = alpha * s_expert_next + (1 - alpha) * s_policy_next

    s_hat.requires_grad_(True)
    s_hat_next.requires_grad_(True)
    
    d_hat = discriminator(s_hat, s_hat_next)
    ones = torch.ones(d_hat.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=[s_hat, s_hat_next],
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    grad_s, grad_s_next = gradients
    grad = torch.cat([grad_s, grad_s_next], dim=1)
    grad_norm = grad.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

def main():
    parser = argparse.ArgumentParser(description="Train GAIfO using TRPO (state-only imitation).")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"],
                        default="cartpole", help="Environment: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Number of expert episodes to use for training")
    args = parser.parse_args()

    SEED = args.seed
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    elif args.env == "halfcheetah":
        ENV_NAME = "HalfCheetah-v4"
    else:
        raise ValueError("Invalid environment.")

    # DEMO_DIR = os.path.join("..", "data", "demonstrations", str(args.demo_episodes))
    DEMO_DIR = os.path.join("..", "data", "demonstrations")
    DEMO_FILENAME = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODELS_DIR = f"models/gaifo_{args.env}_{args.demo_episodes}_1M_expert_3"
    LOG_DIR = os.path.join("logs", f"gaifo_{args.env}_{args.demo_episodes}_1M_expert_3")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(LOG_DIR)

    # Create the vectorized environment wrapped with Monitor
    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME), LOG_DIR)])

    # Load expert demonstrations (expected to be an array of trajectories or states)
    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Extract state trajectory from the demonstrations
    if len(demonstrations) > 0 and hasattr(demonstrations[0], 'obs'):
        demo_traj_list = []
        for traj in demonstrations:
            demo_traj_list.extend([np.array(s, dtype=np.float32) for s in traj.obs])
        demo_traj = np.array(demo_traj_list)
    else:
        demo_traj = np.array([np.array(x, dtype=np.float32) for x in demonstrations])

    # Generate expert transitions (pairs of consecutive states)
    expert_transitions = []
    for i in range(len(demo_traj) - 1):
        expert_transitions.append((demo_traj[i], demo_traj[i + 1]))
    expert_transitions = np.array(expert_transitions)

    expert_s = torch.tensor(np.stack([t[0] for t in expert_transitions]), dtype=torch.float32).to(device)
    expert_s_next = torch.tensor(np.stack([t[1] for t in expert_transitions]), dtype=torch.float32).to(device)

    # Get flat observation dimension (assuming a single environment)
    flat_obs_dim = env.observation_space.shape[0]

    # Initialize the policy model using TRPO
    learner = TRPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=LOG_DIR)
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    learner.set_logger(new_logger)
    obs = env.reset()[0]  # env.reset() returns (observations, info)
    if len(obs.shape) == 1:
        obs = obs[None, :]
    learner._last_obs = obs
    learner.ep_info_buffer = []
    learner.ep_success_buffer = []

    # Initialize the discriminator with flat_obs_dim and move it to device
    discriminator = GAIfODiscriminator(flat_obs_dim, hidden_dim=64).to(device)
    # Increase the discriminator's learning rate
    disc_lr = 3.989020006157259e-05 # Adjusted learning rate for the discriminator
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))
    bce_loss = nn.BCELoss()

    pre_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    pre_reward_mean = np.mean(pre_train_rewards)
    print("Mean reward before training:", pre_reward_mean)
    writer.add_scalar("Reward/PreTrain", pre_reward_mean, 0)

    dummy_callback = DummyCallback()
    dummy_callback.model = learner

    # Training loop: use the number of iterations specified by the argument
    num_iterations = args.iterations
    rollout_length = 2048  # timesteps per rollout
    lambda_gp = 1.660941233998641  # gradient penalty coefficient
    for itr in range(num_iterations):
        print(f"Iteration {itr+1}/{num_iterations}")
        try:
            rollout = learner.collect_rollouts(learner.env, dummy_callback, learner.rollout_buffer, n_rollout_steps=rollout_length)
        except RuntimeError as e:
            if "needs reset" in str(e):
                obs = env.reset()[0]
                if len(obs.shape) == 1:
                    obs = obs[None, :]
                learner._last_obs = obs
                rollout = learner.collect_rollouts(learner.env, dummy_callback, learner.rollout_buffer, n_rollout_steps=rollout_length)
            else:
                raise e

        # rollout_obs has shape (rollout_length, n_envs, obs_dim)
        rollout_obs = learner.rollout_buffer.observations
        # Reconstruct next observations: concatenate rollout_obs[1:] with _last_obs
        last_obs_expanded = learner._last_obs[np.newaxis, ...]  # shape (1, n_envs, obs_dim)
        rollout_obs_next = np.concatenate([rollout_obs[1:], last_obs_expanded], axis=0)
        # Flatten the observations: from (rollout_length, n_envs, obs_dim) to (rollout_length*n_envs, obs_dim)
        rollout_s = torch.tensor(rollout_obs.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)
        rollout_s_next = torch.tensor(rollout_obs_next.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)

        # Train the discriminator
        disc_epochs = 10
        batch_size = 256
        for epoch in range(disc_epochs):
            idx_policy = np.random.choice(rollout_s.shape[0], batch_size, replace=True)
            idx_expert = np.random.choice(expert_s.shape[0], batch_size, replace=True)
            s_policy = rollout_s[idx_policy]
            s_next_policy = rollout_s_next[idx_policy]
            s_expert = expert_s[idx_expert]
            s_next_expert = expert_s_next[idx_expert]
            pred_policy = discriminator(s_policy, s_next_policy)
            pred_expert = discriminator(s_expert, s_next_expert)
            loss_policy = bce_loss(pred_policy, torch.ones_like(pred_policy))
            loss_expert = bce_loss(pred_expert, torch.zeros_like(pred_expert))
            # Compute gradient penalty on interpolated samples between expert and policy data
            gp = compute_gradient_penalty(discriminator, s_expert, s_policy, s_next_expert, s_next_policy, device)
            disc_loss = loss_policy + loss_expert + lambda_gp * gp
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        print(f"Discriminator loss: {disc_loss.item():.4f}")
        writer.add_scalar("Discriminator/Loss", disc_loss.item(), itr+1)

        with torch.no_grad():
            d_vals = discriminator(rollout_s, rollout_s_next)
            rewards = -torch.log(d_vals + 1e-8)
            rewards_np = rewards.cpu().numpy().flatten()
        # Log the mean rollout reward before updating the policy
        avg_rollout_reward = np.mean(rewards_np)
        writer.add_scalar("Reward/Rollout", avg_rollout_reward, itr+1)

        learner.rollout_buffer.rewards = rewards_np
        learner.train()

        if (itr + 1) % 10 == 0:
            post_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
            mean_reward = np.mean(post_train_rewards)
            print(f"Iteration {itr+1}, Mean reward: {mean_reward:.2f}")
            writer.add_scalar("Reward/Evaluation", mean_reward, itr+1)

    model_save_path = os.path.join(MODELS_DIR, f"gaifo_{args.env}_{args.demo_episodes}_{num_iterations}")
    learner.save(model_save_path)
    print(f"GAIfO model saved at {model_save_path}.zip")

    final_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    final_mean = np.mean(final_rewards)
    print("Mean reward after training:", final_mean)
    writer.add_scalar("Reward/Final", final_mean, num_iterations)
    env.close()
    writer.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_gaifo.py --env halfcheetah --iterations 300 --seed 44 --demo-episodes 50")
    main()
    