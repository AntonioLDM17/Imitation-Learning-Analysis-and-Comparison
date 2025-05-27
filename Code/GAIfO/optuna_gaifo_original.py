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
import optuna

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
        # Assume s and s_next have shape (batch_size, flat_obs_dim)
        x = torch.cat([s, s_next], dim=1)
        return self.net(x)

# Function to compute gradient penalty for the discriminator
def compute_gradient_penalty(discriminator, s_expert, s_policy, s_expert_next, s_policy_next, device):
    batch_size = s_expert.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    # Interpolate between expert and policy samples for current and next states
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

def objective(trial: optuna.Trial):
    # Suggest hyperparameters to tune
    disc_epochs = trial.suggest_categorical("disc_epochs", [3, 5, 7, 10])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    # Rollout length remains fixed at 2048
    rollout_length = 2048
    lambda_gp = trial.suggest_float("lambda_gp", 1.0, 20.0, log=True)
    disc_lr = trial.suggest_float("disc_lr", 1e-5, 1e-3, log=True)
    num_iterations = trial.suggest_categorical("num_iterations", [100, 300, 500])

    # Fixed parameters and directories
    SEED = 44 + trial.number
    ENV_NAME = args.env
    DEMO_EPISODES = args.demo_episodes
    if ENV_NAME == "HalfCheetah-v4":
        suffix = "halfcheetah"
    elif ENV_NAME == "CartPole-v1":
        suffix = "cartpole"
    else:
        raise ValueError(f"Unsupported environment: {ENV_NAME}")
    
    DEMO_DIR = os.path.join("..", "data", "demonstrations", str(DEMO_EPISODES))
    DEMO_FILENAME = f"{suffix}_demonstrations_{DEMO_EPISODES}.npy"
    MODELS_DIR = "models/gaifo_" + suffix + f"_{DEMO_EPISODES}_{trial.number}"
    LOG_DIR = os.path.join("logs", f"gaifo_{suffix}_{DEMO_EPISODES}_{trial.number}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Create TensorBoard writer
    writer = SummaryWriter(LOG_DIR)

    # Create vectorized environment wrapped with Monitor
    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME), LOG_DIR)])

    # Load expert demonstrations (expected to be an array of trajectories or states)
    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Extract state trajectory from demonstrations
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

    # Get flat observation dimension (assumes a single environment)
    flat_obs_dim = env.observation_space.shape[0]

    # Initialize the policy model using TRPO for GAIfO training
    learner = TRPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=LOG_DIR)
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    learner.set_logger(new_logger)
    obs = env.reset()[0]
    if len(obs.shape) == 1:
        obs = obs[None, :]
    learner._last_obs = obs
    learner.ep_info_buffer = []
    learner.ep_success_buffer = []

    # Initialize the discriminator and move it to device
    discriminator = GAIfODiscriminator(flat_obs_dim, hidden_dim=64).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))
    bce_loss = nn.BCELoss()

    pre_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    pre_reward_mean = np.mean(pre_train_rewards)
    print("Mean reward before training:", pre_reward_mean)
    writer.add_scalar("Reward/PreTrain", pre_reward_mean, 0)

    dummy_callback = DummyCallback()
    dummy_callback.model = learner

    # Training loop: number of iterations is tunable
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
        # Reconstruct next observations by concatenating rollout_obs[1:] with _last_obs
        last_obs_expanded = learner._last_obs[np.newaxis, ...]  # shape (1, n_envs, obs_dim)
        rollout_obs_next = np.concatenate([rollout_obs[1:], last_obs_expanded], axis=0)
        # Flatten observations: from (rollout_length, n_envs, obs_dim) to (rollout_length * n_envs, obs_dim)
        rollout_s = torch.tensor(rollout_obs.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)
        rollout_s_next = torch.tensor(rollout_obs_next.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)

        # Train the discriminator for the specified number of epochs
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
        avg_rollout_reward = np.mean(rewards_np)
        writer.add_scalar("Reward/Rollout", avg_rollout_reward, itr+1)

        learner.rollout_buffer.rewards = rewards_np
        learner.train()

        if (itr + 1) % 10 == 0:
            post_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
            mean_reward = np.mean(post_train_rewards)
            print(f"Iteration {itr+1}, Evaluation Mean Reward: {mean_reward:.2f}")
            writer.add_scalar("Reward/Evaluation", mean_reward, itr+1)

    model_save_path = os.path.join(MODELS_DIR, f"gaifo_{suffix}_{DEMO_EPISODES}_{trial.number}_disc_epochs{disc_epochs}_batch_size{batch_size}_lambda_gp{lambda_gp}_disc_lr{disc_lr}_iterations{num_iterations}")
    learner.save(model_save_path)
    print(f"GAIfO model saved at {model_save_path}.zip")

    final_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    final_mean = np.mean(final_rewards)
    print("Mean reward after training:", final_mean)
    writer.add_scalar("Reward/Final", final_mean, num_iterations)
    env.close()
    writer.close()
    return final_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for GAIfO")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="Environment name")
    parser.add_argument("--demo_episodes", type=int, default=50, help="Number of expert episodes used for training")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"Objective value: {trial.value:.2f}")
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
