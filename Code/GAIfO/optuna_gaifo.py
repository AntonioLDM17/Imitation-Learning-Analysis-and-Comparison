"""
optuna_gaifo.py  ·  Hyper-parameter search for GAIfO with Optuna
---------------------------------------------------------------
This version uses **total environment steps** as the training budget
instead of the former “num_iterations”.  All logs therefore use the
global step counter so that TensorBoard’s x-axis is in *real* env-steps.

Only the step/iteration-related code has been changed; all GAIfO logic,
hyper-parameters, Optuna integration, etc. remain untouched.
"""

import os, sys, types, argparse, math          # math is needed for ceil()
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

# --------------------------------------------------------------------- #
#  Dummy mujoco_py stubs to avoid compiling MuJoCo when not installed   #
# --------------------------------------------------------------------- #
dummy = types.ModuleType("mujoco_py")
dummy.builder     = types.ModuleType("mujoco_py.builder")
dummy.locomotion  = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"]          = dummy
sys.modules["mujoco_py.builder"]  = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------- #
#  Helper classes and functions                                         #
# --------------------------------------------------------------------- #
class DummyCallback(BaseCallback):
    """No-op callback required by SB3’s collect_rollouts()."""
    def _on_step(self)            -> bool:  return True
    def _on_rollout_start(self)   -> None:  pass
    def _on_rollout_end(self)     -> None:  pass


class GAIfODiscriminator(nn.Module):
    """State-only discriminator (s, s′) → probability."""
    def __init__(self, flat_obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),       nn.ReLU(),
            nn.Linear(hidden_dim, 1),                nn.Sigmoid()
        )
    def forward(self, s, s_next):
        return self.net(torch.cat([s, s_next], dim=1))


def compute_gradient_penalty(discriminator, s_expert, s_policy,
                             s_expert_next, s_policy_next, device):
    """WGAN-GP style penalty on interpolated samples."""
    batch_size = s_expert.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    s_hat       = alpha * s_expert       + (1 - alpha) * s_policy
    s_hat_next  = alpha * s_expert_next  + (1 - alpha) * s_policy_next
    s_hat.requires_grad_(True);  s_hat_next.requires_grad_(True)

    d_hat = discriminator(s_hat, s_hat_next)
    ones  = torch.ones_like(d_hat, device=device)

    grad_s, grad_s_next = torch.autograd.grad(
        outputs=d_hat, inputs=[s_hat, s_hat_next], grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )
    grad      = torch.cat([grad_s, grad_s_next], dim=1)
    grad_norm = grad.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# --------------------------------------------------------------------- #
#  Optuna objective                                                     #
# --------------------------------------------------------------------- #
def objective(trial: optuna.Trial):
    # 1) Hyper-parameters to search ------------------------------------ #
    disc_epochs   = trial.suggest_categorical("disc_epochs", [10, 20, 30])
    batch_size    = trial.suggest_categorical("batch_size",  [256, 512, 1024])
    rollout_length = 2048                              # ← fixed
    lambda_gp     = trial.suggest_float("lambda_gp", 0.3, 10, log=True)
    disc_lr       = trial.suggest_float("disc_lr",  1e-5, 1e-3, log=True)

    # Training budget *in environment steps* (was iterations)
    total_steps   = trial.suggest_categorical(
        "total_steps",
        [ 489 * rollout_length * 2])    # 1 001 472 * 2 = 2 002 944 steps

    # 2) Fixed settings (derived from CLI) ----------------------------- #
    SEED         = 44  #if you want to use different seeds
    ENV_NAME     = args.env
    DEMO_EPISODES = args.demo_episodes

    if ENV_NAME == "HalfCheetah-v4":
        suffix = "halfcheetah"
    elif ENV_NAME == "CartPole-v1":
        suffix = "cartpole"
    else:
        raise ValueError(f"Unsupported environment: {ENV_NAME}")

    DEMO_DIR   = os.path.join("..", "data", "demonstrations", str(DEMO_EPISODES))
    DEMO_FILE  = f"{suffix}_demonstrations_{DEMO_EPISODES}.npy"
    MODELS_DIR = f"models/gaifo_{suffix}_{DEMO_EPISODES}_{trial.number}_exec_2"
    LOG_DIR    = f"logs/gaifo_{suffix}_{DEMO_EPISODES}_{trial.number}_exec_2"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    writer = SummaryWriter(LOG_DIR)       # TensorBoard logger

    # 3) Environment and expert data ----------------------------------- #
    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME), LOG_DIR)])
    demo_path = os.path.join(DEMO_DIR, DEMO_FILE)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Flatten expert trajectories into a single (N, obs_dim) array
    if demonstrations and hasattr(demonstrations[0], 'obs'):
        demo_traj = np.concatenate([np.array(t.obs, dtype=np.float32)
                                    for t in demonstrations])
    else:
        demo_traj = np.array([np.array(s, dtype=np.float32) for s in demonstrations])

    # Build (s, s′) transition pairs
    expert_transitions = np.array([(demo_traj[i], demo_traj[i + 1])
                                   for i in range(len(demo_traj) - 1)])
    expert_s       = torch.tensor(np.stack(expert_transitions[:, 0]), dtype=torch.float32).to(device)
    expert_s_next  = torch.tensor(np.stack(expert_transitions[:, 1]), dtype=torch.float32).to(device)

    flat_obs_dim = env.observation_space.shape[0]

    # 4) Initialise learner (TRPO) and discriminator ------------------- #
    learner = TRPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=LOG_DIR)
    learner.set_logger(configure(LOG_DIR, ["stdout", "tensorboard"]))
    obs = env.reset()[0];  learner._last_obs = obs if obs.ndim == 2 else obs[None, :]
    learner.ep_info_buffer = [];  learner.ep_success_buffer = []

    discriminator = GAIfODiscriminator(flat_obs_dim).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))
    bce_loss = nn.BCELoss()

    # Pre-training evaluation
    pre_reward_mean = np.mean(evaluate_policy(learner, env, 10, return_episode_rewards=True)[0])
    writer.add_scalar("Reward/PreTrain", pre_reward_mean, 0)

    dummy_callback = DummyCallback();  dummy_callback.model = learner

    # ------------------------------------------------------------------ #
    #  Step-based training loop                                          #
    # ------------------------------------------------------------------ #
    n_envs           = env.num_envs                  # =1 with DummyVecEnv
    steps_per_iter   = rollout_length * n_envs
    num_iterations   = math.ceil(total_steps / steps_per_iter)
    steps_so_far     = 0

    for itr in range(num_iterations):
        print(f"Progress: {steps_so_far}/{total_steps} env-steps")

        # Rollout collection (gracefully handle VecEnv reset errors)
        try:
            learner.collect_rollouts(env, dummy_callback, learner.rollout_buffer,
                                     n_rollout_steps=rollout_length)
        except RuntimeError as e:
            if "needs reset" in str(e):
                obs = env.reset()[0];  learner._last_obs = obs if obs.ndim == 2 else obs[None, :]
                learner.collect_rollouts(env, dummy_callback, learner.rollout_buffer,
                                         n_rollout_steps=rollout_length)
            else:
                raise

        steps_so_far += steps_per_iter

        # ---------------------------------------------------------------- #
        #  Discriminator training                                          #
        # ---------------------------------------------------------------- #
        rollout_obs      = learner.rollout_buffer.observations
        last_obs_expanded = learner._last_obs[np.newaxis, ...]
        rollout_obs_next = np.concatenate([rollout_obs[1:], last_obs_expanded], axis=0)

        rollout_s       = torch.tensor(rollout_obs.reshape(-1, flat_obs_dim),      dtype=torch.float32).to(device)
        rollout_s_next  = torch.tensor(rollout_obs_next.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)

        for _ in range(disc_epochs):
            idx_pol = np.random.choice(rollout_s.shape[0], batch_size, replace=True)
            idx_exp = np.random.choice(expert_s.shape[0],  batch_size, replace=True)

            s_pol,  s_pol_next  = rollout_s[idx_pol],  rollout_s_next[idx_pol]
            s_exp,  s_exp_next  = expert_s[idx_exp],   expert_s_next[idx_exp]

            pred_pol = discriminator(s_pol,  s_pol_next)
            pred_exp = discriminator(s_exp,  s_exp_next)

            loss_pol  = bce_loss(pred_pol, torch.ones_like(pred_pol))
            loss_exp  = bce_loss(pred_exp, torch.zeros_like(pred_exp))
            gp        = compute_gradient_penalty(discriminator, s_exp, s_pol,
                                                 s_exp_next, s_pol_next, device)
            disc_loss = loss_pol + loss_exp + lambda_gp * gp

            disc_optimizer.zero_grad();  disc_loss.backward();  disc_optimizer.step()

        writer.add_scalar("Discriminator/Loss", disc_loss.item(), steps_so_far)

        # ---------------------------------------------------------------- #
        #  Build GAIfO rewards & update policy                             #
        # ---------------------------------------------------------------- #
        with torch.no_grad():
            rewards_np = (-torch.log(discriminator(rollout_s, rollout_s_next) + 1e-8)
                          ).cpu().numpy().flatten()

        learner.rollout_buffer.rewards = rewards_np
        avg_rollout_reward = np.mean(rewards_np)
        writer.add_scalar("Reward/Rollout", avg_rollout_reward, steps_so_far)

        learner.train()

        # Periodic evaluation every ~10 rollouts
        if (itr + 1) % 10 == 0:
            mean_reward = np.mean(evaluate_policy(learner, env, 10, True)[0])
            print(f"After {steps_so_far} steps → eval mean reward: {mean_reward:.2f}")
            writer.add_scalar("Reward/Evaluation", mean_reward, steps_so_far)

        # Stop as soon as the target budget is met or exceeded
        if steps_so_far >= total_steps:
            break

    # ------------------------------------------------------------------ #
    #  Save results & final evaluation                                   #
    # ------------------------------------------------------------------ #
    model_path = os.path.join(
        MODELS_DIR,
        f"gaifo_{suffix}_{DEMO_EPISODES}_{trial.number}"
        f"_disc_epochs{disc_epochs}_batch_size{batch_size}"
        f"_lambda_gp{lambda_gp}_disc_lr{disc_lr}_steps{total_steps}"
    )
    learner.save(model_path)
    final_mean = np.mean(evaluate_policy(learner, env, 10, True)[0])
    writer.add_scalar("Reward/Final", final_mean, steps_so_far)

    print(f"GAIfO model saved to {model_path}.zip")
    print(f"Mean reward after training: {final_mean:.2f}")

    env.close();  writer.close()
    return final_mean   # Optuna maximises this value

# ---------------------------------------------------------------------- #
#  CLI entry-point                                                       #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for GAIfO (step-based budget)")
    parser.add_argument("--n_trials",       type=int, default=20,               help="Optuna trials")
    parser.add_argument("--env",            type=str, default="HalfCheetah-v4", help="Gymnasium env id")
    parser.add_argument("--demo_episodes",  type=int, default=50,               help="Expert episodes")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value: {best.value:.2f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
