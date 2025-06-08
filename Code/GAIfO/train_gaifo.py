import os, sys, types, argparse, math
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


# -----------------------------------------------------------------------------
# Work‑around for `mujoco_py` import issues
# -----------------------------------------------------------------------------
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Callback skeleton
# -----------------------------------------------------------------------------
class DummyCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass

# -----------------------------------------------------------------------------
# Discriminator
# -----------------------------------------------------------------------------
class GAIfODiscriminator(nn.Module):
    """ 
    Discriminator for GAIfO, which takes two states (current and next) as input
    and outputs a probability that the transition is from the expert policy.
    Args:
        flat_obs_dim (int): Dimension of the flattened observation space.
        hidden_dim (int): Dimension of the hidden layers in the network.
    Returns:
        torch.Tensor: Output probability in the range (0, 1) indicating the likelihood
                      that the transition is from the expert policy.
    """
    def __init__(self, flat_obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output in (0, 1)
        )

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, s_next], dim=1)
        return self.net(x)

# -----------------------------------------------------------------------------
# Gradient‑penalty helper
# -----------------------------------------------------------------------------

def compute_gradient_penalty(
    discriminator: nn.Module,
    s_expert: torch.Tensor,
    s_policy: torch.Tensor,
    s_expert_next: torch.Tensor,
    s_policy_next: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    WGAN‑GP style gradient penalty.
    Args:
        discriminator (nn.Module): The discriminator
        s_expert (torch.Tensor): Expert states (current).
        s_policy (torch.Tensor): Policy states (current).
        s_expert_next (torch.Tensor): Expert states (next).
        s_policy_next (torch.Tensor): Policy states (next).
        device (torch.device): Device to perform computations on.
    Returns:
        torch.Tensor: Gradient penalty loss.
    """
    batch_size = s_expert.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    s_hat = alpha * s_expert + (1 - alpha) * s_policy
    s_hat_next = alpha * s_expert_next + (1 - alpha) * s_policy_next

    s_hat.requires_grad_(True)
    s_hat_next.requires_grad_(True)

    d_hat = discriminator(s_hat, s_hat_next)
    ones = torch.ones_like(d_hat, device=device)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=[s_hat, s_hat_next],
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grad_s, grad_s_next = gradients
    grad = torch.cat([grad_s, grad_s_next], dim=1)
    grad_norm = grad.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GAIfO using TRPO (state‑only imitation) – step‑based version.",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole", "halfcheetah"],
        default="cartpole",
        help="Environment identifier.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=498 * 2048*2,  # 1 001 472 steps * 2 (for 2M total steps)
        help="Total environment interaction steps to train.",
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--demo_episodes", type=int, default=50, help="Number of expert episodes to use for training")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    elif args.env == "halfcheetah":
        ENV_NAME = "HalfCheetah-v4"
    else:
        raise ValueError("Unsupported environment.")

    DEMO_DIR = os.path.join("..", "data", "demonstrations", str(args.demo_episodes))
    DEMO_FILENAME = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODELS_DIR = f"models/gaifo_{args.env}_{args.demo_episodes}_2M"
    LOG_DIR = os.path.join("logs", f"gaifo_{args.env}_{args.demo_episodes}_2M")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME), LOG_DIR)])
    n_envs = env.num_envs  # usually 1
    SEED = args.seed

    # ------------------------------------------------------------------
    # Expert demonstration loading
    # ------------------------------------------------------------------
    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    demonstrations = np.load(demo_path, allow_pickle=True)
    demonstrations = demonstrations.tolist() if isinstance(demonstrations, np.ndarray) else demonstrations
    # Flatten demonstrations
    if demonstrations and hasattr(demonstrations[0], "obs"):
        demo_traj = np.concatenate([np.array(traj.obs, dtype=np.float32) for traj in demonstrations])
    else:
        demo_traj = np.array([np.array(x, dtype=np.float32) for x in demonstrations])
    # Create transitions from demonstrations
    expert_transitions = np.array([(demo_traj[i], demo_traj[i + 1]) for i in range(len(demo_traj) - 1)])
    expert_s = torch.tensor(np.stack(expert_transitions[:, 0]), dtype=torch.float32).to(device)
    expert_s_next = torch.tensor(np.stack(expert_transitions[:, 1]), dtype=torch.float32).to(device)

    flat_obs_dim = env.observation_space.shape[0]
    # Instantiate the learner
    learner = TRPO("MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log=LOG_DIR)
    learner.set_logger(configure(LOG_DIR, ["stdout", "tensorboard"]))
    # Observation space must be flattened for GAIfO
    obs = env.reset()[0]
    learner._last_obs = obs[None, :] if obs.ndim == 1 else obs
    learner.ep_info_buffer, learner.ep_success_buffer = [], []
    # Instantiate the discriminator
    discriminator = GAIfODiscriminator(flat_obs_dim).to(device)
    disc_lr =5.4868671601784924e-05
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))  # Adjusted learning rate
    bce_loss = nn.BCELoss()
    # Evaluate the policy before training
    pre_reward_mean = np.mean(evaluate_policy(learner, env, n_eval_episodes=10)[0])
    print(f"Mean reward before training: {pre_reward_mean:.2f}")
    writer.add_scalar("Reward/PreTrain", pre_reward_mean, 0)

    dummy_callback = DummyCallback()
    dummy_callback.model = learner

    # ------------------------------------------------------------------
    # Step‑based training loop
    # ------------------------------------------------------------------
    rollout_length = 2048  # steps per rollout
    total_steps_target = args.steps
    total_steps_so_far = 0
    num_iterations = math.ceil(total_steps_target / (rollout_length * n_envs))
    gp_lambda = 5.165201675071828 # gradient‑penalty coefficient

    for itr in range(num_iterations):
        # --------------------------------------------------------------
        # Collect rollout
        # --------------------------------------------------------------
        try:
            learner.collect_rollouts(
                learner.env,
                dummy_callback,
                learner.rollout_buffer,
                n_rollout_steps=rollout_length,
            )
        except RuntimeError as e:
            if "needs reset" in str(e):
                learner._last_obs = env.reset()[0][None, :]
                learner.collect_rollouts(
                    learner.env,
                    dummy_callback,
                    learner.rollout_buffer,
                    n_rollout_steps=rollout_length,
                )
            else:
                raise

        total_steps_so_far += rollout_length * n_envs
        print(
            f"Iteration {itr + 1}/{num_iterations}  |  Steps: {total_steps_so_far}/{total_steps_target}",
            flush=True,
        )
        # --------------------------------------------------------------
        # Prepare transitions for discriminator
        # --------------------------------------------------------------
        rollout_obs = learner.rollout_buffer.observations  # (T, n_envs, dim)
        last_obs = learner._last_obs[np.newaxis, ...]
        rollout_obs_next = np.concatenate([rollout_obs[1:], last_obs], axis=0)
        rollout_s = torch.tensor(rollout_obs.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)
        rollout_s_next = torch.tensor(rollout_obs_next.reshape(-1, flat_obs_dim), dtype=torch.float32).to(device)

        # --------------------------------------------------------------
        # Discriminator update
        # --------------------------------------------------------------
        disc_epochs, batch_size = 20, 512
        for _ in range(disc_epochs):
            idx_pol = np.random.choice(len(rollout_s), batch_size, replace=True)
            idx_exp = np.random.choice(len(expert_s), batch_size, replace=True)
            s_pol, s_pol_next = rollout_s[idx_pol], rollout_s_next[idx_pol]
            s_exp, s_exp_next = expert_s[idx_exp], expert_s_next[idx_exp]
            pred_pol = discriminator(s_pol, s_pol_next)
            pred_exp = discriminator(s_exp, s_exp_next)
            loss_pol = bce_loss(pred_pol, torch.ones_like(pred_pol))
            loss_exp = bce_loss(pred_exp, torch.zeros_like(pred_exp))
            gp = compute_gradient_penalty(discriminator, s_exp, s_pol, s_exp_next, s_pol_next, device)
            disc_loss = loss_pol + loss_exp + gp_lambda * gp
            disc_optimizer.zero_grad(); disc_loss.backward(); disc_optimizer.step()

        writer.add_scalar("Discriminator/Loss", disc_loss.item(), total_steps_so_far)

        # --------------------------------------------------------------
        # Compute and log synthetic rewards
        # --------------------------------------------------------------
        with torch.no_grad():
            rewards = -torch.log(discriminator(rollout_s, rollout_s_next) + 1e-8)
        avg_rollout_reward = rewards.mean().item()
        writer.add_scalar("Reward/Rollout", avg_rollout_reward, total_steps_so_far)

        learner.rollout_buffer.rewards = rewards.cpu().numpy().flatten()
        learner.train()

        # Evaluation every 10 iterations
        if (itr + 1) % 10 == 0 or total_steps_so_far >= total_steps_target:
            eval_mean = np.mean(evaluate_policy(learner, env, n_eval_episodes=10)[0])
            print(f"Evaluation @ {total_steps_so_far} steps → mean reward: {eval_mean:.2f}")
            writer.add_scalar("Reward/Evaluation", eval_mean, total_steps_so_far)
            writer.flush() 
        if total_steps_so_far >= total_steps_target:
            break  # safeguard, though loop limits should align

    # ------------------------------------------------------------------
    # Save & final evaluation
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, f"gaifo_{args.env}_{args.demo_episodes}_{total_steps_so_far}")
    learner.save(model_path)
    print(f"GAIfO model saved at {model_path}.zip")

    final_mean = np.mean(evaluate_policy(learner, env, n_eval_episodes=10)[0])
    print(f"Mean reward after training: {final_mean:.2f}")
    writer.add_scalar("Reward/Final", final_mean, total_steps_so_far)

    env.close(); writer.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_gaifo.py --env halfcheetah --steps 2000000 --seed 44 --demo_episodes 50")
    main()
