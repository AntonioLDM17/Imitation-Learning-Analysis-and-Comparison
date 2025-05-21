"""
optuna_gaifo.py · Grid-aware GAIfO HPO (interrupt-safe, bug-fixed)
-----------------------------------------------------------------
• demo_size   ∈ {5, 10, 15, 20, 50, 100}
• total_steps ∈ {100 k, 250 k, 500 k, 1 M, 2 M}

Revision history
----------------
* **v2 (2025-05-21)** – fixed dimension mismatch for `ro_next`
* **v3 (this file)** – fixed `AttributeError: '_logger'` by installing
  an explicit SB3 logger right after creating each TRPO instance.

All other logic and search space remain unchanged.
"""

import os, sys, types, argparse, math, json
import numpy as np
import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure            # ← use for logger fix
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback
import optuna
from optuna.trial import TrialState

# ------------------------------------------------------------------ #
#  Dummy mujoco_py stubs                                             #
# ------------------------------------------------------------------ #
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules.update({
    "mujoco_py": dummy,
    "mujoco_py.builder": dummy.builder,
    "mujoco_py.locomotion": dummy.locomotion
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ #
#  Helper classes                                                    #
# ------------------------------------------------------------------ #
class DummyCallback(BaseCallback):
    def _on_step(self): return True
    def _on_rollout_start(self): pass
    def _on_rollout_end(self): pass


class GAIfODiscriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
            nn.Linear(hidden, 1),           nn.Sigmoid()
        )
    def forward(self, s, s_next):
        return self.net(torch.cat([s, s_next], dim=1))


def gradient_penalty(disc, s_e, s_p, sn_e, sn_p):
    alpha = torch.rand(s_e.size(0), 1, device=device)
    s_hat, sn_hat = alpha * s_e + (1 - alpha) * s_p, alpha * sn_e + (1 - alpha) * sn_p
    s_hat.requires_grad_(True); sn_hat.requires_grad_(True)
    d_hat = disc(s_hat, sn_hat)
    ones = torch.ones_like(d_hat)
    grads = torch.autograd.grad(d_hat, [s_hat, sn_hat], ones,
                                create_graph=True, retain_graph=True, only_inputs=True)
    grad = torch.cat(grads, dim=1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

# ------------------------------------------------------------------ #
#  Optuna objective                                                  #
# ------------------------------------------------------------------ #
def objective(trial: optuna.Trial):
    # Grid dimensions
    demo_size   = trial.suggest_categorical("demo_size", [5, 10, 20, 50, 100])
    total_steps = trial.suggest_categorical("total_steps", [100_000, 250_000,
                                                            500_000, 1_000_000,
                                                            2_000_000])

    # Tunable GAIfO hyper-parameters
    disc_epochs = trial.suggest_categorical("disc_epochs", [3, 5, 7, 10])
    batch_size  = trial.suggest_categorical("batch_size",  [128, 256, 512])
    lambda_gp   = trial.suggest_float      ("lambda_gp",   1.0, 20.0, log=True)
    disc_lr     = trial.suggest_float      ("disc_lr",     1e-5, 1e-3, log=True)

    rollout_len = 2048
    SEED = 42

    # Demo path helpers
    suffix   = "halfcheetah" if args.env.startswith("HalfCheetah") else "cartpole"
    demo_dir = os.path.join("..", "data", "demonstrations", str(demo_size))
    demo_file = os.path.join(demo_dir, f"{suffix}_demonstrations_{demo_size}.npy")

    # Environment & demonstrations
    env = DummyVecEnv([lambda: Monitor(gym.make(args.env), None)])
    demos = np.load(demo_file, allow_pickle=True).tolist()
    if demos and hasattr(demos[0], "obs"):
        demos = np.concatenate([np.array(t.obs, dtype=np.float32) for t in demos])
    else:
        demos = np.array([np.array(s, dtype=np.float32) for s in demos])

    exp_pairs = np.array([(demos[i], demos[i + 1]) for i in range(len(demos) - 1)])
    exp_s, exp_sn = (torch.tensor(np.stack(exp_pairs[:, i]), dtype=torch.float32).to(device)
                     for i in (0, 1))

    # ------------------------------------------------------------------ #
    #  TRPO learner with explicit logger (fixes AttributeError)          #
    # ------------------------------------------------------------------ #
    learner = TRPO("MlpPolicy", env, seed=SEED, verbose=0)
    learner.ep_info_buffer, learner.ep_success_buffer = [], []

    # Install a minimal SB3 logger so learner.train() can record scalars
    tmp_log = configure(folder=None, format_strings=[])  # log to memory only
    learner.set_logger(tmp_log)

    obs = env.reset()[0]
    learner._last_obs = obs[None, :]  # (1, obs_dim)

    # Discriminator
    obs_dim = env.observation_space.shape[0]
    discrim = GAIfODiscriminator(obs_dim).to(device)
    d_opt   = optim.Adam(discrim.parameters(), lr=disc_lr)
    bce     = nn.BCELoss()
    cb      = DummyCallback(); cb.model = learner

    # Step-based training loop
    steps_per_iter = rollout_len * env.num_envs
    steps_done = 0
    for _ in range(math.ceil(total_steps / steps_per_iter)):
        learner.collect_rollouts(env, cb, learner.rollout_buffer, rollout_len)
        steps_done += steps_per_iter

        ro = learner.rollout_buffer.observations                  # (T, 1, obs_dim)
        last_obs_exp = learner._last_obs[np.newaxis, ...]         # (1, 1, obs_dim)
        ro_next = np.concatenate([ro[1:], last_obs_exp], axis=0)

        pol_s  = torch.tensor(ro.reshape(-1, obs_dim),      dtype=torch.float32).to(device)
        pol_sn = torch.tensor(ro_next.reshape(-1, obs_dim), dtype=torch.float32).to(device)

        # Discriminator update
        for _ in range(disc_epochs):
            idx_p = np.random.choice(pol_s.size(0), batch_size, True)
            idx_e = np.random.choice(exp_s.size(0),  batch_size, True)
            s_p, sn_p = pol_s[idx_p],  pol_sn[idx_p]
            s_e, sn_e = exp_s[idx_e],  exp_sn[idx_e]

            d_p, d_e = discrim(s_p, sn_p), discrim(s_e, sn_e)
            loss = (bce(d_p, torch.ones_like(d_p)) +
                    bce(d_e, torch.zeros_like(d_e)) +
                    lambda_gp * gradient_penalty(discrim, s_e, s_p, sn_e, sn_p))
            d_opt.zero_grad(); loss.backward(); d_opt.step()

        # Policy update
        with torch.no_grad():
            rewards = -torch.log(discrim(pol_s, pol_sn) + 1e-8).cpu().numpy().flatten()
        learner.rollout_buffer.rewards = rewards
        learner.train()

        if steps_done >= total_steps:
            break

    env.close()
    eval_reward = np.mean(evaluate_policy(
        learner, DummyVecEnv([lambda: gym.make(args.env)]), 10, True)[0]
    )
    return eval_reward

# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrupt-safe GAIfO Optuna HPO")
    parser.add_argument("--n_trials", type=int, default=240, help="Optuna trial budget")
    parser.add_argument("--env",     type=str, default="HalfCheetah-v4", help="Gym env id")
    args = parser.parse_args()

    storage = f"sqlite:///gaifo_hpo_{args.env}.db"
    study = optuna.create_study(direction="maximize",
                                study_name="gaifo_hpo",
                                storage=storage,
                                load_if_exists=True)

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted – summarising results…")

    # Best params per (demo_size, steps)
    best = {}
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        key = (t.params["demo_size"], t.params["total_steps"])
        if key not in best or t.value > best[key]["value"]:
            best[key] = {
                "value":  t.value,
                "params": {k: t.params[k] for k in
                           ("disc_epochs", "batch_size", "lambda_gp", "disc_lr")},
                "trial":  t.number
            }

    with open("best_gaifo_params.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n🏆  Best hyper-parameters per (demo_size, steps):")
    for (ds, st), info in sorted(best.items()):
        p = info["params"]
        print(f"  demos={ds:3d}, steps={st:7,d} → reward={info['value']:.2f} (trial {info['trial']})")
        print(f"     disc_epochs={p['disc_epochs']}, batch_size={p['batch_size']}, "
              f"lambda_gp={p['lambda_gp']:.3g}, disc_lr={p['disc_lr']:.3g}")
