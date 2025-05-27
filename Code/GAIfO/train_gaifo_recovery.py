"""
train_gaifo_recovery.py · GAIfO re-training with rescued TRPO hyper-params
--------------------------------------------------------------------------
Ejemplo de uso
--------------
python train_gaifo_recovery.py \
       --env halfcheetah \
       --steps 614400 \
       --demo_episodes 100 \
       --seed 44 \
       --checkpoint gaifo_halfcheetah_…_iterations300.zip
"""
import os, sys, types, argparse, json, pickle, zipfile, pathlib, inspect
import numpy as np
import gymnasium as gym

from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor  import Monitor
from stable_baselines3.common.logger   import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ─────────────────────  Mujoco stub  ─────────────────────
dummy = types.ModuleType("mujoco_py")
dummy.builder   = types.ModuleType("mujoco_py.builder")
dummy.locomotion= types.ModuleType("mujoco_py.locomotion")
sys.modules.update({"mujoco_py": dummy,
                    "mujoco_py.builder": dummy.builder,
                    "mujoco_py.locomotion": dummy.locomotion})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────── Utility ──────────────────
def load_trpo_hyperparams(zip_path: pathlib.Path) -> dict | None:
    """Extrae el bloque de hiper-parámetros TRPO de *cualquier* checkpoint SB3."""
    for candidate in ("data/params.pkl", "data.pkl", "data"):
        with zipfile.ZipFile(zip_path) as zf:
            if candidate in zf.namelist():
                raw = zf.read(candidate)
                break
    else:
        print(f"⚠  {zip_path} no contiene metadatos SB3 reconocibles", file=sys.stderr)
        return None

    try:
        meta = pickle.loads(raw)
    except Exception:
        try:
            meta = json.loads(raw.decode())
        except Exception:
            print("⚠  No se puede decodificar metadatos", file=sys.stderr)
            return None

    def dig(d: dict):
        if not isinstance(d, dict):
            return None
        if "hyperparameters" in d:
            return d["hyperparameters"]
        if "params" in d:
            return d["params"]
        if {"n_steps", "learning_rate", "gamma"} <= d.keys():
            return d
        return dig(d.get("data")) if "data" in d else None

    return dig(meta)

# ──────────────────  Componentes GAIfO  ───────────────────
class DummyCallback(BaseCallback):
    def _on_step(self):           return True
    def _on_rollout_start(self):  pass
    def _on_rollout_end(self):    pass

class GAIfODiscriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
            nn.Linear(hidden, 1),           nn.Sigmoid()
        )
    def forward(self, s, sn):
        return self.net(torch.cat([s, sn], dim=1))

def grad_penalty(disc, s_e, s_p, sn_e, sn_p):
    alpha = torch.rand(s_e.size(0), 1, device=device)
    sh, snh = alpha * s_e + (1 - alpha) * s_p, alpha * sn_e + (1 - alpha) * sn_p
    sh.requires_grad_(True); snh.requires_grad_(True)
    d_hat = disc(sh, snh)
    grad = torch.autograd.grad(
        d_hat, [sh, snh], torch.ones_like(d_hat),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    grad = torch.cat(grad, dim=1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

# ─────────────────────────  MAIN  ─────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--env", choices=["cartpole", "halfcheetah"], default="cartpole")
    pa.add_argument("--steps", type=int, default=300 * 2048)
    pa.add_argument("--seed", type=int, default=44)
    pa.add_argument("--demo_episodes", type=int, default=50)
    pa.add_argument("--checkpoint", type=pathlib.Path)
    args = pa.parse_args()

    # 1) Hiper-parámetros base
    trpo_kwargs = dict(
        n_steps=2048, batch_size=64, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, cg_max_steps=10, cg_damping=0.1,
        target_kl=0.01, n_critic_updates=10, normalize_advantage=True
    )

    # 1-bis) Mezclar con los rescatados
    if args.checkpoint:
        rescued = load_trpo_hyperparams(args.checkpoint)
        if rescued:
            print("✔ Rescued TRPO kwargs from checkpoint:")
            for k, v in rescued.items():
                print(f"   {k:<18}: {v}")
            trpo_kwargs.update(rescued)
        else:
            print("⚠  No fue posible rescatar hiper-parámetros.")
    else:
        print("⚠  No checkpoint proporcionado – se usan defaults.")

    # ── Filtrado: sólo claves aceptadas por la versión actual de TRPO ──
    sig = inspect.signature(TRPO.__init__)
    accepted = set(sig.parameters) - {"policy", "env"}
    filtered = {k: v for k, v in trpo_kwargs.items() if k in accepted}

    # Quitar duplicados
    for dup in ("seed", "verbose", "tensorboard_log"):
        filtered.pop(dup, None)

    # Quitar rollout_buffer_class si no es realmente una clase/función
    if not callable(filtered.get("rollout_buffer_class", None)):
        filtered.pop("rollout_buffer_class", None)

    dropped = set(trpo_kwargs) - set(filtered)
    if dropped:
        print("ℹ  Ignored keys not in current TRPO:", ", ".join(dropped))

    # 2) Rutas
    ENV_NAME = "HalfCheetah-v4" if args.env == "halfcheetah" else "CartPole-v1"
    tag = f"{args.env}_{args.demo_episodes}eps_{args.steps}steps_seed{args.seed}"
    DEMO_DIR = os.path.join("..", "data", "demonstrations")
    DEMO_FILE = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    LOG_DIR = f"logs/gaifo_{tag}"
    MODEL_DIR = f"models/gaifo_{tag}"
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)
    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME), LOG_DIR)])

    # 3) Cargar demostraciones
    demos = np.load(os.path.join(DEMO_DIR, DEMO_FILE), allow_pickle=True)
    demos = demos.tolist() if isinstance(demos, np.ndarray) else demos
    demo_tr = (np.concatenate([np.asarray(t.obs, np.float32) for t in demos])
               if demos and hasattr(demos[0], "obs")
               else np.array([np.asarray(d, np.float32) for d in demos]))
    pairs = np.array([(demo_tr[i], demo_tr[i + 1]) for i in range(len(demo_tr) - 1)])
    exp_s, exp_sn = (torch.tensor(np.stack(pairs[:, i]), dtype=torch.float32, device=device)
                     for i in (0, 1))

    # 4) Learner & Discriminator
    learner = TRPO(
        "MlpPolicy", env,
        seed=args.seed, verbose=1, tensorboard_log=LOG_DIR, **filtered
    )
    learner.set_logger(configure(LOG_DIR, ["stdout", "tensorboard"]))
    learner._last_obs = env.reset()[0][None, :]

    # ←────────────────────────── MODIFICACIÓN clave ──────────────────────────→
    learner._setup_learn(total_timesteps=args.steps)
    # ↑ crea ep_info_buffer, ep_success_buffer, LR scheduler, etc.             ↑
    # -------------------------------------------------------------------------

    disc = GAIfODiscriminator(env.observation_space.shape[0]).to(device)
    # disc_opt = optim.Adam(disc.parameters(), lr=3.989e-5)
    disc_lr = 3.989020006157259e-05 # Adjusted learning rate for the discriminator
    disc_opt = optim.Adam(disc.parameters(), lr=disc_lr, betas=(0.9, 0.999))
    bce = nn.BCELoss()
    cb = DummyCallback(); cb.model = learner

    rollout_len = filtered["n_steps"]
    gp_lambda = 1.660941233998641
    done = 0

    # 5) Training loop
    while done < args.steps:
        learner.collect_rollouts(env, cb, learner.rollout_buffer, rollout_len)
        done += rollout_len

        ro = learner.rollout_buffer.observations
        ro_next = np.concatenate([ro[1:], learner._last_obs[None, ...]], axis=0)
        pol_s = torch.tensor(ro.reshape(-1, ro.shape[-1]), dtype=torch.float32, device=device)
        pol_sn = torch.tensor(ro_next.reshape(-1, ro_next.shape[-1]), dtype=torch.float32, device=device)

        for _ in range(10):
            idx_p = np.random.choice(len(pol_s), 256)
            idx_e = np.random.choice(len(exp_s), 256)
            s_p, sn_p = pol_s[idx_p], pol_sn[idx_p]
            s_e, sn_e = exp_s[idx_e], exp_sn[idx_e]
            loss = (bce(disc(s_p, sn_p), torch.ones_like(s_p[:, :1])) +
                    bce(disc(s_e, sn_e), torch.zeros_like(s_e[:, :1])) +
                    gp_lambda * grad_penalty(disc, s_e, s_p, sn_e, sn_p))
            disc_opt.zero_grad(); loss.backward(); disc_opt.step()

        with torch.no_grad():
            learner.rollout_buffer.rewards = (
                -torch.log(disc(pol_s, pol_sn) + 1e-8).cpu().numpy().flatten())

        learner.train()

        if done % (10 * rollout_len) == 0 or done >= args.steps:
            r = np.mean(evaluate_policy(learner, env, 10, True)[0])
            print(f"[{done}/{args.steps}] eval reward = {r:.1f}")
            writer.add_scalar("Reward/Eval", r, done)
        writer.add_scalar("Discriminator/Loss", loss.item(), done)

    # 6) Guardar
    out = os.path.join(MODEL_DIR, f"gaifo_{done // 1000}k_gamma")
    learner.save(out)
    print(f"✓  Modelo guardado en {out}.zip")
    writer.close(); env.close()

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
