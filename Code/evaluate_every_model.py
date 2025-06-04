import os, re, glob, argparse, types, sys, warnings
import numpy as np, pandas as pd, torch
import gymnasium as gym
from collections import defaultdict
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from imitation.algorithms.bc import BC
from BCO.bco import PolicyNetwork
from SQIL.sqil_agent import SQILAgent
import inspect   #  ← nuevo
# ---------- suprimir mujoco_py -------------
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"]         = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion
warnings.filterwarnings("ignore", category=UserWarning)

ALG_TYPES = {
    "gail":  "sb3_trpo",
    "gaifo": "sb3_trpo",
    "airl":  "sb3_ppo",
    "bc":    "bc_torch",
    "bco":   "bco_torch",
    "sqil":  "sqil_torch",
}

ENV_MAP = {"halfcheetah": "HalfCheetah-v4",
           "cartpole":    "CartPole-v1"}

# ---------- wrapper con dispositivo coherente y determinismo ------------
class PolicyWrapper:
    def __init__(self, net, discrete):
        self.net, self.discrete = net, discrete

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if obs.ndim == 1:
            obs = obs[None, :]
        device = next(self.net.parameters()).device
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            sig = inspect.signature(self.net.forward)
            if "deterministic" in sig.parameters:
                out = self.net.forward(obs_t, deterministic=deterministic)
            else:
                out = self.net(obs_t)

            if isinstance(out, (tuple, list)):
                out = out[0]      # p. ej. (mu, log_std)

        act = (torch.argmax(out, dim=-1) if self.discrete else out).cpu().numpy()
        return act, None



def load_bc(model_path, env):
    # reconstruye la clase de política como en evaluate_bc.py
    dummy_demo = [{"obs": env.observation_space.sample(),
                   "acts": env.action_space.sample(),
                   "dones": False,
                   "next_obs": env.observation_space.sample(),
                   "infos": {}}]
    bc_tmp = BC(observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=dummy_demo,
                rng=np.random.default_rng(44))
    PolicyCls = bc_tmp.policy.__class__
    # carga pesos
    policy = PolicyCls(observation_space=env.observation_space,
                       action_space=env.action_space,
                       lr_schedule=lambda _: 1e-3)
    ckpt = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt.state_dict())
    return PolicyWrapper(policy, isinstance(env.action_space, gym.spaces.Discrete))

def load_bco(model_path, env):
    disc = isinstance(env.action_space, gym.spaces.Discrete)
    net  = PolicyNetwork(env.observation_space.shape[0],
                         env.action_space.n if disc else env.action_space.shape[0],
                         discrete=disc)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return PolicyWrapper(net, disc)

def load_sqil(model_dir, env):
    disc = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if disc else env.action_space.shape[0]
    action_range = 1.0 if disc else float(env.action_space.high[0])
    agent = SQILAgent(env.observation_space.shape[0], action_dim,
                      action_range=action_range, batch_size=256)
    torch.manual_seed(44)
    actor = glob.glob(os.path.join(model_dir, "*actor*.pth"))[0]
    critic = glob.glob(os.path.join(model_dir, "*critic*.pth"))[0]
    agent.actor.load_state_dict(torch.load(actor, map_location="cpu"))
    agent.actor.eval()
    return PolicyWrapper(agent.actor, disc)

# 3. FUNCIÓN de evaluación (sustituye la actual):
def eval_policy(policy, env_id, episodes):
    env = gym.make(env_id)
    env.reset(seed=44)
    np.random.seed(44)
    torch.manual_seed(44)
    mean, std = evaluate_policy(
        policy, env,
        n_eval_episodes=episodes,
        deterministic=True,
        render=False
    )
    env.close()
    return mean, std


def main(root, episodes):
    rows = []
    for sub in os.scandir(root):
        if not sub.is_dir(): continue
        m = re.match(r"(?P<alg>.+?)_(?P<env>.+?)_(?P<traj>\d+)", sub.name)
        if not m: continue
        alg, env_key, traj = m.group("alg"), m.group("env"), int(m.group("traj"))
        env_id = ENV_MAP.get(env_key, env_key)
        alg_type = ALG_TYPES.get(alg)
        if not alg_type:
            print("Algoritmo no soportado:", alg); continue

        # localizar archivos
        if alg_type.startswith("sb3"):
            model_file = glob.glob(os.path.join(sub.path, "*.zip"))[0]
            model = (TRPO if alg_type.endswith("trpo") else PPO).load(model_file, device="cpu")
            policy = model
        elif alg_type == "bc_torch":
            model_file = glob.glob(os.path.join(sub.path, "*.pt"))[0]
            policy = load_bc(model_file, gym.make(env_id))
        elif alg_type == "bco_torch":
            model_file = glob.glob(os.path.join(sub.path, "*.pt"))[0]
            policy = load_bco(model_file, gym.make(env_id))
        elif alg_type == "sqil_torch":
            policy = load_sqil(sub.path, gym.make(env_id))

        else:
            continue

        mean, std = eval_policy(policy, env_id, episodes)
        rows.append({"algoritmo": alg.upper(),
                     "trayectorias": traj,
                     "env": env_id,
                     "media": round(mean, 2),
                     "std": round(std, 2)})
        print(f"{alg.upper():5} | {traj:3d} traj | {mean:7.1f} ± {std:6.1f}")

    if not rows:
        print("No se evaluó ningún modelo."); return

    # ---------- exportar ---------------------------------------------------------
    df = pd.DataFrame(rows).sort_values(["algoritmo", "trayectorias"])
    excel_path = os.path.join(root, f"eval_results_{episodes}eps.xlsx")

    try:
        # intenta escribir a Excel
        df.to_excel(excel_path, index=False)
        print("\nResultados exportados a:", excel_path)

    except ModuleNotFoundError as err:
        if "openpyxl" in str(err):
            print("openpyxl no encontrado. Instalándolo con pip...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
            # reintenta
            df.to_excel(excel_path, index=False)
            print("\nResultados exportados a:", excel_path)
        else:
            # si falla por otro motivo, salvamos a CSV
            csv_path = excel_path.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False)
            print("\nNo se pudo crear el Excel; resultados guardados en:", csv_path)

    print("\nResultados exportados a:", excel_path)

# ------------------ CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="modelos_finales",
                        help="Carpeta con los modelos a evaluar")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodios de evaluación por modelo")
    args = parser.parse_args()
    main(args.root, args.episodes)
