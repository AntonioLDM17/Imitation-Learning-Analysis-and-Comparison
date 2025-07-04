# optuna_bco.py  – versión corregida
# ---------------------------------------------------------
#  Búsqueda de hiperparámetros con Optuna para BCO
#  • Mantiene la constraint de ≈2 M global steps
#  • Usa el mismo intérprete que la sesión actual
# ---------------------------------------------------------
import os                       #  ←  NUEVO (faltaba)
import subprocess
import json
import re
import pathlib
import optuna
import sys
from datetime import datetime
from typing import Tuple, Union

import numpy as np

ROOT   = pathlib.Path(__file__).resolve().parent
DATA   = ROOT.parent / "data" / "demonstrations"
TRAIN  = ROOT / "train_bco.py"
PYTHON = sys.executable         # active Python interpreter

STEP_RE   = re.compile(r"Global steps\s*:\s*([\d,]+)")
REWARD_RE = re.compile(r"Final reward\s*:\s*([-\d\.]+)")

TARGET_STEPS = 2_000_000
TOLERANCE    = 20_000          # ±20 k
MIN_STEPS    = TARGET_STEPS - TOLERANCE
MAX_STEPS    = TARGET_STEPS + TOLERANCE


# Count the number of transitions in demonstration episodes
def count_demo_transitions(env: str, demo_episodes: int) -> int:
    """
    Returns the total number of transitions in the demonstration episodes
    for the specified environment and number of episodes.
    """
    demo_path = DATA / str(demo_episodes) / f"{env}_demonstrations_{demo_episodes}.npy"
    if not demo_path.is_file():
        # Does not exist, return a default value
        return 110_000
    arr = np.load(demo_path, allow_pickle=True)
    # The array can be a numpy array of objects (Trajectories)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.tolist()
    if isinstance(arr, list):
        # each element has an 'obs' attribute (TrajectoryWithRew)
        total = 0
        for traj in arr:
            total += len(traj.obs) if hasattr(traj, "obs") else len(traj)
        return total
    # return the states count
    return len(arr)


def print_steps_summary(
    N: int,
    I_pre: int,
    alpha: float,
    T: int,
    E: int,
    R: int,
    env_steps: int,
    bc_steps: int,
    total_steps: int
) -> None:
    """Prints the summary of steps and parameters."""
    fmt_int = lambda x: f"{x:,}".replace(",", " ")
    print("=== STEPS SUMMARY ===")
    print(f"  N (demo transitions)       = {fmt_int(N)}")
    print(f"  I_pre                      = {fmt_int(I_pre)}")
    print(f"  alpha                      = {alpha}")
    print(f"  num_iterations (T)         = {T}")
    print(f"  policy_epochs (E)          = {E}")
    print(f"  iter_policy_epochs (R)     = {R}")
    print(f"  → env_steps esperado       = {fmt_int(env_steps)}")
    print(f"  → bc_steps esperado        = {fmt_int(bc_steps)}")
    print(f"  → total_global_steps ≈     = {fmt_int(total_steps)}\n")


def run_train_bco(args_dict: dict) -> Tuple[int, float]:
    """
    Executes the train_bco.py script with the provided arguments.
    """
    cmd = [PYTHON, str(TRAIN)]
    for k, v in args_dict.items():
        cmd += [f"--{k}", str(v)]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )

    log_txt = proc.stdout
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    (ROOT / "optuna_logs").mkdir(exist_ok=True)
    (ROOT / "optuna_logs" / f"log_{ts}.txt").write_text(log_txt)

    if proc.returncode != 0:
        raise RuntimeError(f"train_bco ended with code {proc.returncode}")

    m_steps = STEP_RE.search(log_txt)
    m_rew   = REWARD_RE.search(log_txt)
    if not m_steps or not m_rew:
        raise RuntimeError("No metrics found in output log.")

    steps  = int(m_steps.group(1).replace(",", ""))
    reward = float(m_rew.group(1))
    return steps, reward


# Optuna objective function
def objective(trial: optuna.Trial) -> float:
    # Hiperparameters to optimize
    pre_inter   = trial.suggest_int("pre_interactions", 50_000, 300_000, step=25_000)
    alpha       = trial.suggest_float("alpha", 0.0, 0.5, step=0.05)
    iterations  = trial.suggest_int("num_iterations", 0, 4)
    inv_epochs  = trial.suggest_int("inv_epochs", 8, 20)
    pol_epochs  = trial.suggest_int("policy_epochs", 5, 30)
    it_inv_ep   = trial.suggest_int("iter_inv_epochs", 2, 10)
    it_pol_ep   = trial.suggest_int("iter_policy_epochs", 4, 20)
    pol_lr      = trial.suggest_float("policy_lr", 5e-4, 3e-3, log=True)
    demo_eps    = 100   # Fixed for all trials

    # Previous steps computation
    N = count_demo_transitions("halfcheetah", demo_eps)

    env_steps = int(pre_inter * (1 + alpha * iterations))
    bc_steps  = int(N * (pol_epochs + it_pol_ep * iterations))
    total_est = env_steps + bc_steps

    print_steps_summary(N, pre_inter, alpha, iterations,
                        pol_epochs, it_pol_ep,
                        env_steps, bc_steps, total_est)

    # Timesteps constraint
    if not (MIN_STEPS <= total_est <= MAX_STEPS):
        raise optuna.exceptions.TrialPruned(
            f"Previstos {total_est} pasos -> fuera de rango [{MIN_STEPS}, {MAX_STEPS}]"
        )

    # Args for train_bco.py
    args = dict(
        env="halfcheetah",
        pre_interactions = pre_inter,
        alpha            = alpha,
        num_iterations   = iterations,
        inv_epochs       = inv_epochs,
        policy_epochs    = pol_epochs,
        iter_inv_epochs  = it_inv_ep,
        iter_policy_epochs = it_pol_ep,
        policy_lr        = pol_lr,
        demo_episodes    = demo_eps,
        seed             = 44
    )
    try:
        steps_done, reward = run_train_bco(args)
    except Exception as exc:
        raise optuna.exceptions.TrialPruned(str(exc))

    trial.set_user_attr("steps_real", steps_done)
    return reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                study_name="BCO_hypersearch")
    study.optimize(objective, n_trials=300)

    print("\n========== BEST RESULT ==========")
    print("Reward :", study.best_value)
    best = study.best_trial
    print("Steps previstos:", best.user_attrs.get("steps", "–"))
    print("Steps reales   :", best.user_attrs.get("steps_real", "–"))
    print("Config :")
    print(json.dumps(best.params, indent=2))
