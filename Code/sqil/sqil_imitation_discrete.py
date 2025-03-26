# SQIL usando la librería `imitation` con CartPole

import sys, types, os, copy
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# Dummy mujoco_py
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


import datasets
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import sqil
from imitation.data import huggingface_utils
from imitation.data import rollout

# Descargar expert trajectories desde HuggingFace (puedes cambiar el entorno)
dataset = datasets.load_dataset("HumanCompatibleAI/ppo-CartPole-v1")
expert_trajectories = huggingface_utils.TrajectoryDatasetSequence(dataset["train"])

# Ver estadísticas de las trayectorias
stats = rollout.rollout_stats(expert_trajectories)
print(f"Trajectories: {stats['n_traj']}, Length: {stats['len_mean']}, Return: {stats['return_mean']}")

# Crear entorno vectorizado
venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# Inicializar el entrenador SQIL
trainer = sqil.SQIL(
    venv=venv,
    demonstrations=expert_trajectories,
    policy="MlpPolicy",
)

# Evaluar antes de entrenar
reward_before, _ = evaluate_policy(trainer.policy, venv, n_eval_episodes=10)
print(f"Reward antes de entrenar: {reward_before:.2f}")

# Entrenar el modelo
trainer.train(total_timesteps=100_000)

# Evaluar después de entrenar
reward_after, _ = evaluate_policy(trainer.policy, venv, n_eval_episodes=10)
print(f"Reward después de entrenar: {reward_after:.2f}")
