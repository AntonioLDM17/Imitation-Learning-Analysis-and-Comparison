import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Ruta al archivo de logs generado por EvalCallback
LOG_DIR = "logs/sqil_v3"
EVAL_LOG_PATH = os.path.join(LOG_DIR, "evaluations.npz")
OUTPUT_IMAGE = os.path.join(LOG_DIR, "eval_plot.png")

# Cargar logs
if not os.path.exists(EVAL_LOG_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de evaluación en: {EVAL_LOG_PATH}")

data = np.load(EVAL_LOG_PATH)

# Extraer pasos de evaluación y recompensas
timesteps = data["timesteps"]
results = data["results"]  # shape: (n_evals, n_episodes)
mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(timesteps, mean_rewards, label="Mean Reward")
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="Std Dev")
plt.xlabel("Timesteps")
plt.ylabel("Evaluation Reward")
plt.title("SQIL Evaluation Performance Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostrar y guardar imagen
plt.savefig(OUTPUT_IMAGE)
print(f"Gráfico guardado en: {OUTPUT_IMAGE}")
plt.show()
