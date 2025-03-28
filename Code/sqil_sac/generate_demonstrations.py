import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from tqdm import trange

# Parameters
ENV_NAME = "HalfCheetah-v5"
NUM_STEPS = 100000  # Number of steps to generate

# Path to the newly trained expert model (v5)
MODEL_PATH = "sac_halfcheetah_v5_expert.zip"
OUTPUT_PATH = f"demonstrations_halfcheetah_v5_{NUM_STEPS}.npy"


# 10k steps to valida that the agent is learning from the expert
# 10k to 50k steps to have a stable learning curve and to have a good expert model
# 100k or more to reproduce the results in the original paper

# Load environment and expert model
env = gym.make(ENV_NAME)
model = SAC.load(MODEL_PATH)

transitions = []
obs, _ = env.reset()

for _ in trange(NUM_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    transitions.append((obs, action, 1.0, next_obs, done))  # SQIL uses reward = 1.0
    obs = next_obs

    if done:
        obs, _ = env.reset()

env.close()

# Save demonstrations
np.save(OUTPUT_PATH, np.array(transitions, dtype=object))
print(f"Demonstrations saved to {OUTPUT_PATH}.")
