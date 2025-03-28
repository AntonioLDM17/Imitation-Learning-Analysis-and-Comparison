import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Config
ENV_NAME = "HalfCheetah-v5"
TIMESTEPS = 1_000_000
MODEL_PATH = "sac_halfcheetah_v5_expert"

# Create environment
env = gym.make(ENV_NAME)

# Define the model
model = SAC("MlpPolicy", env, verbose=1)

# Optional: save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints",
    name_prefix="sac_v5_checkpoint"
)

# Train the expert model
model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

# Save the final trained model
model.save(MODEL_PATH)
print(f"Expert SAC model saved to '{MODEL_PATH}.zip'")