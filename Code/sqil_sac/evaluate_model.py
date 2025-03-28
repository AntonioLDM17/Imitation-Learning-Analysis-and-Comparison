import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Config
ENV_NAME = "HalfCheetah-v5"
MODEL_PATH = "sqil_halfcheetah_500k.zip" # Path to the SQIL model
# MODEL_PATH = "sac_halfcheetah_v5_expert.zip"  # Path to the expert model
N_EPISODES = 100  # Number of episodes for evaluation

# Load environment and model
env = gym.make(ENV_NAME)
model = SAC.load(MODEL_PATH)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=N_EPISODES,
    deterministic=True,
    render=False,
    return_episode_rewards=False
)

print(f"Evaluation over {N_EPISODES} episodes:")
print(f"  Mean reward: {mean_reward:.2f}")
print(f"  Std reward:  {std_reward:.2f}")
