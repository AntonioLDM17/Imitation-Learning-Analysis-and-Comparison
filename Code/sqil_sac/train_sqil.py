import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

# ---------- Custom Replay Buffer for SQIL ----------
class SQILReplayBuffer(ReplayBuffer):
    def __init__(self, *args, demo_transitions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_size = 0
        if demo_transitions is not None:
            self._add_demonstrations(demo_transitions)

    def _add_demonstrations(self, demo_transitions):
        # Add expert demonstrations with reward = +1.0
        for obs, act, _, next_obs, done in demo_transitions:
            infos = [{"TimeLimit.truncated": False}]
            self.add(obs, next_obs, act, 1.0, done, infos)
        self.demo_size = len(demo_transitions)

    def add_agent_transition(self, obs, action, reward, next_obs, done):
        # Add agent's own experience with reward = 0.0
        infos = [{"TimeLimit.truncated": False}]
        super().add(obs, next_obs, action, 0.0, done, infos)

    def sample(self, batch_size, env=None):
        # Sample 50% expert and 50% agent transitions
        half = batch_size // 2
        demo_idxs = np.random.randint(0, self.demo_size, size=half)
        agent_idxs = np.random.randint(self.demo_size, self.size(), size=half)
        idxs = np.concatenate([demo_idxs, agent_idxs])
        return self._get_samples(idxs, env)

# ---------- SAC modified to behave like SQIL ----------
class SQILSAC(SAC):
    def __init__(self, *args, demo_transitions=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace standard replay buffer with SQIL version
        self.replay_buffer = SQILReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            demo_transitions=demo_transitions
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        # Store the agent's transition with reward = 0.0
        replay_buffer.add_agent_transition(self._last_obs, buffer_action, reward, new_obs, done)

# ---------- Load demonstrations and train ----------
ENV_NAME = "HalfCheetah-v5"
DEMO_PATH = "demonstrations_halfcheetah_v5_100000.npy"  # Path to the expert demonstrations
TOTAL_STEPS = 500_000

# Load expert demonstrations
demo_data = np.load(DEMO_PATH, allow_pickle=True)

# Create environment and initialize SQIL-SAC model
env = gym.make(ENV_NAME)
model = SQILSAC("MlpPolicy", env, verbose=1, demo_transitions=demo_data)

# Train using SQIL (Soft Q Imitation Learning)
model.learn(total_timesteps=TOTAL_STEPS)

MODEL_PATH = "sqil_halfcheetah_500k"
# Save the trained model
model.save(MODEL_PATH)
print(f"Trained SQIL model saved as '{MODEL_PATH}.zip'.")
# Close the environment
env.close()
