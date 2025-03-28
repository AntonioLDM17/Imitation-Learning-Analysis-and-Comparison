import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

# ---------- Custom Replay Buffer for SQIL with λsamp and fixed demos ----------
class SQILReplayBuffer(ReplayBuffer):
    def __init__(self, *args, demo_transitions=None, lambda_samp=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_size = 0
        self.lambda_samp = lambda_samp
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
        # Adjust sampling ratio based on lambda_samp
        demo_batch = int(batch_size * self.lambda_samp)
        agent_batch = batch_size - demo_batch
        demo_idxs = np.random.randint(0, self.demo_size, size=demo_batch)
        agent_idxs = np.random.randint(self.demo_size, self.size(), size=agent_batch)
        idxs = np.concatenate([demo_idxs, agent_idxs])
        return self._get_samples(idxs, env)

# ---------- Modified SAC for SQIL ----------
class SQILSAC(SAC):
    def __init__(self, *args, demo_transitions=None, lambda_samp=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # Use custom SQIL buffer
        self.replay_buffer = SQILReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            demo_transitions=demo_transitions,
            lambda_samp=lambda_samp
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        # Override reward to 0 for agent transitions (SQIL)
        replay_buffer.add_agent_transition(self._last_obs, buffer_action, 0.0, new_obs, done)

# ---------- Load expert data and train ----------
ENV_NAME = "HalfCheetah-v5"
DEMO_PATH = "demonstrations_halfcheetah_v5_100000.npy"
TOTAL_STEPS = 1_000_000
LAMBDA_SAMP = 0.9  # Use 90% demo transitions in early training

# Load demonstrations
demo_data = np.load(DEMO_PATH, allow_pickle=True)

# Create environment and model
env = gym.make(ENV_NAME)
model = SQILSAC("MlpPolicy", env, verbose=1, demo_transitions=demo_data, lambda_samp=LAMBDA_SAMP)

# Train
model.learn(total_timesteps=TOTAL_STEPS)

# Save
MODEL_PATH = "sqil_halfcheetah_v2"
model.save(MODEL_PATH)
print(f"Trained SQIL v2 model saved as '{MODEL_PATH}.zip'")

env.close()
