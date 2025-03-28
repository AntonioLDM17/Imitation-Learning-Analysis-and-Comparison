import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os

# ---------- Custom Replay Buffer for SQIL ----------
class SQILReplayBuffer(ReplayBuffer):
    def __init__(self, *args, demo_transitions=None, lambda_samp=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_size = 0
        self.lambda_samp = lambda_samp
        if demo_transitions is not None:
            self._add_demonstrations(demo_transitions)

    def _add_demonstrations(self, demo_transitions):
        for obs, act, _, next_obs, done in demo_transitions:
            infos = [{"TimeLimit.truncated": False}]
            self.add(obs, next_obs, act, 1.0, done, infos)
        self.demo_size = len(demo_transitions)

    def add_agent_transition(self, obs, action, reward, next_obs, done):
        infos = [{"TimeLimit.truncated": False}]
        super().add(obs, next_obs, action, 0.0, done, infos)

    def sample(self, batch_size, env=None):
        demo_batch = int(batch_size * self.lambda_samp)
        agent_batch = batch_size - demo_batch
        demo_idxs = np.random.randint(0, self.demo_size, size=demo_batch)
        agent_idxs = np.random.randint(self.demo_size, self.size(), size=agent_batch)
        idxs = np.concatenate([demo_idxs, agent_idxs])
        return self._get_samples(idxs, env)

# ---------- Modified SAC for SQIL ----------
class SQILSAC(SAC):
    def __init__(self, *args, demo_transitions=None, lambda_samp=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = SQILReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            demo_transitions=demo_transitions,
            lambda_samp=lambda_samp
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        replay_buffer.add_agent_transition(self._last_obs, buffer_action, 0.0, new_obs, done)

# ---------- Paths and Parameters ----------
ENV_NAME = "HalfCheetah-v5"
DEMO_PATH = "demonstrations_halfcheetah_v5_100000.npy"
TOTAL_STEPS = 500_000
LAMBDA_SAMP = 0.9
EVAL_FREQ = 10_000
LOG_DIR = "logs/sqil_v3"
MODEL_PATH = "sqil_v3_halfcheetah"

# ---------- Load demonstrations ----------
demo_data = np.load(DEMO_PATH, allow_pickle=True)

# ---------- Environment setup with normalization and monitoring ----------
venv = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME))])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

# ---------- Evaluation environment ----------
eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_NAME))])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# ---------- Configure logger for TensorBoard ----------
new_logger = configure(LOG_DIR, ['stdout', 'tensorboard'])

# ---------- Callback for periodic evaluation ----------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False
)

# ---------- Train the model ----------
model = SQILSAC("MlpPolicy", venv, verbose=1, demo_transitions=demo_data, lambda_samp=LAMBDA_SAMP, tensorboard_log=LOG_DIR)
model.set_logger(new_logger)
model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
model.save(MODEL_PATH)
print(f"SQIL v3 model saved as '{MODEL_PATH}.zip'")

venv.close()
eval_env.close()