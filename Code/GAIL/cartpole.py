import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm

SEED = 42

# Crea el entorno vectorizado
env = make_vec_env(
    "CartPole-v1",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)

# Entrena un experto con PPO (si no tienes uno preentrenado)
expert = PPO(policy=MlpPolicy, env=env, seed=SEED)
expert.learn(100_000)

# Genera trayectorias del experto
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=np.random.default_rng(SEED),
)

# Crea el aprendiz (gen_algo)
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)

# Red de recompensa (discriminador)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# Entrenador GAIL
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# Evalúa antes
learner_rewards_before, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)

# Entrena GAIL
gail_trainer.train(200_000)

# Evalúa después
learner_rewards_after, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)

print("Recompensa media antes del entrenamiento:", np.mean(learner_rewards_before))
print("Recompensa media después del entrenamiento:", np.mean(learner_rewards_after))
