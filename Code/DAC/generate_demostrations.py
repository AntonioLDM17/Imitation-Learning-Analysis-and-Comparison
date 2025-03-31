import os
import sys
import types
import argparse

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
# You can add minimal attributes if needed; for now, we leave them empty.
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import numpy as np
import gymnasium as gym
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

def main():
    parser = argparse.ArgumentParser(description="Generate expert demonstrations using a selected policy algorithm")
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="halfcheetah",
                        help="Environment to use: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--policy", type=str, choices=["ppo", "trpo", "sac"], default="sac",
                        help="Expert policy algorithm used: ppo, trpo, or sac")
    parser.add_argument("--num_episodes", type=int, default=60, help="Number of episodes for demonstration generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    SEED = args.seed

    # Determinar nombre del entorno basado en el argumento
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    elif args.env == "halfcheetah":
        ENV_NAME = "HalfCheetah-v4"
    else:
        raise ValueError("The --env parameter must be 'cartpole' or 'halfcheetah'.")

    # Definir ruta del modelo experto basado en entorno y política
    if args.env == "cartpole":
        if args.policy.lower() == "trpo":
            EXPERT_MODEL_PATH = os.path.join("experts", "cartpole_expert_trpo.zip")
        elif args.policy.lower() == "ppo":
            EXPERT_MODEL_PATH = os.path.join("experts", "cartpole_expert.zip")
        else:
            raise ValueError("For CartPole, please use 'ppo' or 'trpo'.")
    elif args.env == "halfcheetah":
        if args.policy.lower() == "sac":
            EXPERT_MODEL_PATH = os.path.join("experts", "halfcheetah_expert_sac.zip")
        elif args.policy.lower() == "trpo":
            EXPERT_MODEL_PATH = os.path.join("experts", "halfcheetah_expert_trpo.zip")
        elif args.policy.lower() == "ppo":
            EXPERT_MODEL_PATH = os.path.join("experts", "halfcheetah_expert.zip")
        else:
            raise ValueError("Unsupported policy for halfcheetah.")

    DEMO_DIR = "demonstrations"
    DEMO_FILENAME = f"{args.env}_demonstrations.npy"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Crear entorno vectorizado
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
    )

    # Cargar modelo experto según el algoritmo seleccionado
    if args.env == "cartpole":
        if args.policy.lower() == "ppo":
            from stable_baselines3 import PPO
            expert = PPO.load(EXPERT_MODEL_PATH, env=env)
        elif args.policy.lower() == "trpo":
            from sb3_contrib import TRPO
            expert = TRPO.load(EXPERT_MODEL_PATH, env=env)
    elif args.env == "halfcheetah":
        if args.policy.lower() == "sac":
            from stable_baselines3 import SAC
            expert = SAC.load(EXPERT_MODEL_PATH, env=env)
        elif args.policy.lower() == "trpo":
            from sb3_contrib import TRPO
            expert = TRPO.load(EXPERT_MODEL_PATH, env=env)
        elif args.policy.lower() == "ppo":
            from stable_baselines3 import PPO
            expert = PPO.load(EXPERT_MODEL_PATH, env=env)

    demonstrations = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.num_episodes),
        rng=np.random.default_rng(SEED),
    )

    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    np.save(demo_path, np.array(demonstrations, dtype=object))
    print(f"Demonstrations saved to {demo_path}")
    env.close()

if __name__ == "__main__":
    print("Usage example:")
    print("python .\generate_demonstrations.py --env halfcheetah --policy sac --num_episodes 100 --seed 42")
    main()
