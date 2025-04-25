import os, sys, types, argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import TRPO  # Using TRPO as in the original GAIL paper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout  # For generating rollouts if needed
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm

def main():
    parser = argparse.ArgumentParser(
        description="Train a GAIL model using TRPO as the RL algorithm."
    )
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment to use: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total timesteps to train GAIL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--policy", type=str, choices=["ppo", "trpo", "sac"], default="ppo",
                        help="Expert policy algorithm to use")
    args = parser.parse_args()

    SEED = args.seed

    # Determine environment name
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    elif args.env == "halfcheetah":
        ENV_NAME = "HalfCheetah-v4"
    else:
        raise ValueError("The --env parameter must be 'cartpole' or 'halfcheetah'.")

    TOTAL_TIMESTEPS = args.timesteps

    # Update paths based on new structure:
    DEMO_DIR = os.path.join("..", "data", "demonstrations")
    DEMO_FILENAME = f"{args.env}_demonstrations.npy"
    if args.env == "cartpole":
        if args.policy.lower() == "trpo":
            EXPERT_MODEL_PATH = os.path.join("..", "data", "experts", "cartpole_expert_trpo.zip")
        elif args.policy.lower() == "ppo":
            EXPERT_MODEL_PATH = os.path.join("..", "data", "experts", "cartpole_expert_ppo.zip")
        else:
            raise ValueError("For CartPole, please use 'ppo' or 'trpo' (SAC is not compatible with discrete actions).")
    elif args.env == "halfcheetah":
        if args.policy.lower() == "sac":
            EXPERT_MODEL_PATH = os.path.join("..", "data", "experts", "halfcheetah_expert_sac.zip")
        elif args.policy.lower() == "trpo":
            EXPERT_MODEL_PATH = os.path.join("..", "data", "experts", "halfcheetah_expert_trpo.zip")
        elif args.policy.lower() == "ppo":
            EXPERT_MODEL_PATH = os.path.join("..", "data", "experts", "halfcheetah_expert_ppo.zip")
        else:
            raise ValueError("Unsupported policy for halfcheetah.")
    else:
        raise ValueError("The --env parameter must be 'cartpole' or 'halfcheetah'.")

    MODELS_DIR = "models"
    MODEL_NAME = f"gail_{args.env}"
    LOG_DIR = os.path.join("logs", f"gail_{args.env}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a vectorized environment with RolloutInfoWrapper
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
    )

    # Load expert demonstrations from DEMO_DIR (the data folder)
    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Set up the learner (policy to be trained with TRPO) and the reward network (the discriminator)
    learner = TRPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Configure logger for TensorBoard
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    learner.set_logger(new_logger)

    # Create the GAIL trainer
    gail_trainer = GAIL(
        demonstrations=demonstrations,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Initialize collapse monitoring
    collapse_threshold = 0.0  # Set the threshold for collapse (e.g., -0.1 for HalfCheetah)
    collapse_counter = 0
    collapse_interval = 16384  # Minimum steps required before checking for collapse
    total_steps = 0

    # Optionally evaluate before training
    pre_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    print("Mean reward before training:", np.mean(pre_train_rewards))

    # Train GAIL
    for _ in range(TOTAL_TIMESTEPS // collapse_interval):
        gail_trainer.train(collapse_interval)
        total_steps += collapse_interval

        # Only start checking for collapse after 16,384 steps
        if total_steps >= 16384:
            # Monitor collapse based on discriminator's performance
            post_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)

            # Check if the reward falls below the threshold (collapse detected)
            if np.mean(post_train_rewards) < collapse_threshold:
                collapse_counter += 1
                print(f"Collapse detected at step {total_steps}, collapse count: {collapse_counter}")
                # Log the collapse event to TensorBoard
                new_logger.record("collapse_count", collapse_counter)

    # Save the trained GAIL model
    model_save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    learner.save(model_save_path)
    print(f"GAIL model saved at {model_save_path}.zip")

    # Evaluate after training
    post_train_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    print("Mean reward after training:", np.mean(post_train_rewards))
    env.close()

if __name__ == "__main__":
    print("Usage example:")
    print("python train_gail_collapse.py --env halfcheetah --timesteps 1000000 --seed 42")
    main()
