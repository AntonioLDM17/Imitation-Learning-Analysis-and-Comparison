import os, sys, types, argparse
import numpy as np
import torch
import gymnasium as gym
from sb3_contrib import TRPO  # Optional: keep for compatibility if needed
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

# Create dummy modules for "mujoco_py" to avoid compiling its extensions
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env


def main():
    parser = argparse.ArgumentParser(
        description="Train an AIRL model using pre-generated expert demonstrations"
    )
    parser.add_argument(
        "--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
        help="Environment to use: 'cartpole' or 'halfcheetah'"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total number of timesteps for AIRL training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--demo-batch-size", type=int, default=2048,
        help="Batch size of expert demonstrations for discriminator training"
    )
    parser.add_argument(
        "demo_episodes", type=int, default=50,
        help="Number of expert episodes to use for training"
    )
    args = parser.parse_args()

    SEED = args.seed
    TOTAL_TIMESTEPS = args.timesteps

    # Data and output directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_DIR = os.path.join(BASE_DIR, os.pardir, "data", "demonstrations", str(args.demo_episodes))
    os.makedirs(DEMO_DIR, exist_ok=True)
    DEMO_FILE = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODELS_DIR = os.path.join(BASE_DIR, "models", f"airl_{args.env}_{args.demo_episodes}")
    LOG_DIR = os.path.join(BASE_DIR, "logs", f"airl_{args.env}_{args.demo_episodes}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Configure environment
    if args.env == "cartpole":
        ENV_NAME = "CartPole-v1"
    else:
        ENV_NAME = "HalfCheetah-v4"

    # Load expert demonstrations
    demo_path = os.path.join(DEMO_DIR, DEMO_FILE)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Vectorized environment with rollout info wrapper
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Generator algorithm (PPO)
    learner = PPO(
        "MlpPolicy",
        env,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=5e-4,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    # TensorBoard logger
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    learner.set_logger(new_logger)

    # Reward network (discriminator)
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Instantiate AIRL trainer
    airl_trainer = AIRL(
        demonstrations=demonstrations,
        demo_batch_size=args.demo_batch_size,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        log_dir=LOG_DIR,
        init_tensorboard=True,            # Enable TensorBoard logging for discriminator
        init_tensorboard_graph=False       # Disable graph logging to reduce startup overhead
    )

    # Pre-training evaluation
    pre_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    print(f"Mean reward before training: {np.mean(pre_rewards)}")

    # Train AIRL
    airl_trainer.train(TOTAL_TIMESTEPS)

    # Save trained models
    learner.save(os.path.join(MODELS_DIR, f"airl_{args.env}_{args.timesteps}"))
    torch.save(reward_net.state_dict(), os.path.join(MODELS_DIR, f"airl_reward_{args.env}_{args.timesteps}.pth"))

    # Post-training evaluation
    post_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    print(f"Mean reward after training: {np.mean(post_rewards)}")

    env.close()


if __name__ == "__main__":
    print("Example usage: python train_airl.py --env cartpole --timesteps 2_000_000 --seed 42")
    print("Example usage: python train_airl.py --env halfcheetah --timesteps 2_000_000 --seed 42")
    print("To view the training process, run: tensorboard --logdir logs")
    main()
