import os
import sys
import types
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Gym and SB3 imports
import gymnasium as gym
from sb3_contrib import TRPO  # Optional: keep for compatibility if needed
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure as sb3_configure

# Imitation library imports
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util.logger import configure as il_configure

# Create dummy modules for "mujoco_py" to avoid compiling its extensions
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def main():
    parser = argparse.ArgumentParser(
        description="Train an AIRL model using pre-generated expert demonstrations"
    )
    parser.add_argument(
        "--env", choices=["cartpole", "halfcheetah"], default="cartpole",
        help="Environment: 'cartpole' or 'halfcheetah'"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total timesteps for AIRL training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--demo-batch-size", type=int, default=2048,
        help="Expert demo batch size for discriminator"
    )
    args = parser.parse_args()

    SEED = args.seed
    TOTAL_TIMESTEPS = args.timesteps

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_DIR = os.path.join(BASE_DIR, os.pardir, "data", "demonstrations")
    DEMO_FILE = f"{args.env}_demonstrations.npy"
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs", f"airl_{args.env}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Environment
    ENV_NAME = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"

    # Load expert demonstrations
    demo_path = os.path.join(DEMO_DIR, DEMO_FILE)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # Create vectorized environment with RolloutInfoWrapper
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # SB3 PPO generator with TensorBoard logging
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

    # Configure SB3 logger
    sb3_logger = sb3_configure(LOG_DIR, ["stdout", "tensorboard"])
    learner.set_logger(sb3_logger)

    # Configure Imitation HierarchicalLogger (positional args)
    il_logger = il_configure(LOG_DIR, ["stdout", "tensorboard"])

    # Reward network (discriminator)
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Instantiate AIRL trainer with HierarchicalLogger
    airl_trainer = AIRL(
        demonstrations=demonstrations,
        demo_batch_size=args.demo_batch_size,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        init_tensorboard=True,
        init_tensorboard_graph=False,
        custom_logger=il_logger,
    )

    # Evaluate pre-training
    pre_rewards, _ = evaluate_policy(
        learner, env, n_eval_episodes=10, return_episode_rewards=True
    )
    mean_pre = float(np.mean(pre_rewards))
    print(f"Mean reward before training: {mean_pre}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(LOG_DIR)
    writer.add_scalar('evaluation/pre_training_reward', mean_pre, 0)

    # Manual adversarial training loop with metric logging
    gen_ts = airl_trainer.gen_train_timesteps or learner.n_steps
    disc_updates = airl_trainer.n_disc_updates_per_round
    n_rounds = TOTAL_TIMESTEPS // gen_ts
    for round_idx in range(n_rounds):
        # Generator update
        airl_trainer.train_gen(gen_ts)

        # Discriminator updates: collect stats
        disc_losses, disc_accs = [], []
        for _ in range(disc_updates):
            stats = airl_trainer.train_disc()
            if 'loss' in stats:
                disc_losses.append(stats['loss'])
            if 'accuracy' in stats:
                disc_accs.append(stats['accuracy'])

        # Log discriminator metrics
        if disc_losses:
            writer.add_scalar('discriminator/loss', np.mean(disc_losses), round_idx)
        if disc_accs:
            writer.add_scalar('discriminator/accuracy', np.mean(disc_accs), round_idx)

        # Policy evaluation
        eval_rewards, _ = evaluate_policy(
            learner, env, n_eval_episodes=10, return_episode_rewards=True
        )
        mean_eval = float(np.mean(eval_rewards))
        writer.add_scalar('evaluation/mean_reward', mean_eval, round_idx)

    # Evaluate post-training
    post_rewards, _ = evaluate_policy(
        learner, env, n_eval_episodes=10, return_episode_rewards=True
    )
    mean_post = float(np.mean(post_rewards))
    print(f"Mean reward after training: {mean_post}")
    writer.add_scalar('evaluation/post_training_reward', mean_post, n_rounds)

    # Close writer
    writer.close()

    # Save trained models
    learner.save(os.path.join(MODELS_DIR, f"airl_{args.env}"))
    torch.save(
        reward_net.state_dict(),
        os.path.join(MODELS_DIR, f"airl_reward_{args.env}.pth"),
    )

    env.close()


if __name__ == "__main__":
    print("Example: python train_airl.py --env cartpole --timesteps 2000000 --seed 42")
    print("To monitor: tensorboard --logdir logs")
    main()