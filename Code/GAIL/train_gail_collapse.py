import os
import sys
import types
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Gym and SB3 imports
import gymnasium as gym
from sb3_contrib import TRPO  # Using TRPO as in the original GAIL paper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure as sb3_configure

# Imitation library imports
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm
from imitation.util.logger import configure as il_configure

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def main():
    parser = argparse.ArgumentParser(
        description="Train a GAIL model with collapse monitoring and TensorBoard logging."
    )
    parser.add_argument("--env", choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total timesteps to train GAIL")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--policy", choices=["ppo","trpo","sac"], default="ppo",
                        help="Expert policy algorithm for demonstrations")
    parser.add_argument("--demo-episodes", type=int, default=50,
                        help="Number of expert episodes for training")
    args = parser.parse_args()

    SEED = args.seed
    TOTAL_TIMESTEPS = args.timesteps

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_DIR = os.path.join(BASE_DIR, "..", "data", "demonstrations", str(args.demo_episodes))
    DEMO_FILE = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODELS_DIR = os.path.join(BASE_DIR, f"models/gail_{args.env}_{args.demo_episodes}")
    LOG_DIR = os.path.join(BASE_DIR, "logs", f"gail_{args.env}_{args.demo_episodes}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Environment name
    ENV_NAME = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"

    # Create vectorized environment
    env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Load expert demonstrations
    demo_path = os.path.join(DEMO_DIR, DEMO_FILE)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()

    # SB3 TRPO generator
    learner = TRPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    sb3_logger = sb3_configure(LOG_DIR, ["stdout","tensorboard"])
    learner.set_logger(sb3_logger)

    # Imitation logger
    il_logger = il_configure(LOG_DIR, ["stdout","tensorboard"])

    # Reward network (discriminator)
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Instantiate GAIL trainer
    gail_trainer = GAIL(
        demonstrations=demonstrations,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        init_tensorboard=True,
        init_tensorboard_graph=False,
        custom_logger=il_logger,
    )

    # Initialize collapse monitoring parameters
    collapse_threshold = 0.0
    collapse_counter = 0
    collapse_interval = 16384
    total_steps = 0

    # SummaryWriter for custom metrics
    writer = SummaryWriter(LOG_DIR)

    # Pre-training evaluation
    pre_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    mean_pre = float(np.mean(pre_rewards))
    writer.add_scalar('evaluation/pre_training_reward', mean_pre, 0)
    print(f"Mean reward before training: {mean_pre}")

    # Adversarial training loop with collapse monitoring
    gen_ts = gail_trainer.gen_train_timesteps or learner.n_steps
    disc_updates = gail_trainer.n_disc_updates_per_round
    n_rounds = TOTAL_TIMESTEPS // gen_ts

    for round_idx in range(n_rounds):
        # Generator update
        gail_trainer.train_gen(gen_ts)

        # Discriminator updates
        losses, accs = [], []
        for _ in range(disc_updates):
            stats = gail_trainer.train_disc()
            losses.append(stats.get('loss', 0))
            accs.append(stats.get('accuracy', 0))
        avg_loss = float(np.mean(losses))
        avg_acc = float(np.mean(accs))

        # Log discriminator metrics
        writer.add_scalar('discriminator/loss', avg_loss, round_idx)
        writer.add_scalar('discriminator/accuracy', avg_acc, round_idx)

        # Policy evaluation
        eval_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
        mean_eval = float(np.mean(eval_rewards))
        writer.add_scalar('evaluation/mean_reward', mean_eval, round_idx)

        # Collapse check
        total_steps += gen_ts
        if total_steps >= collapse_interval and mean_eval < collapse_threshold:
            collapse_counter += 1
            writer.add_scalar('collapse/count', collapse_counter, round_idx)
            print(f"Collapse detected at {total_steps} steps, count: {collapse_counter}")

        writer.flush()

    # Post-training evaluation
    post_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    mean_post = float(np.mean(post_rewards))
    writer.add_scalar('evaluation/post_training_reward', mean_post, n_rounds)
    print(f"Mean reward after training: {mean_post}")

    writer.close()

    # Save trained model
    learner.save(os.path.join(MODELS_DIR, f"gail_{args.env}_{args.demo_episodes}_{TOTAL_TIMESTEPS}"))
    torch.save(reward_net.state_dict(), os.path.join(MODELS_DIR, f"gail_reward_{args.env}_{args.demo_episodes}_{TOTAL_TIMESTEPS}.pth"))
    env.close()


if __name__ == "__main__":
    print("Usage example: python train_gail_collapse.py --env halfcheetah --timesteps 200000 --seed 42 --demo-episodes 50")
    print("To monitor: tensorboard --logdir logs")
    main()
