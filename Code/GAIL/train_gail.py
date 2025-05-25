import os
import sys
import types
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Gym and SB3 imports
import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure as sb3_configure

# Imitation library imports
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm
from imitation.util.logger import configure as il_configure

# Create dummy modules for "mujoco_py"
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


def main():
    parser = argparse.ArgumentParser(
        description="Train a GAIL model using TRPO with detailed TensorBoard logging."
    )
    parser.add_argument("--env", choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total timesteps to train GAIL")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Number of expert episodes for training")
    args = parser.parse_args()

    SEED = args.seed
    TOTAL_TIMESTEPS = args.timesteps

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_DIR = os.path.join(BASE_DIR, "..", "data", "demonstrations", str(args.demo_episodes))
    DEMO_FILE = f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    MODELS_DIR = os.path.join(BASE_DIR, f"models/gail_{args.env}_{args.demo_episodes}_TRPO_2M_simple")
    LOG_DIR = os.path.join(BASE_DIR, "logs", f"gail_{args.env}_{args.demo_episodes}_TRPO_2M_simple")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Environment name
    ENV_NAME = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"

    # Vectorized environment
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

    # Generator algorithm (TRPO)
    learner = TRPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    """
    policy_kwargs = dict(net_arch=[256, 256, 128])
    learner = TRPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=512,
        cg_max_steps=20,
        cg_damping=0.05,
        line_search_shrinking_factor=0.8,
        line_search_max_iter = 10,
        target_kl=0.005,
        n_critic_updates=10,
        sub_sampling_factor=1,
        n_steps=2048,  # Original 2048
        gamma=0.99,
        gae_lambda = 0.97,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    """
    """
    learner = PPO(
        "MlpPolicy",
        env,
        batch_size=512,
        ent_coef=0.01,
        learning_rate=3e-4,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    """
    # SB3 logger
    sb3_logger = sb3_configure(LOG_DIR, ["stdout","tensorboard"])
    learner.set_logger(sb3_logger)

    # Imitation hierarchical logger
    il_logger = il_configure(LOG_DIR, ["stdout","tensorboard"])

    # Reward network (discriminator)
    """
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    """
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hid_sizes=(256, 256, 128),
        normalize_input_layer=RunningNorm,
    )

    """
    gail_trainer = GAIL(
        demonstrations=demonstrations,
        demo_batch_size=1024, # Original 1024
        gen_replay_buffer_capacity=512, # Original 512
        n_disc_updates_per_round=16, # Original 8
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        init_tensorboard=True,
        init_tensorboard_graph=False,
        custom_logger=il_logger,
    )
    """
    # Instantiate GAIL trainer
    gail_trainer = GAIL(
        demonstrations=demonstrations,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=16,
        disc_opt_kwargs={"lr": 3e-4},
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        init_tensorboard=True,
        init_tensorboard_graph=False,
        custom_logger=il_logger,
    )

    # SummaryWriter for custom logging
    writer = SummaryWriter(LOG_DIR)

    # Pre-training evaluation
    pre_rewards,_ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    mean_pre = float(np.mean(pre_rewards))
    writer.add_scalar('evaluation/pre_training_reward', mean_pre, 0)
    print(f"Mean reward before training: {mean_pre}")

    # Adversarial training loop
    gen_ts = gail_trainer.gen_train_timesteps or learner.n_steps
    disc_updates = gail_trainer.n_disc_updates_per_round
    n_rounds = TOTAL_TIMESTEPS // gen_ts
    for round_idx in range(n_rounds):
        # Generator update
        gail_trainer.train_gen(gen_ts)
        # Discriminator updates
        losses, accs = [], []
        expert_accuracies, gen_accuracies = [], []
        for _ in range(disc_updates):
            stats = gail_trainer.train_disc()
            # Collect discriminator stats
            losses.append(stats.get('disc_loss',0))
            accs.append(stats.get('disc_accuracy',0))
            expert_accuracies.append(stats.get('disc_acc_expert',0))
            gen_accuracies.append(stats.get('disc_acc_gen',0))
        # Log metrics
        writer.add_scalar('discriminator/loss', np.mean(losses), round_idx)
        writer.add_scalar('discriminator/accuracy', np.mean(accs), round_idx)
        writer.add_scalar('discriminator/expert_accuracy', np.mean(expert_accuracies), round_idx)
        writer.add_scalar('discriminator/gen_accuracy', np.mean(gen_accuracies), round_idx)
        # Policy evaluation
        eval_r,_ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
        mean_eval = float(np.mean(eval_r))
        writer.add_scalar('evaluation/mean_reward', mean_eval, round_idx)
        writer.add_scalar('evaluation/mean_reward_steps', mean_eval, round_idx * gen_ts)
        writer.flush()

    # Post-training evaluation
    post_rewards,_ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    mean_post = float(np.mean(post_rewards))
    writer.add_scalar('evaluation/post_training_reward', mean_post, n_rounds)
    print(f"Mean reward after training: {mean_post}")

    writer.close()

    # Save models
    learner.save(os.path.join(MODELS_DIR, f"gail_{args.env}_{args.timesteps}_TRPO_2M_simple"))
    torch.save(reward_net.state_dict(), os.path.join(MODELS_DIR, f"gail_reward_{args.env}_{args.timesteps}_TRPO_2M_simple.pth"))
    env.close()


if __name__ == "__main__":
    print("Example usage: python train_gail.py --env cartpole --timesteps 200000 --seed 42 --demo_episodes 50")
    print("To monitor: tensorboard --logdir logs")
    main()
