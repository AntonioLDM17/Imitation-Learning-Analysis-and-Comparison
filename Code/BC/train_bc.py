import os
import sys
import types
import argparse
import time
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.logger import configure
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def main():
    parser = argparse.ArgumentParser(
        description="Train a Behavioral Cloning (BC) model with TensorBoard logging."
    )
    parser.add_argument("--env", type=str, choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment to use: 'cartpole' or 'halfcheetah'")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs for BC")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    ENV_NAME = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"
    SEED = args.seed

    # Load demonstrations from the centralized folder (relative to the parent directory)
    DEMO_DIR = os.path.join("..", "data", "demonstrations")
    DEMO_FILENAME = f"{args.env}_demonstrations.npy"
    # Models for BC are saved in the local "models" folder
    MODEL_DIR = "models"
    MODEL_NAME = f"bc_{args.env}.zip"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Set up TensorBoard logging
    LOG_DIR = os.path.join("logs", f"bc_{args.env}")
    os.makedirs(LOG_DIR, exist_ok=True)
    tb_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    env = make_vec_env(ENV_NAME, rng=np.random.default_rng(SEED), n_envs=8)

    # Load demonstrations
    demo_path = os.path.join(DEMO_DIR, DEMO_FILENAME)
    demonstrations = np.load(demo_path, allow_pickle=True)
    if isinstance(demonstrations, np.ndarray):
        demonstrations = demonstrations.tolist()
    transitions = rollout.flatten_trajectories(demonstrations)

    # Initialize the BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(SEED),
        custom_logger=tb_logger,
    )

    print("Starting Behavioral Cloning training...")
    start_time = time.time()

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs"):
        epoch_start = time.time()
        bc_trainer.train(n_epochs=1)
        epoch_duration = time.time() - epoch_start

        reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
        tb_logger.record("Epoch", epoch)
        tb_logger.record("Average Reward", reward)
        tb_logger.record("Epoch Duration (s)", epoch_duration)
        tb_logger.dump(epoch)

        tqdm.write(f"Epoch {epoch}/{args.epochs}: Reward = {reward:.2f}, Epoch Time = {epoch_duration:.2f}s")

    # Save the trained BC model in the local models folder
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    bc_trainer.policy.save(model_path)
    print(f"Behavioral Cloning model saved at {model_path}")

    final_reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Final average reward after training: {final_reward:.2f}")

    env.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_bc.py --env halfcheetah --epochs 20 --seed 42")
    main()
