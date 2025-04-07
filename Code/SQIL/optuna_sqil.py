import os
import argparse
import time
import numpy as np
import gymnasium as gym
import torch
import optuna

from sqil_agent import SQILAgent

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
import sys, types
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

from torch.utils.tensorboard import SummaryWriter

# Fixed parameters for the environment and training/evaluation episodes
# Set ENV_NAME to either "HalfCheetah-v4" or "CartPole-v1"
ENV_NAME = "HalfCheetah-v4"  # or "CartPole-v1"
TRAIN_EPISODES = 500
EVAL_EPISODES = 50

def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    actor_lr = trial.suggest_loguniform("actor_lr", 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform("critic_lr", 1e-5, 1e-3)
    alpha_lr  = trial.suggest_loguniform("alpha_lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    update_every = trial.suggest_categorical("update_every", [10, 50, 100])
    max_steps = trial.suggest_categorical("max_steps", [500, 1000, 1500])
    
    gamma = 0.99
    tau = 0.005
    target_entropy = None  # Will be computed automatically in SQILAgent
    demo_buffer_capacity = 100000
    agent_buffer_capacity = 100000
    hidden_dim = 256

    # Set seed (vary the seed between trials)
    seed = 42 + trial.number
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine subdirectories for logs and models based on the environment
    if ENV_NAME == "HalfCheetah-v4":
        demo_filename = "halfcheetah_demonstrations.npy"
        log_subdir = "halfcheetah"
        model_subdir = "halfcheetah"
    elif ENV_NAME == "CartPole-v1":
        demo_filename = "cartpole_demonstrations.npy"
        log_subdir = "cartpole"
        model_subdir = "cartpole"
    else:
        raise ValueError("Unsupported ENV_NAME.")

    # Create a SummaryWriter specific for this trial under the proper logs subdirectory
    writer = SummaryWriter(log_dir=os.path.join("logs", log_subdir, f"optuna_trial_{trial.number}"))
    
    # Create environment
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    else:
        raise NotImplementedError("This script is designed for continuous environments.")
    
    # Instantiate the SQIL agent with suggested hyperparameters
    agent = SQILAgent(
        state_dim, action_dim, action_range=action_range,
        actor_hidden=hidden_dim, critic_hidden=hidden_dim,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau, target_entropy=target_entropy,
        demo_buffer_capacity=demo_buffer_capacity,
        agent_buffer_capacity=agent_buffer_capacity,
        batch_size=batch_size
    )
    
    # Load demonstrations (adjust path according to project structure)
    demo_path = os.path.join("..", "data", "demonstrations", demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstration file not found at {demo_path}")
    demos = np.load(demo_path, allow_pickle=True)
    print(f"Trial {trial.number}: Loading {len(demos)} demonstrations from {demo_path}")
    # Extract transitions from each trajectory and store them in the demo buffer
    for traj in demos:
        obs = traj.obs
        acts = traj.acts
        n = len(obs)
        for i in range(n - 1):
            done = (i == n - 2) and getattr(traj, "terminal", False)
            agent.store_demo(obs[i], acts[i], obs[i + 1], done)
    print(f"Trial {trial.number}: Demo buffer loaded with {len(agent.demo_buffer)} transitions.")
    writer.add_text("Demo/Info", f"Demo buffer: {len(agent.demo_buffer)} transitions.", global_step=0)
    
    # Train the agent for TRAIN_EPISODES episodes
    total_steps = 0
    reward_history = []
    for episode in range(1, TRAIN_EPISODES + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # SQIL ignores the environment reward; assign 0 for agent transitions
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            if total_steps % update_every == 0:
                losses = agent.update()
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            if done or truncated:
                break
        reward_history.append(episode_reward)
        writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
        # Debug: every 50 episodes, log average reward over last 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Trial {trial.number} - Episode {episode}: Average reward over last 50 episodes = {avg_reward:.2f}")
            writer.add_text("Train/Debug", f"Episode {episode}: Avg reward (last 50) = {avg_reward:.2f}", global_step=episode)
    
    # Save the trained model (actor) in the appropriate directory
    model_dir = os.path.join("models", model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"sqil_actor_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_alpha_lr{alpha_lr:.0e}_bs{batch_size}_upd{update_every}_ms{max_steps}.pth"
    actor_save_path = os.path.join(model_dir, model_name)
    torch.save(agent.actor.state_dict(), actor_save_path)
    print(f"Trial {trial.number}: Model saved at {actor_save_path}")
    writer.add_text("Model/Info", f"Model saved at {actor_save_path}", global_step=TRAIN_EPISODES)
    
    # Evaluate the trained model on EVAL_EPISODES episodes
    eval_rewards = []
    for ep in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset(seed=seed + 1000 + ep)
        ep_reward = 0
        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done or truncated:
                break
        eval_rewards.append(ep_reward)
        writer.add_scalar("Eval/EpisodeReward", ep_reward, ep)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"Trial {trial.number}: Evaluation over {EVAL_EPISODES} episodes: Mean reward = {mean_reward:.2f} (std: {std_reward:.2f})")
    writer.add_text("Eval/Info", f"Mean reward: {mean_reward:.2f}, std: {std_reward:.2f}", global_step=TRAIN_EPISODES)
    
    env.close()
    writer.close()
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for SQILAgent")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Objective value: {trial.value:.2f}")
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
