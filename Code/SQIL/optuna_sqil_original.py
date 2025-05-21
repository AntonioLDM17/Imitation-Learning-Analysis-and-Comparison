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

# Fixed parameters for environment and training/evaluation episodes.
# Set ENV_NAME to either "HalfCheetah-v4" (continuous) or "CartPole-v1" (discrete)
# ENV_NAME = "CartPole-v1"  
ENV_NAME = "HalfCheetah-v4"
TRAIN_EPISODES = 500
EVAL_EPISODES = 50


def one_hot(action, num_actions):
    """Convert an action index to a one-hot vector."""
    one_hot_vec = np.zeros(num_actions, dtype=np.float32)
    one_hot_vec[action] = 1.0
    return one_hot_vec


def select_action_discrete(agent, state, action_dim, evaluate=False):
    """
    For discrete action spaces, use the actor network's shared layers to produce logits,
    and choose the action with highest probability.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.actor.net[0].weight.device)
    features = agent.actor.net(state_tensor)
    logits = agent.actor.mean_linear(features)  # Use mean_linear as logits
    probabilities = torch.softmax(logits, dim=-1)
    action_index = torch.argmax(probabilities, dim=-1).item()
    return action_index


def load_demonstrations(demo_path):
    """
    Expects the demonstrations file (NumPy .npy) to contain a list of TrajectoryWithRew objects.
    """
    demos = np.load(demo_path, allow_pickle=True)
    return demos


def extract_transitions_from_trajectory(traj):
    """
    Given a TrajectoryWithRew object, extracts transitions as tuples:
    (state, action, next_state, done).
    """
    transitions = []
    obs = traj.obs
    acts = traj.acts
    n = len(obs)
    for i in range(n - 1):
        done = (i == n - 2) and getattr(traj, "terminal", False)
        transitions.append((obs[i], acts[i], obs[i + 1], done))
    return transitions


def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    actor_lr = trial.suggest_loguniform("actor_lr", 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform("critic_lr", 1e-5, 1e-3)
    alpha_lr = trial.suggest_loguniform("alpha_lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    update_every = trial.suggest_categorical("update_every", [5, 10, 20])
    max_steps = trial.suggest_categorical("max_steps", [500, 1000, 1500])

    gamma = 0.99
    tau = 0.005
    target_entropy = None  # Will be computed automatically in SQILAgent
    demo_buffer_capacity = 100000
    agent_buffer_capacity = 100000
    hidden_dim = 256

    # Set seed (varying seed between trials)
    seed = 42 + trial.number
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine subdirectories based on ENV_NAME
    if ENV_NAME == "HalfCheetah-v4":
        demo_filename = f"halfcheetah_demonstrations_{args.demo_episodes}.npy"
        log_subdir = f"halfcheetah_{args.demo_episodes}"
        model_subdir = f"halfcheetah_{args.demo_episodes}"
        discrete = False
    elif ENV_NAME == "CartPole-v1":
        demo_filename = f"cartpole_demonstrations_{args.demo_episodes}.npy"
        log_subdir = f"cartpole_{args.demo_episodes}"
        model_subdir = f"cartpole_{args.demo_episodes}"
        discrete = True
    else:
        raise ValueError("Unsupported ENV_NAME.")

    # Create a SummaryWriter for TensorBoard in the proper log directory
    writer = SummaryWriter(log_dir=os.path.join("logs", log_subdir, f"optuna_trial_{trial.number}"))

    # Create environment
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    if not discrete:
        # Continuous action space
        if isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
            action_range = float(env.action_space.high[0])
        else:
            raise NotImplementedError("This script is designed for continuous environments.")
    else:
        # Discrete action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            action_range = 1.0  # Not used in discrete mode
        else:
            raise NotImplementedError("Discrete version requires a Discrete action space.")

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

    # Load demonstrations
    demo_path = os.path.join("..", "data", "demonstrations", str(args.demo_episodes), demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstration file not found at {demo_path}")
    demos = load_demonstrations(demo_path)
    print(f"Trial {trial.number}: Loading {len(demos)} demonstrations from {demo_path}")
    for traj in demos:
        transitions = extract_transitions_from_trajectory(traj)
        for transition in transitions:
            state, action, next_state, done = transition
            if discrete:
                action = one_hot(action, action_dim)
            agent.store_demo(state, action, next_state, done)
    print(f"Trial {trial.number}: Demo buffer loaded with {len(agent.demo_buffer)} transitions.")
    writer.add_text("Demo/Info", f"Demo buffer: {len(agent.demo_buffer)} transitions.", global_step=0)

    # Train the agent for TRAIN_EPISODES episodes
    total_steps = 0
    reward_history = []
    for episode in range(1, TRAIN_EPISODES + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        steps_this_episode = 0  # Track steps within the current episode
        for step in range(max_steps):
            if not discrete:
                action = agent.select_action(state)
            else:
                action_index = select_action_discrete(agent, state, action_dim)
                action = one_hot(action_index, action_dim)
            next_state, reward, done, truncated, _ = env.step(action_index if discrete else action)
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            steps_this_episode += 1
            if total_steps % update_every == 0:
                try:
                    losses = agent.update()
                except Exception as e:
                    with open("debug_log.txt", "a") as f:
                        f.write(f"Error at step {total_steps}:\n")
                        f.write(f"State shape: {np.array(state).shape}\n")
                        f.write(f"Action shape: {np.array(action).shape}\n")
                        f.write(str(e) + "\n")
                    raise e
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            if done or truncated:
                break
        # End of episode logging
        print(f"Trial {trial.number} - Episode {episode} finished. Steps this episode: {steps_this_episode}, Total steps: {total_steps}")
        reward_history.append(episode_reward)
        writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
        writer.add_scalar("Reward/Steps", episode_reward, total_steps)  # Reward per total steps scalar
        writer.add_scalar("Train/EpisodeSteps", steps_this_episode, episode)

        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Trial {trial.number} - Episode {episode}: Average reward over last 50 episodes = {avg_reward:.2f}")
            writer.add_text("Train/Debug", f"Episode {episode}: Avg reward (last 50) = {avg_reward:.2f}", global_step=episode)

    # Save the trained model (actor) in the appropriate model subdirectory
    model_dir = os.path.join("models", model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    model_name = (
        f"sqil_{args.demo_episodes}__actor_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_alpha_lr{alpha_lr:.0e}_"
        f"bs{batch_size}_upd{update_every}_ms{max_steps}.pth"
    )
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
            if not discrete:
                action = agent.select_action(state, evaluate=True)
            else:
                action_index = select_action_discrete(agent, state, action_dim, evaluate=True)
                action = one_hot(action_index, action_dim)
            next_state, reward, done, truncated, _ = env.step(action_index if discrete else action)
            ep_reward += reward
            state = next_state
            if done or truncated:
                break
        eval_rewards.append(ep_reward)
        writer.add_scalar("Eval/EpisodeReward", ep_reward, ep)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(
        f"Trial {trial.number}: Evaluation over {EVAL_EPISODES} episodes: Mean reward = {mean_reward:.2f} (std: {std_reward:.2f})"
    )
    writer.add_text(
        "Eval/Info",
        f"Mean reward: {mean_reward:.2f}, std: {std_reward:.2f}",
        global_step=TRAIN_EPISODES,
    )

    env.close()
    writer.close()
    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for SQILAgent")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--demo_episodes", type=int, default=50, help="Number of demonstration episodes")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"Objective value: {trial.value:.2f}")
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
