import gymnasium as gym
import numpy as np
import torch
import time
import os, sys, types, argparse
from torch.utils.tensorboard import SummaryWriter
from sqil_agent import SQILAgent

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

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
    Assumes:
      - 'obs' is an array of observations,
      - 'acts' is an array of actions,
      - 'terminal' indicates whether the trajectory is terminal.
    Generates transitions for each consecutive pair of observations.
    The final transition is marked as done if traj.terminal is True.
    """
    transitions = []
    obs = traj.obs
    acts = traj.acts
    n = len(obs)
    for i in range(n - 1):
        done = (i == n - 2) and getattr(traj, "terminal", False)
        transitions.append((obs[i], acts[i], obs[i + 1], done))
    return transitions

def main():
    parser = argparse.ArgumentParser(description="Train SQIL based on DI‑engine (SAC) in PyTorch")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Gym environment name (e.g., HalfCheetah-v4 or CartPole-v1)")
    parser.add_argument("--demo_path", type=str, default=None,
                        help="Path to the demonstrations file (.npy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--update_every", type=int, default=10, help="Number of steps between updates")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine subdirectories based on environment
    if args.env == "HalfCheetah-v4":
        demo_filename = "halfcheetah_demonstrations.npy"
        model_subdir = "halfcheetah"
        log_subdir = "halfcheetah"
    elif args.env == "CartPole-v1":
        demo_filename = "cartpole_demonstrations.npy"
        model_subdir = "cartpole"
        log_subdir = "cartpole"
    else:
        raise ValueError("Unsupported environment. Use 'HalfCheetah-v4' or 'CartPole-v1'.")
    
    # Create SummaryWriter for TensorBoard in the proper log directory
    writer = SummaryWriter(log_dir=os.path.join("logs", log_subdir))
    
    # Create environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    
    # Assume continuous action space (Box)
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    else:
        raise NotImplementedError("This SQIL implementation is designed for continuous environments.")
    
    # Instantiate the SQIL agent
    agent = SQILAgent(state_dim, action_dim, action_range=action_range, batch_size=args.batch_size)
    
    # Load demonstrations if provided
    if args.demo_path is not None:
        demo_path = args.demo_path
    else:
        demo_path = os.path.join("..", "data", "demonstrations", demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstrations file not found at {demo_path}")
    demos = load_demonstrations(demo_path)
    print(f"Loading {len(demos)} demonstrations from {demo_path}")
    for traj in demos:
        transitions = extract_transitions_from_trajectory(traj)
        for transition in transitions:
            state, action, next_state, done = transition
            agent.store_demo(state, action, next_state, done)
    print(f"Demo buffer loaded with {len(agent.demo_buffer)} transitions.")
    writer.add_text("Demo/Info", f"Demo buffer: {len(agent.demo_buffer)} transitions.", global_step=0)
    
    total_steps = 0
    start_time = time.time()
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed)
        episode_reward = 0
        for step in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # SQIL ignores the environment reward; agent transitions get reward 0
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            if total_steps % args.update_every == 0:
                losses = agent.update()
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    print(f"Step {total_steps}: Critic Loss = {critic_loss:.4f}, Actor Loss = {actor_loss:.4f}, Alpha = {alpha_loss:.4f}")
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            if done or truncated:
                break
        print(f"Episode {episode} finished. Accumulated reward: {episode_reward}")
        writer.add_scalar("Reward/Episode", episode_reward, episode)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    
    # Save the trained models in the proper model subdirectory
    model_dir = os.path.join("models", model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    model_path_actor = os.path.join(model_dir, "sqil_actor.pth")
    model_path_critic = os.path.join(model_dir, "sqil_critic.pth")
    torch.save(agent.actor.state_dict(), model_path_actor)
    torch.save(agent.critic.state_dict(), model_path_critic)
    print(f"Models saved in {model_dir}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    print("Example usage for HalfCheetah:")
    print("python train_sqil.py --env HalfCheetah-v4 --demo_path ../data/demonstrations/halfcheetah_demonstrations.npy --seed 42 --episodes 1000")
    print("Example usage for CartPole:")
    print("python train_sqil.py --env CartPole-v1 --demo_path ../data/demonstrations/cartpole_demonstrations.npy --seed 42 --episodes 1000")
    print("To view the training process, run:")
    print("tensorboard --logdir logs")
    main()
