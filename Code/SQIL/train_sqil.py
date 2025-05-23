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

def one_hot(action, num_actions):
    """Converts an action index to a one-hot vector."""
    one_hot_vec = np.zeros(num_actions, dtype=np.float32)
    one_hot_vec[action] = 1.0
    return one_hot_vec

# For discrete environments, define a custom action selection function.
def select_action_discrete(agent, state, action_dim, evaluate=False):
    """
    For discrete action spaces, uses the actor network's shared layers to produce logits
    and then returns the action with highest probability (as an integer). For storing into
    the replay buffer, this action is later converted to one-hot.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.actor.net[0].weight.device)
    features = agent.actor.net(state_tensor)
    logits = agent.actor.mean_linear(features)  # Use mean_linear as logits for discrete actions
    probabilities = torch.softmax(logits, dim=-1)
    action_index = torch.argmax(probabilities, dim=-1).item()
    return action_index

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
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Number of expert episodes for training")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine subdirectories and settings based on environment
    if args.env == "HalfCheetah-v4":
        demo_filename = f"halfcheetah_demonstrations_{args.demo_episodes}.npy"
        model_subdir = f"halfcheetah_{args.demo_episodes}_2"
        log_subdir = f"halfcheetah_{args.demo_episodes}_2"
        discrete = False
    elif args.env == "CartPole-v1":
        demo_filename = f"cartpole_demonstrations_{args.demo_episodes}.npy"
        model_subdir = f"cartpole_{args.demo_episodes}_3"
        log_subdir = f"cartpole_{args.demo_episodes}_3"
        discrete = True
    else:
        raise ValueError("Unsupported environment. Use 'HalfCheetah-v4' or 'CartPole-v1'.")
    
    # Create SummaryWriter for TensorBoard in the proper log directory
    writer = SummaryWriter(log_dir=os.path.join("logs", log_subdir))
    
    # Create environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    
    if not discrete:
        # Continuous action space
        if isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
            action_range = float(env.action_space.high[0])
        else:
            raise NotImplementedError("This SQIL implementation is designed for continuous environments.")
    else:
        # Discrete action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            action_range = 1.0  # Not used in discrete mode
        else:
            raise NotImplementedError("This discrete version requires a Discrete action space.")
    
    # Instantiate the SQIL agent
    agent = SQILAgent(state_dim, action_dim, action_range=action_range, batch_size=args.batch_size)
    
    # Load demonstrations
    if args.demo_path is not None:
        demo_path = args.demo_path
    else:
        demo_path = os.path.join("..", "data", "demonstrations", str(args.demo_episodes), demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstrations file not found at {demo_path}")
    demos = load_demonstrations(demo_path)
    print(f"Loading {len(demos)} demonstrations from {demo_path}")
    for traj in demos:
        transitions = extract_transitions_from_trajectory(traj)
        for transition in transitions:
            state, action, next_state, done = transition
            # For discrete environments, convert the action to one-hot vector before storing
            if discrete:
                action = one_hot(action, action_dim)
            agent.store_demo(state, action, next_state, done)
    print(f"Demo buffer loaded with {len(agent.demo_buffer)} transitions.")
    writer.add_text("Demo/Info", f"Demo buffer: {len(agent.demo_buffer)} transitions.", global_step=0)
    
    total_steps = 0
    start_time = time.time()
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed)
        episode_reward = 0
        steps_this_episode = 0  # NEW: counter for steps in current episode
        for step in range(args.max_steps):
            if not discrete:
                action = agent.select_action(state)
            else:
                # Get discrete action index and then convert it to one-hot for storage
                action_index = select_action_discrete(agent, state, action_dim)
                action = one_hot(action_index, action_dim)
            next_state, reward, done, truncated, _ = env.step(action_index if discrete else action)
            # SQIL ignores the environment reward; agent transitions get reward 0
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            steps_this_episode += 1  # NEW: increment per-step counter
            if total_steps % args.update_every == 0:
                try:
                    losses = agent.update()
                except Exception as e:
                    # Debug: log shapes of state and action if an error occurs
                    with open("debug_log.txt", "a") as f:
                        f.write(f"Error at step {total_steps}:\n")
                        f.write(f"State shape: {np.array(state).shape}\n")
                        f.write(f"Action shape: {np.array(action).shape}\n")
                        f.write(str(e) + "\n")
                    raise e
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    print(f"Step {total_steps}: Critic Loss = {critic_loss:.4f}, Actor Loss = {actor_loss:.4f}, Alpha = {alpha_loss:.4f}")
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            if done or truncated:
                break
        # Print summary including equivalent steps
        print(f"Episode {episode} finished. Steps this episode: {steps_this_episode}, Total steps so far: {total_steps}. Accumulated reward: {episode_reward}")
        # Log reward by episode and by steps
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/Steps", episode_reward, total_steps)  # NEW: reward vs. global steps
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    
    # Save the trained models in the proper model subdirectory
    model_dir = os.path.join("models", model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    name_actor = f"3_sqil_actor_{args.env}_{args.demo_episodes}_{total_steps}.pth"
    name_critic = f"3_sqil_critic_{args.env}_{args.demo_episodes}_{total_steps}.pth"
    model_path_actor = os.path.join(model_dir, name_actor)
    model_path_critic = os.path.join(model_dir, name_critic)
    torch.save(agent.actor.state_dict(), model_path_actor)
    torch.save(agent.critic.state_dict(), model_path_critic)
    print(f"Models saved in {model_dir}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    print("Example usage for HalfCheetah:")
    print("python train_sqil.py --env HalfCheetah-v4 --demo_path ../data/demonstrations/halfcheetah_demonstrations.npy --seed 42 --episodes 1000 --demo_episodes 50")
    print("Example usage for CartPole:")
    print("python train_sqil.py --env CartPole-v1 --demo_path ../data/demonstrations/cartpole_demonstrations.npy --seed 42 --episodes 350 --demo_episodes 50")
    print("To view the training process, run:")
    print("tensorboard --logdir logs")
    main()
