import os
import argparse
import time
import numpy as np
import gymnasium as gym
import torch
import optuna
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
import sys, types
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from dac import DAC

# --- Updated DAC_Discriminator for continuous actions ---
class DAC_Discriminator(nn.Module):
    def __init__(self, observation_space, action_space):
        """
        Discriminator network that receives the concatenation of an observation and an action,
        and estimates the probability that the pair comes from an expert.
        Extracts dimensions from the provided spaces.
        """
        super(DAC_Discriminator, self).__init__()
        if not hasattr(observation_space, "shape") or not hasattr(action_space, "shape"):
            raise ValueError("Both observation_space and action_space must have a 'shape' attribute")
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- Define Networks for Continuous DAC ---
class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ContinuousActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

class ContinuousCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ContinuousCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- Helper to flatten demonstration trajectories ---
def flatten_demonstrations(demos):
    transitions = []
    for traj in demos:
        N = len(traj.acts)
        for i in range(N):
            obs_i = traj.obs[i]
            act_i = traj.acts[i]
            rew_i = traj.rews[i]
            next_obs = traj.obs[i+1]
            done_flag = 1.0 if i == N - 1 else 0.0
            transitions.append((obs_i, act_i, rew_i, next_obs, done_flag))
    return transitions

# --- Evaluation Function for DAC Agent ---
def evaluate_dac(actor, env, eval_episodes=5):
    rewards = []
    # Get the device of the actor
    device = next(actor.parameters()).device
    for ep in range(eval_episodes):
        state, _ = env.reset()
        # If the observation has an extra dimension, use the first element
        if isinstance(state, np.ndarray) and state.ndim == 2:
            state = state[0]
        ep_reward = 0.0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                # Actor output is batched; extract the action (shape: (act_dim,))
                action = actor(state_tensor).cpu().numpy()[0]
            # For non-vectorized env, send the action directly
            result = env.step(action)
            # Gymnasium for non-vectorized env returns (obs, reward, terminated, truncated, info)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            elif len(result) == 4:
                next_state, reward, done, _ = result
            else:
                raise ValueError("Unexpected number of values from env.step()")
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
    return np.mean(rewards)

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
    disc_lr   = trial.suggest_float("disc_lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    max_timesteps = trial.suggest_categorical("max_timesteps", [10000, 20000, 30000])
    gamma = 0.99

    # Use HalfCheetah-v4 for continuous experiments
    ENV_NAME = "HalfCheetah-v4"
    demo_filename = "halfcheetah_demonstrations.npy"
    
    # Create a vectorized environment with a single environment (for simplicity)
    train_env = make_vec_env(
        ENV_NAME,
        rng=np.random.default_rng(42 + trial.number),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
    )
    
    # Create a TensorBoard writer for this trial
    writer = SummaryWriter(log_dir=os.path.join("logs", f"dac_optuna_trial_{trial.number}"))
    
    # Load demonstration data
    demo_path = os.path.join("..", "data", "demonstrations", demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstration file not found at {demo_path}")
    demos = np.load(demo_path, allow_pickle=True)
    if isinstance(demos, np.ndarray):
        demos = demos.tolist()
    try:
        iter(demos[0])
    except TypeError:
        demos = flatten_demonstrations(demos)
    
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks and DAC trainer
    actor_net = ContinuousActor(obs_dim, act_dim).to(device)
    critic_net = ContinuousCritic(obs_dim, act_dim).to(device)
    discriminator = DAC_Discriminator(train_env.observation_space, train_env.action_space).to(device)
    
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)
    
    dac_trainer = DAC(actor_net, critic_net, discriminator,
                      actor_optimizer, critic_optimizer, disc_optimizer,
                      device=device)
    
    replay_buffer = dac_trainer.replay_buffer

    # Extract the initial (flattened) state (as in train_dac.py)
    raw_state = train_env.reset()[0]
    if isinstance(raw_state, np.ndarray) and raw_state.ndim == 2:
        state = raw_state[0]
    else:
        state = raw_state

    timestep = 0
    while timestep < max_timesteps:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor_net(state_tensor).cpu().numpy()[0]
        result = train_env.step(np.array([action]))
        if len(result) == 5:
            next_obs, reward_batch, terminated, truncated, _ = result
            done = terminated[0] or truncated[0]
            next_obs = next_obs[0]
        elif len(result) == 4:
            next_obs, reward_batch, done, info = result
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
            next_obs = next_obs[0]
        else:
            raise ValueError("Unexpected output from env.step()")
        reward = reward_batch[0]
        replay_buffer.append((state, action, reward, next_obs, float(done)))
        state = next_obs
        timestep += 1
        if done:
            raw_state = train_env.reset()[0]
            if isinstance(raw_state, np.ndarray) and raw_state.ndim == 2:
                state = raw_state[0]
            else:
                state = raw_state

    # If there are enough samples, perform updates and log losses
    if len(replay_buffer) >= batch_size:
        indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]
        obs_b, act_b, reward_b, next_obs_b, done_b = zip(*batch)
        obs_b = torch.tensor(np.array(obs_b), dtype=torch.float32).to(device)
        act_b = torch.tensor(np.array(act_b), dtype=torch.float32).to(device)
        reward_b = torch.tensor(np.array(reward_b), dtype=torch.float32).unsqueeze(1).to(device)
        next_obs_b = torch.tensor(np.array(next_obs_b), dtype=torch.float32).to(device)
        done_b = torch.tensor(np.array(done_b), dtype=torch.float32).unsqueeze(1).to(device)
        actor_loss, critic_loss = dac_trainer.update_actor_critic((obs_b, act_b, reward_b, next_obs_b, done_b))
        writer.add_scalar("Loss/Actor", actor_loss, timestep)
        writer.add_scalar("Loss/Critic", critic_loss, timestep)
        
        disc_losses = []
        for _ in range(5):
            indices = np.random.choice(len(replay_buffer), batch_size, replace=True)
            batch = [replay_buffer[i] for i in indices]
            obs_ex, act_ex, _, _, _ = zip(*batch)
            demo_indices = np.random.choice(len(demos), batch_size, replace=True)
            demo_batch = [demos[i] for i in demo_indices]
            obs_demo, act_demo, _, _, _ = zip(*demo_batch)
            expert_obs = torch.tensor(np.array(obs_demo), dtype=torch.float32).to(device)
            expert_act = torch.tensor(np.array(act_demo), dtype=torch.float32).to(device)
            policy_obs = torch.tensor(np.array(obs_ex), dtype=torch.float32).to(device)
            policy_act = torch.tensor(np.array(act_ex), dtype=torch.float32).to(device)
            disc_loss = dac_trainer.update_discriminator((expert_obs, expert_act), (policy_obs, policy_act))
            disc_losses.append(disc_loss)
            writer.add_scalar("Loss/Discriminator", disc_loss, timestep)
        avg_disc_loss = np.mean(disc_losses)
        writer.add_scalar("Loss/Discriminator_Avg", avg_disc_loss, timestep)
    
    # Use a non-vectorized environment for evaluation to avoid dimension issues
    eval_env = gym.make(ENV_NAME)
    eval_reward = evaluate_dac(actor_net, eval_env, eval_episodes=5)
    writer.add_scalar("Reward/Eval", eval_reward, timestep)
    train_env.close()
    eval_env.close()
    
    if ENV_NAME == "HalfCheetah-v4":
        # Save the models in a subdirectory named after the environment
        model_subdir = "halfcheetah"
    elif ENV_NAME == "CartPole-v1":
        model_subdir = "cartpole"
    else:
        raise ValueError("Unsupported environment. Use 'HalfCheetah-v4' or 'CartPole-v1'.")
    # Save the models with filenames that encode the hyperparameter combination
    model_dir = os.path.join("models", model_subdir, f"trial_{trial.number}")
    os.makedirs(model_dir, exist_ok=True)
    model_name_actor = os.path.join(
        model_dir, 
        f"dac_actor_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_disc_lr{disc_lr:.0e}_bs{batch_size}_ms{max_timesteps}.pt"
    )
    torch.save(actor_net.state_dict(), model_name_actor)
    model_name_critic = os.path.join(
        model_dir, 
        f"dac_critic_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_disc_lr{disc_lr:.0e}_bs{batch_size}_ms{max_timesteps}.pt"
    )
    torch.save(critic_net.state_dict(), model_name_critic)
    model_name_disc = os.path.join(
        model_dir, 
        f"dac_disc_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_disc_lr{disc_lr:.0e}_bs{batch_size}_ms{max_timesteps}.pt"
    )
    torch.save(discriminator.state_dict(), model_name_disc)
    
    writer.add_text("Model Info", f"Saved actor at {model_name_actor}")
    writer.add_text("Model Info", f"Saved critic at {model_name_critic}")
    writer.add_text("Model Info", f"Saved discriminator at {model_name_disc}")
    writer.close()
    
    print(f"Trial {trial.number}: Evaluation reward = {eval_reward:.2f}")
    return eval_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for DAC")
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
