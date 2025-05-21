import os
import sys
import math
import json
import pickle
import argparse
import time
import numpy as np
import gymnasium as gym
import torch
import optuna
import pandas as pd                                     # for dataframe export

from sqil_agent import SQILAgent
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Dummy Mujoco stub (avoids building native extensions when they’re absent)
# ---------------------------------------------------------------------------
import types
dummy = types.ModuleType("mujoco_py")
dummy.builder    = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"]          = dummy
sys.modules["mujoco_py.builder"]  = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

# ---------------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------------
ENV_NAME      = "HalfCheetah-v4"       # or "CartPole-v1"
STEPS_TARGET  = 1_000_000             # hard interaction budget (steps)
EVAL_EPISODES = 50                    # fixed eval length

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def one_hot(action: int, num_actions: int) -> np.ndarray:
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action] = 1.0
    return vec

def select_action_discrete(agent, state, action_dim, evaluate=False):
    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.actor.net[0].weight.device)
    feats   = agent.actor.net(state_t)
    logits  = agent.actor.mean_linear(feats)
    probs   = torch.softmax(logits, dim=-1)
    return torch.argmax(probs, dim=-1).item()

def load_demonstrations(path: str):
    return np.load(path, allow_pickle=True)

def extract_transitions_from_trajectory(traj):
    obs, acts = traj.obs, traj.acts
    for i in range(len(obs) - 1):
        done = (i == len(obs) - 2) and getattr(traj, "terminal", False)
        yield obs[i], acts[i], obs[i + 1], done

# ---------------------------------------------------------------------------
# Optuna study persistence (for safe interruption)
# ---------------------------------------------------------------------------
def save_study(study: optuna.Study, tag: str = "optuna_sqil"):
    os.makedirs("optuna_results", exist_ok=True)
    # 1) CSV table of every trial
    df = study.trials_dataframe()
    df.to_csv(f"optuna_results/{tag}_trials.csv", index=False)
    # 2) Best trial JSON
    best = {
        "value":  float(study.best_trial.value) if study.best_trial else None,
        "params": study.best_trial.params        if study.best_trial else {},
    }
    with open(f"optuna_results/{tag}_best.json", "w") as f:
        json.dump(best, f, indent=2)
    # 3) Full pickle dump
    with open(f"optuna_results/{tag}.pkl", "wb") as f:
        pickle.dump(study, f)
    print(f"[✓] Study saved under optuna_results/{tag}_*")

# ---------------------------------------------------------------------------
# Optuna objective -- a single training + evaluation run
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial):
    # ───────── Hyper-parameter search space ─────────
    actor_lr   = trial.suggest_loguniform("actor_lr",  1e-6, 1e-3)
    critic_lr  = trial.suggest_loguniform("critic_lr", 1e-6, 1e-3)
    alpha_lr   = trial.suggest_loguniform("alpha_lr",  1e-6, 1e-3)

    batch_size   = trial.suggest_categorical("batch_size",   [1024, 2048])
    update_every = trial.suggest_categorical("update_every", [5, 10, 20])
    max_steps    = 1000

    # ───────── Fixed agent params ─────────
    gamma, tau, hidden = 0.99, 0.005, 256
    target_entropy = None
    demo_buffer_capacity, agent_buffer_capacity = 100_000, 100_000

    # ───────── Seeding ─────────
    seed = 42 + trial.number
    np.random.seed(seed); torch.manual_seed(seed)

    # ───────── Environment-specific paths ─────────
    if ENV_NAME == "HalfCheetah-v4":
        discrete = False
        demo_filename = f"halfcheetah_demonstrations_{args.demo_episodes}.npy"
        tag = f"halfcheetah_{args.demo_episodes}"
    elif ENV_NAME == "CartPole-v1":
        discrete = True
        demo_filename = f"cartpole_demonstrations_{args.demo_episodes}.npy"
        tag = f"cartpole_{args.demo_episodes}"
    else:
        raise ValueError("Unsupported ENV_NAME.")

    log_dir   = os.path.join("logs", tag, f"optuna_trial_{trial.number}")
    model_dir = os.path.join("models", tag); os.makedirs(model_dir, exist_ok=True)
    writer    = SummaryWriter(log_dir=log_dir)

    # ───────── Environment ─────────
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    if not discrete:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError
        action_dim, action_range = env.action_space.shape[0], float(env.action_space.high[0])
    else:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise NotImplementedError
        action_dim, action_range = env.action_space.n, 1.0

    # ───────── Agent ─────────
    agent = SQILAgent(
        state_dim, action_dim, action_range=action_range,
        actor_hidden=hidden, critic_hidden=hidden,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau, target_entropy=target_entropy,
        demo_buffer_capacity=demo_buffer_capacity,
        agent_buffer_capacity=agent_buffer_capacity,
        batch_size=batch_size
    )

    # ───────── Load demonstrations ─────────
    demo_path = os.path.join("..", "data", "demonstrations", str(args.demo_episodes), demo_filename)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(demo_path)
    for traj in load_demonstrations(demo_path):
        for s, a, ns, d in extract_transitions_from_trajectory(traj):
            agent.store_demo(s, one_hot(a, action_dim) if discrete else a, ns, d)
    writer.add_text("Demo/Info", f"{len(agent.demo_buffer)} demo transitions", 0)

    # ───────── Training loop (with safe interrupt) ─────────
    total_steps, episode = 0, 0
    reward_hist = []
    try:
        while total_steps < STEPS_TARGET:
            episode += 1
            state, _ = env.reset(seed=seed + episode)
            ep_reward, steps_ep = 0.0, 0

            for _ in range(max_steps):
                if discrete:
                    act_idx = select_action_discrete(agent, state, action_dim)
                    env_action = one_hot(act_idx, action_dim)
                else:
                    env_action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(
                    act_idx if discrete else env_action
                )

                agent.store_agent(state, env_action, next_state, float(done or truncated))
                state = next_state
                ep_reward  += reward
                steps_ep   += 1
                total_steps += 1

                if total_steps % update_every == 0:
                    res = agent.update()
                    if res:
                        c_l, a_l, al_l = res
                        writer.add_scalar("Loss/Critic", c_l, total_steps)
                        writer.add_scalar("Loss/Actor",  a_l, total_steps)
                        writer.add_scalar("Loss/Alpha",  al_l, total_steps)

                if done or truncated or total_steps >= STEPS_TARGET:
                    break

            reward_hist.append(ep_reward)
            writer.add_scalar("Train/EpisodeReward", ep_reward, episode)
            writer.add_scalar("Reward/Steps",        ep_reward, total_steps)
            writer.add_scalar("Train/EpisodeSteps",  steps_ep,   episode)
            print(f"Trial {trial.number} | Ep {episode} | {steps_ep} steps | "
                  f"Total {total_steps}/{STEPS_TARGET} | R {ep_reward:.2f}")
    except KeyboardInterrupt:
        print(f"\n[!] Trial {trial.number} interrupted at {total_steps} steps.")
        partial_name = (f"sqil_{args.demo_episodes}_{total_steps//1000}k_"
                        f"trial{trial.number}_PARTIAL.pth")
        torch.save(agent.actor.state_dict(), os.path.join(model_dir, partial_name))
        writer.close(); env.close()
        mean_so_far = np.mean(reward_hist) if reward_hist else float('-inf')
        raise optuna.TrialPruned(f"Interrupted at {total_steps} steps, mean R={mean_so_far:.2f}")

    # ───────── Save final model ─────────
    model_name = (f"sqil_{args.demo_episodes}_{total_steps//1000}k_trial{trial.number}_"
                  f"act{actor_lr:.0e}_crt{critic_lr:.0e}_alp{alpha_lr:.0e}_"
                  f"bs{batch_size}_ms{max_steps}.pth")
    save_path = os.path.join(model_dir, model_name)
    torch.save(agent.actor.state_dict(), save_path)
    writer.add_text("Model/Info", f"Saved at {save_path}", total_steps)

    # ───────── Evaluation ─────────
    eval_rewards = []
    for ep in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset(seed=seed + 1000 + ep)
        ep_r = 0.0
        while True:
            if discrete:
                idx = select_action_discrete(agent, state, action_dim, evaluate=True)
                env_action = one_hot(idx, action_dim)
            else:
                env_action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(
                idx if discrete else env_action
            )
            ep_r += reward
            state = next_state
            if done or truncated:
                break
        eval_rewards.append(ep_r)
        writer.add_scalar("Eval/EpisodeReward", ep_r, ep)

    mean_r, std_r = np.mean(eval_rewards), np.std(eval_rewards)
    writer.add_text("Eval/Info", f"Mean {mean_r:.2f} ± {std_r:.2f}", total_steps)
    env.close(); writer.close()
    return mean_r

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Optuna hyper-parameter search for SQIL (fixed 1M-step budget)"
    )
    parser.add_argument("--n_trials",      type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Number of demonstration episodes to load")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n[!] Global interruption – saving partial study…")
        save_study(study, tag=f"{ENV_NAME}_{args.demo_episodes}")
        sys.exit(0)

    # Normal termination → save full study
    save_study(study, tag=f"{ENV_NAME}_{args.demo_episodes}")
    best = study.best_trial
    print(f"Best trial value: {best.value:.2f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
