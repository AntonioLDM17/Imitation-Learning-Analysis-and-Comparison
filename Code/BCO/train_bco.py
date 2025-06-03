"""
Behavioral Cloning from Observation (BCO)
========================================
Re-implementation + logging improvements.

  * Original algorithm:  Torabi, Warnell & Stone (IJCAI 2018)
  * This version:
      – Supports CartPole-v1 (discrete) and HalfCheetah-v4 (continuous)
      – Adds unified progress tracking in TensorBoard
      – Handles both one-shot BCO(0) and iterative BCO(alpha)
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
import sys
import types
import argparse
import numpy as np

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

# Local utilities (defined in bco.py from the same folder)
from bco import (
    set_seed,
    InverseDynamicsModel,
    PolicyNetwork,
    collect_exploration_data,
    create_dataloader,
    train_inverse_model,
    infer_expert_actions,
    collect_policy_data,
)

# --------------------------------------------------------------------------- #
# Stub-out `mujoco_py` so that importing HalfCheetah-v4 works on machines
# without MuJoCo binaries.  (Gymnasium no longer needs mujoco_py, but other
# packages might try to import it; we silence the import here.)
# --------------------------------------------------------------------------- #
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion


# --------------------------------------------------------------------------- #
# Evaluation helper
# --------------------------------------------------------------------------- #
def evaluate_policy(net, env, discrete: bool = True, n_episodes: int = 5):
    """
    Roll out *n_episodes* of the current policy and return mean/std reward.

    Parameters
    ----------
    net        : PolicyNetwork – trained (or untrained) policy.
    env        : gymnasium.Env – environment instance.
    discrete   : bool – whether the action space is discrete.
    n_episodes : int  – number of independent episodes.

    Returns
    -------
    mean_reward, std_reward : float, float
    """
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_r = False, 0.0
        while not done:
            # Forward pass (no grad needed during evaluation)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = net(s)

            # Pick action – argmax for discrete, raw output for continuous
            action = int(logits.argmax(1)) if discrete else logits.squeeze().cpu().numpy()

            obs, r, done, truncated, _ = env.step(action)
            ep_r += r

            # Gymnasium: `truncated` = TimeLimit reached
            if done or truncated:
                break
        rewards.append(ep_r)

    return float(np.mean(rewards)), float(np.std(rewards))


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    # -------------------------- Argument parsing --------------------------- #
    parser = argparse.ArgumentParser("BCO with unified TensorBoard logging")
    parser.add_argument("--env", choices=["cartpole", "halfcheetah"],
                        default="cartpole", help="Environment to train on")
    parser.add_argument("--pre_interactions", type=int, default=300_000,
                        help="Random interactions before demonstrations (I_pre)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="BCO(alpha) parameter – 0.0 runs one-shot BCO")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterative improvements when alpha>0")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="Optional path to .npy expert demonstrations")
    parser.add_argument("--demo_episodes", type=int, default=100,
                        help="Only used for naming logs if demo_file not given")
    parser.add_argument("--seed", type=int, default=44)

    # Inverse dynamics model HP
    parser.add_argument("--inv_epochs", type=int, default=10)
    parser.add_argument("--inv_lr", type=float, default=1e-3)

    # Behavioural-cloning HP (initial)
    parser.add_argument("--policy_epochs", type=int, default=20)
    parser.add_argument("--policy_lr", type=float, default=1e-3)

    # Iterative BCO(alpha) HP
    parser.add_argument("--iter_inv_epochs", type=int, default=5)
    parser.add_argument("--iter_policy_epochs", type=int, default=10)

    # Misc
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=8000,
                        help="Evaluate every N BC steps")
    args = parser.parse_args()

    # ------------------ Determinism / reproducibility ---------------------- #
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------- Environment setup ----------------------------- #
    env_id = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"
    discrete = args.env == "cartpole"
    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # ----------------------- TensorBoard writer --------------------------- #
    log_dir = (
        f"logs/bco_{args.env}_demo{args.demo_episodes}"
        f"_alpha{args.alpha}_iter{args.num_iterations}_seed{args.seed}_OPTUNA"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ----------------------------- Counters ------------------------------- #
    env_steps = 0      # actual environment interactions
    bc_steps = 0       # total supervised training examples consumed

    def global_steps() -> int:
        """Return a *single* monotonically-increasing step counter."""
        return env_steps + bc_steps

    # ===================================================================== #
    # 1) PRE-DEMONSTRATION : random rollouts + inverse model
    # ===================================================================== #
    print(f"Collecting {args.pre_interactions:,} random transitions …")
    s_pre, s_next_pre, a_pre = collect_exploration_data(env, args.pre_interactions)
    env_steps += args.pre_interactions

    inv_model = InverseDynamicsModel(obs_dim, act_dim, discrete)

    print("Training inverse dynamics model (pre-demo data) …")
    inv_model = train_inverse_model(
        inv_model,
        create_dataloader(s_pre, s_next_pre, a_pre, args.batch_size),
        discrete=discrete,
        epochs=args.inv_epochs,
        lr=args.inv_lr,
        writer=writer,
    )

    # ===================================================================== #
    # 2) LOAD DEMONSTRATIONS AND INFER ACTIONS
    # ===================================================================== #
    demo_path = args.demo_file or os.path.join(
        "..", "data", "demonstrations", str(args.demo_episodes), f"{args.env}_demonstrations_{args.demo_episodes}.npy"
    )
    print("Loading demonstrations:", demo_path)
    demo = np.load(demo_path, allow_pickle=True)

    # Support either a single trajectory object or a list of them (.tolist())
    if isinstance(demo, np.ndarray) and demo.dtype == object:
        states = np.concatenate(
            [np.asarray(getattr(tr, "obs", tr), dtype=np.float32) for tr in demo.tolist()]
        )
    else:
        states = np.asarray(getattr(demo, "obs", demo), dtype=np.float32)

    print(f"  → {len(states):,} states in demonstrations")

    # Infer the hidden expert actions with inverse model
    states_bc, inf_actions = infer_expert_actions(inv_model, states, discrete)
    demo_transitions = len(states_bc)  # N in the paper
    print(f"  → {demo_transitions:,} inferred actions from demonstrations")

    # ===================================================================== #
    # 3) EVALUATION & LOGGING UTILITIES
    # ===================================================================== #
    EVAL_INTERVAL = args.eval_every
    print(f"Evaluation interval: {EVAL_INTERVAL:,} BC steps")
    next_eval = EVAL_INTERVAL  # Next evaluation step

    def maybe_eval(net, tag: str):
        """
        Run evaluation every *EVAL_INTERVAL* BC samples.
        Log all metrics with the *global* step counter.
        """
        """if bc_steps % EVAL_INTERVAL != 0:
            return # Skip evaluation if not due"""
        nonlocal next_eval, env_steps, bc_steps
        gs = global_steps()
        if bc_steps < next_eval:
            return
        # Evaluate the current policy
        mean_r, std_r = evaluate_policy(net, env, discrete, n_episodes=5)

        writer.add_scalar("eval/mean_reward", mean_r, gs)
        writer.add_scalar("eval/std_reward",  std_r, gs)

        # Additional useful plots
        writer.add_scalar("progress/env_steps", env_steps, gs)
        writer.add_scalar("progress/bc_steps",  bc_steps, gs)

        print(f"[{tag}] global={gs:,} env={env_steps:,} bc={bc_steps:,} "
              f"mean_r={mean_r:7.2f} ± {std_r:5.2f}")
        next_eval += EVAL_INTERVAL  # Update next evaluation step
    # --------------------------------------------------------------------- #
    # Helper: supervised BC training with on-the-fly logging
    # --------------------------------------------------------------------- #
    def train_bc(net, st, act, epochs, lr, tag):
        """Standard behavioural-cloning loop + unified logging."""
        nonlocal bc_steps
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss() if discrete else torch.nn.MSELoss()

        ds = torch.utils.data.TensorDataset(torch.tensor(st, dtype=torch.float32),
                                            torch.tensor(act))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        for _ in range(epochs):
            for sb, ab in dl:
                opt.zero_grad()
                pred = net(sb)
                loss = loss_fn(pred, ab.long() if discrete else ab.float())
                loss.backward()
                opt.step()

                bc_steps += sb.size(0)
                maybe_eval(net, tag)

    # ===================================================================== #
    # 4) INITIALISE POLICY AND LOG BASELINE
    # ===================================================================== #
    policy = PolicyNetwork(obs_dim, act_dim, discrete)
    init_mean, init_std = evaluate_policy(policy, env, discrete, 5)
    writer.add_scalar("eval/mean_reward", init_mean, global_steps())
    writer.add_scalar("eval/std_reward",  init_std,  global_steps())
    print(f"[init] mean_r={init_mean:.2f} ± {init_std:.2f}")

    # ===================================================================== #
    # 5) MAIN TRAINING – BCO(0) or BCO(alpha)
    # ===================================================================== #
    if args.alpha == 0.0:
        print(">>> Running BCO(0) (single pass)")
        train_bc(policy, states_bc, inf_actions,
                 args.policy_epochs, args.policy_lr, "BCO0")

    else:
        print(f">>> Running BCO(alpha={args.alpha}) with {args.num_iterations} iterations")
        # ---- initial BC pass (iteration 0) ----
        train_bc(policy, states_bc, inf_actions,
                 args.policy_epochs, args.policy_lr, "iter0")

        # Compute how many env steps to collect *per iteration*
        post_budget = int(args.alpha * args.pre_interactions)

        for it in range(1, args.num_iterations + 1):
            print(f"-- Iter {it}: collecting {post_budget:,} on-policy transitions")
            s_post, s_next_post, a_post = collect_policy_data(policy, env, post_budget, discrete)
            env_steps += post_budget

            # ---- re-train inverse model on combined data ----
            s_all      = np.concatenate([s_pre, s_post])
            s_next_all = np.concatenate([s_next_pre, s_next_post])
            a_all      = np.concatenate([a_pre, a_post])

            inv_model = train_inverse_model(
                inv_model,
                create_dataloader(s_all, s_next_all, a_all, args.batch_size),
                discrete=discrete,
                epochs=args.iter_inv_epochs,
                lr=args.inv_lr,
                writer=writer,
            )

            # ---- re-infer actions & run another BC phase ----
            states_bc, inf_actions = infer_expert_actions(inv_model, states, discrete)

            train_bc(policy, states_bc, inf_actions,
                     args.iter_policy_epochs, args.policy_lr, f"iter{it}")

    # ===================================================================== #
    # 6) SAVE MODEL + FINAL EVALUATION
    # ===================================================================== #
    os.makedirs("models", exist_ok=True)
    model_dir = f"bco_{args.env}_{args.demo_episodes}_alpha{args.alpha}iter{args.num_iterations}_seed{args.seed}_OPTUNA"
    os.makedirs(os.path.join("models", model_dir), exist_ok=True)
    model_name = (
        f"bco_{args.env}_alpha{args.alpha}_iter{args.num_iterations}"
        f"_steps{global_steps()}.pt"
    )
    torch.save(policy.state_dict(), os.path.join("models",model_dir, model_name))

    final_mean, final_std = evaluate_policy(policy, env, discrete, 5)
    writer.add_scalar("eval/final_mean_reward", final_mean, global_steps())
    writer.add_scalar("eval/final_std_reward",  final_std,  global_steps())
    writer.close()
    env.close()

    # --------------------------- Summary ---------------------------------- #
    print("\nTraining complete!")
    writer.add_scalar("eval/mean_reward", final_mean, global_steps())
    writer.add_scalar("eval/std_reward", final_std, global_steps())
    print(f"Global steps   : {global_steps():,} (env {env_steps:,} + bc {bc_steps:,})")
    print(f"Final reward   : {final_mean:.2f} ± {final_std:.2f}")
    print("Model saved to :", model_name)


# Entry-point
if __name__ == "__main__":
    main()
    print("Example of running normal BCO:")
    print(" python train_bco.py --env halfcheetah --pre_interactions 300000 --alpha 0.0 --policy_epochs 17 --demo_episodes 100 --seed 44")
    print("Example of running BCO(alpha):")
    print(" python train_bco.py --env halfcheetah --pre_interactions 20000 --inv_epochs 50 --policy_epochs 100 --policy_lr 3e-4 --alpha 0.5 --num_iterations 5 --seed 44")
