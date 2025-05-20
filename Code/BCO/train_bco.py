""" 
Credits to:
@inproceedings{torabi2018bco,
  author = {Faraz Torabi and Garrett Warnell and Peter Stone}, 
  title = {{Behavioral Cloning from Observation}}, 
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)}, 
  year = {2018} 
}
Where the code is based on the original BCO implementation by Faraz Torabi.
"""
import os
import argparse
import types
import sys
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

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
# Stub-out mujoco_py (not required for CartPole-v1 / HalfCheetah-v4)
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
def evaluate_policy(policy_net, env, discrete=True, n_episodes: int = 5):
    """Run the policy for *n_episodes* and return (mean_reward, std_reward)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy_net(s)
            action = int(logits.argmax(dim=1).item()) if discrete else logits.squeeze().cpu().numpy()
            obs, r, done, truncated, _ = env.step(action)
            ep_r += r
            if done or truncated:
                break
        rewards.append(ep_r)
    return float(np.mean(rewards)), float(np.std(rewards))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning from Observation (BCO) with dual-step and reward logging"
    )
    parser.add_argument("--env", choices=["cartpole", "halfcheetah"], default="cartpole",
                        help="Environment to use")
    parser.add_argument("--pre_interactions", type=int, default=2_000,
                        help="Random interactions before demos (I_pre)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Alpha for iterative BCO (0 = one-shot)")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of BCO(alpha) refinements")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="Path to .npy demonstrations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Number of expert episodes for training")

    # Training hyper-parameters
    parser.add_argument("--inv_epochs", type=int, default=10)
    parser.add_argument("--inv_lr", type=float, default=1e-3)
    parser.add_argument("--policy_epochs", type=int, default=20)
    parser.add_argument("--policy_lr", type=float, default=1e-3)
    parser.add_argument("--iter_inv_epochs", type=int, default=5)
    parser.add_argument("--iter_policy_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=10_000,
                        help="Evaluation interval in BC steps")
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # Reproducibility
    # ----------------------------------------------------------------------- #
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----------------------------------------------------------------------- #
    # Environment
    # ----------------------------------------------------------------------- #
    env_id = "CartPole-v1" if args.env == "cartpole" else "HalfCheetah-v4"
    discrete = args.env == "cartpole"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # ----------------------------------------------------------------------- #
    # TensorBoard
    # ----------------------------------------------------------------------- #
    log_dir = f"logs/bco_{args.env}_{args.demo_episodes}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ----------------------------------------------------------------------- #
    # Global counters
    # ----------------------------------------------------------------------- #
    env_steps = 0        # real env interactions
    total_bc_steps = 0   # supervised samples processed
    EVAL_INTERVAL = args.eval_interval  # BC steps between evaluations

    # ----------------------------------------------------------------------- #
    # 1) Pre-demonstration phase
    # ----------------------------------------------------------------------- #
    print("Collecting random exploration data …")
    s_pre, s_next_pre, a_pre = collect_exploration_data(env, args.pre_interactions)
    env_steps += args.pre_interactions
    loader_pre = create_dataloader(s_pre, s_next_pre, a_pre, args.batch_size)

    inv_model = InverseDynamicsModel(obs_dim, act_dim, discrete=discrete)
    print("Training inverse dynamics model …")
    inv_model = train_inverse_model(inv_model, loader_pre,
                                    discrete=discrete,
                                    epochs=args.inv_epochs,
                                    lr=args.inv_lr,
                                    writer=writer)

    # ----------------------------------------------------------------------- #
    # 2) Load demonstrations & infer actions
    # ----------------------------------------------------------------------- #
    demo_path = args.demo_file or os.path.join("..", "data", "demonstrations", str(args.demo_episodes), f"{args.env}_demonstrations_{args.demo_episodes}.npy")
    print(f"Loading demonstrations from {demo_path}")
    demo = np.load(demo_path, allow_pickle=True)

    if isinstance(demo, np.ndarray) and demo.dtype == object:
        states = np.concatenate([np.asarray(getattr(tr, "obs", tr), dtype=np.float32)
                                 for tr in demo.tolist()], axis=0)
    else:
        states = np.asarray(getattr(demo, "obs", demo), dtype=np.float32)

    print(f"  Loaded {len(states)} states from demonstrations")
    states_bc, inf_actions = infer_expert_actions(inv_model, states, discrete)

    # ----------------------------------------------------------------------- #
    # 3) Evaluation helper
    # ----------------------------------------------------------------------- #
    def maybe_evaluate(pol_net, cur_bc_steps: int, tag: str):
        """Evaluate/record every *EVAL_INTERVAL* BC samples."""
        if cur_bc_steps == 0 or cur_bc_steps % EVAL_INTERVAL:
            return

        mean_r, std_r = evaluate_policy(pol_net, env, discrete, n_episodes=5)
        # log rewards indexed by different x-axes
        writer.add_scalar("eval/mean_reward", mean_r, cur_bc_steps)
        writer.add_scalar("eval/std_reward",  std_r,  cur_bc_steps)

        writer.add_scalar("progress/env_steps",      env_steps,      cur_bc_steps)
        writer.add_scalar("progress/total_bc_steps", cur_bc_steps,   cur_bc_steps)

        writer.add_scalar("reward/env_steps",        mean_r, env_steps)
        writer.add_scalar("reward/total_bc_steps",   mean_r, cur_bc_steps)

        print(f"[{tag}] BC={cur_bc_steps:>7d} | ENV={env_steps:>7d} "
              f"| mean_r={mean_r:7.2f} ± {std_r:5.2f}")

    # ----------------------------------------------------------------------- #
    # 4) Initial evaluation
    # ----------------------------------------------------------------------- #
    policy = PolicyNetwork(obs_dim, act_dim, discrete)
    mean_r, std_r = evaluate_policy(policy, env, discrete, n_episodes=5)

    writer.add_scalar("eval/mean_reward", mean_r, 0)
    writer.add_scalar("eval/std_reward",  std_r, 0)
    writer.add_scalar("progress/env_steps",      env_steps, 0)
    writer.add_scalar("progress/total_bc_steps", total_bc_steps, 0)
    writer.add_scalar("reward/env_steps",        mean_r, env_steps)
    writer.add_scalar("reward/total_bc_steps",   mean_r, total_bc_steps)

    print(f"[init] BC=0 | ENV={env_steps} | mean_r={mean_r:.2f} ± {std_r:.2f}")

    # ----------------------------------------------------------------------- #
    # 5) Helper: supervised BC training with logging
    # ----------------------------------------------------------------------- #
    def train_policy_with_logging(pol_net, st, act, epochs, lr, tag):
        nonlocal total_bc_steps
        opt = torch.optim.Adam(pol_net.parameters(), lr=lr)
        crit = torch.nn.CrossEntropyLoss() if discrete else torch.nn.MSELoss()

        ds = torch.utils.data.TensorDataset(torch.tensor(st, dtype=torch.float32),
                                            torch.tensor(act))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        for _ in range(epochs):
            for s_b, a_b in dl:
                opt.zero_grad()
                pred = pol_net(s_b)
                loss = crit(pred, a_b.long() if discrete else a_b.float())
                loss.backward()
                opt.step()

                total_bc_steps += s_b.size(0)
                maybe_evaluate(pol_net, total_bc_steps, tag)

    # ----------------------------------------------------------------------- #
    # 6) Main training loop
    # ----------------------------------------------------------------------- #
    if args.alpha == 0.0:
        print(">>> Running BCO(0)")
        train_policy_with_logging(policy, states_bc, inf_actions,
                                  args.policy_epochs, args.policy_lr, "BCO0")
    else:
        print(">>> Running BCO(alpha)")
        train_policy_with_logging(policy, states_bc, inf_actions,
                                  args.policy_epochs, args.policy_lr, "iter0")

        post_budget = int(args.alpha * args.pre_interactions)

        for it in range(1, args.num_iterations + 1):
            print(f"--- Iter {it}: collect {post_budget} env steps ---")
            s_post, s_next_post, a_post = collect_policy_data(policy, env,
                                                             post_budget, discrete)
            env_steps += post_budget

            # Retrain inverse model on combined data
            s_all      = np.concatenate([s_pre, s_post], axis=0)
            s_next_all = np.concatenate([s_next_pre, s_next_post], axis=0)
            a_all      = np.concatenate([a_pre, a_post], axis=0)
            loader_all = create_dataloader(s_all, s_next_all, a_all, args.batch_size)

            print("  > Retraining inverse model")
            inv_model = train_inverse_model(inv_model, loader_all,
                                            discrete=discrete,
                                            epochs=args.iter_inv_epochs,
                                            lr=args.inv_lr,
                                            writer=writer)

            print("  > Re-inferring actions & retraining policy")
            states_bc, inf_actions = infer_expert_actions(inv_model, states, discrete)
            train_policy_with_logging(policy, states_bc, inf_actions,
                                      args.iter_policy_epochs, args.policy_lr,
                                      f"iter{it}")

    # ----------------------------------------------------------------------- #
    # 7) Save final policy
    # ----------------------------------------------------------------------- #
    if args.env == "cartpole":
        name_suffix = f"bco_cartpole_{args.demo_episodes}"
    else:
        name_suffix = f"bco_halfcheetah_{args.demo_episodes}"
    os.makedirs(f"models/{name_suffix}", exist_ok=True)
    fname = f"{name_suffix}_{env_steps}_{args.alpha}.pt" if args.alpha > 0 else f"{name_suffix}_{env_steps}.pt"
    torch.save(policy.state_dict(), os.path.join(f"models/{name_suffix}", fname))
    print(f"Saved policy to models/{fname}")

    writer.close()
    env.close()


if __name__ == "__main__":
    print("Example of running normal BCO:")
    print(" python train_bco.py --env cartpole --pre_interactions 200 --inv_epochs 5 --policy_epochs 5 --policy_lr 3e-4  --seed 42 --eval_interval 100")
    print("Example of running BCO(alpha):")
    print("python train_bco.py --env halfcheetah --pre_interactions 20000 --inv_epochs 50 --policy_epochs 100 --policy_lr 3e-4 --alpha 0.5 --num_iterations 5 --seed 42")
    main()
