import argparse
import os
import json
import numpy as np
from datetime import datetime

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))


def make_env(env_name, seed):
    """Create a Safety-Gymnasium environment."""
    import safety_gymnasium
    import gymnasium as gym

    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env

    return _init


def run_single_seed(env_name, seed, total_steps, log_dir, config):
    """Run one training seed and return the log directory."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.logger import configure
    from algos.safe_ppo import SafePPO_GBRL

    seed_dir = os.path.join(log_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Create vectorised environment
    num_envs = config.get("num_envs", 16)
    env = SubprocVecEnv([make_env(env_name, seed + i) for i in range(num_envs)])

    # Model with corrected configuration (all four requirements)
    tree_optimizer = {
        "policy_optimizer": {
            "lr": config.get("policy_lr", 0.031),
            "T": total_steps,
        },
        "value_optimizer": {
            "lr": config.get("value_lr", 0.05),
            "T": total_steps,
        },
        "cost_value_optimizer": {
            "lr": config.get("cost_value_lr", 0.05),
            "T": total_steps,
        },
    }

    model = SafePPO_GBRL(
        policy="GBTMultiPolicy",
        env=env,
        n_steps=config.get("n_steps", 512),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 20),
        gamma=config.get("gamma", 0.999),
        gae_lambda=config.get("gae_lambda", 0.98),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.0033),
        seed=seed,
        verbose=1,
        # Corrected configuration (Requirements 1-4)
        policy_kwargs=dict(
            shared_tree_struct=False,  # Req 3: independent ensembles
        ),
        normalize_reward=False,  # Req 4: symmetric normalisation
        cost_threshold=config.get("cost_threshold", 25.0),
        use_safety_projection=True,
        tree_optimizer=tree_optimizer,
        # Tree cap (Req 5: bounded ensemble)
        # max_policy_trees=config.get("max_trees", 25000),
        # max_value_trees=config.get("max_trees", 25000),
    )

    logger = configure(seed_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=total_steps, progress_bar=True)

    # Save final model
    model.save(os.path.join(seed_dir, "final_model"))
    env.close()

    return seed_dir


def aggregate_results(log_dir, seeds):
    """
    Read CSV logs from each seed directory and compute
    summary statistics (median, IQR, mean, std).
    """
    import pandas as pd

    all_data = {}

    for seed in seeds:
        csv_path = os.path.join(log_dir, f"seed_{seed}", "progress.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data[seed] = df

    if not all_data:
        print("No results found.")
        return

    # Metrics to aggregate
    metrics = [
        "rollout/ep_cost_mean",
        "rollout/ep_rew_mean",
        "train/explained_variance",
        "safety/phi_deg",
        "safety/alpha_predicted",
        "safety/alpha_observed",
        "ev_bound/ev_upper_bound",
        "ev_bound/ev_actual",
    ]

    summary = {}
    for metric in metrics:
        values_per_seed = []
        for seed, df in all_data.items():
            if metric in df.columns:
                # Take the last 20% of training as "converged" performance
                n = len(df)
                tail = df[metric].iloc[int(0.8 * n):].dropna()
                if len(tail) > 0:
                    values_per_seed.append(tail.mean())

        if values_per_seed:
            arr = np.array(values_per_seed)
            summary[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n_seeds": len(values_per_seed),
            }

    # Save summary
    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Metric':<40} {'Mean±Std':<20} {'Median [IQR]':<25}")
    print("=" * 80)
    for metric, stats in summary.items():
        short = metric.split("/")[-1]
        mean_std = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
        median_iqr = f"{stats['median']:.3f} [{stats['q25']:.3f}, {stats['q75']:.3f}]"
        print(f"{short:<40} {mean_std:<20} {median_iqr:<25}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed SafePPO-GBRL runner")
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds")
    parser.add_argument("--total_steps", type=int, default=1_500_000)
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Starting seed (will use base_seed, base_seed+1, ...)")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--sequential", action="store_true",
                        help="Run seeds sequentially (default: sequential)")
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/multiseed_{args.env}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)

    config = {
        "num_envs": 16,
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "policy_lr": 0.031,
        "value_lr": 0.05,
        "cost_value_lr": 0.05,
        "cost_threshold": 25.0,
        "max_trees": 25000,
        "grow_policy": "oblovious",
        "max_policy_grad_norm": 150,
        "max_value_grad_norm": 10,
    }

    # Save config
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump({"env": args.env, "seeds": seeds, "total_steps": args.total_steps,
                   **config}, f, indent=2)

    print(f"Running {len(seeds)} seeds on {args.env}")
    print(f"Logging to {args.log_dir}")

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"  SEED {seed}")
        print(f"{'=' * 60}")
        run_single_seed(args.env, seed, args.total_steps, args.log_dir, config)

    aggregate_results(args.log_dir, seeds)


if __name__ == "__main__":
    main()