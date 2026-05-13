"""
Usage:
    python scripts/multi_seed_runner.py \
        --seeds 5 --base_seed 42 \
        --algo_type=sppo_gbrl --env_type=gym \
        --env_name=SafetyPointCircle1Gymnasium-v0 \
        --batch_size=512 --clip_range=0.2 --device=cuda \
        --ent_coef=0.03 --gae_lambda=0.98 --gamma=0.99 \
        --grow_policy=oblivious --n_epochs=20 --n_steps=512 \
        --num_envs=16 --total_n_steps=1200000 \
        --max_policy_grad_norm=150 --max_value_grad_norm=10
"""

import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

import safety_gymnasium

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

warnings.filterwarnings("ignore")

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from algos.safe_ppo import SafePPO_GBRL
from config.args import parse_args, process_logging, process_policy_kwargs
from policies.actor_critic_safe_policy import ActorCriticSafePolicy
from utils.helpers import set_seed


def run_single_seed(args, seed, log_dir):
    """
    Run one training seed using the same construction logic as train.py.
    """
    seed_dir = os.path.join(log_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Override seed and tensorboard log path
    args.seed = seed
    tensorboard_log = seed_dir

    # Create environment (same as train.py gym/safety-gym path)
    env_kwargs = args.env_kwargs if args.env_kwargs is not None else {}
    env = make_vec_env(
        args.env_name,
        n_envs=args.num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    # Optional VecNormalize wrapper (same as train.py)
    if getattr(args, 'wrapper', None) == 'normalize':
        wrapper_kwargs = getattr(args, 'wrapper_kwargs', {})
        wrapper_kwargs['gamma'] = args.gamma
        env = VecNormalize(env, **wrapper_kwargs)

    set_seed(seed)

    # Build algo_kwargs (same as train.py)
    algo_kwargs = process_policy_kwargs(args)
    print(f"[Seed {seed}] Training with algo_kwargs: {algo_kwargs}")

    # Construct model (same as train.py for sppo_gbrl)
    model = SafePPO_GBRL(
        policy=ActorCriticSafePolicy,
        env=env,
        tensorboard_log=tensorboard_log,
        _init_setup_model=True,
        **algo_kwargs,
    )

    model.learn(
        total_timesteps=args.total_n_steps,
        callback=None,
        log_interval=getattr(args, 'log_interval', 1),
        progress_bar=False,
    )

    # Save final model
    try:
        model.save(os.path.join(seed_dir, f"final_model_seed_{seed}"))
        print(f"[Seed {seed}] Complete. Saved to {seed_dir}")
    except Exception as e:
        print(f"[Seed {seed}] Could not save model: {e}")
    env.close()

    return seed_dir


def aggregate_results(log_dir, seeds):
    """Read CSV logs from each seed and compute summary statistics."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, skipping aggregation.")
        return

    all_data = {}
    for seed in seeds:
        # SB3 tensorboard logger also writes progress.csv
        csv_path = os.path.join(log_dir, f"seed_{seed}", "progress.csv")
        if os.path.exists(csv_path):
            all_data[seed] = pd.read_csv(csv_path)

    if not all_data:
        print("No progress.csv files found. Check if SB3 CSV logger is enabled.")
        return

    metrics = [
        "rollout/ep_cost_mean",
        "rollout/ep_rew_mean",
        "train/explained_variance",
        "safety/phi_deg_mean",
        "safety/alpha_predicted_deg_mean",
        "safety/alpha_observed_deg_mean",
        "safety/alpha_mae_deg",
        "ev_bound/ev_upper_bound",
        "ev_bound/ev_actual",
    ]

    summary = {}
    for metric in metrics:
        values_per_seed = []
        for seed, df in all_data.items():
            if metric in df.columns:
                tail = df[metric].iloc[int(0.8 * len(df)):].dropna()
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
                "n_seeds": len(values_per_seed),
            }

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Metric':<40} {'Mean ± Std':<20} {'Median [IQR]'}")
    print(f"{'='*70}")
    for metric, stats in summary.items():
        short = metric.split("/")[-1]
        print(f"{short:<40} {stats['mean']:.3f} ± {stats['std']:.3f}"
              f"      {stats['median']:.3f} [{stats['q25']:.3f}, {stats['q75']:.3f}]")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    # Inject --seeds and --base_seed before parse_args sees them
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--seeds", type=int, default=5)
    pre_parser.add_argument("--base_seed", type=int, default=42)
    pre_parser.add_argument("--multiseed_log_dir", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Temporarily replace sys.argv so parse_args() sees the train.py-style args
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()

    seeds = list(range(pre_args.base_seed, pre_args.base_seed + pre_args.seeds))

    if pre_args.multiseed_log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            str(ROOT_PATH), "logs",
            f"multiseed_{args.env_name}_{timestamp}"
        )
    else:
        log_dir = pre_args.multiseed_log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"seeds": seeds, "args": str(args)}, f, indent=2)

    print(f"Running {len(seeds)} seeds on {args.env_name}")
    print(f"Logging to {log_dir}")

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        run_single_seed(args, seed, log_dir)

    aggregate_results(log_dir, seeds)
    print("\nAll seeds complete.")