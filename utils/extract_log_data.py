import argparse
import os
import glob
import json
import numpy as np

try:
    from tbparse import SummaryReader
except ImportError:
    print("Install tbparse: pip install tbparse --break-system-packages")
    exit(1)


def extract_experiment(log_dir, label):
    if log_dir is None or not os.path.exists(log_dir):
        return None

    seed_dirs = sorted(glob.glob(os.path.join(log_dir, "seed_*")))
    if not seed_dirs:
        print(f"  [{label}] No seed directories found in {log_dir}")
        return None

    experiment = {"label": label, "seeds": {}}

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        try:
            reader = SummaryReader(seed_dir, pivot=True)
            df = reader.scalars
            if len(df) == 0:
                print(f"  [{label}/{seed_name}] No data")
                continue

            seed_data = {}
            for col in df.columns:
                if col in ['step', 'wall_time']:
                    continue
                series = df[['step', col]].dropna()
                if len(series) == 0:
                    continue
                seed_data[col] = {
                    "steps": series['step'].tolist(),
                    "values": series[col].tolist(),
                }

            experiment["seeds"][seed_name] = seed_data
            n_metrics = len(seed_data)
            n_steps = max(len(v["steps"]) for v in seed_data.values()) if seed_data else 0
            print(f"  [{label}/{seed_name}] {n_metrics} metrics, {n_steps} timesteps")

        except Exception as e:
            print(f"  [{label}/{seed_name}] Error: {e}")

    # Compute cross-seed summary for key metrics
    key_metrics = [
        "rollout/ep_cost_mean", "rollout/ep_rew_mean",
        "train/explained_variance", "train/approx_kl",
        "train/policy_gradient_loss", "train/value_loss",
        "safety/phi_deg_mean", "safety/alpha_observed_deg_mean",
        "safety/rho_mean", "safety/lambda_star_mean",
        "safety/projection_active_frac", "safety/violation_mean",
        "safety/b_rms_mean", "safety/alpha_mae_deg",
        "ev_bound/ev_actual", "ev_bound/ev_upper_bound",
        "ev_bound/var_delta", "ev_bound/var_R2",
        "ev_bound/regime_switch",
        "param/std", "param/log_std",
        "param/theta_max", "param/theta_min",
        "train/policy_num_trees",
    ]

    summary = {}
    for metric in key_metrics:
        per_seed_means = []
        per_seed_finals = []
        all_values = []

        for seed_name, seed_data in experiment["seeds"].items():
            if metric in seed_data:
                vals = seed_data[metric]["values"]
                # Filter out NaN/Inf
                clean = [v for v in vals if v is not None and np.isfinite(v)]
                if clean:
                    per_seed_means.append(np.mean(clean))
                    per_seed_finals.append(clean[-1])
                    all_values.extend(clean)

        if per_seed_means:
            summary[metric] = {
                "n_seeds": len(per_seed_means),
                "mean_of_means": float(np.mean(per_seed_means)),
                "std_of_means": float(np.std(per_seed_means)),
                "mean_of_finals": float(np.mean(per_seed_finals)),
                "std_of_finals": float(np.std(per_seed_finals)),
                "global_min": float(np.min(all_values)),
                "global_max": float(np.max(all_values)),
                "global_median": float(np.median(all_values)),
            }

    experiment["summary"] = summary
    return experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal1_dir", type=str, default=None)
    parser.add_argument("--circle1_dir", type=str, default=None)
    parser.add_argument("--lagrangian_goal1_dir", type=str, default=None)
    parser.add_argument("--lagrangian_circle1_dir", type=str, default=None)
    parser.add_argument("--nocap_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="paper_data.json")
    args = parser.parse_args()

    data = {}

    for label, dir_path in [
        ("projection_goal1", args.goal1_dir),
        ("projection_circle1", args.circle1_dir),
        ("lagrangian_goal1", args.lagrangian_goal1_dir),
        ("lagrangian_circle1", args.lagrangian_circle1_dir),
        ("nocap_goal1", args.nocap_dir),
    ]:
        if dir_path:
            print(f"\nExtracting {label} from {dir_path}...")
            result = extract_experiment(dir_path, label)
            if result:
                data[label] = result

    # Save
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    # Print file size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved to {args.output} ({size_mb:.1f} MB)")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Metric':<35}", end="")
    for label in data:
        short = label.replace("projection_", "P:").replace("lagrangian_", "L:").replace("nocap_", "NC:")
        print(f"  {short:<20}", end="")
    print()
    print("=" * 90)

    all_metrics = set()
    for exp in data.values():
        all_metrics.update(exp.get("summary", {}).keys())

    for metric in sorted(all_metrics):
        short = metric.split("/")[-1]
        print(f"{short:<35}", end="")
        for label, exp in data.items():
            s = exp.get("summary", {}).get(metric)
            if s:
                print(f"  {s['mean_of_means']:>8.3f} ± {s['std_of_means']:<8.3f}", end="")
            else:
                print(f"  {'--':<20}", end="")
        print()

    print(f"\nUpload {args.output} to Claude for paper revision.")


if __name__ == "__main__":
    main()