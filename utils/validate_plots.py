import argparse
import os
import glob
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    return pd.read_csv(path)


def plot_rotation_validation(log_dir, output_path):
    csv_files = sorted(glob.glob(os.path.join(log_dir, "seed_*/progress.csv")))
    if not csv_files:
        print("No CSV files found for rotation plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter of predicted vs observed alpha
    ax = axes[0]
    all_predicted, all_observed = [], []

    for csv_path in csv_files:
        df = load_csv(csv_path)
        if "safety/alpha_predicted" in df.columns and "safety/alpha_observed" in df.columns:
            mask = df["safety/alpha_predicted"].notna() & df["safety/alpha_observed"].notna()
            pred = df.loc[mask, "safety/alpha_predicted"].values
            obs = df.loc[mask, "safety/alpha_observed"].values
            all_predicted.extend(pred)
            all_observed.extend(obs)
            ax.scatter(np.degrees(pred), np.degrees(obs), alpha=0.3, s=10)

    if all_predicted:
        max_val = max(np.max(np.degrees(all_predicted)), np.max(np.degrees(all_observed)))
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="identity")
        ax.set_xlabel(r"Predicted $\alpha$ (degrees)", fontsize=12)
        ax.set_ylabel(r"Observed $\alpha$ (degrees)", fontsize=12)
        ax.set_title("Gradient Rotation Lemma Validation", fontsize=13)
        ax.legend()

    # Right: alpha vs phi, colored by rho
    ax = axes[1]
    for csv_path in csv_files:
        df = load_csv(csv_path)
        cols = ["safety/phi_deg", "safety/alpha_observed", "safety/rho"]
        if all(c in df.columns for c in cols):
            mask = df[cols[0]].notna()
            sc = ax.scatter(
                df.loc[mask, "safety/phi_deg"],
                np.degrees(df.loc[mask, "safety/alpha_observed"]),
                c=df.loc[mask, "safety/rho"],
                cmap="viridis", alpha=0.4, s=10,
            )

    ax.set_xlabel(r"Objective conflict $\varphi$ (degrees)", fontsize=12)
    ax.set_ylabel(r"Rotation $\alpha$ (degrees)", fontsize=12)
    ax.set_title(r"$\alpha$ vs $\varphi$ (color = $\rho$)", fontsize=13)
    if all_predicted:
        plt.colorbar(sc, ax=ax, label=r"$\rho$")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Rotation validation plot saved to {output_path}")
    plt.close()


def plot_ev_bound_validation(log_dir, output_path):
    csv_files = sorted(glob.glob(os.path.join(log_dir, "seed_*/progress.csv")))
    if not csv_files:
        print("No CSV files found for EV bound plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: EV bound and actual EV over time (one seed for clarity)
    ax = axes[0]
    df = load_csv(csv_files[0])  # use first seed for time series
    if "ev_bound/ev_actual" in df.columns and "ev_bound/ev_upper_bound" in df.columns:
        steps = df["time/total_timesteps"].values if "time/total_timesteps" in df.columns else np.arange(len(df))
        mask = df["ev_bound/ev_actual"].notna() & df["ev_bound/ev_upper_bound"].notna()

        ax.plot(steps[mask], df.loc[mask, "ev_bound/ev_upper_bound"],
                "r-", alpha=0.7, linewidth=1, label=r"Upper bound: $1 - \mathrm{Var}(\delta)/\mathrm{Var}(R_2)$")
        ax.plot(steps[mask], df.loc[mask, "ev_bound/ev_actual"],
                "b-", alpha=0.7, linewidth=1, label="Actual EV")

        # Mark regime switches
        if "ev_bound/regime_switch" in df.columns:
            switches = df.loc[mask & (df["ev_bound/regime_switch"] == 1.0)]
            if len(switches) > 0:
                switch_steps = steps[switches.index] if "time/total_timesteps" in df.columns else switches.index
                for s in switch_steps[:20]:  # limit markers
                    ax.axvline(s, color="gray", alpha=0.3, linestyle=":")

        ax.set_xlabel("Training steps", fontsize=12)
        ax.set_ylabel("Explained Variance", fontsize=12)
        ax.set_title("EV Degradation Bound (seed 0)", fontsize=13)
        ax.legend(fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)

    # Right: scatter of bound vs actual across all seeds and timesteps
    ax = axes[1]
    all_bounds, all_actual = [], []
    for csv_path in csv_files:
        df = load_csv(csv_path)
        if "ev_bound/ev_actual" in df.columns and "ev_bound/ev_upper_bound" in df.columns:
            mask = df["ev_bound/ev_actual"].notna() & df["ev_bound/ev_upper_bound"].notna()
            all_bounds.extend(df.loc[mask, "ev_bound/ev_upper_bound"].values)
            all_actual.extend(df.loc[mask, "ev_bound/ev_actual"].values)

    if all_bounds:
        ax.scatter(all_bounds, all_actual, alpha=0.2, s=5, color="steelblue")
        lims = [min(min(all_bounds), min(all_actual)) - 0.1,
                max(max(all_bounds), max(all_actual)) + 0.1]
        ax.plot(lims, lims, "k--", linewidth=1, label="EV = bound")
        ax.fill_between(lims, lims, [lims[0], lims[0]], alpha=0.05, color="green",
                        label="Bound holds (EV ≤ bound)")
        ax.set_xlabel("EV upper bound", fontsize=12)
        ax.set_ylabel("Actual EV", fontsize=12)
        ax.set_title("Proposition Validation (all seeds)", fontsize=13)
        ax.legend(fontsize=10)

        # Count violations
        violations = sum(1 for a, b in zip(all_actual, all_bounds) if a > b + 0.02)
        total = len(all_actual)
        ax.text(0.05, 0.95, f"Bound holds: {total - violations}/{total} ({100 * (total - violations) / total:.1f}%)",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"EV bound validation plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.log_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_rotation_validation(args.log_dir, os.path.join(output_dir, "rotation_validation.pdf"))
    plot_ev_bound_validation(args.log_dir, os.path.join(output_dir, "ev_bound_validation.pdf"))


if __name__ == "__main__":
    main()