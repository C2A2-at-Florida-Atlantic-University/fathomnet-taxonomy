#!/usr/bin/env python3
"""
Plot alpha sensitivity curve with bootstrap confidence intervals.

Reads submission CSVs for each alpha value and generates a publication-quality
plot showing mean taxonomic distance vs alpha with 95% CI error bands.

Usage:
    python scripts/plot_alpha_sensitivity.py
    python scripts/plot_alpha_sensitivity.py --output paper/figures/alpha_sensitivity.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.bootstrap_confidence import load_per_sample_distances, bootstrap_ci


def main():
    parser = argparse.ArgumentParser(
        description="Plot alpha sensitivity curve with bootstrap CIs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/alpha_sensitivity.pdf",
        help="Output figure path",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="solution_kaggle_2025.csv",
        help="Path to ground truth solution CSV",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
        help="Path to taxonomic distance matrix",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples",
    )
    args = parser.parse_args()

    # Alpha values and their submission files
    # alpha=0.3 uses the no-parent-cond 10fold (our best config)
    alpha_submissions = {
        0.0: "submission_10fold_alpha_0.0.csv",
        0.1: "submission_10fold_alpha_0.1.csv",
        0.2: "submission_10fold_alpha_0.2.csv",
        0.3: "submission_no_parent_cond_10fold.csv",
        0.5: "submission_10fold_alpha_0.5.csv",
    }

    alphas = []
    means = []
    ci_lowers = []
    ci_uppers = []

    for alpha, sub_file in sorted(alpha_submissions.items()):
        if not os.path.exists(sub_file):
            print(f"  Skipping alpha={alpha}: {sub_file} not found")
            continue

        print(f"Processing alpha={alpha} ({sub_file})...")
        distances, _ = load_per_sample_distances(
            sub_file, args.solution, args.distance_matrix
        )
        ci = bootstrap_ci(distances, n_bootstrap=args.n_bootstrap)

        alphas.append(alpha)
        means.append(ci["mean"])
        ci_lowers.append(ci["ci_lower"])
        ci_uppers.append(ci["ci_upper"])

        print(f"  Mean: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    if len(alphas) < 2:
        print("Need at least 2 alpha values to plot. Exiting.")
        return

    alphas = np.array(alphas)
    means = np.array(means)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Find best alpha
    best_idx = np.argmin(means)
    best_alpha = alphas[best_idx]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # CI band
    ax.fill_between(alphas, ci_lowers, ci_uppers, alpha=0.2, color="C0", label="95% CI")

    # Mean line
    ax.plot(alphas, means, "o-", color="C0", linewidth=2, markersize=6, label="Mean")

    # Mark best
    ax.plot(
        alphas[best_idx], means[best_idx], "*",
        color="C3", markersize=14, zorder=5,
        label=f"Best ($\\alpha$={best_alpha})",
    )

    ax.set_xlabel(r"$\alpha$ (distance loss weight)", fontsize=12)
    ax.set_ylabel("Mean Taxonomic Distance", fontsize=12)
    ax.set_title(r"$\alpha$ Sensitivity (10-fold ensemble, min-risk + hflip TTA)", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # x-axis ticks at tested values
    ax.set_xticks(alphas)
    ax.set_xlim(-0.03, max(alphas) + 0.03)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {args.output}")

    # Also save as PNG for quick viewing
    png_path = args.output.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"PNG preview saved to {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
