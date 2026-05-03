#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Taxonomic Distance Evaluation.

Computes BCa (bias-corrected and accelerated) bootstrap confidence intervals
for submission CSVs, and paired bootstrap tests for comparing models.

Usage:
    # Single submission CI
    python scripts/bootstrap_confidence.py \
        --submissions submission_minrisk_tta_hflip.csv \
        --solution solution_kaggle_2025.csv

    # Pairwise comparison
    python scripts/bootstrap_confidence.py \
        --submissions submission_no_parent_cond_10fold.csv submission_minrisk_tta_hflip.csv \
        --solution solution_kaggle_2025.csv

    # All key ablation CIs with LaTeX output
    python scripts/bootstrap_confidence.py \
        --submissions submission_ensemble_10fold.csv submission_minrisk_tta_hflip.csv \
                      submission_no_parent_cond_10fold.csv \
        --solution solution_kaggle_2025.csv \
        --latex paper/tables/bootstrap_ci.tex
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_per_sample_distances(
    submission_path: str,
    solution_path: str,
    distance_matrix_path: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute per-sample taxonomic distances between submission and ground truth.

    Args:
        submission_path: Path to submission CSV (annotation_id, concept_name)
        solution_path: Path to solution CSV (annotation_id, concept_name)
        distance_matrix_path: Path to taxonomic distance matrix CSV

    Returns:
        (distances, annotation_ids): Per-sample distance array and annotation IDs
    """
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)

    # Merge on annotation_id
    merged = solution.merge(
        submission,
        on="annotation_id",
        how="left",
        suffixes=("_true", "_pred"),
    )

    # Load distance matrix
    dist_df = pd.read_csv(distance_matrix_path, index_col=0)

    # Compute per-sample distances
    distances = []
    annotation_ids = []
    for _, row in merged.iterrows():
        true_sp = row["concept_name_true"]
        pred_sp = row.get("concept_name_pred", "unknown")
        if pd.isna(pred_sp):
            pred_sp = "unknown"

        if true_sp == pred_sp:
            d = 0.0
        elif true_sp in dist_df.index and pred_sp in dist_df.columns:
            d = float(dist_df.loc[true_sp, pred_sp])
        else:
            d = 7.0  # Max distance for unknown pairs

        distances.append(d)
        annotation_ids.append(row["annotation_id"])

    return np.array(distances), annotation_ids


def bootstrap_ci(
    distances: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = "bca",
    seed: int = 42,
) -> Dict:
    """
    Compute bootstrap confidence interval for mean taxonomic distance.

    Args:
        distances: Per-sample distance array (N,)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95)
        method: "bca" for bias-corrected accelerated, "percentile" for basic
        seed: Random seed

    Returns:
        Dict with mean, se, ci_lower, ci_upper, method
    """
    rng = np.random.RandomState(seed)
    n = len(distances)
    observed_mean = np.mean(distances)

    # Bootstrap resampling
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = distances[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    boot_se = np.std(boot_means, ddof=1)

    if method == "percentile":
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(boot_means, 100 * alpha)
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha))
    elif method == "bca":
        # BCa: Bias-corrected and accelerated
        alpha = (1 - confidence) / 2

        # Bias correction factor z0
        prop_below = np.mean(boot_means < observed_mean)
        # Clamp to avoid infinite z-score
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        from scipy.stats import norm
        z0 = norm.ppf(prop_below)

        # Acceleration factor a (jackknife estimate)
        jackknife_means = np.empty(n)
        for i in range(n):
            jackknife_means[i] = (np.sum(distances) - distances[i]) / (n - 1)
        jack_mean = np.mean(jackknife_means)
        jack_diff = jack_mean - jackknife_means
        a = np.sum(jack_diff ** 3) / (6 * (np.sum(jack_diff ** 2)) ** 1.5 + 1e-10)

        # Adjusted percentiles
        z_alpha = norm.ppf(alpha)
        z_1alpha = norm.ppf(1 - alpha)

        alpha1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        ci_lower = np.percentile(boot_means, 100 * alpha1)
        ci_upper = np.percentile(boot_means, 100 * alpha2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bca' or 'percentile'.")

    return {
        "mean": observed_mean,
        "se": boot_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "method": method,
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def paired_bootstrap_test(
    distances_a: np.ndarray,
    distances_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Paired bootstrap test: is model A significantly different from model B?

    Tests H0: mean(A) = mean(B) using paired bootstrap on per-sample differences.

    Args:
        distances_a: Per-sample distances for model A (N,)
        distances_b: Per-sample distances for model B (N,)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with delta (A-B), p_value, ci_lower, ci_upper for the difference
    """
    assert len(distances_a) == len(distances_b), "Must have same number of samples"

    rng = np.random.RandomState(seed)
    n = len(distances_a)
    diffs = distances_a - distances_b
    observed_delta = np.mean(diffs)

    # Bootstrap the differences
    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_deltas[i] = np.mean(diffs[idx])

    # p-value: proportion of bootstrap samples with sign opposite to observed
    if observed_delta > 0:
        p_value = np.mean(boot_deltas <= 0) * 2  # two-sided
    elif observed_delta < 0:
        p_value = np.mean(boot_deltas >= 0) * 2  # two-sided
    else:
        p_value = 1.0
    p_value = min(p_value, 1.0)

    # 95% CI for the difference
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    return {
        "mean_a": np.mean(distances_a),
        "mean_b": np.mean(distances_b),
        "delta": observed_delta,
        "delta_pct": observed_delta / np.mean(distances_b) * 100 if np.mean(distances_b) > 0 else 0,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def format_ci(result: Dict, precision: int = 3) -> str:
    """Format CI result as string."""
    return (
        f"{result['mean']:.{precision}f} "
        f"[{result['ci_lower']:.{precision}f}, {result['ci_upper']:.{precision}f}]"
    )


def generate_latex_table(
    results: List[Tuple[str, Dict]],
    pairwise: Optional[List[Tuple[str, str, Dict]]] = None,
) -> str:
    """Generate LaTeX table for CIs and pairwise comparisons."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bootstrap confidence intervals (95\%, BCa, $N=10{,}000$) for mean taxonomic distance on the full test set ($n=788$).}")
    lines.append(r"\label{tab:bootstrap_ci}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Mean & SE & 95\% CI & Width \\")
    lines.append(r"\midrule")

    for name, r in results:
        short_name = name.replace("submission_", "").replace(".csv", "").replace("_", r"\_")
        lines.append(
            f"  {short_name} & {r['mean']:.3f} & {r['se']:.3f} & "
            f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] & {r['ci_width']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if pairwise:
        lines.append("")
        lines.append(r"\vspace{1em}")
        lines.append(r"\begin{tabular}{llccc}")
        lines.append(r"\toprule")
        lines.append(r"Model A & Model B & $\Delta$ (A$-$B) & 95\% CI & $p$-value \\")
        lines.append(r"\midrule")

        for name_a, name_b, pr in pairwise:
            sa = name_a.replace("submission_", "").replace(".csv", "").replace("_", r"\_")
            sb = name_b.replace("submission_", "").replace(".csv", "").replace("_", r"\_")
            sig = "*" if pr["significant_005"] else ""
            if pr["significant_001"]:
                sig = "**"
            lines.append(
                f"  {sa} & {sb} & {pr['delta']:+.3f} & "
                f"[{pr['ci_lower']:+.3f}, {pr['ci_upper']:+.3f}] & "
                f"{pr['p_value']:.4f}{sig} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for taxonomic distance evaluation"
    )
    parser.add_argument(
        "--submissions",
        type=str,
        nargs="+",
        required=True,
        help="Submission CSV file(s) to evaluate",
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
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bca",
        choices=["bca", "percentile"],
        help="Bootstrap CI method (default: bca)",
    )
    parser.add_argument(
        "--latex",
        type=str,
        default=None,
        help="Output LaTeX table file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Bootstrap Confidence Intervals for Taxonomic Distance")
    print(f"  Method: {args.method.upper()}")
    print(f"  Bootstrap samples: {args.n_bootstrap:,}")
    print(f"  Confidence level: 95%")
    print("=" * 70)

    # Load per-sample distances for each submission
    all_distances = {}
    results = []

    for sub_path in args.submissions:
        name = os.path.basename(sub_path)
        print(f"\nLoading {name}...")

        distances, ann_ids = load_per_sample_distances(
            sub_path, args.solution, args.distance_matrix
        )
        all_distances[name] = distances

        # Compute CI
        ci = bootstrap_ci(
            distances,
            n_bootstrap=args.n_bootstrap,
            method=args.method,
            seed=args.seed,
        )
        results.append((name, ci))

        print(f"  N = {ci['n_samples']}")
        print(f"  Mean taxonomic distance: {ci['mean']:.4f}")
        print(f"  Standard error: {ci['se']:.4f}")
        print(f"  95% CI ({ci['method'].upper()}): [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  CI width: {ci['ci_width']:.4f}")
        print(f"  Accuracy: {np.mean(distances == 0) * 100:.1f}%")

    # Pairwise comparisons if multiple submissions
    pairwise_results = []
    if len(args.submissions) > 1:
        print("\n" + "=" * 70)
        print("Pairwise Comparisons (Paired Bootstrap Test)")
        print("=" * 70)

        names = list(all_distances.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]
                pr = paired_bootstrap_test(
                    all_distances[name_a],
                    all_distances[name_b],
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                pairwise_results.append((name_a, name_b, pr))

                sig_str = ""
                if pr["significant_001"]:
                    sig_str = " **"
                elif pr["significant_005"]:
                    sig_str = " *"

                print(f"\n  {name_a}")
                print(f"    vs {name_b}")
                print(f"    Delta (A-B): {pr['delta']:+.4f} ({pr['delta_pct']:+.1f}%)")
                print(f"    95% CI: [{pr['ci_lower']:+.4f}, {pr['ci_upper']:+.4f}]")
                print(f"    p-value: {pr['p_value']:.4f}{sig_str}")

    # Generate LaTeX table
    if args.latex:
        latex = generate_latex_table(results, pairwise_results if pairwise_results else None)
        os.makedirs(os.path.dirname(args.latex) if os.path.dirname(args.latex) else ".", exist_ok=True)
        with open(args.latex, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table written to {args.latex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
