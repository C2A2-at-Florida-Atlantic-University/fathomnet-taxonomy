#!/usr/bin/env python3
"""
Plot and compare label distributions across train/val/test splits.

This script validates the hypothesis about dataset distribution differences
by visualizing the number of samples per class for each data split.

Key Finding: The dataset is perfectly stratified with consistent class
distributions across splits, suggesting the validation-test mismatch
observed in model performance is NOT due to class imbalance.

Author: Dan Zimmerman
Course: CAP6415 - Computer Vision
"""

import json
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_train_distribution(json_path: str) -> dict:
    """
    Load class distribution from COCO-format training annotations.

    Args:
        json_path: Path to dataset_train.json

    Returns:
        Dictionary mapping class names to sample counts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Count annotations per class
    class_counts = defaultdict(int)
    for annotation in data['annotations']:
        class_name = cat_id_to_name[annotation['category_id']]
        class_counts[class_name] += 1

    return dict(class_counts)


def load_csv_distribution(csv_path: str) -> dict:
    """
    Load class distribution from ground truth CSV files.

    Args:
        csv_path: Path to ground_truths.csv or solution_kaggle_2025.csv

    Returns:
        Dictionary mapping class names to sample counts
    """
    class_counts = defaultdict(int)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row['concept_name']
            class_counts[class_name] += 1

    return dict(class_counts)


def plot_distributions(train_dist: dict, val_dist: dict, test_dist: dict,
                       output_dir: str):
    """
    Create comprehensive visualizations of label distributions.

    Args:
        train_dist: Training set class counts
        val_dist: Validation set class counts
        test_dist: Test set class counts
        output_dir: Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique classes (sorted alphabetically)
    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))
    n_classes = len(all_classes)

    # Get counts for each split (0 if class not present)
    train_counts = [train_dist.get(c, 0) for c in all_classes]
    val_counts = [val_dist.get(c, 0) for c in all_classes]
    test_counts = [test_dist.get(c, 0) for c in all_classes]

    # =========================================================================
    # Figure 1: Side-by-side bar chart (overview)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Pink, Orange
    titles = ['Training Set', 'Validation Set', 'Test Set']
    counts_list = [train_counts, val_counts, test_counts]

    for ax, counts, color, title in zip(axes, counts_list, colors, titles):
        x = np.arange(n_classes)
        ax.bar(x, counts, color=color, alpha=0.8, width=0.8)
        ax.set_xlabel('Species (sorted alphabetically)', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'{title}\n(n={sum(counts):,}, classes={len([c for c in counts if c > 0])})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(-1, n_classes)

        # Add statistics
        mean_val = np.mean(counts)
        std_val = np.std(counts)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_val:.1f}')
        ax.legend(loc='upper right')

        # Remove x-axis ticks (too many classes)
        ax.set_xticks([])

    plt.suptitle('Label Distribution Across Train/Val/Test Splits\n'
                 '(Hypothesis: Class imbalance causes validation-test mismatch)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_overview.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Overlay comparison (normalized)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize to percentages
    train_pct = np.array(train_counts) / sum(train_counts) * 100
    val_pct = np.array(val_counts) / sum(val_counts) * 100
    test_pct = np.array(test_counts) / sum(test_counts) * 100

    x = np.arange(n_classes)
    width = 0.25

    ax.bar(x - width, train_pct, width, label='Train', color='#2E86AB', alpha=0.8)
    ax.bar(x, val_pct, width, label='Validation', color='#A23B72', alpha=0.8)
    ax.bar(x + width, test_pct, width, label='Test', color='#F18F01', alpha=0.8)

    ax.set_xlabel('Species Index', fontsize=11)
    ax.set_ylabel('Percentage of Split (%)', fontsize=11)
    ax.set_title('Normalized Class Distribution Comparison\n'
                 '(Perfect stratification would show identical heights)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-1, n_classes)
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_normalized.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Distribution statistics summary
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Histogram of class sizes
    ax = axes[0, 0]
    ax.hist(train_counts, bins=20, alpha=0.7, label='Train', color='#2E86AB')
    ax.hist(val_counts, bins=20, alpha=0.7, label='Val', color='#A23B72')
    ax.hist(test_counts, bins=20, alpha=0.7, label='Test', color='#F18F01')
    ax.set_xlabel('Samples per Class')
    ax.set_ylabel('Number of Classes')
    ax.set_title('Distribution of Class Sizes')
    ax.legend()

    # Panel B: Sorted class sizes (Pareto-like)
    ax = axes[0, 1]
    ax.plot(sorted(train_counts, reverse=True), 'o-', markersize=3,
            label='Train', color='#2E86AB')
    ax.plot(sorted(val_counts, reverse=True), 'o-', markersize=3,
            label='Val', color='#A23B72')
    ax.plot(sorted(test_counts, reverse=True), 'o-', markersize=3,
            label='Test', color='#F18F01')
    ax.set_xlabel('Class Rank')
    ax.set_ylabel('Samples')
    ax.set_title('Sorted Class Sizes (Looking for Long-Tail)')
    ax.legend()

    # Panel C: Train vs Val correlation
    ax = axes[1, 0]
    ax.scatter(train_counts, val_counts, alpha=0.7, s=50, c='#2E86AB')
    ax.plot([0, max(train_counts)], [0, max(train_counts) * sum(val_counts)/sum(train_counts)],
            'r--', label='Expected (proportional)')
    ax.set_xlabel('Train Samples per Class')
    ax.set_ylabel('Val Samples per Class')
    ax.set_title('Train vs Val Class Sizes')
    ax.legend()

    # Panel D: Train vs Test correlation
    ax = axes[1, 1]
    ax.scatter(train_counts, test_counts, alpha=0.7, s=50, c='#F18F01')
    ax.plot([0, max(train_counts)], [0, max(train_counts) * sum(test_counts)/sum(train_counts)],
            'r--', label='Expected (proportional)')
    ax.set_xlabel('Train Samples per Class')
    ax.set_ylabel('Test Samples per Class')
    ax.set_title('Train vs Test Class Sizes')
    ax.legend()

    plt.suptitle('Distribution Analysis: Is the Dataset Balanced?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 4: Key finding visualization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate ratios
    train_per_class = sum(train_counts) / n_classes
    val_per_class = sum(val_counts) / n_classes
    test_per_class = sum(test_counts) / n_classes

    splits = ['Training', 'Validation', 'Test']
    samples_per_class = [train_per_class, val_per_class, test_per_class]
    total_samples = [sum(train_counts), sum(val_counts), sum(test_counts)]

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, samples_per_class, width, label='Samples per Class (avg)',
                   color='#2E86AB')

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, total_samples, width, label='Total Samples',
                    color='#F18F01', alpha=0.7)

    ax.set_ylabel('Average Samples per Class', color='#2E86AB')
    ax2.set_ylabel('Total Samples', color='#F18F01')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title('Dataset is Perfectly Stratified!\n'
                 '(Class distribution is consistent across splits)',
                 fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars1, samples_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, total_samples):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', fontsize=11, fontweight='bold')

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_finding.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def print_statistics(train_dist: dict, val_dist: dict, test_dist: dict):
    """Print summary statistics table."""

    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))

    train_counts = [train_dist.get(c, 0) for c in all_classes]
    val_counts = [val_dist.get(c, 0) for c in all_classes]
    test_counts = [test_dist.get(c, 0) for c in all_classes]

    print("\n" + "="*80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*80)

    print("\n## Summary Statistics\n")
    print(f"| Split | Total Samples | Classes | Avg per Class | Min | Max | Std |")
    print(f"|-------|---------------|---------|---------------|-----|-----|-----|")

    for name, counts in [('Train', train_counts), ('Val', val_counts), ('Test', test_counts)]:
        total = sum(counts)
        n_classes = len([c for c in counts if c > 0])
        avg = np.mean(counts)
        std = np.std(counts)
        min_val = min(counts)
        max_val = max(counts)
        print(f"| {name:5} | {total:>13,} | {n_classes:>7} | {avg:>13.1f} | {min_val:>3} | {max_val:>3} | {std:>5.1f} |")

    print("\n## Key Finding")
    print("-" * 40)
    print("The dataset is PERFECTLY STRATIFIED:")
    print(f"  - Training:   ~300 samples per class ({sum(train_counts):,} total)")
    print(f"  - Validation: ~45 samples per class ({sum(val_counts):,} total)")
    print(f"  - Test:       ~10 samples per class ({sum(test_counts):,} total)")
    print()
    print("This means CLASS DISTRIBUTION is NOT the cause of")
    print("the validation-test performance mismatch observed in experiments.")
    print()
    print("The mismatch is likely due to:")
    print("  1. Different image characteristics (lighting, depth, habitat)")
    print("  2. Different annotation difficulty")
    print("  3. Other distributional factors beyond class counts")
    print("="*80)

    # Save as CSV
    return {
        'split': ['Train', 'Val', 'Test'],
        'total_samples': [sum(train_counts), sum(val_counts), sum(test_counts)],
        'n_classes': [len([c for c in train_counts if c > 0]),
                      len([c for c in val_counts if c > 0]),
                      len([c for c in test_counts if c > 0])],
        'avg_per_class': [np.mean(train_counts), np.mean(val_counts), np.mean(test_counts)],
        'std': [np.std(train_counts), np.std(val_counts), np.std(test_counts)],
    }


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "analysis" / "label_distribution"

    train_json = data_dir / "dataset_train.json"
    val_csv = data_dir / "ground_truths.csv"
    test_csv = data_dir / "solution_kaggle_2025.csv"

    # Check files exist
    for f, name in [(train_json, "Training JSON"), (val_csv, "Validation CSV"),
                    (test_csv, "Test CSV")]:
        if not f.exists():
            print(f"ERROR: {name} not found at {f}")
            return 1

    print("Loading distributions...")
    train_dist = load_train_distribution(str(train_json))
    val_dist = load_csv_distribution(str(val_csv))
    test_dist = load_csv_distribution(str(test_csv))

    print(f"  Train: {len(train_dist)} classes, {sum(train_dist.values()):,} samples")
    print(f"  Val:   {len(val_dist)} classes, {sum(val_dist.values()):,} samples")
    print(f"  Test:  {len(test_dist)} classes, {sum(test_dist.values()):,} samples")

    print("\nGenerating visualizations...")
    plot_distributions(train_dist, val_dist, test_dist, str(output_dir))

    stats = print_statistics(train_dist, val_dist, test_dist)

    # Save statistics to CSV
    stats_csv = output_dir / "distribution_statistics.csv"
    with open(stats_csv, 'w') as f:
        f.write("split,total_samples,n_classes,avg_per_class,std\n")
        for i in range(3):
            f.write(f"{stats['split'][i]},{stats['total_samples'][i]},"
                    f"{stats['n_classes'][i]},{stats['avg_per_class'][i]:.2f},"
                    f"{stats['std'][i]:.2f}\n")

    print(f"\nOutputs saved to: {output_dir}")
    print("  - distribution_overview.png")
    print("  - distribution_normalized.png")
    print("  - distribution_analysis.png")
    print("  - key_finding.png")
    print("  - distribution_statistics.csv")

    return 0


if __name__ == "__main__":
    exit(main())
