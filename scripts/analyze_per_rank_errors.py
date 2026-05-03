#!/usr/bin/env python3
"""
Analyze per-rank taxonomic errors to address reviewer feedback.

This script:
1. Loads a trained model checkpoint
2. Runs inference on validation set
3. Computes error rates at each taxonomic level
4. Shows where the model makes mistakes (genus vs family vs species)
5. Plots taxonomic distance histogram

Usage:
    python scripts/analyze_per_rank_errors.py \
        --checkpoint outputs/multiscale_4scales_taxloss/checkpoints/best-*.ckpt \
        --config config/experiment-multiscale.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from data.data import build_multiscale_dataloaders, load_and_encode_taxonomy
from src.config import load_config
from src.models import MultiScaleTaxonomicClassifier


def compute_taxonomic_distance(pred_lineage, true_lineage):
    """Compute taxonomic distance between predicted and true lineages."""
    levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # Find lowest common ancestor
    lca_depth = 0
    for i, level in enumerate(levels):
        if pred_lineage.get(level) == true_lineage.get(level):
            lca_depth = i + 1
        else:
            break

    # Distance = 2 * (total_depth - lca_depth)
    distance = 2 * (len(levels) - lca_depth)
    return distance, lca_depth


def analyze_errors(model, dataloader, taxonomy_df, encoders, id_to_name, device):
    """Analyze errors at each taxonomic rank."""

    levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # Collect predictions and ground truth
    all_predictions = {level: [] for level in levels}
    all_ground_truth = {level: [] for level in levels}
    all_distances = []
    error_by_rank = {level: [] for level in levels}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing"):
            images, labels = batch
            images = {k: v.to(device) for k, v in images.items()}

            # Get predictions
            outputs = model(images)

            # Process each level
            for level in levels:
                if level in outputs:
                    preds = outputs[level].argmax(dim=1).cpu().numpy()
                    truths = labels[level].cpu().numpy()

                    all_predictions[level].extend(preds)
                    all_ground_truth[level].extend(truths)

                    # Track errors at this level
                    errors = (preds != truths).astype(int)
                    error_by_rank[level].extend(errors)

            # Compute taxonomic distances
            species_preds = outputs["species"].argmax(dim=1).cpu().numpy()
            species_truths = labels["species"].cpu().numpy()

            for pred_idx, true_idx in zip(species_preds, species_truths):
                # Get full lineage for predicted and true species
                pred_species = id_to_name["species"][pred_idx]
                true_species = id_to_name["species"][true_idx]

                pred_lineage = taxonomy_df[taxonomy_df["species"] == pred_species].iloc[0].to_dict()
                true_lineage = taxonomy_df[taxonomy_df["species"] == true_species].iloc[0].to_dict()

                distance, lca_depth = compute_taxonomic_distance(pred_lineage, true_lineage)
                all_distances.append(distance)

    # Compute metrics
    results = {}

    print("\n" + "="*80)
    print("PER-RANK ERROR ANALYSIS")
    print("="*80)
    print(f"\n{'Rank':<12} {'Accuracy':>10} {'Error Rate':>12} {'Count':>10}")
    print("-"*80)

    for level in levels:
        preds = np.array(all_predictions[level])
        truths = np.array(all_ground_truth[level])

        accuracy = (preds == truths).mean()
        error_rate = (preds != truths).mean()

        results[level] = {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "total": len(preds),
        }

        print(f"{level:<12} {accuracy:>10.4f} {error_rate:>12.4f} {len(preds):>10}")

    # Taxonomic distance analysis
    distances_array = np.array(all_distances)
    mean_distance = distances_array.mean()

    print("\n" + "="*80)
    print("TAXONOMIC DISTANCE ANALYSIS")
    print("="*80)
    print(f"Mean taxonomic distance: {mean_distance:.4f}")
    print(f"Std taxonomic distance: {distances_array.std():.4f}")
    print(f"\nDistance distribution:")
    for d in range(0, 15, 2):
        count = (distances_array == d).sum()
        pct = (distances_array == d).mean() * 100
        print(f"  Distance {d:2d}: {count:5d} ({pct:5.1f}%)")

    return results, distances_array, error_by_rank


def plot_results(results, distances, output_dir):
    """Generate plots for the paper."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Error rate by taxonomic rank
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    error_rates = [results[level]["error_rate"] * 100 for level in levels]

    ax.bar(levels, error_rates, color='steelblue')
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_xlabel('Taxonomic Rank', fontsize=12)
    ax.set_title('Error Rate by Taxonomic Rank', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_rank.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'error_by_rank.png'}")

    # Plot 2: Taxonomic distance histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(distances, bins=range(0, 15, 2), color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Taxonomic Distance', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Taxonomic Distances (Errors)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_histogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'distance_histogram.png'}")

    # Plot 3: Cumulative error cascade
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = [results[level]["accuracy"] * 100 for level in levels]

    ax.plot(levels, accuracies, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Taxonomic Rank', fontsize=12)
    ax.set_title('Hierarchical Accuracy Cascade', fontsize=14)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_cascade.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_cascade.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="analysis/per_rank_errors", help="Output directory")
    args = parser.parse_args()

    # Load config (uses defaults merged with YAML overrides)
    cfg = load_config(args.config)

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        taxonomy_csv=cfg.paths.taxonomy_csv,
        levels=list(cfg.data.taxonomy_levels),
    )

    # Build validation dataloader
    print("Building dataloader...")
    dataloaders, _ = build_multiscale_dataloaders(
        cfg=cfg,
        taxonomy_df=taxonomy_df,
        encoders=encoders,
        scales=["1x", "3x", "5x", "full"],
    )
    val_loader = dataloaders["val"]

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiScaleTaxonomicClassifier.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=["1x", "3x", "5x", "full"],
        strict=False,
    )
    model = model.to(device)

    # Analyze errors
    print("\nAnalyzing errors...")
    results, distances, error_by_rank = analyze_errors(
        model, val_loader, taxonomy_df, encoders, id_to_name, device
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, distances, args.output_dir)

    # Save results to CSV
    output_dir = Path(args.output_dir)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'per_rank_metrics.csv')
    print(f"\nSaved: {output_dir / 'per_rank_metrics.csv'}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
