#!/usr/bin/env python3
"""
Calibration Analysis for FathomNet Ensemble Predictions.

Computes Expected Calibration Error (ECE), reliability diagrams, and tests
the effect of temperature scaling on min-risk inference.

Usage:
    python scripts/calibration_analysis.py \
        --checkpoints ./outputs/outputs/kfold_10fold/fold_*/checkpoints/*.ckpt \
        --solution solution_kaggle_2025.csv
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data import load_and_encode_taxonomy
from src.config import load_config
from src.inference.ensemble_predict import (
    MultiScaleTestDataset,
    apply_tta,
    load_models,
)
from src.inference.evaluate_on_test import (
    compute_taxonomic_score,
    load_distance_matrix,
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, dict]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: (N, C) probability matrix
        labels: (N,) integer ground-truth labels
        n_bins: Number of calibration bins

    Returns:
        ece: Expected Calibration Error
        bin_data: Dict with per-bin accuracy, confidence, and count
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
        else:
            bin_acc = 0.0
            bin_conf = (lo + hi) / 2
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(int(count))

    # Weighted average of |acc - conf| by bin count
    total = sum(bin_counts)
    ece = sum(
        (count / total) * abs(acc - conf)
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        if count > 0
    )

    return ece, {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return probabilities."""
    scaled = logits / temperature
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def ensemble_predict_with_logits(
    models: list,
    dataloader: DataLoader,
    device: torch.device,
    tta_views: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run ensemble prediction and return raw averaged logits (pre-softmax),
    probabilities, and annotation IDs.
    """
    tta_views = tta_views or ["original"]
    n_models = len(models)

    all_logits = []
    all_probs = []
    all_ann_ids = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Collecting logits"):
            images, annotation_ids, image_ids = batch_data
            images = {k: v.to(device) for k, v in images.items()}

            view_logits = []
            view_probs = []
            for aug_name in tta_views:
                aug_images = apply_tta(images, aug_name)
                batch_logits = []
                batch_probs = []
                for model in models:
                    outputs = model(aug_images)
                    logits = outputs["species"]
                    probs = torch.softmax(logits, dim=1)
                    batch_logits.append(logits)
                    batch_probs.append(probs)

                # Average logits and probs across models
                avg_logits = torch.stack(batch_logits).mean(dim=0)
                avg_probs = torch.stack(batch_probs).mean(dim=0)
                view_logits.append(avg_logits)
                view_probs.append(avg_probs)

            # Average across TTA views
            ensemble_logits = torch.stack(view_logits).mean(dim=0)
            ensemble_probs = torch.stack(view_probs).mean(dim=0)

            all_logits.append(ensemble_logits.cpu().numpy())
            all_probs.append(ensemble_probs.cpu().numpy())
            all_ann_ids.extend(annotation_ids)

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_probs, axis=0),
        all_ann_ids,
    )


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Model checkpoint paths",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="solution_kaggle_2025.csv",
        help="Ground truth solution CSV",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["1x", "3x", "5x", "full"],
    )
    parser.add_argument(
        "--n-bins", type=int, default=15, help="Number of ECE bins"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--tta",
        type=str,
        default=None,
        help="Comma-separated TTA augmentations",
    )
    args = parser.parse_args()

    # Expand glob patterns
    checkpoint_paths = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        checkpoint_paths.extend(sorted(matches) if matches else [pattern])
    checkpoint_paths = sorted(set(checkpoint_paths))
    print(f"Found {len(checkpoint_paths)} checkpoints")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Load taxonomy
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, list(cfg.data.taxonomy_levels)
    )

    # Species list for distance matrix
    species_list = [id_to_name["species"][i] for i in range(class_counts["species"])]
    species_to_idx = {name: i for i, name in enumerate(species_list)}

    # Load models
    models = load_models(
        checkpoint_paths,
        cfg,
        class_counts,
        id_to_name,
        args.scales,
        args.distance_matrix,
        device,
    )

    # Create test dataset
    test_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = MultiScaleTestDataset(
        coco_json_path=cfg.paths.test_coco_json,
        roi_root=cfg.paths.test_roi_root,
        scales=args.scales,
        transform=test_transform,
    )

    def collate_fn(batch):
        images_list, annotation_ids, image_ids = zip(*batch)
        images = {
            scale: torch.stack([img[scale] for img in images_list])
            for scale in args.scales
        }
        return images, list(annotation_ids), torch.tensor(image_ids, dtype=torch.long)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Parse TTA
    tta_views = ["original"]
    if args.tta:
        tta_views += [s.strip() for s in args.tta.split(",")]

    # Get ensemble logits and probabilities
    print("\nCollecting ensemble predictions...")
    logits, probs, annotation_ids = ensemble_predict_with_logits(
        models, test_loader, device, tta_views
    )

    # Load ground truth
    solution_df = pd.read_csv(args.solution)
    # Create mapping: annotation_id -> species name
    gt_map = dict(zip(solution_df["annotation_id"], solution_df["concept_name"]))

    # Convert to integer labels
    gt_labels = np.array([
        species_to_idx.get(gt_map.get(aid, ""), 0)
        for aid in annotation_ids
    ])

    # Load distance matrix
    dist_matrix = load_distance_matrix(args.distance_matrix, species_list)

    # ===== 1. ECE on raw probabilities =====
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)

    ece_raw, bin_data_raw = compute_ece(probs, gt_labels, n_bins=args.n_bins)
    print(f"\nRaw ensemble probabilities:")
    print(f"  ECE = {ece_raw:.4f}")
    print(f"  Mean confidence = {probs.max(axis=1).mean():.4f}")
    print(f"  Accuracy (argmax) = {(probs.argmax(axis=1) == gt_labels).mean():.4f}")

    # Min-risk with raw probs
    expected_dist_raw = probs @ dist_matrix
    minrisk_preds_raw = expected_dist_raw.argmin(axis=1)
    raw_preds = [species_list[p] for p in minrisk_preds_raw]
    raw_gt = [gt_map.get(aid, "") for aid in annotation_ids]
    raw_score = compute_taxonomic_score(raw_preds, raw_gt, dist_matrix, species_to_idx)
    print(f"  Min-risk tax. distance = {raw_score:.3f}")

    # ===== 2. Temperature scaling sweep =====
    print(f"\nTemperature scaling sweep:")
    print(f"  {'Temp':>6s}  {'ECE':>7s}  {'Accuracy':>8s}  {'MinRisk':>8s}  {'MeanConf':>8s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    best_ece_temp = 1.0
    best_ece = ece_raw
    best_minrisk_temp = 1.0
    best_minrisk = raw_score

    for temp in temperatures:
        scaled_probs = apply_temperature(logits, temp)
        ece_temp, _ = compute_ece(scaled_probs, gt_labels, n_bins=args.n_bins)

        # Min-risk with scaled probs
        expected_dist = scaled_probs @ dist_matrix
        minrisk_preds = expected_dist.argmin(axis=1)
        preds = [species_list[p] for p in minrisk_preds]
        gt_list = [gt_map.get(aid, "") for aid in annotation_ids]
        score = compute_taxonomic_score(preds, gt_list, dist_matrix, species_to_idx)

        acc = (scaled_probs.argmax(axis=1) == gt_labels).mean()
        mean_conf = scaled_probs.max(axis=1).mean()

        marker = ""
        if ece_temp < best_ece:
            best_ece = ece_temp
            best_ece_temp = temp
            marker += " *ECE"
        if score < best_minrisk:
            best_minrisk = score
            best_minrisk_temp = temp
            marker += " *MR"

        print(f"  {temp:6.2f}  {ece_temp:7.4f}  {acc:8.4f}  {score:8.3f}  {mean_conf:8.4f}{marker}")

    print(f"\n  Best ECE: T={best_ece_temp:.2f} -> ECE={best_ece:.4f}")
    print(f"  Best min-risk: T={best_minrisk_temp:.2f} -> score={best_minrisk:.3f}")

    # ===== 3. Reliability diagram data =====
    print(f"\nReliability diagram (T=1.0, {args.n_bins} bins):")
    print(f"  {'Bin':>4s}  {'Range':>12s}  {'Count':>6s}  {'Acc':>7s}  {'Conf':>7s}  {'Gap':>7s}")
    for i in range(args.n_bins):
        lo = bin_data_raw["bin_boundaries"][i]
        hi = bin_data_raw["bin_boundaries"][i + 1]
        count = bin_data_raw["bin_counts"][i]
        acc = bin_data_raw["bin_accs"][i]
        conf = bin_data_raw["bin_confs"][i]
        gap = abs(acc - conf)
        if count > 0:
            print(f"  {i+1:4d}  ({lo:.2f}, {hi:.2f}]  {count:6d}  {acc:7.3f}  {conf:7.3f}  {gap:7.3f}")

    # ===== 4. Summary for paper =====
    print(f"\n{'='*60}")
    print("SUMMARY FOR PAPER")
    print(f"{'='*60}")
    print(f"ECE (raw, T=1.0): {ece_raw:.4f}")
    print(f"ECE (best T={best_ece_temp:.1f}): {best_ece:.4f}")
    print(f"Min-risk score (raw, T=1.0): {raw_score:.3f}")
    print(f"Min-risk score (best T={best_minrisk_temp:.1f}): {best_minrisk:.3f}")
    print(f"Temperature scaling {'improves' if best_minrisk < raw_score else 'does not improve'} min-risk inference")


if __name__ == "__main__":
    main()
