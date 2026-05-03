#!/usr/bin/env python3
"""
Cross-Validated Temperature Scaling for FathomNet Ensemble.

Learns temperature T on out-of-fold (OOF) validation predictions, avoiding
any test-set leakage. This addresses the reviewer concern that post-hoc T
selection on the test set constitutes implicit overfitting.

Approach:
  For each fold k, checkpoint k predicts on fold k's held-out validation set.
  All OOF predictions are concatenated to form a full-training-set prediction.
  T is optimized on this OOF data (minimizing NLL or ECE).
  The learned T can then be applied at test time.

Usage:
    python scripts/cross_validated_calibration.py \
        --checkpoint-dir ./outputs/outputs/kfold_10fold_dinov2_llrd \
        --config config/experiment-dinov2.yaml \
        --scales 1x 3x 5x full
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data import (
    MultiScalePrecroppedDataset,
    load_and_encode_taxonomy,
    load_coco_annotations,
    create_transforms,
)
from src.config import load_config
from src.inference.ensemble_predict import load_models


def collect_oof_logits(
    checkpoint_dir: str,
    cfg,
    class_counts: dict,
    id_to_name: dict,
    scales: List[str],
    distance_matrix_path: str,
    device: torch.device,
    batch_size: int = 16,
    parent_conditioning: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect out-of-fold logits and ground truth labels.

    For each fold k, loads checkpoint k and runs inference on fold k's
    validation set. Concatenates all OOF predictions.

    Returns:
        logits: (N, C) array of species logits
        labels: (N,) array of integer ground truth labels
    """
    # Load fold splits
    splits_path = os.path.join(checkpoint_dir, "fold_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    n_folds = len(splits)
    print(f"Found {n_folds} fold splits")

    # Create full training dataset (val transform = no augmentation)
    taxonomy_df, encoders, _, _ = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, list(cfg.data.taxonomy_levels)
    )

    # Load all annotations
    annotations = load_coco_annotations(
        cfg.paths.train_coco_json,
        cfg.paths.train_full_image_dir,
        include_labels=True,
    )

    _, val_transform = create_transforms(cfg)

    full_dataset = MultiScalePrecroppedDataset(
        frame=annotations,
        taxonomy_df=taxonomy_df,
        levels=list(cfg.data.taxonomy_levels),
        encoders=encoders,
        roi_root=cfg.paths.train_roi_root,
        scales=scales,
        transform=val_transform,
    )

    all_logits = []
    all_labels = []

    for fold_idx, split in enumerate(splits):
        val_indices = split["val_indices"]

        # Find checkpoint
        ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}", "checkpoints")
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            print(f"  Fold {fold_idx}: no checkpoint found, skipping")
            continue
        ckpt_path = os.path.join(ckpt_dir, sorted(ckpt_files)[-1])

        print(f"  Fold {fold_idx}: loading {os.path.basename(ckpt_path)} "
              f"({len(val_indices)} val samples)")

        # Load model
        models = load_models(
            [ckpt_path], cfg, class_counts, id_to_name, scales,
            distance_matrix_path, device,
            parent_conditioning=parent_conditioning,
        )
        model = models[0]

        # Create validation subset
        val_subset = Subset(full_dataset, val_indices)

        def collate_fn(batch):
            images_list = []
            labels_list = []
            for item in batch:
                images_list.append(item["scales"])
                labels_list.append(item["labels"])
            images = {
                scale: torch.stack([img[scale] for img in images_list])
                for scale in scales
            }
            labels = {
                level: torch.tensor([l[level] for l in labels_list])
                for level in labels_list[0].keys()
            }
            return images, labels

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        fold_logits = []
        fold_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  Fold {fold_idx}", leave=False):
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(images)
                logits = outputs["species"]
                fold_logits.append(logits.cpu().numpy())
                fold_labels.append(labels["species"].numpy())

        all_logits.append(np.concatenate(fold_logits, axis=0))
        all_labels.append(np.concatenate(fold_labels, axis=0))

        # Free GPU memory
        del model, models
        torch.cuda.empty_cache()

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def nll_loss(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Compute negative log-likelihood with temperature scaling."""
    scaled = logits / temperature
    # Numerically stable log-softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    log_probs = shifted - log_sum_exp[:, None]
    # NLL = -mean(log_prob of true class)
    return -np.mean(log_probs[np.arange(len(labels)), labels])


def ece(logits: np.ndarray, labels: np.ndarray, temperature: float,
        n_bins: int = 15) -> float:
    """Compute Expected Calibration Error with temperature scaling."""
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total = len(labels)
    ece_val = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum()
        if count > 0:
            ece_val += (count / total) * abs(accuracies[mask].mean() - confidences[mask].mean())

    return ece_val


def main():
    parser = argparse.ArgumentParser(description="Cross-validated temperature scaling")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory with fold_splits.json and fold_*/checkpoints/",
    )
    parser.add_argument("--config", type=str, default="config/experiment-multiscale.yaml")
    parser.add_argument("--distance-matrix", type=str, default="data/distance_matrix.csv")
    parser.add_argument("--scales", type=str, nargs="+", default=["1x", "3x", "5x", "full"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-parent-conditioning", action="store_true",
        help="Load models with independent heads",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, list(cfg.data.taxonomy_levels)
    )

    print("Collecting out-of-fold logits...")
    logits, labels = collect_oof_logits(
        args.checkpoint_dir, cfg, class_counts, id_to_name,
        args.scales, args.distance_matrix, device, args.batch_size,
        parent_conditioning=not args.no_parent_conditioning,
    )
    print(f"OOF predictions: {logits.shape[0]} samples, {logits.shape[1]} classes")
    print(f"OOF accuracy: {(logits.argmax(axis=1) == labels).mean():.4f}")

    # Optimize T by minimizing NLL
    print("\nOptimizing temperature (NLL)...")
    result_nll = minimize_scalar(
        lambda t: nll_loss(logits, labels, t),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    t_nll = result_nll.x
    print(f"  Optimal T (NLL): {t_nll:.4f}")

    # Optimize T by minimizing ECE
    print("Optimizing temperature (ECE)...")
    result_ece = minimize_scalar(
        lambda t: ece(logits, labels, t),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    t_ece = result_ece.x
    print(f"  Optimal T (ECE): {t_ece:.4f}")

    # Sweep table
    print(f"\n{'T':>6s}  {'NLL':>8s}  {'ECE':>8s}  {'Accuracy':>8s}  {'MeanConf':>8s}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for t in [0.5, 0.75, 1.0, t_nll, t_ece, 1.5, 2.0, 3.0, 5.0]:
        nll_val = nll_loss(logits, labels, t)
        ece_val = ece(logits, labels, t)
        scaled = logits / t
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        acc = (probs.argmax(axis=1) == labels).mean()
        conf = probs.max(axis=1).mean()
        marker = ""
        if abs(t - t_nll) < 0.001:
            marker += " *NLL"
        if abs(t - t_ece) < 0.001:
            marker += " *ECE"
        print(f"{t:6.3f}  {nll_val:8.4f}  {ece_val:8.4f}  {acc:8.4f}  {conf:8.4f}{marker}")

    print(f"\nCross-validated optimal temperature (NLL): T = {t_nll:.3f}")
    print(f"Cross-validated optimal temperature (ECE): T = {t_ece:.3f}")
    print("These values can be used at test time without data leakage.")


if __name__ == "__main__":
    main()
