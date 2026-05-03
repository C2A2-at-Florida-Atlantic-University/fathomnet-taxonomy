#!/usr/bin/env python3
"""
Compute hierarchy consistency rate for independent-head models.

For each test sample, checks whether the per-level argmax predictions
form a valid path in the taxonomy (e.g., predicted species is a descendant
of the predicted genus).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.inference.ensemble_predict import (
    MultiScaleTestDataset,
    load_models,
)
from data.data import load_and_encode_taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/experiment-multiscale.yaml")
    parser.add_argument("--scales", type=str, nargs="+", default=["1x", "3x", "5x", "full"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-parent-conditioning", action="store_true")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone name (e.g., dinov2_base). If set, all scales use this backbone.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, list(cfg.data.taxonomy_levels)
    )

    # Build ancestor lookup: for each species index, what's the expected ancestor at each level
    levels = list(cfg.data.taxonomy_levels)
    species_encoder = encoders["species"]
    species_to_ancestors = {}  # species_idx -> {level: expected_idx}

    for _, row in taxonomy_df.iterrows():
        sp_name = row["species"]
        if sp_name in species_encoder.classes_:
            sp_idx = list(species_encoder.classes_).index(sp_name)
            ancestors = {}
            for level in levels:
                level_name = row[level]
                level_encoder = encoders[level]
                if level_name in level_encoder.classes_:
                    ancestors[level] = list(level_encoder.classes_).index(level_name)
            species_to_ancestors[sp_idx] = ancestors

    # Load checkpoints
    ckpt_pattern = os.path.join(args.checkpoint_dir, "fold_*/checkpoints/*.ckpt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    print(f"Found {len(checkpoints)} checkpoints")

    # Load models
    distance_matrix_path = "data/distance_matrix.csv"
    scale_backbones = None
    if args.backbone:
        scale_backbones = {s: args.backbone for s in args.scales}
    models = load_models(
        checkpoints, cfg, class_counts, id_to_name,
        scales=args.scales,
        distance_matrix_path=distance_matrix_path,
        device=device,
        scale_backbones=scale_backbones,
        parent_conditioning=not args.no_parent_conditioning,
    )

    # Create test dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = MultiScaleTestDataset(
        coco_json_path=cfg.paths.test_coco_json,
        roi_root=cfg.paths.test_roi_root,
        scales=args.scales,
        transform=transform,
    )

    def collate_fn(batch):
        images_list, annotation_ids, image_ids = zip(*batch)
        images = {
            scale: torch.stack([img[scale] for img in images_list])
            for scale in args.scales
        }
        return {"images": images, "annotation_ids": list(annotation_ids), "image_ids": list(image_ids)}

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4,
                             shuffle=False, collate_fn=collate_fn)

    # Collect per-level ensemble predictions
    print("Running inference...")
    per_level_probs = {level: [] for level in levels}

    for batch in tqdm(test_loader, desc="Inference"):
        images = {k: v.to(device) for k, v in batch["images"].items()}

        # Ensemble: average per-level softmax across models
        level_probs_batch = {level: [] for level in levels}

        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(images)

            for level in levels:
                if level in output:
                    level_probs_batch[level].append(output[level].cpu().numpy())

        for level in levels:
            if level_probs_batch[level]:
                avg = np.mean(level_probs_batch[level], axis=0)
                per_level_probs[level].append(avg)

    # Concatenate
    for level in levels:
        per_level_probs[level] = np.concatenate(per_level_probs[level], axis=0)

    N = per_level_probs["species"].shape[0]
    print(f"\nTotal test samples: {N}")

    # Compute per-level argmax
    per_level_preds = {level: per_level_probs[level].argmax(axis=1) for level in levels}

    # Check consistency: for each adjacent level pair
    print("\n" + "=" * 70)
    print("HIERARCHY CONSISTENCY ANALYSIS")
    print("=" * 70)

    # For each sample, get predicted species, then check if predicted genus
    # matches the genus of the predicted species
    total_consistent = np.ones(N, dtype=bool)

    for child_level, parent_level in [
        ("species", "genus"),
        ("genus", "family"),
        ("family", "order"),
        ("order", "class"),
        ("class", "phylum"),
        ("phylum", "kingdom"),
    ]:
        if child_level not in per_level_preds or parent_level not in per_level_preds:
            continue

        consistent = 0
        total = 0

        for i in range(N):
            child_pred = per_level_preds[child_level][i]
            parent_pred = per_level_preds[parent_level][i]

            # What parent does the predicted child imply?
            if child_pred in species_to_ancestors if child_level == "species" else True:
                # Look up the expected parent for this child
                expected_parent = None
                for sp_idx, ancestors in species_to_ancestors.items():
                    if child_level == "species" and sp_idx == child_pred:
                        expected_parent = ancestors.get(parent_level)
                        break
                    elif child_level != "species":
                        # For non-species levels, need child->parent mapping
                        if ancestors.get(child_level) == child_pred:
                            expected_parent = ancestors.get(parent_level)
                            break

                if expected_parent is not None:
                    is_consistent = (parent_pred == expected_parent)
                    consistent += int(is_consistent)
                    total += 1
                    if not is_consistent:
                        total_consistent[i] = False

        rate = consistent / total * 100 if total > 0 else 0
        print(f"  {child_level:>10s} -> {parent_level:<10s}: {consistent}/{total} consistent ({rate:.1f}%)")

    overall = total_consistent.sum() / N * 100
    print(f"\n  Fully consistent (all levels):   {total_consistent.sum()}/{N} ({overall:.1f}%)")
    print(f"  At least one inconsistency:      {N - total_consistent.sum()}/{N} ({100-overall:.1f}%)")


if __name__ == "__main__":
    main()
