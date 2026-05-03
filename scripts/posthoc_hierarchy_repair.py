#!/usr/bin/env python3
"""
Post-hoc hierarchy repair experiment.

For each test sample, checks if the species-level argmax is consistent with
coarser-level argmaxes. If not, restricts the species prediction to only
those species that are descendants of the predicted genus (or family, etc.).

This tests whether enforcing path consistency improves taxonomic distance.
"""

import argparse
import glob
import json
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


def build_descendant_lookup(taxonomy_df, encoders, levels):
    """
    Build mapping from (level, class_idx) -> set of valid species indices.

    For each level and each class at that level, returns which species
    indices are descendants.
    """
    species_encoder = encoders["species"]
    descendant_map = {}  # (level, idx) -> set of species indices

    for level in levels:
        level_encoder = encoders[level]
        for cls_idx, cls_name in enumerate(level_encoder.classes_):
            # Find all species that have this ancestor
            mask = taxonomy_df[level] == cls_name
            species_names = taxonomy_df.loc[mask, "species"].unique()
            sp_indices = set()
            for sp in species_names:
                if sp in species_encoder.classes_:
                    sp_indices.add(list(species_encoder.classes_).index(sp))
            descendant_map[(level, cls_idx)] = sp_indices

    return descendant_map


def repair_predictions(species_probs, coarse_preds, descendant_map, levels,
                       distance_matrix, decision="min_risk"):
    """
    Repair species predictions by restricting to valid descendants.

    For each sample:
    1. Get the predicted class at each coarse level
    2. Intersect valid species sets from all coarse levels
    3. Zero out species probabilities outside the valid set
    4. Re-normalize and re-predict

    Args:
        species_probs: (N, S) array of species softmax probabilities
        coarse_preds: dict of level -> (N,) argmax predictions
        descendant_map: mapping from (level, idx) -> set of species indices
        levels: list of taxonomic levels (coarsest to finest)
        distance_matrix: (S, S) pairwise distance matrix
        decision: "argmax" or "min_risk"

    Returns:
        repaired_preds: (N,) array of species predictions
        n_changed: number of predictions that changed
    """
    N, S = species_probs.shape
    repaired_preds = np.zeros(N, dtype=int)
    original_preds = np.zeros(N, dtype=int)
    n_changed = 0

    # Coarse levels (everything except species)
    coarse_levels = [l for l in levels if l != "species"]

    for i in range(N):
        # Get valid species from each coarse level
        valid_sets = []
        for level in coarse_levels:
            if level in coarse_preds:
                pred_idx = coarse_preds[level][i]
                key = (level, pred_idx)
                if key in descendant_map:
                    valid_sets.append(descendant_map[key])

        # Intersect all valid sets
        if valid_sets:
            valid_species = valid_sets[0]
            for vs in valid_sets[1:]:
                valid_species = valid_species & vs
        else:
            valid_species = set(range(S))

        # If intersection is empty (coarse preds disagree), fall back to genus only
        if not valid_species and "genus" in coarse_preds:
            key = ("genus", coarse_preds["genus"][i])
            if key in descendant_map:
                valid_species = descendant_map[key]

        # If still empty, use all species
        if not valid_species:
            valid_species = set(range(S))

        # Mask and re-normalize
        masked_probs = species_probs[i].copy()
        mask = np.ones(S, dtype=bool)
        for idx in valid_species:
            mask[idx] = False
        masked_probs[mask] = 0.0

        prob_sum = masked_probs.sum()
        if prob_sum > 0:
            masked_probs /= prob_sum
        else:
            # Fallback: uniform over valid species
            for idx in valid_species:
                masked_probs[idx] = 1.0 / len(valid_species)

        if decision == "min_risk":
            expected_cost = masked_probs @ distance_matrix
            repaired_preds[i] = expected_cost.argmin()
        else:
            repaired_preds[i] = masked_probs.argmax()

        # Original prediction
        if decision == "min_risk":
            orig_cost = species_probs[i] @ distance_matrix
            original_preds[i] = orig_cost.argmin()
        else:
            original_preds[i] = species_probs[i].argmax()

        if repaired_preds[i] != original_preds[i]:
            n_changed += 1

    return repaired_preds, original_preds, n_changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/experiment-multiscale.yaml")
    parser.add_argument("--scales", type=str, nargs="+", default=["1x", "3x", "5x", "full"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-parent-conditioning", action="store_true")
    parser.add_argument("--decision", type=str, choices=["argmax", "min_risk"], default="min_risk")
    parser.add_argument("--tta", type=str, default=None, help="e.g. 'hflip'")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--solution", type=str, default="solution_kaggle_2025.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, list(cfg.data.taxonomy_levels)
    )
    levels = list(cfg.data.taxonomy_levels)

    # Build descendant lookup
    descendant_map = build_descendant_lookup(taxonomy_df, encoders, levels)

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

    # Get distance matrix from model
    dist_matrix = models[0].distance_matrix.cpu().numpy()

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

    # Collect per-level ensemble probabilities
    print("Running inference...")
    per_level_probs = {level: [] for level in levels}
    all_annotation_ids = []

    tta_modes = ["none"]
    if args.tta:
        tta_modes.extend(args.tta.split(","))

    from scipy.special import softmax as scipy_softmax

    for batch in tqdm(test_loader, desc="Inference"):
        all_annotation_ids.extend(batch["annotation_ids"])
        level_probs_batch = {level: [] for level in levels}

        for model in models:
            model.eval()
            for tta_mode in tta_modes:
                images = {k: v.to(device) for k, v in batch["images"].items()}

                if tta_mode == "hflip":
                    images = {k: torch.flip(v, dims=[-1]) for k, v in images.items()}
                elif tta_mode == "vflip":
                    images = {k: torch.flip(v, dims=[-2]) for k, v in images.items()}

                with torch.no_grad():
                    output = model(images)

                for level in levels:
                    if level in output:
                        # Model returns logits — apply softmax to get probabilities
                        logits = output[level].cpu().numpy()
                        probs = scipy_softmax(logits, axis=1)
                        level_probs_batch[level].append(probs)

        for level in levels:
            if level_probs_batch[level]:
                avg = np.mean(level_probs_batch[level], axis=0)
                per_level_probs[level].append(avg)

    # Concatenate
    for level in levels:
        per_level_probs[level] = np.concatenate(per_level_probs[level], axis=0)

    species_probs = per_level_probs["species"]
    N = species_probs.shape[0]
    print(f"\nTotal test samples: {N}")

    # Get coarse-level argmax predictions
    coarse_preds = {}
    for level in levels:
        if level != "species":
            coarse_preds[level] = per_level_probs[level].argmax(axis=1)

    # Run repair
    repaired_preds, original_preds, n_changed = repair_predictions(
        species_probs, coarse_preds, descendant_map, levels,
        dist_matrix, decision=args.decision,
    )
    print(f"\nPredictions changed by repair: {n_changed}/{N} ({n_changed/N*100:.1f}%)")

    # Evaluate both original and repaired
    species_encoder = encoders["species"]
    solution = pd.read_csv(args.solution)

    # Build annotation_id -> species name mapping
    original_names = [species_encoder.classes_[p] for p in original_preds]
    repaired_names = [species_encoder.classes_[p] for p in repaired_preds]

    # Create submission DataFrames
    original_df = pd.DataFrame({
        "annotation_id": all_annotation_ids,
        "concept_name": original_names,
    })
    repaired_df = pd.DataFrame({
        "annotation_id": all_annotation_ids,
        "concept_name": repaired_names,
    })

    # Load distance matrix for scoring
    dist_csv = pd.read_csv(distance_matrix_path, index_col=0)

    def score_submission(sub_df, sol_df):
        merged = sol_df.merge(sub_df, on="annotation_id", how="inner",
                              suffixes=("_true", "_pred"))
        preds = merged["concept_name_pred"].fillna("unknown").tolist()
        truths = merged["concept_name_true"].tolist()
        distances = []
        for p, t in zip(preds, truths):
            if p == t:
                distances.append(0)
            elif p in dist_csv.index and t in dist_csv.columns:
                distances.append(dist_csv.loc[p, t])
            else:
                distances.append(7)
        acc = sum(1 for p, t in zip(preds, truths) if p == t) / len(truths)
        return np.mean(distances), acc, np.array(distances)

    orig_score, orig_acc, orig_dists = score_submission(original_df, solution)
    rep_score, rep_acc, rep_dists = score_submission(repaired_df, solution)

    print(f"\n{'='*60}")
    print(f"HIERARCHY REPAIR RESULTS ({args.decision} inference)")
    print(f"{'='*60}")
    print(f"  Original:  score={orig_score:.3f}, acc={orig_acc:.1%}")
    print(f"  Repaired:  score={rep_score:.3f}, acc={rep_acc:.1%}")
    print(f"  Delta:     {rep_score - orig_score:+.3f} ({(rep_score - orig_score)/orig_score*100:+.1f}%)")
    print(f"  Changed:   {n_changed}/{N} predictions ({n_changed/N*100:.1f}%)")

    # Per-changed-sample analysis
    changed_mask = original_preds != repaired_preds
    if changed_mask.any():
        orig_changed = orig_dists[changed_mask]
        rep_changed = rep_dists[changed_mask]
        improved = (rep_changed < orig_changed).sum()
        worsened = (rep_changed > orig_changed).sum()
        unchanged_dist = (rep_changed == orig_changed).sum()
        print(f"\n  Among {n_changed} changed predictions:")
        print(f"    Improved: {improved} ({improved/n_changed*100:.1f}%)")
        print(f"    Worsened: {worsened} ({worsened/n_changed*100:.1f}%)")
        print(f"    Same dist: {unchanged_dist} ({unchanged_dist/n_changed*100:.1f}%)")
        print(f"    Mean distance before: {orig_changed.mean():.3f}")
        print(f"    Mean distance after:  {rep_changed.mean():.3f}")


if __name__ == "__main__":
    main()
