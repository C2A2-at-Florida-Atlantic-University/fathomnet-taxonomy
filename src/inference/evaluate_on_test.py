#!/usr/bin/env python3
"""
Evaluate model predictions against Kaggle test set ground truth.

This script computes the true generalization performance of models using the
ground truth labels for the public and private Kaggle test sets.

The competition metric is mean taxonomic distance:
    Score = (1/N) × Σ_i D(predicted_species_i, true_species_i)

Where D(s1, s2) is the taxonomic distance between species (lower is better).

Usage:
    # Evaluate a submission CSV
    python evaluate_on_test.py --submission submission.csv

    # Evaluate a model checkpoint directly
    python evaluate_on_test.py --checkpoint outputs/model/best.ckpt

"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data.data import load_and_encode_taxonomy
from src.config import load_config


def load_distance_matrix(path: str, species_list: List[str]) -> np.ndarray:
    """
    Load taxonomic distance matrix and align with species list.

    Args:
        path: Path to distance matrix CSV
        species_list: Ordered list of species names

    Returns:
        N×N numpy array where D[i,j] is distance between species i and j
    """
    df = pd.read_csv(path, index_col=0)

    n_species = len(species_list)
    distance_matrix = np.zeros((n_species, n_species))

    for i, sp1 in enumerate(species_list):
        for j, sp2 in enumerate(species_list):
            if sp1 in df.index and sp2 in df.columns:
                distance_matrix[i, j] = df.loc[sp1, sp2]
            elif sp1 == sp2:
                distance_matrix[i, j] = 0
            else:
                # Maximum distance for unknown pairs
                distance_matrix[i, j] = 7

    return distance_matrix


def compute_taxonomic_score(
    predictions: List[str],
    ground_truth: List[str],
    distance_matrix: np.ndarray,
    species_to_idx: Dict[str, int],
) -> float:
    """
    Compute mean taxonomic distance (competition metric).

    Args:
        predictions: List of predicted species names
        ground_truth: List of true species names
        distance_matrix: N×N distance matrix
        species_to_idx: Mapping from species name to index

    Returns:
        Mean taxonomic distance (lower is better, 0 = perfect)
    """
    total_distance = 0.0
    count = 0

    for pred, true in zip(predictions, ground_truth):
        pred_idx = species_to_idx.get(pred)
        true_idx = species_to_idx.get(true)

        if pred_idx is not None and true_idx is not None:
            total_distance += distance_matrix[pred_idx, true_idx]
        else:
            # Penalize unknown species with max distance
            total_distance += 7 if pred != true else 0

        count += 1

    return total_distance / count if count > 0 else 0.0


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute exact match accuracy."""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if predictions else 0.0


def evaluate_submission(
    submission_path: str,
    solution_path: str,
    distance_matrix_path: str,
) -> Dict:
    """
    Evaluate a submission CSV against ground truth.

    Args:
        submission_path: Path to submission CSV (annotation_id, concept_name)
        solution_path: Path to ground truth CSV
        distance_matrix_path: Path to taxonomic distance matrix

    Returns:
        Dictionary with evaluation metrics
    """
    # Load files
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)

    # Merge on annotation_id
    merged = solution.merge(
        submission,
        on="annotation_id",
        how="left",
        suffixes=("_true", "_pred"),
    )

    # Get species list for distance matrix
    all_species = list(set(merged["concept_name_true"].unique()) |
                       set(merged["concept_name_pred"].dropna().unique()))
    species_to_idx = {sp: i for i, sp in enumerate(all_species)}

    # Load distance matrix
    distance_matrix = load_distance_matrix(distance_matrix_path, all_species)

    # Compute metrics
    predictions = merged["concept_name_pred"].fillna("unknown").tolist()
    ground_truth = merged["concept_name_true"].tolist()

    tax_score = compute_taxonomic_score(
        predictions, ground_truth, distance_matrix, species_to_idx
    )
    accuracy = compute_accuracy(predictions, ground_truth)

    # Per-species analysis
    species_results = {}
    for species in sorted(set(ground_truth)):
        mask = [g == species for g in ground_truth]
        sp_preds = [p for p, m in zip(predictions, mask) if m]
        sp_truth = [g for g, m in zip(ground_truth, mask) if m]

        sp_tax = compute_taxonomic_score(
            sp_preds, sp_truth, distance_matrix, species_to_idx
        )
        sp_acc = compute_accuracy(sp_preds, sp_truth)

        species_results[species] = {
            "count": len(sp_truth),
            "accuracy": sp_acc,
            "tax_score": sp_tax,
        }

    return {
        "n_samples": len(merged),
        "accuracy": accuracy,
        "taxonomic_score": tax_score,
        "species_results": species_results,
    }


class MultiScaleTestDataset(Dataset):
    """Test dataset for loading multi-scale pre-cropped ROIs."""

    def __init__(
        self,
        coco_json_path: str,
        roi_root: str,
        scales: List[str],
        transform=None,
    ):
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        self.annotations = coco_data["annotations"]
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.roi_root = roi_root
        self.scales = scales
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        annotation_id = ann["id"]

        images = {}
        for scale in self.scales:
            scale_dir = os.path.join(self.roi_root, scale)
            img_path = os.path.join(scale_dir, f"{image_id}_{annotation_id}.png")

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), (0, 0, 0))

            if self.transform:
                img = self.transform(img)
            images[scale] = img

        return images, annotation_id


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    solution_path: str,
    distance_matrix_path: str,
    scales: List[str],
    batch_size: int = 16,
) -> Dict:
    """
    Evaluate a model checkpoint on the test set.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config YAML
        solution_path: Path to ground truth CSV
        distance_matrix_path: Path to distance matrix
        scales: List of scales to use
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation metrics
    """
    from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load config
    cfg = load_config(config_path)

    # Load taxonomy
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )

    # Load model
    model = MultiScaleTaxonomicClassifier.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=scales,
        distance_matrix_path=distance_matrix_path,
        strict=False,
        map_location=device,
    )
    model.to(device)
    model.eval()

    # Create test transform
    test_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset
    test_dataset = MultiScaleTestDataset(
        coco_json_path=cfg.paths.test_coco_json,
        roi_root=cfg.paths.test_roi_root,
        scales=scales,
        transform=test_transform,
    )

    def collate_fn(batch):
        images_list, annotation_ids = zip(*batch)
        images = {
            scale: torch.stack([img[scale] for img in images_list])
            for scale in scales
        }
        return images, list(annotation_ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Run inference
    all_predictions = []
    all_annotation_ids = []

    with torch.no_grad():
        for images, annotation_ids in tqdm(test_loader, desc="Evaluating"):
            images = {k: v.to(device) for k, v in images.items()}
            outputs = model(images)
            species_preds = torch.argmax(outputs["species"], dim=1)
            all_predictions.extend(species_preds.cpu().numpy())
            all_annotation_ids.extend(annotation_ids)

    # Convert to species names
    predictions = [id_to_name["species"][pred] for pred in all_predictions]

    # Create temporary submission DataFrame
    submission_df = pd.DataFrame({
        "annotation_id": all_annotation_ids,
        "concept_name": predictions,
    })

    # Save temporary file and evaluate
    temp_path = "/tmp/temp_submission.csv"
    submission_df.to_csv(temp_path, index=False)

    results = evaluate_submission(
        temp_path, solution_path, distance_matrix_path
    )

    # Clean up
    os.remove(temp_path)

    return results


def print_results(results: Dict, verbose: bool = False):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:          {results['n_samples']}")
    print(f"Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Taxonomic Score:  {results['taxonomic_score']:.4f}")
    print("=" * 60)

    if verbose and results.get("species_results"):
        print("\nPer-Species Results (sorted by tax_score, worst first):")
        print("-" * 60)
        sorted_species = sorted(
            results["species_results"].items(),
            key=lambda x: -x[1]["tax_score"],
        )
        for species, stats in sorted_species[:20]:
            print(f"  {species:30s} n={stats['count']:3d}  "
                  f"acc={stats['accuracy']:.2f}  tax={stats['tax_score']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against Kaggle test set ground truth"
    )
    parser.add_argument(
        "--submission",
        type=str,
        help="Path to submission CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (alternative to --submission)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
        help="Path to config file (used with --checkpoint)",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="data/solution_kaggle_2025.csv",
        help="Path to ground truth solution CSV",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
        help="Path to taxonomic distance matrix",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["1x", "3x", "5x", "full"],
        help="Scales to use (with --checkpoint)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (with --checkpoint)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-species breakdown",
    )
    args = parser.parse_args()

    if not args.submission and not args.checkpoint:
        parser.error("Must specify either --submission or --checkpoint")

    if args.submission:
        results = evaluate_submission(
            args.submission,
            args.solution,
            args.distance_matrix,
        )
    else:
        results = evaluate_checkpoint(
            args.checkpoint,
            args.config,
            args.solution,
            args.distance_matrix,
            args.scales,
            args.batch_size,
        )

    print_results(results, args.verbose)


if __name__ == "__main__":
    main()
