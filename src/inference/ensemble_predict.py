#!/usr/bin/env python3
"""
Ensemble Prediction using K-Fold Models.

This script loads multiple models trained via k-fold CV and combines their
predictions for improved generalization. Ensemble methods typically reduce
variance and improve robustness on unseen data.

Ensemble methods supported:
1. Average probabilities (soft voting)
2. Majority voting (hard voting)
3. Weighted average (if fold validation scores provided)

Usage:
    # Ensemble prediction with all fold checkpoints
    python ensemble_predict.py --checkpoints outputs/kfold/fold_*/checkpoints/best*.ckpt

    # Specify output file
    python ensemble_predict.py --checkpoints ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \\
        --output ensemble_submission.csv

    # Use weighted ensemble based on validation scores
    python ensemble_predict.py --checkpoints ckpt1.ckpt ckpt2.ckpt \\
        --weights 0.6 0.4
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data.data import load_and_encode_taxonomy
from src.config import load_config
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


class MultiScaleTestDataset(Dataset):
    """Test dataset for multi-scale inference."""

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

        return images, annotation_id, image_id


def load_models(
    checkpoint_paths: List[str],
    cfg,
    class_counts: Dict,
    id_to_name: Dict,
    scales: List[str],
    distance_matrix_path: str,
    device: torch.device,
    scale_backbones: Optional[Dict[str, str]] = None,
    fusion_type: str = "concat",
    share_backbone: bool = False,
    branch_cnn: bool = False,
    use_gnn: bool = False,
    parent_conditioning: bool = True,
    leaf_only: bool = False,
) -> List[MultiScaleTaxonomicClassifier]:
    """Load multiple models from checkpoints."""
    models = []

    for ckpt_path in checkpoint_paths:
        print(f"Loading model from {ckpt_path}...")
        model = MultiScaleTaxonomicClassifier.load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
            class_counts=class_counts,
            id_to_name=id_to_name,
            scales=scales,
            distance_matrix_path=distance_matrix_path,
            scale_backbones=scale_backbones,
            fusion_type=fusion_type,
            share_backbone=share_backbone,
            branch_cnn=branch_cnn,
            use_gnn=use_gnn,
            parent_conditioning=parent_conditioning,
            leaf_only=leaf_only,
            strict=False,
            map_location=device,
            weights_only=False,  # Required for checkpoints with pandas DataFrames
        )
        model.to(device)
        model.eval()
        models.append(model)

    return models


def apply_tta(images: dict, augmentation: str) -> dict:
    """Apply a TTA augmentation to multi-scale image dict (on GPU)."""
    if augmentation == "hflip":
        return {k: torch.flip(v, dims=[-1]) for k, v in images.items()}
    elif augmentation == "vflip":
        return {k: torch.flip(v, dims=[-2]) for k, v in images.items()}
    else:
        return images


def ensemble_predict(
    models: List[MultiScaleTaxonomicClassifier],
    dataloader: DataLoader,
    device: torch.device,
    weights: Optional[List[float]] = None,
    method: str = "average",
    decision: str = "argmax",
    distance_matrix: Optional[torch.Tensor] = None,
    tta: Optional[List[str]] = None,
    use_gnn: bool = False,
    temperature: float = 1.0,
) -> tuple:
    """
    Run ensemble prediction.

    Args:
        models: List of trained models
        dataloader: Test data loader
        device: Torch device
        weights: Optional weights for each model (for weighted average)
        method: "average" (soft voting) or "voting" (hard voting)
        decision: "argmax" (highest probability) or "min_risk" (minimize
            expected taxonomic distance: argmin_s Σ_i P(i) × D(s, i))
        distance_matrix: (C x C) taxonomic distance matrix, required for
            min_risk decision
        tta: List of TTA augmentations (e.g. ["original", "hflip", "vflip"])
        temperature: Temperature scaling parameter (default 1.0, no scaling).
            Applied via power scaling: p_i^(1/T) / sum(p_j^(1/T))

    Returns:
        all_predictions: List of predicted class indices
        all_annotation_ids: List of annotation IDs
        all_probs: Ensembled probability matrix (N x C)
    """
    if decision == "min_risk" and distance_matrix is None:
        raise ValueError("distance_matrix required for min_risk decision")
    n_models = len(models)
    tta_views = tta or ["original"]
    n_views = len(tta_views)

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    all_predictions = []
    all_annotation_ids = []
    all_probs = []

    desc = f"Ensemble predicting ({n_views} TTA views)" if n_views > 1 else "Ensemble predicting"
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=desc):
            # Unpack: dataset returns (images, annotation_id, image_id)
            images, annotation_ids, image_ids = batch_data
            images = {k: v.to(device) for k, v in images.items()}
            image_ids_device = image_ids.to(device) if use_gnn else None

            # Collect predictions from all models × all TTA views
            # Each TTA view gets equal weight; model weights apply within each view
            view_probs = []
            for aug_name in tta_views:
                aug_images = apply_tta(images, aug_name)
                batch_probs = []
                for model in models:
                    outputs = model(aug_images, image_ids=image_ids_device)
                    logits = outputs["species"]
                    if temperature != 1.0:
                        logits = logits / temperature
                    probs = torch.softmax(logits, dim=1)
                    batch_probs.append(probs)

                # Average across models for this view
                stacked = torch.stack(batch_probs, dim=0)  # (n_models, B, C)
                weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
                view_avg = (stacked * weights_tensor).sum(dim=0)  # (B, C)
                view_probs.append(view_avg)

            # Average across TTA views (equal weight per view)
            ensemble_probs = torch.stack(view_probs, dim=0).mean(dim=0)  # (B, C)

            if decision == "min_risk":
                # Minimum-risk decision: pick species that minimizes expected taxonomic distance. 
                # # expected_dist[b, i] = Σ_j p_j * D_ij   (Eq. 3; D symmetric so D_ij = D_ji)
                expected_dist = ensemble_probs @ distance_matrix  # (B, C)
                predictions = expected_dist.argmin(dim=1)
            else:
                predictions = ensemble_probs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_annotation_ids.extend(annotation_ids)
            all_probs.append(ensemble_probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    return all_predictions, all_annotation_ids, all_probs


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction with k-fold models")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints (supports glob patterns)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
        help="Path to distance matrix",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_ensemble.csv",
        help="Output submission file",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["1x", "3x", "5x", "full"],
        help="Scales to use",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each model (default: equal)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["average", "voting"],
        default="average",
        help="Ensemble method",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also run evaluation on test set ground truth",
    )
    parser.add_argument(
        "--scale-backbones",
        type=str,
        nargs="+",
        default=None,
        help="Per-scale backbone mapping as 'scale=backbone' pairs. "
             "Example: --scale-backbones 1x=convnextv2_large.fcmae_ft_in22k_in1k",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=["concat", "hierarchy_attn", "gated_cross", "patch_self_attn", "part_attn", "patch_part_attn"],
        default="concat",
        help="Fusion type used when training the models",
    )
    parser.add_argument(
        "--decision",
        type=str,
        choices=["argmax", "min_risk"],
        default="argmax",
        help="Decision strategy: 'argmax' (highest probability) or "
             "'min_risk' (minimize expected taxonomic distance)",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default=None,
        help="Comma-separated TTA augmentations (e.g. 'hflip' or 'hflip,vflip'). "
             "Original view is always included.",
    )
    parser.add_argument(
        "--share-backbone",
        action="store_true",
        help="Load models with shared backbone across all scales",
    )
    parser.add_argument(
        "--branch-cnn",
        action="store_true",
        help="Load models with B-CNN multi-stage branching",
    )
    parser.add_argument(
        "--use-gnn",
        action="store_true",
        help="Enable GNN for multi-specimen relationship modeling during inference",
    )
    parser.add_argument(
        "--no-parent-conditioning",
        action="store_true",
        help="Load models with independent heads (no hierarchical conditioning)",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Load models with species-only head (coarse levels via marginalization)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling parameter (default: 1.0, no scaling). "
             "T > 1 softens predictions, T < 1 sharpens them. "
             "Applied via power scaling: p_i^(1/T) / sum(p_j^(1/T))",
    )
    args = parser.parse_args()

    # Parse scale_backbones into a dict
    scale_backbones = None
    if args.scale_backbones:
        scale_backbones = {}
        for item in args.scale_backbones:
            scale, backbone = item.split("=")
            scale_backbones[scale] = backbone

    # Expand glob patterns in checkpoint paths
    checkpoint_paths = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        if matches:
            checkpoint_paths.extend(matches)
        else:
            checkpoint_paths.append(pattern)

    checkpoint_paths = sorted(set(checkpoint_paths))
    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    if len(checkpoint_paths) == 0:
        raise ValueError("No checkpoints found")

    # Validate weights
    if args.weights is not None and len(args.weights) != len(checkpoint_paths):
        raise ValueError(
            f"Number of weights ({len(args.weights)}) must match "
            f"number of checkpoints ({len(checkpoint_paths)})"
        )

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(args.config)

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )

    # Load models
    models = load_models(
        checkpoint_paths,
        cfg,
        class_counts,
        id_to_name,
        args.scales,
        args.distance_matrix,
        device,
        scale_backbones=scale_backbones,
        fusion_type=args.fusion_type,
        share_backbone=args.share_backbone,
        branch_cnn=args.branch_cnn,
        use_gnn=args.use_gnn,
        parent_conditioning=not args.no_parent_conditioning,
        leaf_only=args.leaf_only,
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
    print(f"Test samples: {len(test_dataset)}")

    def collate_fn(batch):
        images_list, annotation_ids, image_ids = zip(*batch)
        images = {
            scale: torch.stack([img[scale] for img in images_list])
            for scale in args.scales
        }
        return images, list(annotation_ids), torch.tensor(image_ids, dtype=torch.long)

    if args.use_gnn:
        # GNN requires all annotations from the same image in the same batch
        from data.data import ImageGroupedBatchSampler
        test_image_ids = [ann["image_id"] for ann in test_dataset.annotations]
        test_sampler = ImageGroupedBatchSampler(
            image_ids=test_image_ids,
            max_batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    # Load distance matrix for min_risk decision
    dist_matrix = None
    if args.decision == "min_risk":
        # Grab from first model (already loaded and aligned with species indices)
        dist_matrix = models[0].distance_matrix  # (C, C) tensor on device
        print(f"Distance matrix shape: {dist_matrix.shape}")

    # Parse TTA views
    tta_views = None
    if args.tta:
        tta_views = ["original"] + [s.strip() for s in args.tta.split(",")]
        print(f"TTA views: {tta_views}")

    # Run ensemble prediction
    if args.temperature != 1.0:
        print(f"\nTemperature scaling: T={args.temperature}")
    print(f"\nRunning ensemble prediction ({args.method}, decision={args.decision})...")
    predictions, annotation_ids, probs = ensemble_predict(
        models, test_loader, device, args.weights, args.method,
        decision=args.decision, distance_matrix=dist_matrix, tta=tta_views,
        use_gnn=args.use_gnn, temperature=args.temperature,
    )

    # Convert to species names
    concept_names = [id_to_name["species"][pred] for pred in predictions]

    # Create submission
    submission_df = pd.DataFrame({
        "annotation_id": annotation_ids,
        "concept_name": concept_names,
    })
    submission_df = submission_df.sort_values("annotation_id")
    submission_df.to_csv(args.output, index=False)

    print(f"\nSubmission saved to {args.output}")
    print(f"Total predictions: {len(submission_df)}")

    # Show distribution
    print("\nPrediction distribution (top 10):")
    for name, count in submission_df["concept_name"].value_counts().head(10).items():
        print(f"  {name}: {count} ({count/len(submission_df)*100:.1f}%)")

    # Optionally evaluate
    if args.evaluate:
        print("\nRunning evaluation...")
        from src.inference.evaluate_on_test import evaluate_submission, print_results

        results = evaluate_submission(
            args.output,
            cfg.paths.test_solution_csv,
            args.distance_matrix,
            subset=None,
        )
        print_results(results, verbose=True)


if __name__ == "__main__":
    main()
