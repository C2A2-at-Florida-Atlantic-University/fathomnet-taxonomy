#!/usr/bin/env python3
"""
K-Fold Cross-Validation Training for FathomNet 2025.

This script trains multiple models using k-fold cross-validation to:
1. Get more robust estimates of model performance
2. Enable ensemble predictions from multiple folds
3. Reduce variance in hyperparameter selection

For final deployment, the ensemble of k models typically generalizes better
than any single model trained on a fixed split.

Usage:
    # 5-fold CV with default settings
    python train_kfold.py --config config/experiment-multiscale.yaml

    # 3-fold CV with specific scales
    python train_kfold.py --config config/experiment-multiscale.yaml \\
        --n-folds 3 --scales 1x 3x 5x full

    # Resume after a crash (skips completed folds automatically)
    python train_kfold.py --n-folds 10 \\
        --resume-from outputs/kfold_10fold

    # Add 5 more folds to an existing 10-fold run (new CV round, seed+1)
    python train_kfold.py --n-folds 15 \\
        --resume-from outputs/kfold_10fold

    # Resume from a specific fold index (legacy, still supported)
    python train_kfold.py --config config/experiment-multiscale.yaml \\
        --start-fold 3

    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python train_kfold.py --config config/experiment-multiscale.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from data.data import (
    MultiScalePrecroppedDataset,
    load_and_encode_taxonomy,
    load_coco_annotations,
    create_transforms,
    gnn_collate_fn,
    ImageGroupedBatchSampler,
)
from src.config import load_config
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


def create_fold_dataloaders(
    cfg: DictConfig,
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    taxonomy_df: pd.DataFrame,
    encoders: Dict,
    scales: List[str],
    use_gnn: bool = False,
    devices: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a specific fold.

    Args:
        cfg: Configuration object
        fold_idx: Current fold index (for logging)
        train_indices: Training sample indices
        val_indices: Validation sample indices
        taxonomy_df: Taxonomy DataFrame
        encoders: Label encoders for each taxonomy level
        scales: List of scales to use
        use_gnn: Whether GNN is enabled
        devices: Number of GPUs (1=single GPU, >1=DDP)

    Returns:
        train_loader, val_loader
    """
    # Load all annotations
    annotations = load_coco_annotations(
        cfg.paths.train_coco_json,
        cfg.paths.train_full_image_dir,
        include_labels=True,
    )

    # Create transforms
    train_transform, val_transform = create_transforms(cfg)

    # Split annotations by indices
    train_annotations = annotations.iloc[train_indices].reset_index(drop=True)
    val_annotations = annotations.iloc[val_indices].reset_index(drop=True)

    # Create separate datasets with appropriate transforms
    train_dataset = MultiScalePrecroppedDataset(
        frame=train_annotations,
        taxonomy_df=taxonomy_df,
        levels=list(cfg.data.taxonomy_levels),
        encoders=encoders,
        roi_root=cfg.paths.train_roi_root,
        scales=scales,
        transform=train_transform,
    )

    val_dataset = MultiScalePrecroppedDataset(
        frame=val_annotations,
        taxonomy_df=taxonomy_df,
        levels=list(cfg.data.taxonomy_levels),
        encoders=encoders,
        roi_root=cfg.paths.train_roi_root,
        scales=scales,
        transform=val_transform,
    )

    print(f"Fold {fold_idx}: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Create dataloaders
    if use_gnn and devices <= 1:
        # Single-GPU GNN: use ImageGroupedBatchSampler to ensure all
        # annotations from the same image are in the same batch, so the
        # GNN sees complete multi-specimen graphs during training.
        # This doesn't work with DDP (causes deadlocks), but is fine
        # on a single GPU.
        train_image_ids = [
            int(train_annotations.iloc[i].get("image_id", i))
            for i in range(len(train_annotations))
        ]
        train_sampler = ImageGroupedBatchSampler(
            image_ids=train_image_ids,
            max_batch_size=cfg.data.batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=gnn_collate_fn,
            pin_memory=True,
        )
        val_image_ids = [
            int(val_annotations.iloc[i].get("image_id", i))
            for i in range(len(val_annotations))
        ]
        val_sampler = ImageGroupedBatchSampler(
            image_ids=val_image_ids,
            max_batch_size=cfg.data.batch_size,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=gnn_collate_fn,
            pin_memory=True,
        )
    elif use_gnn:
        # Multi-GPU GNN: use standard DataLoader with gnn_collate_fn.
        # Lightning's DistributedSampler handles DDP distribution.
        # GNN processes whatever image groups end up in the same batch.
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=gnn_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=gnn_collate_fn,
            pin_memory=True,
        )
    else:
        # Standard collate: returns (images, labels) 2-tuple
        def collate_fn(batch):
            # batch is list of {"scales": {...}, "labels": {...}, "image_id": int}
            images = {
                scale: torch.stack([item["scales"][scale] for item in batch])
                for scale in scales
            }
            labels = {
                level: torch.tensor([item["labels"][level] for item in batch])
                for level in batch[0]["labels"].keys()
            }
            return images, labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader


def save_fold_splits(output_dir: str, splits: List[Dict[str, Any]]) -> str:
    """Save fold train/val index splits to JSON for reproducibility and resuming."""
    path = os.path.join(output_dir, "fold_splits.json")
    serializable = [
        {
            "fold": s["fold"],
            "seed": s["seed"],
            "n_splits": s["n_splits"],
            "train_indices": (
                s["train_indices"].tolist()
                if isinstance(s["train_indices"], np.ndarray)
                else s["train_indices"]
            ),
            "val_indices": (
                s["val_indices"].tolist()
                if isinstance(s["val_indices"], np.ndarray)
                else s["val_indices"]
            ),
        }
        for s in splits
    ]
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"Saved fold splits to {path}")
    return path


def load_fold_splits(output_dir: str) -> List[Dict[str, Any]]:
    """Load previously saved fold splits from a directory."""
    path = os.path.join(output_dir, "fold_splits.json")
    with open(path) as f:
        data = json.load(f)
    for s in data:
        s["train_indices"] = np.array(s["train_indices"])
        s["val_indices"] = np.array(s["val_indices"])
    return data


def find_completed_folds(output_dir: str) -> Dict[int, str]:
    """Find folds that already have a best checkpoint."""
    completed = {}
    for fold_dir in sorted(Path(output_dir).glob("fold_*")):
        try:
            fold_idx = int(fold_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        ckpt_dir = fold_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("fold*.ckpt"))
            if ckpts:
                completed[fold_idx] = str(ckpts[0])
    return completed


def generate_image_level_fold_splits(
    annotations: pd.DataFrame,
    n_folds: int,
    seed: int,
    existing_splits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate stratified fold splits grouped by image_id.

    All annotations from the same image are always placed in the same fold
    (either all in training or all in validation), eliminating shared-image
    leakage between train and validation sets.

    Stratification uses the most-common species label per image.

    Args:
        annotations: DataFrame with ``image_id`` and ``label`` columns.
        n_folds: Number of folds.
        seed: Random seed.
        existing_splits: Previously saved splits to preserve.

    Returns:
        List of fold split dicts (same format as ``generate_fold_splits``).
    """
    if existing_splits and n_folds <= len(existing_splits):
        return existing_splits[:n_folds]

    image_groups = annotations.groupby("image_id")
    unique_images = list(image_groups.groups.keys())

    # Per-image stratification label = most common species
    image_labels = []
    image_ann_indices: Dict[Any, List[int]] = {}
    for img_id in unique_images:
        group = image_groups.get_group(img_id)
        image_labels.append(group["label"].mode().iloc[0])
        image_ann_indices[img_id] = group.index.tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_splits: List[Dict[str, Any]] = []

    for fold_idx, (train_img_idx, val_img_idx) in enumerate(
        skf.split(np.zeros(len(unique_images)), image_labels)
    ):
        train_ann = [
            idx for i in train_img_idx for idx in image_ann_indices[unique_images[i]]
        ]
        val_ann = [
            idx for i in val_img_idx for idx in image_ann_indices[unique_images[i]]
        ]
        all_splits.append(
            {
                "fold": fold_idx,
                "seed": seed,
                "n_splits": n_folds,
                "train_indices": np.array(sorted(train_ann)),
                "val_indices": np.array(sorted(val_ann)),
            }
        )

    # Verify no image leakage
    for split in all_splits:
        train_imgs = set(annotations.iloc[split["train_indices"]]["image_id"])
        val_imgs = set(annotations.iloc[split["val_indices"]]["image_id"])
        overlap = train_imgs & val_imgs
        assert len(overlap) == 0, (
            f"Image-level split has {len(overlap)} images in both train and val!"
        )

    n_images = len(unique_images)
    print(f"Image-level splitting: {n_images} unique images → {n_folds} folds")
    print(f"  No image appears in both train and val (verified)")

    return all_splits


def generate_fold_splits(
    labels: List,
    n_folds: int,
    seed: int,
    existing_splits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate stratified fold splits, extending existing ones if needed.

    Uses "rounds" of StratifiedKFold to generate independent splits.
    Round 0 uses ``seed``, round 1 uses ``seed + 1``, etc.  Each round
    produces ``k`` folds, where ``k`` is determined by the first round
    (i.e. the original ``n_folds`` on the very first run).

    This means you can train 10 folds first, then later extend to 15 or 20
    without invalidating the original folds.  The additional folds come from
    fresh CV rounds with different random seeds.

    Example:
        First run with ``--n-folds 10 --seed 42`` creates round 0:
            folds 0-9  → StratifiedKFold(n_splits=10, seed=42)

        Later ``--n-folds 15 --resume-from ...`` extends with round 1:
            folds 0-9  → unchanged (loaded from fold_splits.json)
            folds 10-14 → StratifiedKFold(n_splits=10, seed=43), first 5

    Args:
        labels: Class labels for stratification.
        n_folds: Total number of folds to produce.
        seed: Base random seed (round N uses seed + N).
        existing_splits: Previously saved splits to preserve.

    Returns:
        List of fold split dicts, each containing fold index, seed,
        n_splits, train_indices, and val_indices.
    """
    if existing_splits and n_folds <= len(existing_splits):
        return existing_splits[:n_folds]

    if existing_splits:
        # Determine round size from the first round's n_splits
        k = existing_splits[0]["n_splits"]
        all_splits: List[Dict[str, Any]] = list(existing_splits)
    else:
        k = n_folds
        all_splits = []

    while len(all_splits) < n_folds:
        round_idx = len(all_splits) // k
        fold_in_round = len(all_splits) % k
        round_seed = seed + round_idx

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=round_seed)
        round_folds = list(skf.split(np.zeros(len(labels)), labels))

        for i in range(fold_in_round, k):
            if len(all_splits) >= n_folds:
                break
            train_idx, val_idx = round_folds[i]
            all_splits.append(
                {
                    "fold": len(all_splits),
                    "seed": round_seed,
                    "n_splits": k,
                    "train_indices": train_idx,
                    "val_indices": val_idx,
                }
            )

    return all_splits


def train_single_fold(
    cfg: DictConfig,
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_counts: Dict,
    id_to_name: Dict,
    scales: List[str],
    output_dir: str,
    distance_matrix_path: str,
    loss_type: str,
    alpha: float,
    smoothing: float,
    scale_backbones: Optional[Dict[str, str]] = None,
    use_uncertainty_weighting: bool = False,
    taxonomy_df: Optional[pd.DataFrame] = None,
    fusion_type: str = "concat",
    devices: int = 1,
    species_class_weights: Optional[torch.Tensor] = None,
    consistency_weight: float = 0.0,
    share_backbone: bool = False,
    branch_cnn: bool = False,
    bt_strategy: bool = False,
    use_gnn: bool = False,
    parent_conditioning: bool = True,
    leaf_only: bool = False,
    stop_gradient: bool = False,
) -> str:
    """
    Train a single fold and return the best checkpoint path.

    Returns:
        Path to the best checkpoint for this fold
    """
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    # Create model
    model = MultiScaleTaxonomicClassifier(
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=scales,
        distance_matrix_path=distance_matrix_path,
        loss_type=loss_type,
        alpha=alpha,
        smoothing=smoothing,
        scale_backbones=scale_backbones,
        use_uncertainty_weighting=use_uncertainty_weighting,
        taxonomy_df=taxonomy_df,
        fusion_type=fusion_type,
        species_class_weights=species_class_weights,
        consistency_weight=consistency_weight,
        share_backbone=share_backbone,
        branch_cnn=branch_cnn,
        bt_strategy=bt_strategy,
        use_gnn=use_gnn,
        parent_conditioning=parent_conditioning,
        leaf_only=leaf_only,
        stop_gradient=stop_gradient,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(fold_output_dir, "checkpoints"),
        filename=f"fold{fold_idx}-{{epoch:02d}}-{{val_tax_score:.3f}}",
        monitor="val_tax_score",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_tax_score",
        mode="min",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(
        save_dir=fold_output_dir,
        name="logs",
        version=f"fold_{fold_idx}",
    )

    # Trainer - use DDP strategy when using multiple GPUs
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_callback.best_model_path


def run_kfold_cv(
    cfg: DictConfig,
    n_folds: int,
    scales: List[str],
    output_dir: str,
    distance_matrix_path: str,
    loss_type: str,
    alpha: float,
    smoothing: float,
    start_fold: int = 0,
    seed: int = 42,
    scale_backbones: Optional[Dict[str, str]] = None,
    use_uncertainty_weighting: bool = False,
    fusion_type: str = "concat",
    devices: int = 1,
    resume_from: Optional[str] = None,
    species_class_weights: Optional[torch.Tensor] = None,
    consistency_weight: float = 0.0,
    share_backbone: bool = False,
    branch_cnn: bool = False,
    bt_strategy: bool = False,
    use_gnn: bool = False,
    parent_conditioning: bool = True,
    leaf_only: bool = False,
    stop_gradient: bool = False,
    stop_after_fold: Optional[int] = None,
    image_level_split: bool = False,
) -> List[str]:
    """
    Run k-fold cross-validation training.

    Args:
        cfg: Configuration object
        n_folds: Number of folds
        scales: List of scales to use
        output_dir: Output directory for all folds
        distance_matrix_path: Path to distance matrix
        loss_type: Loss function type
        alpha: Alpha for distance loss
        smoothing: Label smoothing factor
        start_fold: Fold to start from (for resuming)
        seed: Random seed for reproducibility
        scale_backbones: Optional dict mapping scales to backbone names
        use_uncertainty_weighting: Whether to use learnable uncertainty weighting
        fusion_type: Multi-scale fusion method
        devices: Number of GPUs
        resume_from: Path to a previous run's output directory. Loads
            existing fold splits and skips folds that already have
            checkpoints.  If n_folds exceeds the saved splits, additional
            rounds of k-fold CV are generated with incrementing seeds.
        species_class_weights: Optional inverse-frequency weights for species CE
        consistency_weight: Weight for C-HMCNN hierarchy consistency loss
        share_backbone: Share one backbone across all scales
        branch_cnn: Enable B-CNN multi-stage branching
        bt_strategy: Enable BT-Strategy dynamic weight scheduling
        image_level_split: Group annotations by image_id for fold splitting

    Returns:
        List of best checkpoint paths for each fold
    """
    pl.seed_everything(seed)

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )
    print(f"Class counts: {class_counts}")

    # Load annotations for stratification
    annotations = load_coco_annotations(
        cfg.paths.train_coco_json,
        cfg.paths.train_full_image_dir,
        include_labels=True,
    )
    labels = annotations["label"].tolist()

    # ---- Compute species class weights if requested ----
    if species_class_weights is not None:
        print(f"Species class weighting: inverse-frequency")
        print(f"  Weight range: [{species_class_weights.min():.2f}, {species_class_weights.max():.2f}]")
        print(f"  Weight mean: {species_class_weights.mean():.2f}")
    else:
        print("Species class weighting: none")

    # ---- Generate or load fold splits ----
    # DDP guard: with multi-GPU DDP, all ranks execute this function in
    # parallel.  Only rank 0 should do file I/O to avoid race conditions.
    # Other ranks generate identical splits locally (deterministic with
    # the same seed) and skip saving.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    existing_splits = None
    if local_rank == 0:
        # Only rank 0 checks for and loads existing splits from disk
        for search_dir in [resume_from, output_dir]:
            if search_dir and os.path.exists(os.path.join(search_dir, "fold_splits.json")):
                existing_splits = load_fold_splits(search_dir)
                print(f"Loaded {len(existing_splits)} existing fold splits from {search_dir}")
                break

    if image_level_split:
        splits = generate_image_level_fold_splits(
            annotations, n_folds, seed, existing_splits
        )
    else:
        splits = generate_fold_splits(labels, n_folds, seed, existing_splits)

    if local_rank == 0:
        save_fold_splits(output_dir, splits)

    # ---- Detect completed folds ----
    completed_folds = find_completed_folds(output_dir)
    if resume_from and resume_from != output_dir:
        completed_folds.update(find_completed_folds(resume_from))
    if completed_folds:
        print(f"Found {len(completed_folds)} completed fold(s): {sorted(completed_folds.keys())}")

    best_checkpoints = []

    print(f"\n{'='*60}")
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Total samples: {len(labels)}")
    print(f"Scales: {scales}")
    print(f"Fusion type: {fusion_type}")
    print(f"Loss type: {loss_type}, alpha: {alpha}, smoothing: {smoothing}")
    print(f"Uncertainty weighting: {use_uncertainty_weighting}")
    if scale_backbones:
        print(f"Scale backbones: {scale_backbones}")
    print(f"Output: {output_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"{'='*60}\n")

    for split_info in splits:
        fold_idx = split_info["fold"]
        train_idx = split_info["train_indices"]
        val_idx = split_info["val_indices"]

        if fold_idx < start_fold:
            print(f"Skipping fold {fold_idx} (starting from fold {start_fold})")
            continue

        if fold_idx in completed_folds:
            ckpt = completed_folds[fold_idx]
            print(f"Fold {fold_idx} already complete: {ckpt}")
            best_checkpoints.append(ckpt)
            continue

        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold_idx + 1}/{n_folds}")
        print(f"  Round seed: {split_info['seed']}, "
              f"k={split_info['n_splits']}")
        print(f"{'='*60}")

        # Create dataloaders for this fold
        train_loader, val_loader = create_fold_dataloaders(
            cfg=cfg,
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            taxonomy_df=taxonomy_df,
            encoders=encoders,
            scales=scales,
            use_gnn=use_gnn,
            devices=devices,
        )

        # Train this fold
        best_ckpt = train_single_fold(
            cfg=cfg,
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            class_counts=class_counts,
            id_to_name=id_to_name,
            scales=scales,
            output_dir=output_dir,
            distance_matrix_path=distance_matrix_path,
            loss_type=loss_type,
            alpha=alpha,
            smoothing=smoothing,
            scale_backbones=scale_backbones,
            use_uncertainty_weighting=use_uncertainty_weighting,
            taxonomy_df=taxonomy_df,
            fusion_type=fusion_type,
            devices=devices,
            species_class_weights=species_class_weights,
            consistency_weight=consistency_weight,
            share_backbone=share_backbone,
            branch_cnn=branch_cnn,
            bt_strategy=bt_strategy,
            use_gnn=use_gnn,
            parent_conditioning=parent_conditioning,
            leaf_only=leaf_only,
            stop_gradient=stop_gradient,
        )

        best_checkpoints.append(best_ckpt)
        print(f"\nFold {fold_idx} best checkpoint: {best_ckpt}")

        if stop_after_fold is not None and fold_idx >= stop_after_fold:
            print(f"Stopping after fold {fold_idx} (--stop-after-fold {stop_after_fold})")
            break

    # Save summary (rank 0 only)
    if local_rank != 0:
        return best_checkpoints
    summary_path = os.path.join(output_dir, "kfold_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"K-Fold Cross-Validation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"N-folds: {n_folds}\n")
        f.write(f"Scales: {scales}\n")
        f.write(f"Fusion type: {fusion_type}\n")
        f.write(f"Loss type: {loss_type}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Smoothing: {smoothing}\n")
        f.write(f"Uncertainty weighting: {use_uncertainty_weighting}\n")
        if resume_from:
            f.write(f"Resumed from: {resume_from}\n")
        if scale_backbones:
            f.write(f"Scale backbones: {scale_backbones}\n")
        f.write(f"\nBest Checkpoints:\n")
        for i, ckpt in enumerate(best_checkpoints):
            f.write(f"  Fold {i}: {ckpt}\n")

    print(f"\n{'='*60}")
    print(f"K-FOLD TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Summary saved to: {summary_path}")

    return best_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["1x", "3x", "5x", "full"],
        help="Scales to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/kfold_<timestamp>)",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
        help="Path to distance matrix",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="distance",
        choices=["distance", "smooth", "both", "ce", "hxe"],
        help="Loss function type (hxe = Hierarchical Cross-Entropy from Bertinetto et al. 2020)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha for distance loss",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Fold to start from (for resuming)",
    )
    parser.add_argument(
        "--stop-after-fold",
        type=int,
        default=None,
        help="Stop after completing this fold index (e.g., 0 to train only fold 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--scale-backbones",
        type=str,
        nargs="+",
        default=None,
        help="Per-scale backbone mapping as 'scale=backbone' pairs. "
             "Example: --scale-backbones 1x=convnextv2_large.fcmae_ft_in22k_in1k "
             "3x=convnextv2_base.fcmae_ft_in22k_in1k",
    )
    parser.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        help="Use learnable uncertainty weighting (Kendall et al. 2018) "
             "instead of fixed hierarchy weights",
    )
    parser.add_argument(
        "--weight-scheme",
        type=str,
        default="piecewise",
        choices=["uniform", "linear", "piecewise", "piecewise_1000x"],
        help="Hierarchy weight scheme: "
             "uniform (all 1.0), "
             "linear (1,2,3,4,5,6,7), "
             "piecewise (0.5,0.75,1.0,1.25,1.5,2.0,2.5 - default)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous run's output directory to resume from. "
             "Loads existing fold splits and skips folds that already have "
             "checkpoints. Use with a larger --n-folds to add more folds.",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="concat",
        choices=["concat", "hierarchy_attn", "gated_cross", "patch_self_attn", "part_attn", "patch_part_attn"],
        help="Multi-scale fusion method: "
             "concat (original concatenation), "
             "hierarchy_attn (per-level attention over scales), "
             "gated_cross (self-attention between scales), "
             "patch_self_attn (self-attention within ROI patches), "
             "part_attn (learnable part prototypes for ROI), "
             "patch_part_attn (combined: patch self-attn + part prototypes)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1)",
    )
    parser.add_argument(
        "--class-weighting",
        type=str,
        default="none",
        choices=["none", "inverse"],
        help="Species class weighting: 'none' (default) or "
             "'inverse' (inverse frequency, upweights rare species)",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Weight for C-HMCNN-inspired hierarchy consistency loss (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--share-backbone",
        action="store_true",
        help="Share one backbone across all scales (for Jetson Nano deployment). "
             "Reduces params from N×backbone to 1×backbone.",
    )
    parser.add_argument(
        "--branch-cnn",
        action="store_true",
        help="Enable B-CNN multi-stage branching: coarse taxonomy levels use early "
             "backbone stages, fine levels use deep stages (Zhu & Bain 2017).",
    )
    parser.add_argument(
        "--bt-strategy",
        action="store_true",
        help="Enable BT-Strategy dynamic weight scheduling: shift loss weight "
             "from coarse to fine levels during training (Zhu & Bain 2017).",
    )
    parser.add_argument(
        "--use-gnn",
        action="store_true",
        help="Enable GNN for multi-specimen relationship modeling. "
             "Adds a 2-layer GCN after feature fusion to let specimens "
             "from the same image exchange information.",
    )
    parser.add_argument(
        "--no-parent-conditioning",
        action="store_true",
        help="Use independent classification heads (no hierarchical conditioning). "
             "Each taxonomy level is predicted independently from shared features.",
    )
    parser.add_argument(
        "--stop-gradient",
        action="store_true",
        help="Stop gradients through parent→child conditioning. Parent predictions "
             "are detached before being passed as features to child heads, preventing "
             "error propagation through the cascade during training.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Species-only head with marginalization to coarse levels. "
             "Predicts only at species level; coarse-level probabilities are derived "
             "by summing species probabilities within each ancestor group.",
    )
    parser.add_argument(
        "--image-level-split",
        action="store_true",
        help="Group annotations by image_id for fold splitting. "
             "All annotations from the same image stay in the same fold, "
             "eliminating shared-image leakage between train and val.",
    )
    # ViT fine-tuning (LLRD) arguments
    parser.add_argument(
        "--layer-decay",
        type=float,
        default=1.0,
        help="Layer-wise LR decay for ViT backbones (1.0=off, 0.65-0.75 typical). "
             "Lower layers get smaller LR: lr_i = base_lr × scale × decay^(N-i).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of linear LR warmup epochs (3-5 typical for ViT fine-tuning).",
    )
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.0,
        help="Stochastic depth / drop path rate for ViT backbone (0.1 typical).",
    )
    args = parser.parse_args()

    # Parse scale_backbones into a dict
    scale_backbones = None
    if args.scale_backbones:
        scale_backbones = {}
        for item in args.scale_backbones:
            scale, backbone = item.split("=")
            scale_backbones[scale] = backbone

    # Load config
    cfg = load_config(args.config)

    # Override hierarchy weights based on weight scheme
    WEIGHT_SCHEMES = {
        "uniform": {
            "kingdom": 1.0,
            "phylum": 1.0,
            "class": 1.0,
            "order": 1.0,
            "family": 1.0,
            "genus": 1.0,
            "species": 1.0,
        },
        "linear": {
            "kingdom": 1.0,
            "phylum": 2.0,
            "class": 3.0,
            "order": 4.0,
            "family": 5.0,
            "genus": 6.0,
            "species": 7.0,
        },
        "piecewise": {
            "kingdom": 0.5,
            "phylum": 0.75,
            "class": 1.0,
            "order": 1.25,
            "family": 1.5,
            "genus": 2.0,
            "species": 2.5,
        },
        "piecewise_1000x": {
            "kingdom": 500,
            "phylum": 750,
            "class": 1000,
            "order": 1250,
            "family": 1500,
            "genus": 2000,
            "species": 2500,
        },
    }
    cfg.loss.hierarchy_weights = WEIGHT_SCHEMES[args.weight_scheme]
    print(f"Using weight scheme: {args.weight_scheme}")
    print(f"  Weights: {dict(cfg.loss.hierarchy_weights)}")

    # Apply ViT fine-tuning overrides
    cfg.training.layer_decay = args.layer_decay
    cfg.training.warmup_epochs = args.warmup_epochs
    cfg.training.drop_path_rate = args.drop_path_rate
    if args.layer_decay < 1.0:
        print(f"ViT LLRD enabled: layer_decay={args.layer_decay}, "
              f"warmup={args.warmup_epochs}, drop_path={args.drop_path_rate}")

    # Set output directory
    if args.output_dir is None:
        if args.resume_from:
            # Default to resuming in the same directory
            output_dir = args.resume_from
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/kfold_{args.n_folds}fold_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_save_path)

    # Compute species class weights if requested
    species_class_weights = None
    if args.class_weighting == "inverse":
        # Load annotations to compute species frequency
        _annotations = load_coco_annotations(
            cfg.paths.train_coco_json,
            cfg.paths.train_full_image_dir,
            include_labels=True,
        )
        _, _encoders, _class_counts, _ = load_and_encode_taxonomy(
            cfg.paths.taxonomy_csv,
            list(cfg.data.taxonomy_levels),
        )
        # Encode string labels to integers
        species_labels = _encoders["species"].transform(_annotations["label"].values)
        n_species = _class_counts["species"]
        counts = np.bincount(species_labels, minlength=n_species).astype(float)
        counts = np.maximum(counts, 1.0)  # avoid div-by-zero
        weights = len(species_labels) / (n_species * counts)
        # Cap extreme weights (e.g., "unknown" with 1 sample gets 63x)
        max_weight = 10.0
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.mean()  # renormalize so mean = 1.0
        species_class_weights = torch.from_numpy(weights).float()
        print(f"Computed inverse-frequency weights for {n_species} species (capped at {max_weight}x)")
        print(f"  Range: [{species_class_weights.min():.2f}, {species_class_weights.max():.2f}]")

    # Run k-fold training
    best_checkpoints = run_kfold_cv(
        cfg=cfg,
        n_folds=args.n_folds,
        scales=args.scales,
        output_dir=output_dir,
        distance_matrix_path=args.distance_matrix,
        loss_type=args.loss_type,
        alpha=args.alpha,
        smoothing=args.smoothing,
        start_fold=args.start_fold,
        seed=args.seed,
        scale_backbones=scale_backbones,
        use_uncertainty_weighting=args.uncertainty_weighting,
        fusion_type=args.fusion_type,
        devices=args.devices,
        resume_from=args.resume_from,
        species_class_weights=species_class_weights,
        consistency_weight=args.consistency_weight,
        share_backbone=args.share_backbone,
        branch_cnn=args.branch_cnn,
        bt_strategy=args.bt_strategy,
        use_gnn=args.use_gnn,
        parent_conditioning=not args.no_parent_conditioning,
        leaf_only=args.leaf_only,
        stop_gradient=args.stop_gradient,
        stop_after_fold=args.stop_after_fold,
        image_level_split=args.image_level_split,
    )


if __name__ == "__main__":
    main()
