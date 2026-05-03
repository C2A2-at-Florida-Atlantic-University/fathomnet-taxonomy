#!/usr/bin/env python3
"""
Comparison experiment: Attention vs Taxonomic Loss vs Combined.

This script trains and compares three models:
1. Baseline: Simple concatenation + standard CE loss
2. Attention: Cross-attention fusion + standard CE loss
3. TaxLoss: Simple concatenation + taxonomic distance loss
4. Combined: Cross-attention fusion + taxonomic distance loss

Usage:
    python scripts/compare_models.py --config config/experiment-multiscale.yaml

    # Quick test with fewer epochs
    python scripts/compare_models.py --max-epochs 5 --batch-size 8

    # Skip models you've already trained
    python scripts/compare_models.py --skip baseline --skip attention
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.data import build_multiscale_dataloaders, load_and_encode_taxonomy
from src.config import Config, load_config
from src.models import (
    MultiScaleTaxonomyClassifier,
    MultiScaleCrossAttentionClassifier,
    MultiScaleTaxonomicClassifier,
    MultiScaleAttentionTaxonomicClassifier,
)


def train_model(
    model_class,
    model_name: str,
    cfg,
    class_counts: Dict,
    id_to_name: Dict,
    train_loader,
    val_loader,
    scales: List[str],
    max_epochs: int,
    **model_kwargs,
):
    """
    Train a single model and return results.

    Args:
        model_class: Model class to instantiate
        model_name: Name for logging
        cfg: Configuration object
        class_counts: Dict of class counts per level
        id_to_name: Dict of id-to-name mappings per level
        train_loader: Training dataloader
        val_loader: Validation dataloader
        scales: List of scales to use
        max_epochs: Maximum training epochs
        **model_kwargs: Additional kwargs for model constructor

    Returns:
        Dict with training results
    """
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    # Create model
    model = model_class(
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=scales,
        **model_kwargs,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup logging and checkpoints
    output_dir = Path("outputs") / f"comparison_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="lightning_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Get results
    results = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_checkpoint": str(checkpoint_callback.best_model_path),
        "final_val_loss": float(checkpoint_callback.best_model_score),
        "epochs_trained": trainer.current_epoch + 1,
    }

    # Try to get taxonomic score if available
    if hasattr(model, "_compute_taxonomic_score"):
        # Would need to recompute on validation set
        # For now, just log that it's available
        results["has_tax_score"] = True
    else:
        results["has_tax_score"] = False

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["1x", "3x", "5x"],
        help="Scales to use",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["baseline", "attention", "taxloss", "combined"],
        help="Models to skip",
    )
    args = parser.parse_args()

    # Load config (merges YAML overrides with defaults from Config)
    cfg = load_config(args.config)

    # Override config if specified
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    print(f"Configuration:")
    print(f"  Config file: {args.config}")
    print(f"  Scales: {args.scales}")
    print(f"  Max epochs: {cfg.training.max_epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Skipping: {args.skip if args.skip else 'None'}")

    # Set seed
    pl.seed_everything(cfg.training.seed, workers=True)

    # Load taxonomy
    print("\n=== Loading taxonomy ===")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        csv_path=cfg.data.taxonomy_csv
    )

    # Build dataloaders
    print("\n=== Building dataloaders ===")
    train_loader, val_loader = build_multiscale_dataloaders(
        scales=args.scales,
        rois_base_dir=cfg.data.train_rois_dir,
        annotations_csv=cfg.data.train_annotations_csv,
        encoders=encoders,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        train_split=cfg.data.train_split,
        augmentation_level=cfg.data.augmentation_level,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Distance matrix path
    distance_matrix_path = cfg.data.get("distance_matrix_csv", "data/distance_matrix.csv")
    if not Path(distance_matrix_path).exists():
        print(f"WARNING: Distance matrix not found at {distance_matrix_path}")
        distance_matrix_path = None

    # =================================================================
    # TRAIN MODELS
    # =================================================================

    results = []

    # 1. Baseline: Simple concatenation + CE loss
    if "baseline" not in args.skip:
        baseline_results = train_model(
            model_class=MultiScaleTaxonomyClassifier,
            model_name="baseline",
            cfg=cfg,
            class_counts=class_counts,
            id_to_name=id_to_name,
            train_loader=train_loader,
            val_loader=val_loader,
            scales=args.scales,
            max_epochs=cfg.training.max_epochs,
        )
        results.append(baseline_results)

    # 2. Attention: Cross-attention + CE loss
    if "attention" not in args.skip:
        attention_results = train_model(
            model_class=MultiScaleCrossAttentionClassifier,
            model_name="attention",
            cfg=cfg,
            class_counts=class_counts,
            id_to_name=id_to_name,
            train_loader=train_loader,
            val_loader=val_loader,
            scales=args.scales,
            max_epochs=cfg.training.max_epochs,
            num_attention_layers=4,
            num_attention_heads=8,
        )
        results.append(attention_results)

    # 3. TaxLoss: Simple concatenation + taxonomic loss
    if "taxloss" not in args.skip and distance_matrix_path:
        taxloss_results = train_model(
            model_class=MultiScaleTaxonomicClassifier,
            model_name="taxloss",
            cfg=cfg,
            class_counts=class_counts,
            id_to_name=id_to_name,
            train_loader=train_loader,
            val_loader=val_loader,
            scales=args.scales,
            max_epochs=cfg.training.max_epochs,
            distance_matrix_path=distance_matrix_path,
            loss_type="distance",
            alpha=0.3,
        )
        results.append(taxloss_results)

    # 4. Combined: Cross-attention + taxonomic loss
    if "combined" not in args.skip and distance_matrix_path:
        combined_results = train_model(
            model_class=MultiScaleAttentionTaxonomicClassifier,
            model_name="combined",
            cfg=cfg,
            class_counts=class_counts,
            id_to_name=id_to_name,
            train_loader=train_loader,
            val_loader=val_loader,
            scales=args.scales,
            max_epochs=cfg.training.max_epochs,
            num_attention_layers=4,
            num_attention_heads=8,
            distance_matrix_path=distance_matrix_path,
            loss_type="distance",
            alpha=0.3,
        )
        results.append(combined_results)

    # =================================================================
    # SAVE COMPARISON RESULTS
    # =================================================================

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by validation loss
    df = df.sort_values("final_val_loss")

    print("\nResults (sorted by val_loss):")
    print(df.to_string(index=False))

    # Save to file
    output_file = Path("outputs") / "model_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save detailed JSON
    json_file = Path("outputs") / "model_comparison.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    best = df.iloc[0]
    print(f"\nBest model: {best['model_name']}")
    print(f"Val loss: {best['final_val_loss']:.4f}")
    print(f"Parameters: {best['total_params']:,}")
    print(f"Checkpoint: {best['best_checkpoint']}")

    # Compare to baseline
    if "baseline" not in args.skip:
        baseline = df[df["model_name"] == "baseline"].iloc[0]
        improvement = ((baseline["final_val_loss"] - best["final_val_loss"])
                      / baseline["final_val_loss"] * 100)
        print(f"\nImprovement over baseline: {improvement:.2f}%")


if __name__ == "__main__":
    main()
