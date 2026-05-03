#!/usr/bin/env python3
"""
Training script for Multi-Scale Cross-Attention model with Taxonomic Loss.

This combines the best of both worlds:
1. Cross-attention for dynamic context fusion
2. Taxonomic distance-aware loss for biologically informed training

Usage:
    # Train with default settings (4-layer attention, distance loss)
    python -m src.training.train_attention_taxloss

    # Custom configuration
    python -m src.training.train_attention_taxloss \
        --config config/experiment-multiscale.yaml \
        --scales 1x 3x 5x full \
        --num-attention-layers 6 \
        --num-attention-heads 8 \
        --loss-type both \
        --alpha 0.3 \
        --exp-name attention_taxloss_6layer

    # Asymmetric backbones (larger ROI encoder)
    python -m src.training.train_attention_taxloss \
        --roi-backbone convnextv2_large.fcmae_ft_in22k_in1k_384 \
        --context-backbone convnextv2_base.fcmae_ft_in22k_in1k \
        --exp-name attention_taxloss_asymmetric

Arguments:
    --config: Path to config YAML file (default: config/experiment-multiscale.yaml)
    --scales: Scales to use (default: 1x 3x 5x)
    --num-attention-layers: Number of cross-attention layers (default: 4)
    --num-attention-heads: Number of attention heads (default: 8)
    --roi-backbone: Backbone for ROI encoder (default: from config)
    --context-backbone: Backbone for context encoders (default: from config)
    --loss-type: Loss type - "distance", "smooth", "both", or "ce" (default: distance)
    --alpha: Weight for distance term in loss (default: 0.3)
    --smoothing: Label smoothing amount (default: 0.1)
    --exp-name: Experiment name for logging (default: attention_taxloss)
    --resume: Resume from checkpoint path
"""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from data.data import build_multiscale_dataloaders, load_and_encode_taxonomy
from src.config import Config, load_config
from src.models import MultiScaleAttentionTaxonomicClassifier


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Scales to use (e.g., 1x 3x 5x full)",
    )
    parser.add_argument(
        "--num-attention-layers",
        type=int,
        default=4,
        help="Number of cross-attention layers",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--roi-backbone",
        type=str,
        default=None,
        help="Backbone for ROI encoder (overrides config)",
    )
    parser.add_argument(
        "--context-backbone",
        type=str,
        default=None,
        help="Backbone for context encoders (overrides config)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="distance",
        choices=["distance", "smooth", "both", "ce"],
        help="Loss type for species level",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=Config.ALPHA,
        help=f"Weight for distance term in TaxonomicDistanceLoss (default: {Config.ALPHA})",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Amount of label smoothing for TaxonomicLabelSmoothing",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="attention_taxloss",
        help="Experiment name for logging and checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration (merges YAML overrides with defaults from Config)
    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Scales: {args.scales}")
    print(f"Attention layers: {args.num_attention_layers}, heads: {args.num_attention_heads}")
    print(f"Loss type: {args.loss_type}, alpha: {args.alpha}")

    # Set random seed for reproducibility
    seed = cfg.get("project", {}).get("seed", 42)  # Use project.seed or default to 42
    if "training" in cfg and "seed" in cfg.training:
        seed = cfg.training.seed
    pl.seed_everything(seed, workers=True)

    # =================================================================
    # LOAD TAXONOMY AND BUILD DATALOADERS
    # =================================================================

    print("\n=== Loading taxonomy ===")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        taxonomy_csv=cfg.data.taxonomy_csv,
        levels=list(cfg.data.taxonomy_levels),
    )

    print("\n=== Building dataloaders ===")
    dataloaders, split_indices = build_multiscale_dataloaders(
        cfg=cfg,
        taxonomy_df=taxonomy_df,
        encoders=encoders,
        scales=args.scales,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # =================================================================
    # CREATE MODEL
    # =================================================================

    print("\n=== Creating model ===")
    print(f"Model: MultiScaleAttentionTaxonomicClassifier")

    # Distance matrix path
    distance_matrix_path = cfg.data.get("distance_matrix_csv", "data/distance_matrix.csv")
    if not os.path.exists(distance_matrix_path):
        print(f"WARNING: Distance matrix not found at {distance_matrix_path}")
        print("Will fall back to standard cross-entropy loss")
        distance_matrix_path = None

    model = MultiScaleAttentionTaxonomicClassifier(
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=args.scales,
        num_attention_layers=args.num_attention_layers,
        num_attention_heads=args.num_attention_heads,
        roi_backbone=args.roi_backbone,
        context_backbone=args.context_backbone,
        distance_matrix_path=distance_matrix_path,
        loss_type=args.loss_type,
        alpha=args.alpha,
        smoothing=args.smoothing,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =================================================================
    # SETUP CALLBACKS AND LOGGER
    # =================================================================

    output_dir = Path("outputs") / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="lightning_logs",
        version=None,
    )

    # Checkpoint callback - save best models based on val_tax_score
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_tax_score:.3f}",
        monitor="val_tax_score",
        mode="min",  # Lower taxonomic score is better
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    # Early stopping based on val_tax_score
    early_stop_callback = EarlyStopping(
        monitor="val_tax_score",
        mode="min",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    # =================================================================
    # SETUP TRAINER
    # =================================================================

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=4,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision=cfg.training.get("precision", "32-true"),
        deterministic=True,
        log_every_n_steps=10,
    )

    # =================================================================
    # TRAIN MODEL
    # =================================================================

    print("\n=== Starting training ===")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"Batch size: {cfg.data.batch_size}")  # batch_size is in cfg.data
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Backbone LR scale: {cfg.training.backbone_lr_scale}")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(model, train_loader, val_loader)

    # =================================================================
    # TRAINING COMPLETE
    # =================================================================

    print("\n=== Training complete ===")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_tax_score: {checkpoint_callback.best_model_score:.4f}")
    print(f"Logs saved to: {output_dir}")


if __name__ == "__main__":
    main()
