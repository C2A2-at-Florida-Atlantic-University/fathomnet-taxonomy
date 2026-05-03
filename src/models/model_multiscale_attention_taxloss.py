"""
Multi-scale Cross-Attention Model with Taxonomic Distance-Aware Loss.

This model combines the best of both worlds:
1. Cross-attention feature fusion (from model_multiscale_attention.py)
2. Taxonomic distance-aware loss (from model_multiscale_taxloss.py)

Architecture:
    ROI Encoder ──┐
    3x Encoder ───┼──> Cross-Attention ──> Fusion ──> Hierarchical Heads
    5x Encoder ───┘                                         │
                                                            ▼
                                                    Taxonomic Loss
                                                    (distance-aware)

Key Features:
-------------
1. Cross-attention allows ROI to selectively attend to relevant context
2. Taxonomic loss penalizes predictions based on biological distance
3. Validation metric tracks expected taxonomic score (competition metric)

Usage:
------
    model = MultiScaleAttentionTaxonomicClassifier(
        cfg=config,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=["1x", "3x", "5x"],
        num_attention_layers=4,
        num_attention_heads=8,
        distance_matrix_path="data/distance_matrix.csv",
        loss_type="distance",  # or "smooth", "both", "ce"
        alpha=0.3,
    )
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..config import Config
from ..losses import TaxonomicDistanceLoss, TaxonomicLabelSmoothing

# Import cross-attention components from existing model
from .model_multiscale_attention import (
    CrossAttentionBlock,
    MultiLayerCrossAttention,
    PatchFeatureExtractor,
)


class MultiScaleAttentionTaxonomicClassifier(pl.LightningModule):
    """
    Multi-scale classifier with cross-attention fusion and taxonomic loss.

    Combines cross-attention mechanism with taxonomic distance-aware training
    for optimal performance on the FathomNet competition.

    Args:
        cfg: OmegaConf configuration
        class_counts: Dict mapping level names to number of classes
        id_to_name: Dict mapping level names to {class_id: class_name} dicts
        scales: List of scale names (default: ["1x", "3x", "5x"])
        num_attention_layers: Number of cross-attention layers
        num_attention_heads: Number of attention heads
        roi_backbone: Optional ROI backbone name
        context_backbone: Optional context backbone name
        distance_matrix_path: Path to taxonomic distance CSV
        loss_type: "distance", "smooth", "both", or "ce"
        alpha: Weight for distance term in loss
        smoothing: Amount of label smoothing
    """

    def __init__(
        self,
        cfg: DictConfig,
        class_counts: Dict[str, int],
        id_to_name: Dict[str, Dict[int, str]],
        scales: List[str] = None,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        roi_backbone: Optional[str] = None,
        context_backbone: Optional[str] = None,
        distance_matrix_path: Optional[str] = None,
        loss_type: str = "distance",
        alpha: float = Config.ALPHA,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "id_to_name"])

        self.cfg = cfg
        self.class_counts = class_counts
        self.id_to_name = id_to_name
        self.levels = list(class_counts.keys())
        self.scales = scales or ["1x", "3x", "5x"]
        self.loss_type = loss_type

        # =================================================================
        # ENCODER SETUP (same as MultiScaleCrossAttentionClassifier)
        # =================================================================

        self.roi_scale = "1x"
        self.context_scales = [s for s in self.scales if s != self.roi_scale]

        default_backbone = cfg.model.backbone
        roi_backbone_name = roi_backbone or default_backbone
        context_backbone_name = context_backbone or default_backbone

        print(f"Creating ROI encoder: {roi_backbone_name}")
        self.roi_encoder = PatchFeatureExtractor(roi_backbone_name, pretrained=True)
        roi_feature_dim = self.roi_encoder.feature_dim

        print(f"Creating {len(self.context_scales)} context encoders: {context_backbone_name}")
        self.context_encoders = nn.ModuleDict()
        for scale in self.context_scales:
            self.context_encoders[scale] = PatchFeatureExtractor(
                context_backbone_name, pretrained=True
            )
        context_feature_dim = self.context_encoders[self.context_scales[0]].feature_dim

        # =================================================================
        # ASYMMETRIC BACKBONE HANDLING
        # =================================================================

        self.asymmetric = roi_feature_dim != context_feature_dim
        attention_dim = roi_feature_dim

        if self.asymmetric:
            print(f"Asymmetric mode: ROI dim={roi_feature_dim}, Context dim={context_feature_dim}")
            print(f"Projecting context features to {attention_dim}")
            self.context_proj = nn.Linear(context_feature_dim, attention_dim)
        else:
            self.context_proj = nn.Identity()

        # =================================================================
        # CROSS-ATTENTION MODULE
        # =================================================================

        print(f"Creating cross-attention with {num_attention_layers} layers, {num_attention_heads} heads")
        self.cross_attention = MultiLayerCrossAttention(
            embed_dim=attention_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=0.1,
        )

        # =================================================================
        # FEATURE PROJECTION AND CLASSIFICATION HEADS
        # =================================================================

        feature_dim = attention_dim
        hidden_dim = cfg.model.hidden_dim
        head_dim = cfg.model.head_dim

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self._build_heads(hidden_dim, head_dim)

        # =================================================================
        # TAXONOMIC LOSS SETUP (from MultiScaleTaxonomicClassifier)
        # =================================================================

        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)
        self._setup_losses(distance_matrix_path, loss_type, alpha, smoothing)

        self.val_outputs = []

    def _setup_losses(
        self,
        distance_matrix_path: Optional[str],
        loss_type: str,
        alpha: float,
        smoothing: float,
    ):
        """Setup taxonomic-aware losses (copied from model_multiscale_taxloss.py)."""
        # Standard cross-entropy for non-species levels
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=self.cfg.training.label_smoothing
        )

        # Setup taxonomic-aware loss for species level
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            species_names = [
                self.id_to_name["species"][i]
                for i in range(self.class_counts["species"])
            ]

            if loss_type in ["distance", "both"]:
                self.tax_distance_loss = TaxonomicDistanceLoss(
                    distance_matrix_path,
                    species_names,
                    alpha=alpha,
                )
            else:
                self.tax_distance_loss = None

            if loss_type in ["smooth", "both"]:
                self.tax_smooth_loss = TaxonomicLabelSmoothing(
                    distance_matrix_path,
                    species_names,
                    smoothing=smoothing,
                )
            else:
                self.tax_smooth_loss = None

            # Load distance matrix for validation metric
            dm = pd.read_csv(distance_matrix_path, index_col=0)
            n_species = self.class_counts["species"]
            dist_tensor = torch.zeros(n_species, n_species)

            for i, name_i in enumerate(species_names):
                for j, name_j in enumerate(species_names):
                    if name_i in dm.index and name_j in dm.columns:
                        dist_tensor[i, j] = dm.loc[name_i, name_j]
                    elif i == j:
                        dist_tensor[i, j] = 0.0
                    else:
                        dist_tensor[i, j] = dm.values.max()

            self.register_buffer("distance_matrix", dist_tensor)
            self.use_taxonomic_loss = loss_type != "ce"

        else:
            self.tax_distance_loss = None
            self.tax_smooth_loss = None
            self.use_taxonomic_loss = False
            self.register_buffer(
                "distance_matrix",
                torch.zeros(self.class_counts["species"], self.class_counts["species"])
            )

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """Build hierarchical classification heads (same as base models)."""
        def block(out_dim: int):
            return nn.Linear(head_dim * 2, out_dim)

        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])

        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _fuse_context_patches(
        self,
        context_patches: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combine patch features from multiple context scales."""
        all_patches = []
        for scale in self.context_scales:
            if scale in context_patches:
                patches = context_patches[scale]
                patches = self.context_proj(patches)
                all_patches.append(patches)

        if not all_patches:
            raise ValueError("No context patches available")

        return torch.cat(all_patches, dim=1)

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-attention fusion.

        Same as MultiScaleCrossAttentionClassifier but outputs go through
        taxonomic loss instead of standard CE.
        """
        # =================================================================
        # FEATURE EXTRACTION AND CROSS-ATTENTION
        # =================================================================

        roi_global, roi_patches = self.roi_encoder(images[self.roi_scale])
        query = roi_global.unsqueeze(1)

        context_patches = {}
        for scale in self.context_scales:
            if scale in images:
                _, patches = self.context_encoders[scale](images[scale])
                context_patches[scale] = patches

        all_context = self._fuse_context_patches(context_patches)
        attended = self.cross_attention(query, all_context)
        attended = attended.squeeze(1)

        # =================================================================
        # CLASSIFICATION
        # =================================================================

        shared = self.feature_proj(attended)

        # Hierarchical classification (same as other models)
        kingdom_logits = self.kingdom_head(shared)
        kingdom_probs = torch.softmax(kingdom_logits, dim=1)

        phylum_input = self.kingdom_to_phylum(kingdom_probs)
        phylum_feats = self.phylum_features(shared)
        phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
        phylum_probs = torch.softmax(phylum_logits, dim=1)

        class_input = self.phylum_to_class(phylum_probs)
        class_feats = self.class_features(shared)
        class_logits = self.class_head(torch.cat([class_input, class_feats], dim=1))
        class_probs = torch.softmax(class_logits, dim=1)

        order_input = self.class_to_order(class_probs)
        order_feats = self.order_features(shared)
        order_logits = self.order_head(torch.cat([order_input, order_feats], dim=1))
        order_probs = torch.softmax(order_logits, dim=1)

        family_input = self.order_to_family(order_probs)
        family_feats = self.family_features(shared)
        family_logits = self.family_head(torch.cat([family_input, family_feats], dim=1))
        family_probs = torch.softmax(family_logits, dim=1)

        genus_input = self.family_to_genus(family_probs)
        genus_feats = self.genus_features(shared)
        genus_logits = self.genus_head(torch.cat([genus_input, genus_feats], dim=1))
        genus_probs = torch.softmax(genus_logits, dim=1)

        species_input = self.genus_to_species(genus_probs)
        species_feats = self.species_features(shared)
        species_logits = self.species_head(torch.cat([species_input, species_feats], dim=1))

        return {
            "kingdom": kingdom_logits,
            "phylum": phylum_logits,
            "class": class_logits,
            "order": order_logits,
            "family": family_logits,
            "genus": genus_logits,
            "species": species_logits,
        }

    def hierarchical_loss(self, outputs, targets):
        """
        Compute weighted loss with taxonomic awareness for species.

        Same as model_multiscale_taxloss.py.
        """
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0

        for level in self.levels:
            if level not in outputs or level not in targets:
                continue

            logits = outputs[level]
            target = targets[level]
            w = self.hierarchy_weights.get(level, 1.0)

            # Species level gets taxonomic-aware loss
            if level == "species" and self.use_taxonomic_loss:
                if self.tax_distance_loss and self.tax_smooth_loss:
                    loss = 0.5 * self.tax_distance_loss(logits, target) + \
                           0.5 * self.tax_smooth_loss(logits, target)
                elif self.tax_distance_loss:
                    loss = self.tax_distance_loss(logits, target)
                elif self.tax_smooth_loss:
                    loss = self.tax_smooth_loss(logits, target)
                else:
                    loss = self.ce_loss(logits, target)
            else:
                loss = self.ce_loss(logits, target)

            total_loss += w * loss
            weight_sum += w
            level_losses[level] = loss

        total_loss = total_loss / max(weight_sum, 1e-6)
        return total_loss, level_losses

    def _compute_taxonomic_score(self, species_logits, species_targets):
        """Compute expected taxonomic distance (competition metric)."""
        preds = species_logits.argmax(dim=1)
        distances = self.distance_matrix[preds, species_targets]
        return distances.float().mean()

    def training_step(self, batch, batch_idx):
        """Training step with taxonomic score tracking."""
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)

        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)

        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("train_tax_score", tax_score, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step with taxonomic score tracking."""
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)

            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()
            self.log(f"val_{lvl}_acc", acc, prog_bar=(lvl == "species"), sync_dist=True)

        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("val_tax_score", tax_score, prog_bar=True, sync_dist=True)

        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Clear stored validation outputs."""
        self.val_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizers with parameter-specific learning rates.

        Three groups:
        1. Encoders: base_lr × backbone_lr_scale (slow)
        2. Cross-attention: base_lr × 0.5 (moderate)
        3. Heads: base_lr (fast)
        """
        encoder_params = []
        attention_params = []
        head_params = []

        for name, param in self.named_parameters():
            if "roi_encoder" in name or "context_encoders" in name:
                encoder_params.append(param)
            elif "cross_attention" in name:
                attention_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": self.cfg.training.learning_rate * self.cfg.training.backbone_lr_scale,
                },
                {
                    "params": attention_params,
                    "lr": self.cfg.training.learning_rate * 0.5,
                },
                {
                    "params": head_params,
                    "lr": self.cfg.training.learning_rate,
                },
            ],
            weight_decay=self.cfg.training.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
