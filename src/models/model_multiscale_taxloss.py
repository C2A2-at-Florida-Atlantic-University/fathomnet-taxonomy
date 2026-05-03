"""
Multi-scale Model with Taxonomic Distance-Aware Loss for FathomNet 2025.

================================================================================
MOTIVATION AND COMPETITION METRIC
================================================================================

The FathomNet 2025 competition uses a hierarchical scoring metric that
penalizes misclassifications based on taxonomic distance:

    Score = (1/N) × Σ_i d(predicted_species_i, true_species_i)

Where d(s1, s2) is the taxonomic distance between species, defined as the
minimum number of edges in the taxonomy tree to traverse from s1 to s2.

Example distances:
    - Same species:        d = 0
    - Same genus:          d = 2  (up to genus, down to other species)
    - Same family:         d = 4  (up 2 levels, down 2 levels)
    - Same order:          d = 6  (up 3, down 3)
    - Different phylum:    d = 12 (up 6, down 6)

Key Insight:
------------
Standard cross-entropy loss treats all misclassifications equally. But for
this competition, predicting a closely related species (e.g., same genus)
is MUCH better than predicting a distant species (e.g., different phylum).

This model incorporates the taxonomic distance directly into the loss
function, encouraging the model to make "smarter" mistakes when uncertain.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This model extends MultiScaleTaxonomyClassifier with:

1. TAXONOMIC DISTANCE LOSS
   - Penalizes predictions proportional to their taxonomic distance from truth
   - Mathematical formulation: L = (1-α)×CE + α×E[D|p]

2. TAXONOMIC LABEL SMOOTHING
   - Smooths labels based on taxonomic similarity
   - Related species get higher soft targets than distant ones

3. VALIDATION METRIC TRACKING
   - Logs expected taxonomic score during validation
   - Allows direct monitoring of competition metric

Architecture Diagram:
---------------------
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Images (Multiple Scales)                    │
│                   1x (ROI), 3x, 5x [, full]                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Backbone   │     │   Backbone   │     │   Backbone   │
│  (1x Scale)  │     │  (3x Scale)  │     │  (5x Scale)  │
│  ConvNeXtV2  │     │  ConvNeXtV2  │     │  ConvNeXtV2  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Feature Concatenation  │
              │   [f_1x; f_3x; f_5x]     │
              │   (B, 3×1024) = (B, 3072)│
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │    Fusion Network       │
              │    LayerNorm → Linear   │
              │    → GELU → Dropout     │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Hierarchical Classifier │
              │  Kingdom → ... → Species │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   Taxonomic-Aware Loss  │
              │   Species: Tax Loss     │
              │   Others: Standard CE   │
              └─────────────────────────┘

================================================================================
LOSS FUNCTION DETAILS
================================================================================

This model supports 4 loss configurations for the species level:

1. "ce" - Standard Cross-Entropy
   L = -log(p_y)

2. "distance" - Taxonomic Distance Loss
   L = (1-α) × CE + α × E_p[D(·, y)]
     = (1-α) × (-log p_y) + α × Σ_i p_i × D(i, y)

   The second term penalizes probability mass on distant species.
   α controls the trade-off (default: 0.3).

3. "smooth" - Taxonomic Label Smoothing
   Instead of hard labels [0,0,1,0,0], use soft labels based on similarity:
   q_i = (1-ε)×I(i=y) + ε × sim(i, y) / Z

   Where sim(i, y) = exp(-β × D(i, y)) for some temperature β.
   Related species get higher soft targets.

4. "both" - Combined Distance + Smoothing
   L = 0.5 × TaxDistanceLoss + 0.5 × TaxLabelSmoothing

================================================================================
EXPECTED TAXONOMIC SCORE COMPUTATION
================================================================================

During validation, we compute an approximation of the competition metric:

    val_tax_score = (1/N) × Σ_i D(argmax(p_i), y_i)

This uses argmax predictions (not soft predictions) to match the actual
submission format. Lower is better.

Key benefit: We can monitor the actual competition metric during training,
rather than relying on cross-entropy loss as a proxy.

================================================================================
DISTANCE MATRIX
================================================================================

The taxonomic distance matrix is precomputed and stored in a CSV file:
    taxonomic_distance_matrix.csv

Format:
    - Rows and columns are species names
    - Cell (i, j) contains D(species_i, species_j)
    - Symmetric matrix with zeros on diagonal

The matrix is loaded and converted to a PyTorch tensor registered as a
buffer (moves to GPU automatically, not treated as a parameter).

================================================================================
EXPERIMENTAL RESULTS
================================================================================

Training Configuration:
    - Scales: 1x, 3x, 5x
    - Backbone: ConvNeXtV2-Base
    - Loss type: "distance" (α=0.3)
    - Batch size: 16, 50 epochs

Results:
    - val_tax_score converged to ~1.7-1.8 (vs ~2.0+ for standard CE)
    - Private leaderboard: 2.00 (our best submission)
    - Improvement from taxonomic awareness: ~0.2-0.3 points

The taxonomic loss consistently outperformed standard cross-entropy,
especially on difficult samples where the model was uncertain.

================================================================================
USAGE
================================================================================

Basic usage:
    model = MultiScaleTaxonomicClassifier(
        cfg=config,
        class_counts={"kingdom": 2, ..., "species": 80},
        id_to_name=id_to_name_mapping,  # Required for loss setup
        scales=["1x", "3x", "5x"],
        distance_matrix_path="taxonomic_distance_matrix.csv",
        loss_type="distance",  # or "smooth", "both", "ce"
        alpha=0.3,             # Weight for distance term
        smoothing=0.1,         # Amount of label smoothing
    )

Training:
    trainer = pl.Trainer(...)
    trainer.fit(model, train_loader, val_loader)

================================================================================
MULTI-SCALE FUSION METHODS
================================================================================

This model supports four fusion strategies for combining multi-scale features:

1. "concat" (Default) - Simple Concatenation
   Features from all scales are concatenated into a single vector.
   This is parameter-efficient but treats all levels identically.

2. "hierarchy_attn" - Hierarchy-Aware Attention Fusion
   Each taxonomy level learns its own attention weights over scales.
   Hypothesis: Species may prefer fine details (1x), Kingdom may use context (3x, 5x).

   Based on:
   - Ding et al., "CHRF: Cross-Hierarchical Region Feature Learning", ECCV 2022
   - Park et al., "MAHL: Multi-Scale Attention-Driven Hierarchical Learning", 2025
   - Zhang et al., "Hierarchical Attention Vision Transformer", Expert Systems, 2023

3. "gated_cross" - Gated Cross-Scale Fusion
   Scales attend to each other via self-attention before per-level gating.
   More expressive but more parameters.

   Based on:
   - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
   - Dynamic Gated Fusion Network concepts for multi-source feature combination

4. "patch_self_attn" - Patch Self-Attention Fusion
   Self-attention within ROI patches to learn part relationships.
   ROI spatial features (7×7=49 patches) attend to each other, enabling
   the model to learn spatial relationships between discriminative regions.

   Architecture:
   - ROI (1x): Spatial features → 2-layer self-attention → attended features
   - Context (3x, 5x): Global pooled features
   - Fusion: Concatenate and project

   Key insight: Fine-grained classification benefits from learning part
   relationships (e.g., fins + spots + body shape for marine species).

   Based on:
   - Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021
   - He et al., "TransFG: A Transformer for Fine-grained Recognition", AAAI 2022

5. "part_attn" - Part-Based Attention with Learnable Prototypes
   Learns K prototype vectors that discover discriminative parts in an
   unsupervised manner. Each prototype attends to different spatial regions
   to extract part-specific features (e.g., fins, head, patterns).

   Architecture:
   - Learn K=8 prototype vectors (each learns to detect a specific part)
   - Compute attention: each prototype attends to spatial positions
   - Extract part features: weighted sum of features for each prototype
   - Aggregate: concatenate and project part features

   Key insight: Fine-grained species classification relies on subtle
   part differences. Learning prototypes allows automatic part discovery.

   Based on:
   - Zheng et al., "Learning Multi-Attention CNN for Fine-Grained
     Image Recognition", ICCV 2017
   - Sun et al., "Multi-Attention Multi-Class Constraint", ECCV 2018
   - Fu et al., "Look Closer to See Better", CVPR 2017

See hierarchy_fusion.py and attention_module.py for implementations.

================================================================================
REFERENCES
================================================================================

Loss Functions:
    - Deng et al., "Hierarchical Semantic Indexing for Large Scale Image Retrieval"
    - Redmon and Farhadi, "YOLO9000: Better, Faster, Stronger" (WordTree)
    - Wu et al., "Making Better Mistakes: Leveraging Class Hierarchies"
    - Bertinetto et al., "Making Better Mistakes" (HXE loss), CVPR 2020

Multi-Scale Fusion:
    - Ding et al., "CHRF: Cross-Hierarchical Region Feature Learning", ECCV 2022
    - Park et al., "MAHL: Multi-Scale Attention-Driven Hierarchical Learning", 2025
    - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
"""

import os
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..config import Config
from ..losses import TaxonomicDistanceLoss, TaxonomicLabelSmoothing, HierarchicalCrossEntropyLoss
from .hierarchy_fusion import HierarchyAwareFusion, GatedCrossScaleFusion
from .attention_module import PatchSelfAttention, PartBasedAttention
from .dinov2_backbone import DINOv2Backbone, create_dinov2_backbone, DINOV2_MODELS


def create_backbone(
    backbone_name: str,
    pretrained: bool = True,
    features_only: bool = False,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """
    Create a backbone model, supporting both timm models and DINOv2.

    Args:
        backbone_name: Model name (e.g., "convnextv2_base.fcmae_ft_in22k_in1k" or "dinov2_base")
        pretrained: Whether to load pretrained weights
        features_only: If True, return intermediate stage features (for B-CNN branching).
            Not supported with DINOv2 backbones.
        drop_path_rate: Stochastic depth rate (for DINOv2 ViT fine-tuning)

    Returns:
        Backbone model with num_features attribute and forward_features() method.
        If features_only=True, forward() returns a list of per-stage spatial feature maps.
    """
    name_lower = backbone_name.lower()

    if "dinov2" in name_lower:
        if features_only:
            raise ValueError("B-CNN branching (features_only) not supported with DINOv2 backbones")
        # Use our DINOv2 wrapper
        return create_dinov2_backbone(
            backbone_name, pretrained=pretrained, drop_path_rate=drop_path_rate
        )
    else:
        # Use timm for other models (ConvNeXtV2, etc.)
        kwargs = {}
        if drop_path_rate > 0:
            kwargs["drop_path_rate"] = drop_path_rate
        return timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, keep feature extractor
            features_only=features_only,
            **kwargs,
        )


class MultiScaleTaxonomicClassifier(pl.LightningModule):
    """
    Multi-scale classifier with taxonomic distance-aware training.

    This model combines multi-scale feature extraction with loss functions
    that incorporate taxonomic knowledge. The key innovation is that the
    species-level loss penalizes predictions proportional to their taxonomic
    distance from the ground truth.

    Key Features:
    -------------
    1. Multi-scale backbones: Separate ConvNeXtV2 encoders for each scale
    2. Feature fusion: Concatenation followed by learned projection
    3. Hierarchical classification: Parent-conditioned predictions
    4. Taxonomic loss: Distance-aware penalty for species misclassification
    5. Metric tracking: Logs expected taxonomic score during validation

    Loss Types:
    -----------
    - "ce": Standard cross-entropy (baseline)
    - "distance": CE + expected distance penalty
    - "smooth": Taxonomic label smoothing
    - "both": Combination of distance and smoothing

    Args:
        cfg: OmegaConf configuration with model/training settings
        class_counts: Dict mapping level names to number of classes
        id_to_name: Dict mapping level names to {class_id: class_name} dicts
            Required for setting up the taxonomic distance matrix
        scales: List of scale names (default: ["1x", "3x", "5x"])
        distance_matrix_path: Path to CSV with taxonomic distances
        loss_type: Which loss to use ("distance", "smooth", "both", "ce")
        alpha: Weight for distance term in TaxonomicDistanceLoss (default: 0.3)
        smoothing: Amount of label smoothing for TaxonomicLabelSmoothing

    Input Format:
        Dictionary with scale keys mapping to image tensors:
        {
            "1x": torch.Tensor of shape (B, 3, 224, 224),
            "3x": torch.Tensor of shape (B, 3, 224, 224),
            "5x": torch.Tensor of shape (B, 3, 224, 224),
        }

    Output Format:
        Dictionary with logits for each taxonomy level:
        {
            "kingdom": torch.Tensor of shape (B, 2),
            "phylum": torch.Tensor of shape (B, 8),
            ...
            "species": torch.Tensor of shape (B, 80),
        }

    Example:
        >>> from src.data import load_and_encode_taxonomy
        >>> taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(...)
        >>> model = MultiScaleTaxonomicClassifier(
        ...     cfg=config,
        ...     class_counts=class_counts,
        ...     id_to_name=id_to_name,
        ...     distance_matrix_path="taxonomic_distance_matrix.csv",
        ...     loss_type="distance",
        ... )
    """

    def __init__(
        self,
        cfg: DictConfig,
        class_counts: Dict[str, int],
        id_to_name: Dict[str, Dict[int, str]],
        scales: List[str] = None,
        distance_matrix_path: Optional[str] = None,
        loss_type: str = "distance",  # "distance", "smooth", "both", "ce", "hxe"
        alpha: float = Config.ALPHA,
        smoothing: float = 0.1,
        scale_backbones: Optional[Dict[str, str]] = None,  # Per-scale backbone mapping
        use_uncertainty_weighting: bool = False,  # Kendall et al. learnable weights
        taxonomy_df: Optional[pd.DataFrame] = None,  # Required for HXE loss
        fusion_type: str = "concat",  # "concat", "hierarchy_attn", "gated_cross"
        species_class_weights: Optional[torch.Tensor] = None,  # Inverse-freq weights for species CE
        consistency_weight: float = 0.0,  # C-HMCNN hierarchy consistency loss weight
        share_backbone: bool = False,  # Share one backbone across all scales (for Jetson deployment)
        branch_cnn: bool = False,  # B-CNN: branch coarse levels from early backbone stages
        bt_strategy: bool = False,  # BT-Strategy: dynamic coarse→fine weight scheduling
        use_gnn: bool = False,  # GNN for multi-specimen relationship modeling
        parent_conditioning: bool = True,  # Cascading heads; False = independent heads
        leaf_only: bool = False,  # Species-only head with marginalization to coarse levels
        stop_gradient: bool = False,  # Stop gradients through parent→child conditioning
    ):
        super().__init__()
        # Save hyperparameters for checkpoint loading
        # Exclude cfg (not serializable) and id_to_name (large dict)
        self.save_hyperparameters(ignore=["cfg", "id_to_name", "species_class_weights"])

        # Store configuration
        self.cfg = cfg
        self.class_counts = class_counts
        self.id_to_name = id_to_name  # Needed to map class IDs to species names
        self.levels = list(class_counts.keys())  # ["kingdom", "phylum", ..., "species"]
        self.scales = scales or ["1x", "3x", "5x"]
        self.loss_type = loss_type
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.fusion_type = fusion_type
        self.consistency_weight = consistency_weight
        self.share_backbone = share_backbone
        self.branch_cnn = branch_cnn
        self.bt_strategy = bt_strategy
        self.use_gnn = use_gnn
        self.leaf_only = leaf_only
        self.parent_conditioning = parent_conditioning if not leaf_only else False
        self.stop_gradient = stop_gradient

        # =================================================================
        # MULTI-SCALE BACKBONE SETUP
        # =================================================================

        # Default backbone from config
        default_backbone = cfg.model.backbone  # e.g., "convnextv2_base.fcmae_ft_in22k_in1k"

        # Per-scale backbone mapping (allows different backbones per scale)
        # Example: {"1x": "convnextv2_large...", "3x": "convnextv2_base...", ...}
        self.scale_backbones = scale_backbones or {}

        if share_backbone:
            # SHARED BACKBONE: One backbone processes all scales sequentially.
            # Reduces params from N×backbone to 1×backbone, enabling Jetson
            # deployment with larger backbones. The same weights extract features
            # from ROI, context, and full-image crops.
            backbone_name = next(iter(self.scale_backbones.values()), default_backbone)
            drop_path = getattr(cfg.training, 'drop_path_rate', 0.0)
            shared_bb = create_backbone(backbone_name, pretrained=True, features_only=branch_cnn, drop_path_rate=drop_path)
            if branch_cnn:
                print(f"  SHARED backbone (B-CNN features_only) for all scales: {backbone_name}")
            else:
                print(f"  SHARED backbone for all scales: {backbone_name} (dim={shared_bb.num_features})")
            # Assign same module to all scale keys - PyTorch de-duplicates params
            self.backbones = nn.ModuleDict({scale: shared_bb for scale in self.scales})
        else:
            # SEPARATE BACKBONES: Each scale has its own backbone instance.
            # Each backbone is independently pretrained and will be fine-tuned.
            self.backbones = nn.ModuleDict()
            for scale in self.scales:
                backbone_name = self.scale_backbones.get(scale, default_backbone)
                drop_path = getattr(cfg.training, 'drop_path_rate', 0.0)
                self.backbones[scale] = create_backbone(backbone_name, pretrained=True, features_only=branch_cnn, drop_path_rate=drop_path)
                if branch_cnn:
                    print(f"  Scale {scale}: {backbone_name} (B-CNN features_only)")
                else:
                    print(f"  Scale {scale}: {backbone_name} (dim={self.backbones[scale].num_features})")

        # =================================================================
        # FEATURE FUSION NETWORK
        # =================================================================

        hidden_dim = cfg.model.hidden_dim  # e.g., 2048
        head_dim = cfg.model.head_dim      # e.g., 512

        if branch_cnn:
            # =============================================================
            # B-CNN MULTI-STAGE BRANCHING (Zhu & Bain 2017)
            # =============================================================
            # Instead of routing all taxonomy levels through the final stage,
            # branch coarser levels from earlier backbone stages:
            #   Stage 0 (192-dim) → kingdom, phylum
            #   Stage 1 (384-dim) → class, order
            #   Stage 2 (768-dim) → family
            #   Stage 3 (1536-dim) → genus, species
            #
            # Each stage's features are GAP-pooled, concatenated across scales,
            # and projected to hidden_dim before feeding to the classification heads.

            first_bb = next(iter(self.backbones.values()))
            self.stage_dims = first_bb.feature_info.channels()  # e.g., [192, 384, 768, 1536]
            self.n_stages = len(self.stage_dims)

            # Stage → taxonomy level mapping
            self.stage_level_map = {
                0: ["kingdom", "phylum"],
                1: ["class", "order"],
                2: ["family"],
                3: ["genus", "species"],
            }

            print(f"  B-CNN stage dims: {self.stage_dims}")
            for stage_idx, levels in self.stage_level_map.items():
                print(f"    Stage {stage_idx} ({self.stage_dims[stage_idx]}-dim) → {levels}")

            # Per-stage projection: concat multi-scale features → hidden_dim
            self.stage_projections = nn.ModuleDict()
            n_scales = len(self.scales)
            for stage_idx in range(self.n_stages):
                stage_in = self.stage_dims[stage_idx] * n_scales
                self.stage_projections[f"stage_{stage_idx}"] = nn.Sequential(
                    nn.LayerNorm(stage_in),
                    nn.Linear(stage_in, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                )

            # Not used with B-CNN branching
            self.hierarchy_fusion = None
            self.level_projections = None
            self.roi_self_attention = None
            self.part_attention = None
            self.shared_features = None
            self.scale_dims = None  # Not needed — stage_dims used instead

        else:
            # Standard fusion (non-branching) — existing code path
            # Get feature dimensions for each scale
            # e.g., ConvNeXtV2-L=1536, ConvNeXtV2-B=1024
            self.scale_dims = [self.backbones[scale].num_features for scale in self.scales]
            in_features = sum(self.scale_dims)
            self.stage_projections = None  # Not used without B-CNN

        # Choose fusion method based on fusion_type (skipped for B-CNN which uses stage_projections)
        if self.branch_cnn:
            pass  # B-CNN uses per-stage projections instead of fusion network
        elif fusion_type == "hierarchy_attn":
            # =============================================================
            # HIERARCHY-AWARE ATTENTION FUSION (Option 2)
            # =============================================================
            # Each taxonomy level learns its own attention weights over scales.
            # This allows species to focus on fine details (1x ROI) while
            # kingdom/phylum can use more context (3x, 5x).
            #
            # Benefits:
            # - Reduces head input from 4608-D to 1024-D (4.5× smaller)
            # - Per-level scale emphasis is learned
            # - Interpretable attention weights for analysis

            fusion_common_dim = 1024  # Common dimension for all scales
            self.hierarchy_fusion = HierarchyAwareFusion(
                scale_dims=self.scale_dims,
                common_dim=fusion_common_dim,
                num_levels=len(self.levels),
                level_names=self.levels,
                attention_hidden_dim=256,
                dropout=0.1,
            )
            # With hierarchy fusion, each level gets its own 1024-D representation
            # We still need a projection to hidden_dim for the classification heads
            self.level_projections = nn.ModuleDict({
                level: nn.Sequential(
                    nn.LayerNorm(fusion_common_dim),
                    nn.Linear(fusion_common_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                )
                for level in self.levels
            })
            self.shared_features = None  # Not used with hierarchy fusion
            self.roi_self_attention = None
            self.part_attention = None

        elif fusion_type == "gated_cross":
            # =============================================================
            # GATED CROSS-SCALE FUSION (Option 3)
            # =============================================================
            # Scales attend to each other via self-attention before fusion.
            # More expressive but also more parameters.

            fusion_common_dim = 1024
            self.hierarchy_fusion = GatedCrossScaleFusion(
                scale_dims=self.scale_dims,
                common_dim=fusion_common_dim,
                num_levels=len(self.levels),
                num_heads=8,
                dropout=0.1,
            )
            self.level_projections = nn.ModuleDict({
                level: nn.Sequential(
                    nn.LayerNorm(fusion_common_dim),
                    nn.Linear(fusion_common_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                )
                for level in self.levels
            })
            self.shared_features = None
            self.roi_self_attention = None
            self.part_attention = None

        elif fusion_type == "patch_self_attn":
            # =============================================================
            # PATCH SELF-ATTENTION FUSION
            # =============================================================
            # Apply self-attention within ROI patches to learn part relationships.
            # Patches within the ROI attend to each other, enabling the model
            # to learn spatial relationships between discriminative regions.
            #
            # Architecture:
            # 1. ROI: Extract spatial features, apply self-attention, pool
            # 2. Context: Use global features as usual
            # 3. Concatenate ROI attended + context features
            #
            # Key insight: Fine-grained classification benefits from learning
            # part relationships (e.g., fins + spots + body shape).
            #
            # References:
            # - Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT)
            # - He et al., "TransFG: A Transformer for Fine-grained Recognition"

            self.hierarchy_fusion = None
            self.level_projections = None

            # ROI self-attention module
            roi_dim = self.scale_dims[0]  # Assume 1x is first scale
            self.roi_self_attention = PatchSelfAttention(
                input_dim=roi_dim,
                hidden_dim=512,
                output_dim=roi_dim,  # Keep same dim for compatibility
                num_heads=8,
                num_layers=2,
                dropout=0.1,
                max_patches=49,  # 7x7 for 224px input
            )

            # Context features remain global pooled
            # Total features = ROI attended (roi_dim) + sum of context dims
            context_dims = self.scale_dims[1:]  # All scales except 1x
            total_features = roi_dim + sum(context_dims)

            self.shared_features = nn.Sequential(
                nn.LayerNorm(total_features),
                nn.Linear(total_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            self.part_attention = None

        elif fusion_type == "patch_part_attn":
            # =============================================================
            # COMBINED: PATCH SELF-ATTENTION + PART-BASED ATTENTION
            # =============================================================
            # First apply self-attention within patches to learn relationships,
            # then apply part prototypes to discover discriminative regions.
            #
            # Architecture:
            # 1. ROI: Spatial features → Patch self-attention → Part attention
            # 2. Context: Global pooled features
            # 3. Concatenate enhanced ROI + context features
            #
            # Key insight: Combines two complementary mechanisms:
            # - Patch self-attention: learns spatial relationships between regions
            # - Part attention: discovers and focuses on discriminative parts

            self.hierarchy_fusion = None
            self.level_projections = None

            roi_dim = self.scale_dims[0]  # Assume 1x is first scale

            # First stage: Patch self-attention
            self.roi_self_attention = PatchSelfAttention(
                input_dim=roi_dim,
                hidden_dim=512,
                output_dim=roi_dim,
                num_heads=8,
                num_layers=2,
                dropout=0.1,
                max_patches=49,
            )

            # Second stage: Part-based attention on self-attended features
            # Note: Part attention expects spatial input, but self-attn outputs (B, D)
            # So we apply part attention BEFORE self-attention pools to (B, D)
            # Actually, let's apply them in parallel and combine
            self.part_attention = PartBasedAttention(
                input_dim=roi_dim,
                num_parts=8,
                hidden_dim=256,
                output_dim=roi_dim,
                dropout=0.1,
                diversity_loss=True,
            )

            # Context features remain global pooled
            context_dims = self.scale_dims[1:]
            # ROI contributes 2x (self-attn + part-attn outputs)
            total_features = roi_dim * 2 + sum(context_dims)

            self.shared_features = nn.Sequential(
                nn.LayerNorm(total_features),
                nn.Linear(total_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
            )

        elif fusion_type == "part_attn":
            # =============================================================
            # PART-BASED ATTENTION WITH LEARNABLE PROTOTYPES
            # =============================================================
            # Learns K prototype vectors that discover discriminative parts
            # (e.g., fins, head, patterns) in an unsupervised manner.
            # Each prototype attends to different spatial regions.
            #
            # Architecture:
            # 1. ROI: Spatial features → Part-based attention with K prototypes
            # 2. Context: Global pooled features
            # 3. Concatenate part-aware ROI + context features
            #
            # Key insight: Fine-grained classification relies on subtle part
            # differences. Learning part prototypes allows the model to focus
            # on discriminative regions automatically.
            #
            # References:
            # - Zheng et al., "Learning Multi-Attention CNN for Fine-Grained
            #   Image Recognition", ICCV 2017
            # - Sun et al., "Multi-Attention Multi-Class Constraint", ECCV 2018
            # - Fu et al., "Look Closer to See Better", CVPR 2017

            self.hierarchy_fusion = None
            self.level_projections = None
            self.roi_self_attention = None

            # Part-based attention module for ROI
            roi_dim = self.scale_dims[0]  # Assume 1x is first scale
            num_parts = 8  # Number of learnable part prototypes
            part_hidden_dim = 256  # Hidden dimension for part features

            self.part_attention = PartBasedAttention(
                input_dim=roi_dim,
                num_parts=num_parts,
                hidden_dim=part_hidden_dim,
                output_dim=roi_dim,  # Keep same dim for compatibility
                dropout=0.1,
                diversity_loss=True,  # Encourage diverse prototypes
            )

            # Context features remain global pooled
            context_dims = self.scale_dims[1:]  # All scales except 1x
            total_features = roi_dim + sum(context_dims)

            self.shared_features = nn.Sequential(
                nn.LayerNorm(total_features),
                nn.Linear(total_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
            )

        else:
            # =============================================================
            # CONCATENATION FUSION (Original approach)
            # =============================================================
            # Simple concatenation of all scale features.
            # All taxonomy levels receive the same fused representation.

            self.hierarchy_fusion = None
            self.level_projections = None
            self.roi_self_attention = None
            self.part_attention = None
            # Fusion layer: project concatenated features to classification space
            # LayerNorm helps with different feature magnitudes across scales
            self.shared_features = nn.Sequential(
                nn.LayerNorm(in_features),          # Normalize concatenated features
                nn.Linear(in_features, hidden_dim), # Project to hidden dim
                nn.GELU(),                          # Non-linearity
                nn.Dropout(0.3),                    # Regularization
            )

        # =================================================================
        # OPTIONAL GNN FOR MULTI-SPECIMEN RELATIONSHIPS
        # =================================================================
        self.specimen_gnn = None
        if self.use_gnn:
            from .gnn_module import SpecimenGNN
            self.specimen_gnn = SpecimenGNN(feature_dim=hidden_dim)
            print(f"  GNN enabled: SpecimenGNN (dim={hidden_dim})")

        # =================================================================
        # HIERARCHICAL CLASSIFICATION HEADS
        # =================================================================

        self._build_heads(hidden_dim, head_dim)

        # Build marginalization matrices for leaf-only mode
        if self.leaf_only and taxonomy_df is not None:
            self._build_marginalization_matrices(taxonomy_df)

        # =================================================================
        # LOSS FUNCTION SETUP
        # =================================================================

        # Hierarchy weights for combining losses across taxonomy levels
        # Used when use_uncertainty_weighting=False
        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)

        # Learnable Uncertainty Weighting (Kendall et al., 2018)
        # "Multi-Task Learning Using Uncertainty to Weigh Losses"
        # We learn log(σ²) for numerical stability: L = (1/2σ²)L_task + log(σ)
        # Parameterized as: L = 0.5 * exp(-s) * L_task + 0.5 * s, where s = log(σ²)
        if self.use_uncertainty_weighting:
            # Initialize log-variances (s = log(σ²)) for each taxonomy level
            # Starting at 0 means σ² = 1, giving equal initial weights
            self.log_vars = nn.ParameterDict({
                level: nn.Parameter(torch.zeros(1))
                for level in self.levels
            })
            # FULLY INTEGRATED: Separate log-vars for CE and taxonomic distance
            # at species level. This allows the model to learn whether CE or
            # taxonomic distance is more useful, replacing fixed alpha.
            # L_species = exp(-s_ce)*CE + exp(-s_tax)*TaxDist + 0.5*(s_ce + s_tax)
            self.log_var_species_ce = nn.Parameter(torch.zeros(1))
            self.log_var_species_tax = nn.Parameter(torch.zeros(1))

        # Setup taxonomic-aware losses
        self._setup_losses(distance_matrix_path, loss_type, alpha, smoothing, taxonomy_df,
                           species_class_weights=species_class_weights)

        # Storage for validation outputs
        self.val_outputs = []

    def _setup_losses(
        self,
        distance_matrix_path: Optional[str],
        loss_type: str,
        alpha: float,
        smoothing: float,
        taxonomy_df: Optional[pd.DataFrame] = None,
        species_class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Setup loss functions based on configuration.

        For non-species levels, we use standard cross-entropy with label
        smoothing. For species level, we optionally use taxonomic-aware losses.

        The distance matrix is loaded from CSV and converted to a tensor
        that's registered as a buffer (for automatic GPU transfer).

        Args:
            distance_matrix_path: Path to taxonomic distance CSV
            loss_type: Which loss to use for species level
            alpha: Weight for distance term in TaxonomicDistanceLoss
            smoothing: Smoothing amount for TaxonomicLabelSmoothing
            taxonomy_df: Taxonomy DataFrame (required for HXE loss)
            species_class_weights: Optional inverse-frequency weights for species CE
        """
        # Standard cross-entropy for non-species levels
        # Uses label smoothing for calibration (from config)
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=self.cfg.training.label_smoothing
        )

        # Setup taxonomic-aware loss for species level
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            # Build list of species names in order of class IDs
            # This ensures distance matrix rows/columns align with logit positions
            species_names = [
                self.id_to_name["species"][i]
                for i in range(self.class_counts["species"])
            ]

            # Create TaxonomicDistanceLoss if requested
            if loss_type in ["distance", "both"]:
                self.tax_distance_loss = TaxonomicDistanceLoss(
                    distance_matrix_path,
                    species_names,
                    alpha=alpha,  # Weight for distance term
                    class_weights=species_class_weights,
                )
            else:
                self.tax_distance_loss = None

            # Create TaxonomicLabelSmoothing if requested
            if loss_type in ["smooth", "both"]:
                self.tax_smooth_loss = TaxonomicLabelSmoothing(
                    distance_matrix_path,
                    species_names,
                    smoothing=smoothing,
                )
            else:
                self.tax_smooth_loss = None

            # Create HierarchicalCrossEntropyLoss (HXE) if requested
            if loss_type == "hxe" and taxonomy_df is not None:
                # HXE uses conditional CE at each hierarchy level
                # with per-level weights from Bertinetto et al. 2020
                hxe_level_weights = {
                    "genus": 1.0,
                    "family": 0.9,
                    "order": 0.8,
                    "class": 0.7,
                    "phylum": 0.6,
                    "kingdom": 0.5,
                }
                self.hxe_loss = HierarchicalCrossEntropyLoss(
                    taxonomy_df=taxonomy_df,
                    class_names=species_names,
                    level_weights=hxe_level_weights,
                    alpha=alpha,  # Balance between CE and HXE terms
                )
            else:
                self.hxe_loss = None

            # =========================================================
            # LOAD DISTANCE MATRIX FOR VALIDATION METRIC
            # =========================================================
            # We need the full distance matrix to compute val_tax_score
            dm = pd.read_csv(distance_matrix_path, index_col=0)
            n_species = self.class_counts["species"]
            dist_tensor = torch.zeros(n_species, n_species)

            # Fill in the distance matrix
            for i, name_i in enumerate(species_names):
                for j, name_j in enumerate(species_names):
                    if name_i in dm.index and name_j in dm.columns:
                        dist_tensor[i, j] = dm.loc[name_i, name_j]
                    elif i == j:
                        # Same species = distance 0
                        dist_tensor[i, j] = 0.0
                    else:
                        # Unknown pair = max distance (conservative)
                        dist_tensor[i, j] = dm.values.max()

            # Register as buffer: moves to GPU with model, not a trainable parameter
            self.register_buffer("distance_matrix", dist_tensor)
            self.use_taxonomic_loss = loss_type not in ["ce"]

        else:
            # No distance matrix available - fall back to standard CE
            self.tax_distance_loss = None
            self.tax_smooth_loss = None
            self.hxe_loss = None
            self.use_taxonomic_loss = False

            # Create dummy distance matrix (all zeros)
            self.register_buffer(
                "distance_matrix",
                torch.zeros(self.class_counts["species"], self.class_counts["species"])
            )

        # C-HMCNN-inspired hierarchy consistency loss
        if taxonomy_df is not None and self.consistency_weight > 0:
            from src.losses import HierarchyConsistencyLoss
            species_names_for_consistency = [
                self.id_to_name["species"][i]
                for i in range(self.class_counts["species"])
            ]
            self.hierarchy_consistency_loss = HierarchyConsistencyLoss(
                taxonomy_df=taxonomy_df,
                class_names=species_names_for_consistency,
            )
        else:
            self.hierarchy_consistency_loss = None

    def _build_marginalization_matrices(self, taxonomy_df: pd.DataFrame):
        """
        Build binary matrices for marginalizing species probabilities to coarse levels.

        For each coarse level L, creates M_L of shape (num_species, num_L_classes)
        where M_L[s, c] = 1 if species s belongs to coarse class c.

        Then P(L=c) = sum_s P(species=s) * M_L[s, c] = species_probs @ M_L.
        """
        import torch
        num_species = self.class_counts["species"]
        for level in self.levels:
            if level == "species":
                continue
            num_classes = self.class_counts[level]
            M = torch.zeros(num_species, num_classes)
            for _, row in taxonomy_df.iterrows():
                sp_id = int(row["species_id"])
                coarse_id = int(row[f"{level}_id"])
                if sp_id < num_species and coarse_id < num_classes:
                    M[sp_id, coarse_id] = 1.0
            # Handle "unknown" species (id may not be in taxonomy_df)
            # If any species has no mapping, assign to unknown coarse class
            unmapped = M.sum(dim=1) == 0
            if unmapped.any():
                # Find "unknown" class id for this level
                unknown_id = None
                for cid, name in self.id_to_name[level].items():
                    if name == "unknown":
                        unknown_id = cid
                        break
                if unknown_id is not None:
                    M[unmapped, unknown_id] = 1.0
            self.register_buffer(f"margin_{level}", M)
            # Pre-compute log-mask for numerically stable logsumexp marginalization
            # log(1)=0 for members, log(0)=-inf for non-members
            log_M = torch.log(M)
            self.register_buffer(f"log_margin_{level}", log_M)
            print(f"  Marginalization matrix {level}: {M.shape}, "
                  f"nonzero={int(M.sum())}")

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """
        Build classification heads.

        When leaf_only=True:
            Only a species head is created. Coarse-level predictions are derived
            by marginalizing species probabilities.

        When parent_conditioning=True (default):
            Each level (except kingdom) receives conditioning from the parent
            level's predicted distribution plus direct features.
            logits_L = Linear(2 × head_dim → L_classes)([parent_cond; features])

        When parent_conditioning=False:
            All levels get independent Linear(hidden_dim → L_classes) heads.

        Args:
            hidden_dim: Dimension of shared features from fusion layer
            head_dim: Dimension for parent conditioning embeddings
        """
        if self.leaf_only:
            # Single species head — coarse levels derived by marginalization
            self.species_head = nn.Linear(hidden_dim, self.class_counts["species"])
            return

        if not self.parent_conditioning:
            # Independent heads — no cascading, same architecture as kingdom
            for level in self.levels:
                setattr(self, f"{level}_head",
                        nn.Linear(hidden_dim, self.class_counts[level]))
            return

        # --- Parent-conditioned (cascading) heads ---

        def block(out_dim: int):
            """Create classification block: [conditioning; features] → logits"""
            return nn.Linear(head_dim * 2, out_dim)

        # Kingdom: Top level, no parent conditioning
        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])

        # Phylum: Conditioned on kingdom
        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        # Class: Conditioned on phylum
        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        # Order: Conditioned on class
        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        # Family: Conditioned on order
        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        # Genus: Conditioned on family
        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        # Species: Conditioned on genus (final level, most important for scoring)
        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _extract_features(
        self,
        images: Dict[str, torch.Tensor],
        return_list: bool = False,
    ) -> torch.Tensor:
        """
        Extract features from all scales.

        Each scale goes through its dedicated backbone, producing a feature
        vector. By default, concatenated; optionally returns as list.

        Args:
            images: Dict mapping scale names to image tensors (B, 3, 224, 224)
            return_list: If True, return list of features instead of concatenated

        Returns:
            If return_list=False: Concatenated features (B, num_scales × feature_dim)
            If return_list=True: List of tensors, one per scale
        """
        scale_features = []
        for scale in self.scales:
            if scale in images:
                # Extract features through scale-specific backbone
                feats = self.backbones[scale](images[scale])  # (B, 1024 or 1536)
                scale_features.append(feats)

        if return_list:
            return scale_features

        # Concatenate along feature dimension
        return torch.cat(scale_features, dim=1)  # (B, total_features)

    def _extract_features_branched(
        self, images: Dict[str, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract per-stage features from all scales for B-CNN branching.

        Each scale's backbone (with features_only=True) returns spatial feature
        maps at each stage. We GAP-pool each stage, concatenate across scales,
        and project to hidden_dim.

        Returns:
            Dict mapping stage index → projected feature tensor (B, hidden_dim)
        """
        stage_features = [[] for _ in range(self.n_stages)]

        for scale in self.scales:
            if scale in images:
                # features_only backbone returns list of spatial feature maps
                feats_list = self.backbones[scale](images[scale])
                for stage_idx, feat in enumerate(feats_list):
                    # Global average pool: (B, C, H, W) → (B, C)
                    pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    stage_features[stage_idx].append(pooled)

        # Concatenate across scales and project per stage
        stage_vectors = {}
        for stage_idx in range(self.n_stages):
            if stage_features[stage_idx]:
                cat = torch.cat(stage_features[stage_idx], dim=1)  # (B, C_stage * n_scales)
                stage_vectors[stage_idx] = self.stage_projections[f"stage_{stage_idx}"](cat)  # (B, hidden_dim)

        return stage_vectors

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        return_attention: bool = False,
        image_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale backbones and hierarchical classifier.

        Data flow:
        1. Extract features from each scale using dedicated backbones
        2. Fuse features (concatenation or hierarchy-aware attention)
        3. (Optional) Apply GNN for multi-specimen relationship modeling
        4. Apply hierarchical classification with parent conditioning

        Args:
            images: Dict with scale keys mapping to image tensors
            return_attention: If True and using hierarchy fusion, also return
                attention weights (useful for visualization/analysis)
            image_ids: Optional (B,) tensor of image IDs for GNN grouping.
                When provided and use_gnn=True, specimens from the same image
                exchange information via graph convolution.

        Returns:
            Dict of logits for each taxonomy level
            If return_attention=True: Also returns attention_weights dict
        """
        # =================================================================
        # FEATURE EXTRACTION AND FUSION
        # =================================================================

        attention_weights = None

        if self.branch_cnn:
            # =============================================================
            # B-CNN MULTI-STAGE BRANCHING
            # =============================================================
            # Each taxonomy level gets features from its assigned backbone stage:
            #   Stage 0 → kingdom, phylum
            #   Stage 1 → class, order
            #   Stage 2 → family
            #   Stage 3 → genus, species
            stage_vectors = self._extract_features_branched(images)
            level_shared = {}
            for stage_idx, levels in self.stage_level_map.items():
                for level in levels:
                    level_shared[level] = stage_vectors[stage_idx]

        elif self.hierarchy_fusion is not None:
            # Hierarchy-aware fusion: get per-level representations
            scale_features = self._extract_features(images, return_list=True)

            if return_attention:
                level_features, attention_weights = self.hierarchy_fusion(
                    scale_features, return_attention=True
                )
            else:
                level_features = self.hierarchy_fusion(scale_features)

            # Project each level's features to hidden_dim
            level_shared = {
                level: self.level_projections[level](level_features[level])
                for level in self.levels
            }
        elif self.roi_self_attention is not None and self.part_attention is not None:
            # Combined patch self-attention + part-based attention
            # Apply both in parallel on spatial features, then concatenate

            roi_scale = self.scales[0]
            context_scales = self.scales[1:]

            # ROI: Extract spatial features
            roi_spatial = self.backbones[roi_scale].forward_features(images[roi_scale])

            # Apply both attention mechanisms in parallel
            roi_self_attended = self.roi_self_attention(roi_spatial)  # (B, roi_dim)
            roi_parts = self.part_attention(roi_spatial)  # (B, roi_dim)

            # Context: Global features
            context_features = []
            for scale in context_scales:
                if scale in images:
                    ctx_feats = self.backbones[scale](images[scale])
                    context_features.append(ctx_feats)

            # Concatenate: self-attended + part-attended + context
            all_features = [roi_self_attended, roi_parts] + context_features
            combined_features = torch.cat(all_features, dim=1)

            shared = self.shared_features(combined_features)
            level_shared = {level: shared for level in self.levels}

        elif self.roi_self_attention is not None:
            # Patch self-attention fusion: attend within ROI patches
            # 1. Get spatial features from ROI (1x scale) - use forward_features
            # 2. Apply self-attention over ROI patches
            # 3. Get global features from context scales
            # 4. Concatenate and fuse

            roi_scale = self.scales[0]  # Assume 1x is first
            context_scales = self.scales[1:]  # 3x, 5x, etc.

            # ROI: Extract spatial features (B, C, H, W) using forward_features
            roi_spatial = self.backbones[roi_scale].forward_features(images[roi_scale])
            # Apply self-attention within ROI patches
            roi_attended = self.roi_self_attention(roi_spatial)  # (B, roi_dim)

            # Context: Extract global features (standard pooled)
            context_features = []
            for scale in context_scales:
                if scale in images:
                    ctx_feats = self.backbones[scale](images[scale])  # (B, ctx_dim)
                    context_features.append(ctx_feats)

            # Concatenate ROI attended + context features
            all_features = [roi_attended] + context_features
            combined_features = torch.cat(all_features, dim=1)

            # Fuse through shared layer
            shared = self.shared_features(combined_features)  # (B, hidden_dim)
            level_shared = {level: shared for level in self.levels}

        elif self.part_attention is not None:
            # Part-based attention fusion: discover and attend to discriminative parts
            # 1. Get spatial features from ROI (1x scale)
            # 2. Apply part-based attention with learnable prototypes
            # 3. Get global features from context scales
            # 4. Concatenate part-aware ROI + context features

            roi_scale = self.scales[0]  # Assume 1x is first
            context_scales = self.scales[1:]  # 3x, 5x, etc.

            # ROI: Extract spatial features (B, C, H, W) using forward_features
            roi_spatial = self.backbones[roi_scale].forward_features(images[roi_scale])
            # Apply part-based attention with learnable prototypes
            roi_parts = self.part_attention(roi_spatial)  # (B, roi_dim)

            # Context: Extract global features (standard pooled)
            context_features = []
            for scale in context_scales:
                if scale in images:
                    ctx_feats = self.backbones[scale](images[scale])  # (B, ctx_dim)
                    context_features.append(ctx_feats)

            # Concatenate part-aware ROI + context features
            all_features = [roi_parts] + context_features
            combined_features = torch.cat(all_features, dim=1)

            # Fuse through shared layer
            shared = self.shared_features(combined_features)  # (B, hidden_dim)
            level_shared = {level: shared for level in self.levels}

        else:
            # Original concatenation fusion
            combined_features = self._extract_features(images)  # (B, total_features)
            shared = self.shared_features(combined_features)  # (B, hidden_dim)
            # All levels use the same shared representation
            level_shared = {level: shared for level in self.levels}

        # =================================================================
        # OPTIONAL GNN: MULTI-SPECIMEN RELATIONSHIP MODELING
        # =================================================================
        if self.specimen_gnn is not None and image_ids is not None:
            from .gnn_module import apply_gnn_to_batch
            # Apply GNN to shared features, grouped by image
            # For concat/attention fusions where all levels share the same features,
            # we update the shared tensor once. For B-CNN where levels have different
            # features, we update each level's features independently.
            updated_levels = {}
            seen_tensors = {}
            for level in self.levels:
                feat_id = id(level_shared[level])
                if feat_id not in seen_tensors:
                    seen_tensors[feat_id] = apply_gnn_to_batch(
                        self.specimen_gnn, level_shared[level], image_ids
                    )
                updated_levels[level] = seen_tensors[feat_id]
            level_shared = updated_levels

        # =================================================================
        # HIERARCHICAL CLASSIFICATION
        # =================================================================
        # Each level gets its own (potentially different) representation from fusion

        if self.leaf_only:
            # Leaf-only: single species head, marginalize to coarse levels
            # Uses logsumexp for numerical stability (avoids softmax→matmul→log chain)
            species_logits = self.species_head(level_shared["species"])
            outputs = {"species": species_logits}
            for level in self.levels:
                if level != "species":
                    # log_margin: (num_species, num_coarse) with 0 for members, -inf for non-members
                    log_M = getattr(self, f"log_margin_{level}")
                    # species_logits: (B, S), log_M: (S, C) → masked: (B, S, C)
                    masked = species_logits.unsqueeze(2) + log_M.unsqueeze(0)
                    # logsumexp over species dim → (B, C) unnormalized coarse logits
                    # CrossEntropyLoss applies log_softmax internally, which normalizes
                    # by subtracting logsumexp(all species) — giving correct -log P(coarse)
                    outputs[level] = torch.logsumexp(masked, dim=1)
        elif not self.parent_conditioning:
            # Independent heads — no cascading between levels
            outputs = {}
            for level in self.levels:
                outputs[level] = getattr(self, f"{level}_head")(level_shared[level])
        else:
            # Parent-conditioned (cascading) heads
            # When stop_gradient=True, detach parent probs before feeding to child.
            # This prevents error propagation through the cascade during training
            # while still providing parent predictions as features.
            _sg = lambda p: p.detach() if self.stop_gradient else p

            # Kingdom (top level - no conditioning)
            kingdom_logits = self.kingdom_head(level_shared["kingdom"])
            kingdom_probs = torch.softmax(kingdom_logits, dim=1)

            # Phylum (conditioned on kingdom)
            phylum_input = self.kingdom_to_phylum(_sg(kingdom_probs))
            phylum_feats = self.phylum_features(level_shared["phylum"])
            phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
            phylum_probs = torch.softmax(phylum_logits, dim=1)

            # Class (conditioned on phylum)
            class_input = self.phylum_to_class(_sg(phylum_probs))
            class_feats = self.class_features(level_shared["class"])
            class_logits = self.class_head(torch.cat([class_input, class_feats], dim=1))
            class_probs = torch.softmax(class_logits, dim=1)

            # Order (conditioned on class)
            order_input = self.class_to_order(_sg(class_probs))
            order_feats = self.order_features(level_shared["order"])
            order_logits = self.order_head(torch.cat([order_input, order_feats], dim=1))
            order_probs = torch.softmax(order_logits, dim=1)

            # Family (conditioned on order)
            family_input = self.order_to_family(_sg(order_probs))
            family_feats = self.family_features(level_shared["family"])
            family_logits = self.family_head(torch.cat([family_input, family_feats], dim=1))
            family_probs = torch.softmax(family_logits, dim=1)

            # Genus (conditioned on family)
            genus_input = self.family_to_genus(_sg(family_probs))
            genus_feats = self.genus_features(level_shared["genus"])
            genus_logits = self.genus_head(torch.cat([genus_input, genus_feats], dim=1))
            genus_probs = torch.softmax(genus_logits, dim=1)

            # Species (conditioned on genus - final and most important level)
            species_input = self.genus_to_species(_sg(genus_probs))
            species_feats = self.species_features(level_shared["species"])
            species_logits = self.species_head(torch.cat([species_input, species_feats], dim=1))

            outputs = {
                "kingdom": kingdom_logits,
                "phylum": phylum_logits,
                "class": class_logits,
                "order": order_logits,
                "family": family_logits,
                "genus": genus_logits,
                "species": species_logits,
            }

        if return_attention and attention_weights is not None:
            return outputs, attention_weights
        return outputs

    def hierarchical_loss(self, outputs, targets):
        """
        Compute weighted loss with taxonomic awareness for species.

        For non-species levels: standard cross-entropy with label smoothing
        For species level: taxonomic-aware loss (if configured)

        Loss combination (fixed weights):
            total_loss = Σ_L (w_L × loss_L) / Σ_L w_L

        Loss combination (uncertainty weighting, Kendall et al. 2018):
            total_loss = Σ_L [0.5 × exp(-s_L) × loss_L + 0.5 × s_L]
            where s_L = log(σ²_L) is learned per level

        When loss_type == "both":
            species_loss = 0.5 × TaxDistanceLoss + 0.5 × TaxLabelSmoothing

        Args:
            outputs: Dict of logits for each level
            targets: Dict of ground truth labels for each level

        Returns:
            total_loss: Weighted average loss across levels
            level_losses: Dict of individual losses per level
        """
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0

        for level in self.levels:
            if level not in outputs or level not in targets:
                continue

            logits = outputs[level]
            target = targets[level]

            # Species level gets special taxonomic-aware loss
            if level == "species" and self.use_taxonomic_loss:
                # FULLY INTEGRATED UNCERTAINTY WEIGHTING for species level:
                # When enabled, we learn separate weights for CE and taxonomic distance
                # L = exp(-s_ce)*CE + exp(-s_tax)*TaxDist + 0.5*(s_ce + s_tax)
                if self.use_uncertainty_weighting and self.tax_distance_loss:
                    # Compute CE and taxonomic distance losses separately
                    ce_loss = self.ce_loss(logits, target)
                    # Get raw distance loss (we need the internal components)
                    # TaxonomicDistanceLoss computes: (1-α)*CE + α*ExpectedDist
                    # We want just the expected distance: Σ p_i * d(i, y)
                    probs = torch.softmax(logits, dim=1)
                    # Index into distance matrix for each target
                    # TaxonomicDistanceLoss stores distances as 'distances' buffer
                    distances = self.tax_distance_loss.distances[target]  # (B, C)
                    expected_dist = (probs * distances).sum(dim=1).mean()

                    # Apply uncertainty weighting to each component
                    prec_ce = torch.exp(-self.log_var_species_ce)
                    prec_tax = torch.exp(-self.log_var_species_tax)

                    # Combined loss with learnable weights
                    loss = (0.5 * prec_ce * ce_loss + 0.5 * self.log_var_species_ce +
                            0.5 * prec_tax * expected_dist + 0.5 * self.log_var_species_tax)

                    # Store component losses for logging (use _component suffix to avoid
                    # confusion with taxonomy levels in validation_step)
                    level_losses["species_ce_component"] = ce_loss.detach()
                    level_losses["species_tax_component"] = expected_dist.detach()
                    level_losses[level] = loss

                    # For species with full uncertainty weighting, skip the normal weighting
                    # (the uncertainty weighting is already applied above)
                    total_loss += loss
                    continue

                elif self.hxe_loss is not None:
                    # HXE loss (Bertinetto et al. 2020): conditional CE at each level
                    loss = self.hxe_loss(logits, target)
                elif self.tax_distance_loss and self.tax_smooth_loss:
                    # Combine both losses equally (fixed weighting)
                    loss = 0.5 * self.tax_distance_loss(logits, target) + \
                           0.5 * self.tax_smooth_loss(logits, target)
                elif self.tax_distance_loss:
                    # Just distance-aware loss (fixed alpha weighting)
                    loss = self.tax_distance_loss(logits, target)
                elif self.tax_smooth_loss:
                    # Just taxonomic label smoothing
                    loss = self.tax_smooth_loss(logits, target)
                else:
                    # Fallback to standard CE
                    loss = self.ce_loss(logits, target)
            else:
                # Non-species levels use standard CE
                loss = self.ce_loss(logits, target)

            level_losses[level] = loss

            # Apply weighting scheme
            if self.use_uncertainty_weighting:
                # Kendall et al. uncertainty weighting:
                # L = 0.5 * exp(-s) * L_task + 0.5 * s
                # This learns to downweight noisy/hard tasks automatically
                log_var = self.log_vars[level]
                precision = torch.exp(-log_var)  # 1/σ²
                total_loss += 0.5 * precision * loss + 0.5 * log_var
            else:
                # Fixed hierarchy weights
                w = self.hierarchy_weights.get(level, 1.0)
                total_loss += w * loss
                weight_sum += w

        # Normalize by total weight (only for fixed weighting)
        if not self.use_uncertainty_weighting:
            total_loss = total_loss / max(weight_sum, 1e-6)

        # C-HMCNN-inspired hierarchy consistency loss (additive regularization)
        if (self.hierarchy_consistency_loss is not None
                and "species" in outputs and "species" in targets):
            consistency_loss = self.hierarchy_consistency_loss(
                outputs["species"], targets["species"]
            )
            total_loss += self.consistency_weight * consistency_loss
            level_losses["consistency"] = consistency_loss.detach()

        return total_loss, level_losses

    def _compute_taxonomic_score(self, species_logits, species_targets):
        """
        Compute expected taxonomic distance (competition metric approximation).

        This is an approximation of the actual competition metric:
            score = mean(D(predicted, true)) for all samples

        We use argmax predictions to match submission format.

        Args:
            species_logits: Model predictions of shape (B, num_species)
            species_targets: Ground truth labels of shape (B,)

        Returns:
            Mean taxonomic distance (lower is better)
        """
        # Get predicted species (argmax)
        preds = species_logits.argmax(dim=1)  # (B,)

        # Look up distance from distance matrix
        # distance_matrix[i, j] = distance between species i and j
        distances = self.distance_matrix[preds, species_targets]

        return distances.float().mean()

    def on_train_epoch_start(self):
        """
        BT-Strategy: dynamically shift hierarchy weights from coarse to fine.

        Linearly interpolates between coarse-heavy weights (early training) and
        species-heavy weights (late training). Inspired by Zhu & Bain (2017).

        Only active when bt_strategy=True. Otherwise a no-op.
        """
        if not self.bt_strategy:
            return

        t = self.current_epoch / max(self.trainer.max_epochs - 1, 1)

        # Start: emphasize coarse levels (helps backbone learn general features)
        start = {
            "kingdom": 3.0, "phylum": 2.5, "class": 2.0, "order": 1.5,
            "family": 1.0, "genus": 0.5, "species": 0.25,
        }
        # End: emphasize fine levels (matches piecewise scheme for final convergence)
        end = {
            "kingdom": 0.5, "phylum": 0.75, "class": 1.0, "order": 1.25,
            "family": 1.5, "genus": 2.0, "species": 2.5,
        }

        for level in self.levels:
            self.hierarchy_weights[level] = (1 - t) * start[level] + t * end[level]

        # Log periodically
        if self.current_epoch % 5 == 0:
            weights_str = ", ".join(
                f"{l}={self.hierarchy_weights[l]:.2f}" for l in self.levels
            )
            print(f"BT-Strategy epoch {self.current_epoch} (t={t:.2f}): {weights_str}")

        # Log weights to tensorboard
        for level in self.levels:
            self.log(f"bt_weight/{level}", self.hierarchy_weights[level], prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Logs:
            - train_loss: Overall weighted loss
            - train_{level}_loss: Loss for each taxonomy level
            - train_tax_score: Taxonomic distance metric

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Training loss for this batch
        """
        images, labels = batch[0], batch[1]
        image_ids = batch[2] if len(batch) > 2 else None
        outputs = self(images, image_ids=image_ids)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log overall loss
        self.log("train_loss", loss, prog_bar=True)

        # Log per-level losses
        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)

        # Log taxonomic score (competition metric approximation)
        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("train_tax_score", tax_score, prog_bar=False)

        # Log learned uncertainty weights (Kendall et al.)
        if self.use_uncertainty_weighting:
            for level in self.levels:
                log_var = self.log_vars[level].item()
                # Convert to effective weight: w = exp(-log_var) = 1/σ²
                effective_weight = torch.exp(-self.log_vars[level]).item()
                self.log(f"uncertainty/{level}_log_var", log_var, prog_bar=False)
                self.log(f"uncertainty/{level}_weight", effective_weight, prog_bar=False)

            # Log fully integrated species CE vs taxonomic distance weights
            if hasattr(self, 'log_var_species_ce'):
                ce_weight = torch.exp(-self.log_var_species_ce).item()
                tax_weight = torch.exp(-self.log_var_species_tax).item()
                # Compute effective alpha: α = tax_weight / (ce_weight + tax_weight)
                effective_alpha = tax_weight / (ce_weight + tax_weight + 1e-8)
                self.log("uncertainty/species_ce_weight", ce_weight, prog_bar=False)
                self.log("uncertainty/species_tax_weight", tax_weight, prog_bar=False)
                self.log("uncertainty/effective_alpha", effective_alpha, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step with accuracy and taxonomic score tracking.

        Logs:
            - val_loss: Overall weighted loss
            - val_{level}_loss: Loss for each taxonomy level
            - val_{level}_acc: Accuracy for each level
            - val_tax_score: Taxonomic distance metric (MOST IMPORTANT)

        The val_tax_score is highlighted in prog_bar because it directly
        corresponds to the competition evaluation metric.

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Validation loss for this batch
        """
        images, labels = batch[0], batch[1]
        image_ids = batch[2] if len(batch) > 2 else None
        outputs = self(images, image_ids=image_ids)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log overall loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log per-level loss and accuracy
        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)

            # Skip non-taxonomy-level losses (components, consistency, etc.)
            if lvl.endswith("_component") or lvl not in outputs:
                continue

            # Compute accuracy
            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()

            # Show species accuracy in progress bar (most important level)
            self.log(f"val_{lvl}_acc", acc, prog_bar=(lvl == "species"), sync_dist=True)

        # Log taxonomic score (MOST IMPORTANT METRIC!)
        # This directly corresponds to the competition evaluation
        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("val_tax_score", tax_score, prog_bar=True, sync_dist=True)

        # Store for epoch-end analysis
        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Clear stored validation outputs at end of epoch."""
        self.val_outputs.clear()

    def _is_dinov2_backbone(self) -> bool:
        """Check if the backbone is a DINOv2 ViT model."""
        first_bb = next(iter(self.backbones.values()))
        return isinstance(first_bb, DINOv2Backbone)

    def _get_dinov2_num_blocks(self) -> int:
        """Get number of transformer blocks in DINOv2 backbone."""
        first_bb = next(iter(self.backbones.values()))
        return getattr(first_bb, 'n_blocks', 12)

    @staticmethod
    def _get_dinov2_layer_id(param_name: str, num_blocks: int) -> int:
        """
        Map a DINOv2 parameter name to its layer group index for LLRD.

        Layer groups (from earliest/lowest LR to latest/highest LR):
          0: patch_embed, cls_token, pos_embed, mask_token
          1..num_blocks: transformer blocks
          num_blocks+1: norm (final layer norm)

        Args:
            param_name: Full parameter name (e.g., "backbones.1x.model.blocks.5.mlp.fc1.weight")
            num_blocks: Number of transformer blocks (e.g., 12 for ViT-B)

        Returns:
            Layer group index (0 = earliest layer, num_blocks+1 = latest)
        """
        # Strip prefix up to ".model." to get the DINOv2-internal name
        if ".model." in param_name:
            inner = param_name.split(".model.", 1)[1]
        else:
            inner = param_name

        if inner.startswith("patch_embed") or inner.startswith("cls_token") or \
           inner.startswith("pos_embed") or inner.startswith("mask_token"):
            return 0
        elif inner.startswith("blocks."):
            # e.g., "blocks.5.mlp.fc1.weight" → block 5 → layer group 6
            block_idx = int(inner.split(".")[1])
            return block_idx + 1
        elif inner.startswith("norm"):
            return num_blocks + 1
        else:
            # Unknown — assign to latest layer group
            return num_blocks + 1

    def configure_optimizers(self):
        """
        Configure optimizers with parameter-specific learning rates.

        Learning Rate Strategy:
        -----------------------
        For CNN backbones (ConvNeXtV2):
            1. Backbone parameters: base_lr × backbone_lr_scale (e.g., 0.1)
            2. Other parameters: base_lr

        For ViT backbones (DINOv2) with layer_decay < 1.0:
            Layer-wise Learning Rate Decay (LLRD):
            - Each transformer block gets lr_i = base_lr × backbone_scale × decay^(N-i)
            - No weight decay on bias and LayerNorm parameters
            - Optional linear warmup

        Scheduler: CosineAnnealingWarmRestarts
            - T_0=5: First restart after 5 epochs
            - T_mult=2: Double period after each restart (5, 10, 20, ...)
            - eta_min=1e-6: Minimum learning rate

        Returns:
            Dict with optimizer and lr_scheduler configuration
        """
        base_lr = self.cfg.training.learning_rate
        layer_decay = getattr(self.cfg.training, 'layer_decay', 1.0)
        warmup_epochs = getattr(self.cfg.training, 'warmup_epochs', 0)
        vit_wd = getattr(self.cfg.training, 'vit_weight_decay', self.cfg.training.weight_decay)

        use_llrd = self._is_dinov2_backbone() and layer_decay < 1.0

        if use_llrd:
            # LLRD for DINOv2 ViT fine-tuning
            num_blocks = self._get_dinov2_num_blocks()
            num_layers = num_blocks + 2  # patch_embed + blocks + norm
            backbone_base_lr = base_lr * self.cfg.training.backbone_lr_scale

            # Build per-layer parameter groups
            # Track param ids to avoid duplicates (shared backbone)
            seen_param_ids = set()
            param_groups = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen_param_ids:
                    continue
                seen_param_ids.add(pid)

                if "backbones" in name:
                    # Backbone parameter — apply LLRD
                    layer_id = self._get_dinov2_layer_id(name, num_blocks)
                    lr_scale = layer_decay ** (num_layers - 1 - layer_id)
                    lr = backbone_base_lr * lr_scale

                    # No weight decay on bias and LayerNorm
                    if "bias" in name or "norm" in name.split(".")[-2:][0] if "." in name else "":
                        wd = 0.0
                    elif any(nd in name for nd in ["bias", ".norm", "LayerNorm", "layer_norm"]):
                        wd = 0.0
                    else:
                        wd = vit_wd

                    param_groups.append({
                        "params": [param],
                        "lr": lr,
                        "weight_decay": wd,
                    })
                else:
                    # Head/fusion parameter — full learning rate
                    param_groups.append({
                        "params": [param],
                        "lr": base_lr,
                        "weight_decay": self.cfg.training.weight_decay,
                    })

            # Print LLRD schedule summary
            print(f"\n  LLRD schedule (layer_decay={layer_decay}, num_blocks={num_blocks}):")
            for layer_id in range(num_layers):
                lr_scale = layer_decay ** (num_layers - 1 - layer_id)
                print(f"    Layer {layer_id:2d}: lr = {backbone_base_lr * lr_scale:.2e}")
            print(f"    Heads:    lr = {base_lr:.2e}")

            optimizer = torch.optim.AdamW(param_groups)
        else:
            # Standard 2-group optimizer for CNN backbones
            backbone_params = []
            other_params = []

            for name, param in self.named_parameters():
                if "backbones" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": base_lr * self.cfg.training.backbone_lr_scale,
                    },
                    {
                        "params": other_params,
                        "lr": base_lr,
                    },
                ],
                weight_decay=self.cfg.training.weight_decay,
            )

        # Cosine annealing with warm restarts
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        if warmup_epochs > 0 and use_llrd:
            # Linear warmup then cosine decay
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
