"""
Hierarchy-Aware Multi-Scale Fusion Module.

================================================================================
MOTIVATION
================================================================================

The current multi-scale fusion approach concatenates features from all scales
(e.g., 1x ROI, 3x context, 5x context) into a single vector that feeds ALL
taxonomy classification heads identically. This is suboptimal because:

1. SCALE-AGNOSTIC: Kingdom vs Species may benefit from different scale emphasis
   - Kingdom/Phylum: High-level body plan → may prefer context scales (3x, 5x)
   - Genus/Species: Fine-grained details → may prefer ROI scale (1x)

2. PARAMETER-HEAVY: With 4 scales × 1024-D features = 4608-D, each head needs
   many parameters. Reducing to 1024-D makes heads 4.5× smaller.

3. NO SCALE INTERACTION: Scales don't communicate before classification.

================================================================================
HIERARCHY-AWARE ATTENTION (Option 2)
================================================================================

Key insight: Let the model LEARN which scales matter for each taxonomy level.

Architecture:
                     ┌─────────────────────────────────────────────────┐
                     │           Scale Features                        │
                     │  1x: [B, 1536]  3x: [B, 1024]  ...              │
                     └────────────────────┬────────────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────────────┐
                     │         Project to Common Dimension              │
                     │  proj_1x(1536→1024)  proj_3x(1024→1024)  ...    │
                     └────────────────────┬────────────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────────────┐
                     │              Stack: [B, 4, 1024]                 │
                     └────────────────────┬────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
        ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  Kingdom Attn    │             │  Genus Attn      │             │  Species Attn    │
│  [B, 4] weights  │             │  [B, 4] weights  │             │  [B, 4] weights  │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                │                                │
         ▼                                ▼                                ▼
  Weighted sum →               Weighted sum →                   Weighted sum →
  [B, 1024]                    [B, 1024]                        [B, 1024]
  Kingdom features             Genus features                   Species features

Each level learns its own attention over scales, producing level-specific
1024-D representations. This allows:
- Species head to focus on fine details (1x ROI)
- Kingdom head to use global context (3x, 5x scales)

================================================================================
RELATED WORK & THEORETICAL FOUNDATION
================================================================================

This module draws on several related concepts from the literature:

1. CROSS-HIERARCHICAL REGION FEATURE (CHRF) LEARNING
   Ding et al., "CHRF: A Cross-Hierarchical Region Feature Learning Network for
   Fine-Grained Image Classification", ECCV 2022.

   Key insight: Different semantic levels (coarse vs fine-grained) benefit from
   different spatial regions. CHRF learns level-specific attention over regions.
   Our approach extends this to multi-SCALE features rather than spatial regions.

2. MULTI-SCALE ATTENTION-DRIVEN HIERARCHICAL LEARNING (MAHL)
   Park et al., "MAHL: Multi-Scale Attention-Driven Hierarchical Learning for
   Fine-Grained Visual Classification", Electronics, 2025.

   Demonstrates that hierarchical classifiers benefit from scale-aware attention
   where different classification levels use different scale combinations.

3. HIERARCHICAL ATTENTION VISION TRANSFORMER (HAVT)
   Zhang et al., "Hierarchical Attention Vision Transformer for Fine-Grained
   Image Classification", ScienceDirect Expert Systems with Applications, 2023.

   Shows effectiveness of hierarchy-aware attention for fine-grained tasks.

4. SQUEEZE-AND-EXCITATION NETWORKS
   Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

   Foundation for channel/feature attention mechanisms. Our per-level attention
   extends SE's "recalibration" concept to multi-scale feature fusion.

5. DYNAMIC GATED FUSION
   Various works on gated multi-modal/multi-scale fusion, including DGFNet
   (Dynamic Gated Fusion Network) for multi-source feature combination.

================================================================================
DESIGN RATIONALE
================================================================================

The key hypothesis is that taxonomy levels have different optimal scale preferences:

- COARSE LEVELS (Kingdom, Phylum): Require holistic body plan understanding
  → May benefit from context scales (3x, 5x, full image)

- FINE LEVELS (Genus, Species): Require discriminative local features
  → May benefit from ROI-focused scale (1x)

By learning per-level attention weights, the model can:
1. Automatically discover optimal scale combinations per level
2. Reduce parameter count (shared backbone, level-specific fusion)
3. Provide interpretable scale preferences for analysis
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class HierarchyAwareFusion(nn.Module):
    """
    Hierarchy-aware multi-scale fusion with per-level attention.

    Instead of concatenating all scale features and feeding the same
    representation to all taxonomy heads, this module:

    1. Projects each scale to a common dimension
    2. For each taxonomy level, computes attention weights over scales
    3. Returns level-specific weighted combinations

    This allows different levels to emphasize different scales:
    - Kingdom/Phylum may use more context (3x, 5x)
    - Genus/Species may focus on fine details (1x ROI)

    Args:
        scale_dims: List of feature dimensions per scale
            e.g., [1536, 1024, 1024, 1024] for [1x, 3x, 5x, full]
        common_dim: Dimension to project all scales to (default: 1024)
        num_levels: Number of taxonomy levels (default: 7)
        level_names: Optional list of level names for interpretability

    Example:
        >>> fusion = HierarchyAwareFusion(
        ...     scale_dims=[1536, 1024, 1024],  # 3 scales
        ...     common_dim=1024,
        ...     num_levels=7,
        ... )
        >>> scale_features = [
        ...     torch.randn(4, 1536),  # 1x features
        ...     torch.randn(4, 1024),  # 3x features
        ...     torch.randn(4, 1024),  # 5x features
        ... ]
        >>> level_features = fusion(scale_features)
        >>> # level_features is dict with 7 tensors of shape [4, 1024]
    """

    def __init__(
        self,
        scale_dims: List[int],
        common_dim: int = 1024,
        num_levels: int = 7,
        level_names: Optional[List[str]] = None,
        attention_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.scale_dims = scale_dims
        self.common_dim = common_dim
        self.num_scales = len(scale_dims)
        self.num_levels = num_levels
        self.level_names = level_names or [
            "kingdom", "phylum", "class", "order", "family", "genus", "species"
        ]

        # =================================================================
        # SCALE PROJECTIONS
        # =================================================================
        # Project each scale to the common dimension
        # This handles the case where backbones have different output dims
        # e.g., ConvNeXtV2-Large = 1536, ConvNeXtV2-Base = 1024

        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, common_dim),
                nn.GELU(),
            )
            for dim in scale_dims
        ])

        # =================================================================
        # PER-LEVEL ATTENTION
        # =================================================================
        # Each taxonomy level has its own attention network that outputs
        # weights over the scales. Input is concatenated projected features.

        # Attention network architecture:
        # [concat_all_scales] → Linear → GELU → Dropout → Linear → Softmax
        attention_input_dim = common_dim * self.num_scales

        self.level_attention = nn.ModuleDict({
            level: nn.Sequential(
                nn.Linear(attention_input_dim, attention_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(attention_hidden_dim, self.num_scales),
                # Softmax applied in forward for flexibility
            )
            for level in self.level_names
        })

        # Temperature for attention (learnable, per-level)
        # Lower temperature = sharper attention, higher = more uniform
        self.attention_temperature = nn.ParameterDict({
            level: nn.Parameter(torch.ones(1))
            for level in self.level_names
        })

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchy-aware fusion of multi-scale features.

        Args:
            scale_features: List of tensors, one per scale
                Each tensor has shape [B, scale_dim]
            return_attention: If True, also return attention weights

        Returns:
            level_features: Dict mapping level names to fused features
                Each tensor has shape [B, common_dim]
            attention_weights: (optional) Dict of attention weights per level
                Each tensor has shape [B, num_scales]
        """
        batch_size = scale_features[0].shape[0]

        # =================================================================
        # PROJECT ALL SCALES TO COMMON DIMENSION
        # =================================================================
        projected = []
        for i, (proj, feats) in enumerate(zip(self.scale_projections, scale_features)):
            projected.append(proj(feats))  # [B, common_dim]

        # Stack for weighted sum: [B, num_scales, common_dim]
        stacked = torch.stack(projected, dim=1)

        # Concatenate for attention input: [B, num_scales * common_dim]
        concat = torch.cat(projected, dim=-1)

        # =================================================================
        # COMPUTE PER-LEVEL ATTENTION AND FUSE
        # =================================================================
        level_features = {}
        attention_weights = {}

        for level in self.level_names:
            # Compute attention logits
            attn_logits = self.level_attention[level](concat)  # [B, num_scales]

            # Apply temperature scaling
            temp = self.attention_temperature[level].clamp(min=0.1)
            attn_logits = attn_logits / temp

            # Softmax to get weights
            weights = torch.softmax(attn_logits, dim=-1)  # [B, num_scales]
            attention_weights[level] = weights

            # Weighted sum over scales
            # weights: [B, num_scales] → [B, num_scales, 1]
            # stacked: [B, num_scales, common_dim]
            # result: [B, common_dim]
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            level_features[level] = fused

        if return_attention:
            return level_features, attention_weights
        return level_features

    def get_attention_summary(
        self,
        scale_features: List[torch.Tensor],
        scale_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get a human-readable summary of attention weights.

        Useful for analyzing which scales the model prefers for each level.

        Args:
            scale_features: List of scale feature tensors
            scale_names: Optional names for scales (e.g., ["1x", "3x", "5x"])

        Returns:
            Dict mapping level names to dicts of {scale_name: avg_weight}

        Example output:
            {
                "kingdom": {"1x": 0.15, "3x": 0.42, "5x": 0.43},
                "species": {"1x": 0.65, "3x": 0.20, "5x": 0.15},
            }
        """
        scale_names = scale_names or [f"scale_{i}" for i in range(self.num_scales)]

        with torch.no_grad():
            _, attention = self.forward(scale_features, return_attention=True)

        summary = {}
        for level, weights in attention.items():
            # Average over batch
            avg_weights = weights.mean(dim=0).cpu().numpy()
            summary[level] = {
                name: float(w) for name, w in zip(scale_names, avg_weights)
            }

        return summary


class GatedCrossScaleFusion(nn.Module):
    """
    Gated cross-scale fusion with self-attention between scales.

    This is Option 3 from the fusion analysis: scales "talk" to each other
    via self-attention before being combined.

    Architecture:
        1. Project scales to common dimension
        2. Self-attention over scales (scales as tokens)
        3. Gated combination for each level

    This allows the model to learn relationships like:
    - "When 1x shows spots, check 3x for body shape"
    - "Context from 5x helps disambiguate similar species in 1x"

    More expressive than HierarchyAwareFusion but also more parameters.
    """

    def __init__(
        self,
        scale_dims: List[int],
        common_dim: int = 1024,
        num_levels: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.common_dim = common_dim
        self.num_scales = len(scale_dims)

        # Project scales to common dimension
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, common_dim),
            )
            for dim in scale_dims
        ])

        # Self-attention over scales (treating scales as tokens)
        self.scale_self_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm after attention
        self.attn_norm = nn.LayerNorm(common_dim)

        # Per-level gating: combines attended features for each level
        self.level_names = [
            "kingdom", "phylum", "class", "order", "family", "genus", "species"
        ]

        # Gating network per level
        self.level_gates = nn.ModuleDict({
            level: nn.Sequential(
                nn.Linear(common_dim * self.num_scales, self.num_scales),
                nn.Sigmoid(),
            )
            for level in self.level_names
        })

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-scale attention.

        Args:
            scale_features: List of [B, scale_dim] tensors
            return_attention: If True, return attention weights

        Returns:
            Dict of fused features per level, each [B, common_dim]
        """
        # Project to common dimension
        projected = [proj(f) for proj, f in zip(self.scale_projections, scale_features)]

        # Stack as sequence: [B, num_scales, common_dim]
        x = torch.stack(projected, dim=1)

        # Self-attention: scales attend to each other
        attended, attn_weights = self.scale_self_attention(x, x, x)

        # Residual connection + norm
        x = self.attn_norm(x + attended)

        # Flatten for gating: [B, num_scales * common_dim]
        flat = x.flatten(start_dim=1)

        # Per-level gated combination
        level_features = {}
        for level in self.level_names:
            gates = self.level_gates[level](flat)  # [B, num_scales]
            # Gated sum
            fused = (x * gates.unsqueeze(-1)).sum(dim=1)  # [B, common_dim]
            level_features[level] = fused

        if return_attention:
            return level_features, attn_weights
        return level_features
