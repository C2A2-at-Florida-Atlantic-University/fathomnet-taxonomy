"""
Multi-Context Environmental Attention Module (MCEAM) for CNN features.

This module implements cross-attention between ROI features and multi-scale
context features, inspired by the ViT-based MCEAM from the reference paper,
but adapted for CNN (ConvNeXt) feature representations.

Key differences from ViT-based MCEAM:
-------------------------------------
1. ViT uses patch embeddings (spatial grid of features)
   CNN uses global pooled features (single vector per image)

2. ViT attention: Q from [CLS] token, K/V from patch tokens
   CNN attention: Q from ROI features, K/V from context features

3. We simulate "patch-like" behavior by using the spatial feature maps
   from intermediate CNN layers instead of just the final pooled features

Architecture:
-------------
Input:
  - ROI features: (B, C, H, W) from ConvNeXt backbone
  - Context features: List of (B, C, H, W) from each context scale

Output:
  - Fused features: (B, hidden_dim) combining ROI + attended contexts

Flow:
  1. Extract spatial features from each scale
  2. For each context scale:
     - Flatten spatial dims to create "pseudo-patches"
     - Cross-attend: Q=ROI features, K/V=context features
     - Pool attended features
  3. Concatenate all attended context features with ROI global features
  4. Project through fusion network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class PatchSelfAttention(nn.Module):
    """
    Self-attention within spatial patches of a feature map.

    This module treats spatial positions in a CNN feature map as "patches"
    and applies self-attention to learn part relationships within the image.

    Key Insight:
    ------------
    For fine-grained classification, different parts of an organism (head, body,
    fins, patterns) provide complementary information. Self-attention allows
    these parts to communicate and form a richer representation.

    For example, in marine species:
    - Seeing spots + seeing a tail → more confident it's a spotted fish
    - Seeing tentacles + seeing a bell → confident it's a jellyfish

    Architecture:
    -------------
    Input: (B, C, H, W) spatial feature map from CNN backbone
           e.g., ConvNeXt with 224x224 input → (B, 1024, 7, 7)

    Processing:
    1. Flatten spatial dims: (B, C, H, W) → (B, H×W, C) = (B, 49, 1024)
    2. Add learnable positional encoding
    3. Apply N layers of self-attention + FFN
    4. Pool to get global representation

    Output: (B, output_dim) global feature vector

    References:
    -----------
    - Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021
    - He et al., "TransFG: A Transformer Architecture for Fine-grained Recognition"
    - Zhang et al., "Multi-branch and Multi-scale Attention Learning"

    Args:
        input_dim: Channel dimension of input features (e.g., 1024)
        hidden_dim: Dimension for attention computation (default: 512)
        output_dim: Output feature dimension (default: same as input_dim)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout probability (default: 0.1)
        max_patches: Maximum number of spatial patches (default: 49 for 7×7)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_patches: int = 49,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding for spatial positions
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, hidden_dim) * 0.02)

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for module in self.output_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Apply self-attention over spatial patches.

        Args:
            x: Input feature map (B, C, H, W) or already flattened (B, N, C)
            return_attention: If True, also return attention weights

        Returns:
            features: (B, output_dim) global feature representation
            attention: (optional) attention weights from last layer
        """
        # Handle both 4D (spatial) and 3D (sequence) inputs
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Flatten spatial dimensions: (B, C, H, W) → (B, H×W, C)
            x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H×W
        else:
            B, N, C = x.shape

        N = x.shape[1]  # Number of patches

        # Project to hidden dimension
        x = self.input_proj(x)  # (B, N, hidden_dim)

        # Add positional encoding (truncate or pad if needed)
        if N <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :N, :]
        else:
            # Interpolate positional encoding if more patches than expected
            pos = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, hidden_dim)

        # Apply transformer encoder (self-attention)
        x = self.transformer(x)  # (B, 1+N, hidden_dim)

        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (B, hidden_dim)

        # Project to output dimension
        output = self.output_proj(cls_output)  # (B, output_dim)

        if return_attention:
            # Get attention from last layer (approximation - actual extraction is complex)
            return output, None  # TODO: proper attention extraction if needed

        return output


class PatchSelfAttentionPooling(nn.Module):
    """
    Alternative: Self-attention with attention pooling instead of [CLS] token.

    This version uses attention pooling where a learnable query attends to all
    patches to produce the final representation. Can be more flexible than [CLS].

    Args:
        input_dim: Channel dimension of input features
        hidden_dim: Dimension for attention computation
        output_dim: Output feature dimension
        num_heads: Number of attention heads
        num_layers: Number of self-attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_patches: int = 49,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, hidden_dim) * 0.02)

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Attention pooling: learnable query that attends to all patches
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features or (B, N, C) sequence

        Returns:
            (B, output_dim) pooled representation
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
        else:
            B = x.shape[0]

        N = x.shape[1]

        # Project and add positional encoding
        x = self.input_proj(x)
        if N <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :N, :]

        # Apply self-attention layers
        for i in range(len(self.self_attn_layers)):
            # Self-attention
            residual = x
            x = self.norms1[i](x)
            attn_out, _ = self.self_attn_layers[i](x, x, x)
            x = residual + attn_out

            # FFN
            residual = x
            x = self.norms2[i](x)
            x = residual + self.ffn_layers[i](x)

        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        pooled, _ = self.pool_attn(query, x, x)  # (B, 1, hidden_dim)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, hidden_dim)

        # Output projection
        return self.output_proj(pooled)


class PartBasedAttention(nn.Module):
    """
    Part-based attention with learnable prototypes for fine-grained classification.

    This module learns K prototype vectors that discover discriminative parts
    (e.g., fins, head, patterns) in an unsupervised manner. Each prototype
    attends to different spatial regions to extract part-specific features.

    Key Insight:
    ------------
    Fine-grained species classification relies on subtle differences in specific
    body parts. Instead of using a single global feature, we learn to:
    1. Discover K discriminative parts automatically via learnable prototypes
    2. Localize each part by computing attention between prototype and spatial features
    3. Extract part-specific features from attended regions
    4. Combine part representations for classification

    For marine species, discovered parts might correspond to:
    - Head shape and eyes
    - Fin structure (dorsal, pectoral, caudal)
    - Body patterns (stripes, spots, coloring)
    - Tail/appendage characteristics

    Architecture:
    -------------
    Input: (B, C, H, W) spatial feature map from CNN backbone

    Processing:
    1. Learn K prototype vectors (K x D)
    2. Flatten features: (B, C, H, W) → (B, H×W, C)
    3. Compute attention: softmax(prototypes @ features.T / sqrt(d))
    4. Extract part features: attention @ features
    5. Aggregate: concat or attention pool over K parts

    Output: (B, output_dim) part-aware feature vector

    References:
    -----------
    - Zheng et al., "Learning Multi-Attention Convolutional Neural Network
      for Fine-Grained Image Recognition", ICCV 2017
    - Sun et al., "Multi-Attention Multi-Class Constraint for Fine-grained
      Image Recognition", ECCV 2018
    - Zhang et al., "Part-based R-CNNs for Fine-grained Category Detection"
    - Fu et al., "Look Closer to See Better: Recurrent Attention CNN", CVPR 2017

    Args:
        input_dim: Channel dimension of input features (e.g., 1024)
        num_parts: Number of learnable part prototypes (default: 8)
        hidden_dim: Dimension for part representations (default: 256)
        output_dim: Output feature dimension (default: same as input_dim)
        dropout: Dropout probability (default: 0.1)
        diversity_loss: Whether to encourage prototype diversity (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        num_parts: int = 8,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        diversity_loss: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_parts = num_parts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.diversity_loss = diversity_loss

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable part prototypes: K vectors that learn to discover parts
        # Each prototype will attend to different spatial regions
        self.part_prototypes = nn.Parameter(torch.randn(num_parts, hidden_dim) * 0.02)

        # Temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1) * (hidden_dim ** -0.5))

        # Part feature refinement (per-part MLPs for discriminative features)
        self.part_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_parts)
        ])

        # Aggregation options:
        # Option 1: Concatenate all part features
        # Option 2: Attention pooling over parts
        # We use both: concat for explicit parts + attention for adaptive weighting

        # Part importance weights (learned)
        self.part_importance = nn.Sequential(
            nn.Linear(hidden_dim * num_parts, num_parts),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_parts),
            nn.Linear(hidden_dim * num_parts, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initialize prototypes with diverse vectors
        self._init_weights()

    def _init_weights(self):
        # Initialize prototypes orthogonally for diversity
        nn.init.orthogonal_(self.part_prototypes)

        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Compute diversity loss to encourage prototypes to discover different parts.

        Returns L_div = ||P @ P.T - I||^2 where P is normalized prototypes.
        This penalizes prototypes that are too similar to each other.
        """
        # Normalize prototypes
        P = F.normalize(self.part_prototypes, dim=-1)  # (K, D)

        # Compute similarity matrix
        similarity = P @ P.T  # (K, K)

        # Target is identity (each prototype only similar to itself)
        identity = torch.eye(self.num_parts, device=P.device)

        # Frobenius norm of difference
        diversity_loss = ((similarity - identity) ** 2).mean()

        return diversity_loss

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_diversity_loss: bool = False,
    ) -> torch.Tensor:
        """
        Apply part-based attention to extract discriminative part features.

        Args:
            x: Input feature map (B, C, H, W) or already flattened (B, N, C)
            return_attention: If True, also return part attention maps
            return_diversity_loss: If True, also return prototype diversity loss

        Returns:
            features: (B, output_dim) part-aware feature representation
            attention: (optional) (B, K, H, W) part attention maps
            diversity_loss: (optional) scalar diversity regularization loss
        """
        # Handle both 4D (spatial) and 3D (sequence) inputs
        if x.dim() == 4:
            B, C, H, W = x.shape
            spatial_shape = (H, W)
            # Flatten spatial dimensions: (B, C, H, W) → (B, H×W, C)
            x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H×W
        else:
            B, N, C = x.shape
            # Approximate spatial shape for visualization
            H = W = int(math.sqrt(N))
            spatial_shape = (H, W)

        N = x.shape[1]  # Number of spatial positions

        # Project features to hidden dimension
        features = self.input_proj(x)  # (B, N, hidden_dim)

        # Normalize features and prototypes for attention
        features_norm = F.normalize(features, dim=-1)  # (B, N, D)
        prototypes_norm = F.normalize(self.part_prototypes, dim=-1)  # (K, D)

        # Compute attention: each prototype attends to spatial positions
        # attention[k, n] = how much prototype k attends to position n
        # (K, D) @ (B, D, N) -> (B, K, N)
        attention_logits = torch.einsum('kd,bnd->bkn', prototypes_norm, features_norm)
        attention_logits = attention_logits * self.temperature.exp()  # Learnable temperature

        # Softmax over spatial positions (each prototype sums to 1)
        attention_weights = F.softmax(attention_logits, dim=-1)  # (B, K, N)

        # Extract part features: weighted sum of features for each prototype
        # (B, K, N) @ (B, N, D) -> (B, K, D)
        part_features = torch.einsum('bkn,bnd->bkd', attention_weights, features)

        # Refine each part's features with part-specific MLPs
        refined_parts = []
        for k in range(self.num_parts):
            refined = self.part_refinement[k](part_features[:, k, :])  # (B, D)
            refined_parts.append(refined)

        # Stack refined parts: (B, K, D)
        refined_parts = torch.stack(refined_parts, dim=1)

        # Concatenate all part features: (B, K*D)
        concat_parts = refined_parts.flatten(1)

        # Output projection
        output = self.output_proj(concat_parts)  # (B, output_dim)

        # Prepare return values
        if return_attention or return_diversity_loss:
            returns = [output]

            if return_attention:
                # Reshape attention to spatial format for visualization
                attn_spatial = attention_weights.view(B, self.num_parts, *spatial_shape)
                returns.append(attn_spatial)

            if return_diversity_loss:
                div_loss = self.compute_diversity_loss()
                returns.append(div_loss)

            return tuple(returns)

        return output


class SpatialCrossAttention(nn.Module):
    """
    Cross-attention between ROI spatial features and context spatial features.

    This simulates the ViT cross-attention mechanism but operates on CNN
    spatial feature maps instead of patch embeddings.

    Args:
        roi_dim: Channel dimension of ROI features
        context_dim: Channel dimension of context features
        hidden_dim: Dimension for Q, K, V projections
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        roi_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Q projection from ROI features
        self.q_proj = nn.Linear(roi_dim, hidden_dim)

        # K, V projections from context features
        self.k_proj = nn.Linear(context_dim, hidden_dim)
        self.v_proj = nn.Linear(context_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, roi_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: (B, C_roi, H, W) - ROI spatial features
            context_features: (B, C_ctx, H, W) - Context spatial features

        Returns:
            attended_features: (B, hidden_dim) - Attended context information
        """
        B, C_roi, H_roi, W_roi = roi_features.shape
        _, C_ctx, H_ctx, W_ctx = context_features.shape

        # Flatten spatial dimensions to create sequence of "patches"
        # ROI: (B, C, H, W) -> (B, H*W, C)
        roi_seq = roi_features.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Context: (B, C, H, W) -> (B, H*W, C)
        ctx_seq = context_features.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project to Q, K, V
        Q = self.q_proj(roi_seq)  # (B, H*W, hidden_dim)
        K = self.k_proj(ctx_seq)  # (B, H*W, hidden_dim)
        V = self.v_proj(ctx_seq)  # (B, H*W, hidden_dim)

        # Reshape for multi-head attention
        # (B, seq_len, hidden_dim) -> (B, num_heads, seq_len, head_dim)
        Q = Q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention: Q @ K^T / sqrt(d_k)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, H*W_roi, H*W_ctx)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values: attn @ V
        out = attn @ V  # (B, heads, H*W_roi, head_dim)

        # Concatenate heads and project
        out = out.transpose(1, 2).reshape(B, -1, self.hidden_dim)  # (B, H*W_roi, hidden_dim)
        out = self.out_proj(out)  # (B, H*W_roi, hidden_dim)

        # Global average pooling across spatial dimension
        out = out.mean(dim=1)  # (B, hidden_dim)

        return out


class MultiContextAttentionModule(nn.Module):
    """
    Multi-Context Environmental Attention Module (MCEAM) for CNN features.

    This module implements the attention-based fusion strategy from the
    reference paper, adapted for CNN features instead of ViT patch embeddings.

    Key Idea:
    ---------
    Instead of simply concatenating multi-scale features, we use cross-attention
    to let the model learn which environmental context is relevant for
    identifying each specific marine animal.

    For example:
    - Some species are only found on rocky substrates -> 3x context useful
    - Some species need depth cues from full image -> full-scale context useful
    - Attention learns to weight contexts based on what's informative

    Args:
        roi_dim: Feature dimension from ROI backbone (e.g., 1536 for ConvNeXt-Large)
        context_dims: List of feature dims for each context scale
        hidden_dim: Hidden dimension for attention and fusion
        num_heads: Number of attention heads per cross-attention layer
        output_dim: Output dimension after fusion
        dropout: Dropout probability
    """

    def __init__(
        self,
        roi_dim: int,
        context_dims: List[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        output_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.roi_dim = roi_dim
        self.context_dims = context_dims
        self.num_contexts = len(context_dims)

        # Cross-attention layers for each context scale
        self.context_attentions = nn.ModuleList([
            SpatialCrossAttention(
                roi_dim=roi_dim,
                context_dim=ctx_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for ctx_dim in context_dims
        ])

        # ROI global pooling and projection
        self.roi_pool = nn.AdaptiveAvgPool2d(1)
        self.roi_proj = nn.Linear(roi_dim, hidden_dim)

        # Fusion network
        # Input: ROI features + attended context features from each scale
        # (hidden_dim) + (num_contexts × hidden_dim)
        fusion_input_dim = hidden_dim * (1 + self.num_contexts)

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        context_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            roi_features: (B, C_roi, H, W) - Spatial features from ROI backbone
            context_features: List of (B, C_ctx, H, W) - Features from each context scale

        Returns:
            fused_features: (B, output_dim) - Attended and fused representation
        """
        # ROI global features
        roi_global = self.roi_pool(roi_features).flatten(1)  # (B, C_roi)
        roi_global = self.roi_proj(roi_global)  # (B, hidden_dim)

        # Cross-attend to each context scale
        attended_contexts = []
        for i, ctx_feats in enumerate(context_features):
            # Cross-attention: Q from ROI, K/V from context
            attended = self.context_attentions[i](roi_features, ctx_feats)  # (B, hidden_dim)
            attended_contexts.append(attended)

        # Concatenate ROI + all attended contexts
        all_features = [roi_global] + attended_contexts  # List of (B, hidden_dim)
        combined = torch.cat(all_features, dim=1)  # (B, hidden_dim × (1 + num_contexts))

        # Fuse through projection network
        fused = self.fusion(combined)  # (B, output_dim)

        return fused


class MultiContextAttentionModuleSimplified(nn.Module):
    """
    Simplified version of MCEAM that works with global pooled features.

    This is easier to integrate into existing code that only uses global
    features (not spatial feature maps). Instead of spatial cross-attention,
    it uses feature-level cross-attention similar to Transformer encoder.

    Use this if you don't want to modify backbone feature extraction.

    Args:
        roi_dim: Feature dimension from ROI backbone
        context_dims: List of feature dimensions for each context scale
        hidden_dim: Hidden dimension for attention
        num_heads: Number of attention heads
        output_dim: Output dimension after fusion
        dropout: Dropout probability
    """

    def __init__(
        self,
        roi_dim: int,
        context_dims: List[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        output_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.roi_dim = roi_dim
        self.context_dims = context_dims
        self.num_contexts = len(context_dims)

        # Project all features to same dimension
        self.roi_proj = nn.Linear(roi_dim, hidden_dim)
        self.context_projs = nn.ModuleList([
            nn.Linear(ctx_dim, hidden_dim) for ctx_dim in context_dims
        ])

        # Multi-head self-attention across all scales
        # Treats ROI and contexts as a sequence to attend over
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        roi_features: torch.Tensor,
        context_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            roi_features: (B, roi_dim) - Global features from ROI
            context_features: List of (B, ctx_dim) - Global features from contexts

        Returns:
            fused_features: (B, output_dim) - Fused representation
        """
        B = roi_features.shape[0]

        # Project all features to same dimension
        roi_proj = self.roi_proj(roi_features)  # (B, hidden_dim)
        ctx_projs = [proj(ctx) for proj, ctx in zip(self.context_projs, context_features)]

        # Stack into sequence: [ROI, ctx1, ctx2, ctx3, ...]
        # (B, num_scales, hidden_dim) where num_scales = 1 + num_contexts
        seq = torch.stack([roi_proj] + ctx_projs, dim=1)  # (B, 1+num_contexts, hidden_dim)

        # Self-attention across scales
        attn_out, _ = self.attention(seq, seq, seq)  # (B, 1+num_contexts, hidden_dim)
        seq = self.norm1(seq + attn_out)

        # Feedforward
        ffn_out = self.ffn(seq)
        seq = self.norm2(seq + ffn_out)

        # Use ROI position (index 0) as the fused representation
        # This is like the [CLS] token in ViT - aggregates information from all scales
        fused = seq[:, 0, :]  # (B, hidden_dim)

        # Project to output dimension
        fused = self.output_proj(fused)  # (B, output_dim)

        return fused
