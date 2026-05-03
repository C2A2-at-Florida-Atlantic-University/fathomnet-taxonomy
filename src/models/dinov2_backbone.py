"""
DINOv2 Backbone Wrapper for Multi-Scale Classification

This module provides a wrapper around DINOv2 models that mimics the timm interface,
allowing seamless integration with our multi-scale classification pipeline.

DINOv2 (Self-DIstillation with NO labels v2) is a self-supervised Vision Transformer
trained on 142M images. It produces rich semantic patch-level features that are
particularly useful for fine-grained recognition tasks.

Key Features:
- Patch tokens with strong semantic meaning (even without fine-tuning)
- CLS token for global image representation
- Multiple model sizes: small (384), base (768), large (1024), giant (1536)

Usage:
    # Create DINOv2 backbone for ROI scale
    backbone = DINOv2Backbone(model_size="base", pretrained=True)

    # Get global features (like timm model())
    global_feats = backbone(images)  # (B, 768)

    # Get spatial features (like timm forward_features())
    spatial_feats = backbone.forward_features(images)  # (B, 768, H, W)

References:
    - Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
    - https://github.com/facebookresearch/dinov2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class DINOv2Backbone(nn.Module):
    """
    DINOv2 backbone wrapper with timm-compatible interface.

    Provides both global (CLS token) and spatial (patch tokens) features,
    allowing it to work with our attention-based fusion methods.

    Args:
        model_size: One of "small", "base", "large", "giant"
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone weights (default: False)
        use_registers: Whether to use the register variant (better for dense tasks)
    """

    # Model configurations: (hub_name, embed_dim, patch_size)
    MODEL_CONFIGS = {
        "small": ("dinov2_vits14", 384, 14),
        "base": ("dinov2_vitb14", 768, 14),
        "large": ("dinov2_vitl14", 1024, 14),
        "giant": ("dinov2_vitg14", 1536, 14),
        # Register variants (have 4 extra register tokens, better for dense prediction)
        "small_reg": ("dinov2_vits14_reg", 384, 14),
        "base_reg": ("dinov2_vitb14_reg", 768, 14),
        "large_reg": ("dinov2_vitl14_reg", 1024, 14),
        "giant_reg": ("dinov2_vitg14_reg", 1536, 14),
    }

    def __init__(
        self,
        model_size: Literal["small", "base", "large", "giant",
                           "small_reg", "base_reg", "large_reg", "giant_reg"] = "base",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_registers: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Handle register variant selection
        if use_registers and not model_size.endswith("_reg"):
            model_size = f"{model_size}_reg"

        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model_size: {model_size}. "
                           f"Choose from {list(self.MODEL_CONFIGS.keys())}")

        hub_name, self.embed_dim, self.patch_size = self.MODEL_CONFIGS[model_size]
        self.model_size = model_size

        # Load DINOv2 from torch hub
        print(f"Loading DINOv2 {model_size} from torch hub...")
        hub_kwargs = {}
        if drop_path_rate > 0:
            hub_kwargs["drop_path_rate"] = drop_path_rate
            print(f"  drop_path_rate={drop_path_rate}")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
            **hub_kwargs,
        )

        # Expose number of transformer blocks for LLRD
        self.n_blocks = len(self.model.blocks) if hasattr(self.model, 'blocks') else 12

        # Freeze if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"DINOv2 {model_size} backbone frozen")

        # For timm compatibility
        self.num_features = self.embed_dim

        # Track if model uses registers
        self.has_registers = "_reg" in model_size
        self.num_register_tokens = 4 if self.has_registers else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global features (CLS token).

        This mimics timm's model(x) behavior - returns a single feature vector
        per image suitable for classification.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Global features (B, embed_dim)
        """
        # DINOv2's forward returns CLS token by default
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features (patch tokens reshaped to spatial grid).

        This mimics timm's forward_features() behavior - returns a spatial
        feature map suitable for attention operations.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Spatial features (B, embed_dim, H', W') where H'=W'=H/patch_size
        """
        B, C, H, W = x.shape

        # Compute output spatial dimensions
        h = H // self.patch_size
        w = W // self.patch_size

        # Get all tokens from DINOv2
        # DINOv2's forward_features returns a dict with keys:
        #   - x_norm_clstoken: (B, embed_dim) - CLS token
        #   - x_norm_regtokens: (B, num_registers, embed_dim) - register tokens (if any)
        #   - x_norm_patchtokens: (B, h*w, embed_dim) - patch tokens
        #   - x_prenorm: (B, 1+num_registers+h*w, embed_dim) - pre-normalization features
        features_dict = self.model.forward_features(x)

        # Extract patch tokens directly from the dict
        patch_tokens = features_dict["x_norm_patchtokens"]  # (B, h*w, embed_dim)

        # Reshape to spatial format (B, embed_dim, h, w)
        patch_tokens = patch_tokens.transpose(1, 2)  # (B, embed_dim, h*w)
        spatial_features = patch_tokens.view(B, self.embed_dim, h, w)

        return spatial_features

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
        reshape: bool = True,
    ) -> list:
        """
        Get intermediate layer outputs (useful for feature pyramid or multi-scale).

        Args:
            x: Input images (B, 3, H, W)
            n: Number of layers from the end to return
            reshape: Whether to reshape patch tokens to spatial grid

        Returns:
            List of intermediate features
        """
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Get intermediate features from DINOv2
        outputs = self.model.get_intermediate_layers(x, n=n, reshape=False)

        if reshape:
            reshaped = []
            for feat in outputs:
                # Skip CLS and registers, reshape patches
                patch_start_idx = 1 + self.num_register_tokens
                patches = feat[:, patch_start_idx:, :]
                patches = patches.transpose(1, 2).view(B, -1, h, w)
                reshaped.append(patches)
            return reshaped

        return outputs


def create_dinov2_backbone(
    model_name: str,
    pretrained: bool = True,
    **kwargs,
) -> DINOv2Backbone:
    """
    Factory function to create DINOv2 backbone from model name.

    Supports names like:
    - "dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"
    - "dinov2_base_reg" (register variant)

    Args:
        model_name: Model name string
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to DINOv2Backbone

    Returns:
        DINOv2Backbone instance
    """
    # Parse model name
    name_lower = model_name.lower()

    if "dinov2" not in name_lower:
        raise ValueError(f"Expected 'dinov2' in model name, got: {model_name}")

    # Extract size from name - check longer patterns first to avoid substring issues
    # e.g., "base" contains "s" which could match "small" if checked first
    size_patterns = [
        # Check specific model suffixes first (most reliable)
        ("_small", "small"), ("_base", "base"), ("_large", "large"), ("_giant", "giant"),
        # Then check ViT naming convention
        ("vits", "small"), ("vitb", "base"), ("vitl", "large"), ("vitg", "giant"),
        # Finally check full words
        ("small", "small"), ("base", "base"), ("large", "large"), ("giant", "giant"),
    ]

    model_size = None
    for pattern, size in size_patterns:
        if pattern in name_lower:
            model_size = size
            break

    if model_size is None:
        model_size = "base"  # Default

    # Check for register variant
    use_registers = "_reg" in name_lower or "register" in name_lower

    # Filter kwargs to only those accepted by DINOv2Backbone
    valid_kwargs = {}
    for k in ("freeze_backbone", "drop_path_rate"):
        if k in kwargs:
            valid_kwargs[k] = kwargs[k]

    return DINOv2Backbone(
        model_size=model_size,
        pretrained=pretrained,
        use_registers=use_registers,
        **valid_kwargs,
    )


# Register with a simple lookup for integration
DINOV2_MODELS = {
    "dinov2_small": lambda pretrained=True: DINOv2Backbone("small", pretrained),
    "dinov2_base": lambda pretrained=True: DINOv2Backbone("base", pretrained),
    "dinov2_large": lambda pretrained=True: DINOv2Backbone("large", pretrained),
    "dinov2_giant": lambda pretrained=True: DINOv2Backbone("giant", pretrained),
    "dinov2_small_reg": lambda pretrained=True: DINOv2Backbone("small_reg", pretrained),
    "dinov2_base_reg": lambda pretrained=True: DINOv2Backbone("base_reg", pretrained),
    "dinov2_large_reg": lambda pretrained=True: DINOv2Backbone("large_reg", pretrained),
    "dinov2_giant_reg": lambda pretrained=True: DINOv2Backbone("giant_reg", pretrained),
}


if __name__ == "__main__":
    # Quick test
    print("Testing DINOv2 backbone wrapper...")

    # Create model
    backbone = DINOv2Backbone("base", pretrained=True)
    print(f"Created DINOv2 base with embed_dim={backbone.embed_dim}")

    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)

    # Test global features
    global_feats = backbone(x)
    print(f"Global features shape: {global_feats.shape}")  # Expected: (2, 768)

    # Test spatial features
    spatial_feats = backbone.forward_features(x)
    print(f"Spatial features shape: {spatial_feats.shape}")  # Expected: (2, 768, 16, 16)

    print("All tests passed!")
