# FathomNet 2025 Default Configuration
"""
Default configuration values for model hyperparameters.

This module serves as the single source of truth for all default configuration
values. YAML experiment files should only override values that differ from
these defaults.

Usage:
    # Direct access to defaults
    from src.config import Config
    lr = Config.LEARNING_RATE

    # Get defaults as OmegaConf (for merging with YAML overrides)
    from src.config import get_default_config, load_config
    cfg = get_default_config()

    # Load YAML with defaults
    cfg = load_config("config/experiment-multiscale.yaml")
"""

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Optional


class Config:
    """Default configuration values."""

    # =========================================================================
    # PROJECT
    # =========================================================================
    PROJECT_NAME = "fathomnet-2025"
    SEED = 42

    # =========================================================================
    # PATHS (relative to project root)
    # =========================================================================
    DATA_ROOT = "."
    TAXONOMY_CSV = "data/taxonomy.csv"
    DISTANCE_MATRIX_CSV = "data/distance_matrix.csv"
    TRAIN_ANNOTATIONS_CSV = "train/annotations.csv"
    TRAIN_ROIS_DIR = "train/rois"
    TRAIN_IMAGES_DIR = "train/images"
    TRAIN_COCO_JSON = "data/dataset_train.json"
    TEST_COCO_JSON = "data/dataset_test.json"
    TEST_IMAGES_DIR = "test/images"
    TEST_ROIS_DIR = "test/rois"
    TEST_SOLUTION_CSV = "data/solution_kaggle_2025.csv"  # Ground truth for test set
    OUTPUT_DIR = "outputs"

    # =========================================================================
    # DATA
    # =========================================================================
    TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    EVAL_RATIO = 0.0  # No eval set - use Kaggle test set ground truth instead
    NUM_WORKERS = 8
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    SCALES = ["1x", "3x", "5x"]

    # =========================================================================
    # AUGMENTATION
    # =========================================================================
    RANDOM_RESIZED_CROP_SCALE = [0.7, 1.0]
    ROTATION_DEGREES = 15
    HORIZONTAL_FLIP = True
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.1
    COLOR_JITTER_HUE = 0.0
    AUGMENTATION_LEVEL = "medium"

    # =========================================================================
    # MODEL
    # =========================================================================
    BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"
    HIDDEN_DIM = 1024
    HEAD_DIM = 512
    DROPOUT = 0.3
    FUSION_METHOD = "concat"  # concat, attention, or weighted

    # =========================================================================
    # TRAINING
    # =========================================================================
    MAX_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    PRECISION = "16-mixed"
    BACKBONE_LR_SCALE = 0.1
    ACCUMULATE_GRAD_BATCHES = 2
    GRADIENT_CLIP_VAL = 1.0
    EARLY_STOPPING_PATIENCE = 10

    # =========================================================================
    # VIT FINE-TUNING (LLRD)
    # =========================================================================
    LAYER_DECAY = 1.0          # Layer-wise LR decay factor (1.0 = no decay; 0.65-0.75 typical for ViT)
    WARMUP_EPOCHS = 0          # Linear LR warmup epochs (3-5 typical for ViT fine-tuning)
    VIT_WEIGHT_DECAY = 0.05    # Weight decay for ViT (no WD on bias/LayerNorm)
    DROP_PATH_RATE = 0.0       # Stochastic depth rate for ViT backbone

    # =========================================================================
    # LOSS
    # =========================================================================
    ALPHA = 0.3  # Weight for distance term in TaxonomicDistanceLoss
    HIERARCHY_WEIGHTS = {
        "kingdom": 0.5,
        "phylum": 0.75,
        "class": 1.0,
        "order": 1.25,
        "family": 1.5,
        "genus": 2.0,
        "species": 2.5,
    }

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    MONITOR_METRIC = "val_loss"
    MONITOR_MODE = "min"

    # =========================================================================
    # INFERENCE
    # =========================================================================
    ENABLE_FALLBACK = True
    CONFIDENCE_THRESHOLD = 0.7


def get_default_config() -> DictConfig:
    """
    Generate an OmegaConf DictConfig from Config class defaults.

    Returns:
        DictConfig with all default values structured for training scripts.
    """
    cfg_dict = {
        "project": {
            "name": Config.PROJECT_NAME,
            "seed": Config.SEED,
        },
        "paths": {
            "data_root": Config.DATA_ROOT,
            "taxonomy_csv": Config.TAXONOMY_CSV,
            "distance_matrix_csv": Config.DISTANCE_MATRIX_CSV,
            "train_annotations_csv": Config.TRAIN_ANNOTATIONS_CSV,
            "train_rois_dir": Config.TRAIN_ROIS_DIR,
            "train_images_dir": Config.TRAIN_IMAGES_DIR,
            "train_full_image_dir": Config.TRAIN_IMAGES_DIR,  # Alias for full images
            "train_roi_root": Config.TRAIN_ROIS_DIR,  # Alias for ROI root
            "train_coco_json": Config.TRAIN_COCO_JSON,
            "test_coco_json": Config.TEST_COCO_JSON,
            "test_images_dir": Config.TEST_IMAGES_DIR,
            "test_full_image_dir": Config.TEST_IMAGES_DIR,  # Alias for full test images
            "test_rois_dir": Config.TEST_ROIS_DIR,
            "test_roi_root": Config.TEST_ROIS_DIR,  # Alias for test ROI root
            "test_solution_csv": Config.TEST_SOLUTION_CSV,
            "output_dir": Config.OUTPUT_DIR,
        },
        "data": {
            "taxonomy_csv": Config.TAXONOMY_CSV,
            "taxonomy_levels": Config.TAXONOMY_LEVELS,
            "train_annotations_csv": Config.TRAIN_ANNOTATIONS_CSV,
            "train_rois_dir": Config.TRAIN_ROIS_DIR,
            "distance_matrix_csv": Config.DISTANCE_MATRIX_CSV,
            "train_ratio": Config.TRAIN_RATIO,
            "train_split": Config.TRAIN_RATIO,  # Alias for compatibility
            "val_ratio": Config.VAL_RATIO,
            "eval_ratio": Config.EVAL_RATIO,
            "num_workers": Config.NUM_WORKERS,
            "batch_size": Config.BATCH_SIZE,
            "img_size": Config.IMAGE_SIZE,
            "scales": Config.SCALES,
            "use_multiscale": True,
            "augmentation_level": Config.AUGMENTATION_LEVEL,
        },
        "augmentation": {
            "random_resized_crop_scale": Config.RANDOM_RESIZED_CROP_SCALE,
            "rotation_degrees": Config.ROTATION_DEGREES,
            "horizontal_flip": Config.HORIZONTAL_FLIP,
            "color_jitter": {
                "brightness": Config.COLOR_JITTER_BRIGHTNESS,
                "contrast": Config.COLOR_JITTER_CONTRAST,
                "saturation": Config.COLOR_JITTER_SATURATION,
                "hue": Config.COLOR_JITTER_HUE,
            },
        },
        "model": {
            "backbone": Config.BACKBONE,
            "hidden_dim": Config.HIDDEN_DIM,
            "head_dim": Config.HEAD_DIM,
            "dropout": Config.DROPOUT,
            "fusion_method": Config.FUSION_METHOD,
        },
        "training": {
            "seed": Config.SEED,
            "max_epochs": Config.MAX_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "label_smoothing": Config.LABEL_SMOOTHING,
            "precision": Config.PRECISION,
            "backbone_lr_scale": Config.BACKBONE_LR_SCALE,
            "accumulate_grad_batches": Config.ACCUMULATE_GRAD_BATCHES,
            "gradient_clip_val": Config.GRADIENT_CLIP_VAL,
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
            "alpha": Config.ALPHA,
            "layer_decay": Config.LAYER_DECAY,
            "warmup_epochs": Config.WARMUP_EPOCHS,
            "vit_weight_decay": Config.VIT_WEIGHT_DECAY,
            "drop_path_rate": Config.DROP_PATH_RATE,
        },
        "callbacks": {
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
            "monitor_metric": Config.MONITOR_METRIC,
            "monitor_mode": Config.MONITOR_MODE,
        },
        "loss": {
            "alpha": Config.ALPHA,
            "hierarchy_weights": Config.HIERARCHY_WEIGHTS,
        },
        "inference": {
            "enable_fallback": Config.ENABLE_FALLBACK,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        },
    }
    return OmegaConf.create(cfg_dict)


def load_config(yaml_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration with defaults, optionally merging with a YAML file.

    The YAML file only needs to specify values that differ from defaults.
    All other values will be filled in from Config class.

    Args:
        yaml_path: Optional path to YAML file with overrides.
                   If None, returns just the defaults.

    Returns:
        DictConfig with merged configuration.

    Example:
        # Load just defaults
        cfg = load_config()

        # Load with experiment overrides
        cfg = load_config("config/experiment-multiscale.yaml")
    """
    defaults = get_default_config()

    if yaml_path is None:
        return defaults

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    overrides = OmegaConf.load(yaml_path)

    # Merge: overrides take precedence over defaults
    merged = OmegaConf.merge(defaults, overrides)

    return merged


# Legacy compatibility: DATA_ROOT for old scripts
DATA_ROOT = Config.DATA_ROOT
