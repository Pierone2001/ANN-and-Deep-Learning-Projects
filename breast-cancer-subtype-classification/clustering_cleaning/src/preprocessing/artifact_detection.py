"""Artifact detection for cartoon-like green stains and other artifacts.

This module detects non-realistic artifacts in histology images,
particularly highly saturated green regions.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import yaml

logger = logging.getLogger(__name__)


def detect_green_artifacts(
    image: np.ndarray,
    hue_range: Tuple[float, float] = (50, 80),
    saturation_threshold: float = 0.6,
    value_threshold: float = 0.3
) -> Tuple[np.ndarray, float]:
    """Detect cartoon-like green artifacts in an image.
    
    Args:
        image: RGB image array (H, W, 3) with values in [0, 255].
        hue_range: (min_hue, max_hue) in HSV for green detection (0-180 in OpenCV).
        saturation_threshold: Minimum saturation to consider (0-1).
        value_threshold: Minimum value (brightness) to consider (0-1).
        
    Returns:
        Tuple of (artifact_mask, artifact_ratio) where:
            - artifact_mask: Binary mask (True = artifact pixel)
            - artifact_ratio: Ratio of artifact pixels to total pixels
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Normalize to [0, 1] for thresholds
    h = hsv[:, :, 0].astype(np.float32)  # Hue: 0-180 in OpenCV
    s = hsv[:, :, 1].astype(np.float32) / 255.0  # Saturation: 0-255 -> 0-1
    v = hsv[:, :, 2].astype(np.float32) / 255.0  # Value: 0-255 -> 0-1
    
    # Create mask for green, highly saturated regions
    hue_mask = (h >= hue_range[0]) & (h <= hue_range[1])
    sat_mask = s >= saturation_threshold
    val_mask = v >= value_threshold
    
    artifact_mask = hue_mask & sat_mask & val_mask
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    artifact_mask = cv2.morphologyEx(
        artifact_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )
    artifact_mask = cv2.morphologyEx(
        artifact_mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    artifact_mask = artifact_mask.astype(bool)
    
    # Calculate ratio
    total_pixels = image.shape[0] * image.shape[1]
    artifact_pixels = artifact_mask.sum()
    artifact_ratio = artifact_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return artifact_mask, artifact_ratio


def compute_artifact_ratio_in_mask(
    image: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    **detection_kwargs
) -> Tuple[float, bool]:
    """Compute artifact ratio, optionally relative to tissue mask.
    
    Args:
        image: RGB image array.
        tissue_mask: Optional binary mask of tissue regions.
        **detection_kwargs: Additional arguments for detect_green_artifacts.
        
    Returns:
        Tuple of (artifact_ratio, has_cartoon_artifacts) where:
            - artifact_ratio: Ratio of artifact pixels
            - has_cartoon_artifacts: Boolean flag if ratio exceeds threshold
    """
    artifact_mask, artifact_ratio = detect_green_artifacts(image, **detection_kwargs)
    
    # If tissue mask provided, compute ratio relative to tissue area
    if tissue_mask is not None:
        tissue_pixels = tissue_mask.sum()
        if tissue_pixels > 0:
            artifact_in_tissue = (artifact_mask & tissue_mask).sum()
            artifact_ratio = artifact_in_tissue / tissue_pixels
        else:
            artifact_ratio = 0.0
    
    return artifact_ratio, artifact_mask


def process_image_artifacts(
    image_path: str,
    mask_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> dict:
    """Process a single image to detect artifacts.
    
    Args:
        image_path: Path to image file.
        mask_path: Optional path to tissue mask file.
        config_path: Optional path to config YAML with thresholds.
        
    Returns:
        Dictionary with:
            - artifact_ratio: Ratio of artifact pixels
            - has_cartoon_artifacts: Boolean flag
            - artifact_mask: Binary mask (if needed)
    """
    # Load config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        hue_range = tuple(config.get('hue_range', [50, 80]))
        saturation_threshold = config.get('saturation_threshold', 0.6)
        value_threshold = config.get('value_threshold', 0.3)
        artifact_threshold = config.get('artifact_threshold', 0.1)
    else:
        hue_range = (50, 80)
        saturation_threshold = 0.6
        value_threshold = 0.3
        artifact_threshold = 0.1
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        image_array = np.array(img)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return {
            'artifact_ratio': 0.0,
            'has_cartoon_artifacts': False
        }
    
    # Load mask if provided
    tissue_mask = None
    if mask_path and Path(mask_path).exists():
        try:
            mask_img = Image.open(mask_path).convert('L')
            tissue_mask = np.array(mask_img) > 128  # Binary threshold
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_path}: {e}")
    
    # Detect artifacts
    artifact_ratio, artifact_mask = compute_artifact_ratio_in_mask(
        image_array,
        tissue_mask,
        hue_range=hue_range,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold
    )
    
    has_cartoon_artifacts = artifact_ratio >= artifact_threshold
    
    return {
        'artifact_ratio': float(artifact_ratio),
        'has_cartoon_artifacts': has_cartoon_artifacts,
        'artifact_mask': artifact_mask  # Can be used for masking later
    }


def mask_artifacts_in_image(
    image: np.ndarray,
    artifact_mask: np.ndarray,
    method: str = 'black'
) -> np.ndarray:
    """Mask out artifact regions in an image.
    
    Args:
        image: RGB image array.
        artifact_mask: Binary mask of artifact regions.
        method: Method to mask ('black', 'mean', or 'inpaint').
        
    Returns:
        Image with artifacts masked out.
    """
    masked_image = image.copy()
    
    if method == 'black':
        masked_image[artifact_mask] = 0
    elif method == 'mean':
        mean_color = image[~artifact_mask].mean(axis=0) if (~artifact_mask).any() else [0, 0, 0]
        masked_image[artifact_mask] = mean_color
    elif method == 'inpaint':
        # Use OpenCV inpainting
        artifact_mask_uint8 = artifact_mask.astype(np.uint8) * 255
        masked_image = cv2.inpaint(image, artifact_mask_uint8, 3, cv2.INPAINT_TELEA)
    else:
        logger.warning(f"Unknown masking method: {method}, using 'black'")
        masked_image[artifact_mask] = 0
    
    return masked_image

