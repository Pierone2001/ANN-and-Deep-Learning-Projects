"""Patch extraction pipeline for histology images.

This module handles image normalization, cropping, and patch extraction
from histology images with masks.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
from PIL import Image
import yaml
import cv2

logger = logging.getLogger(__name__)


def load_image_and_mask(
    image_path: str,
    mask_path: Optional[str]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load image and mask, ensuring they are aligned.
    
    Args:
        image_path: Path to image file.
        mask_path: Optional path to mask file.
        
    Returns:
        Tuple of (image_array, mask_array) where arrays are RGB and grayscale respectively.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    image_array = np.array(img)
    
    # Load mask if provided
    mask_array = None
    if mask_path and Path(mask_path).exists():
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        
        # Ensure same size
        if image_array.shape[:2] != mask_array.shape:
            logger.warning(
                f"Size mismatch: image {image_array.shape[:2]}, "
                f"mask {mask_array.shape}. Resizing mask."
            )
            mask_img_resized = mask_img.resize(
                (image_array.shape[1], image_array.shape[0]),
                Image.Resampling.NEAREST
            )
            mask_array = np.array(mask_img_resized)
    
    return image_array, mask_array


def get_mask_bbox(
    mask: np.ndarray,
    margin: int = 0
) -> Tuple[int, int, int, int]:
    """Get bounding box of mask with optional margin.
    
    Args:
        mask: Binary mask array.
        margin: Pixels to add as margin around bbox.
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max).
    """
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # No foreground pixels, return full image
        return 0, 0, mask.shape[1], mask.shape[0]
    
    y_min, y_max = coords[0].min(), coords[0].max() + 1
    x_min, x_max = coords[1].min(), coords[1].max() + 1
    
    # Add margin
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(mask.shape[0], y_max + margin)
    x_max = min(mask.shape[1], x_max + margin)
    
    return x_min, y_min, x_max, y_max


def crop_to_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    margin: int = 50,
    min_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Crop image and mask to bounding box of mask.
    
    Args:
        image: Image array (H, W, 3).
        mask: Optional mask array (H, W).
        margin: Margin in pixels around bbox.
        min_size: Minimum size (width, height) to keep. If bbox is smaller, return original.
        
    Returns:
        Tuple of (cropped_image, cropped_mask).
    """
    if mask is None:
        return image, None
    
    x_min, y_min, x_max, y_max = get_mask_bbox(mask, margin)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # If bbox too small, return original
    if bbox_width < min_size[0] or bbox_height < min_size[1]:
        logger.debug("Bbox too small, keeping original size")
        return image, mask
    
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_mask


def normalize_image(
    image: np.ndarray,
    method: str = 'imagenet',
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """Normalize image for training.
    
    Args:
        image: Image array (H, W, 3) with values in [0, 255].
        method: Normalization method ('imagenet', 'custom', or 'none').
        mean: Custom mean per channel (if method='custom').
        std: Custom std per channel (if method='custom').
        
    Returns:
        Normalized image array with values typically in [0, 1] or standardized.
    """
    image_float = image.astype(np.float32)
    
    if method == 'imagenet':
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_float = image_float / 255.0
        image_float = (image_float - mean) / std
    elif method == 'custom' and mean is not None and std is not None:
        image_float = image_float / 255.0
        image_float = (image_float - np.array(mean)) / np.array(std)
    elif method == 'none':
        # Just convert to [0, 1]
        image_float = image_float / 255.0
    else:
        logger.warning(f"Unknown normalization method: {method}, using 'none'")
        image_float = image_float / 255.0
    
    return image_float


def extract_patches(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_ratio: float = 0.1,
    artifact_mask: Optional[np.ndarray] = None,
    max_artifact_ratio: float = 0.5
) -> List[Dict]:
    """Extract patches from an image.
    
    Args:
        image: Image array (H, W, 3).
        mask: Optional tissue mask (H, W).
        patch_size: Size of square patches.
        stride: Stride for patch extraction (overlapping if < patch_size).
        min_tissue_ratio: Minimum ratio of tissue pixels in patch to keep it.
        artifact_mask: Optional artifact mask to check.
        max_artifact_ratio: Maximum ratio of artifact pixels allowed.
        
    Returns:
        List of dictionaries, each containing:
            - 'patch': Patch image array
            - 'x': X coordinate of patch top-left
            - 'y': Y coordinate of patch top-left
            - 'tissue_ratio': Ratio of tissue pixels in patch
            - 'artifact_ratio': Ratio of artifact pixels in patch
            - 'valid': Whether patch passes all filters
    """
    patches = []
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch_img = image[y:y + patch_size, x:x + patch_size]
            
            # Check tissue ratio
            tissue_ratio = 1.0
            if mask is not None:
                patch_mask = mask[y:y + patch_size, x:x + patch_size]
                tissue_pixels = (patch_mask > 0).sum()
                tissue_ratio = tissue_pixels / (patch_size * patch_size)
            
            # Check artifact ratio
            artifact_ratio = 0.0
            if artifact_mask is not None:
                patch_artifact = artifact_mask[y:y + patch_size, x:x + patch_size]
                artifact_pixels = patch_artifact.sum()
                artifact_ratio = artifact_pixels / (patch_size * patch_size)
            
            # Determine if valid
            valid = (
                tissue_ratio >= min_tissue_ratio and
                artifact_ratio <= max_artifact_ratio
            )
            
            patches.append({
                'patch': patch_img,
                'x': x,
                'y': y,
                'tissue_ratio': tissue_ratio,
                'artifact_ratio': artifact_ratio,
                'valid': valid
            })
    
    return patches


def process_image_to_patches(
    image_path: str,
    mask_path: Optional[str],
    output_dir: str,
    image_id: str,
    class_label: str,
    config_path: Optional[str] = None,
    artifact_info: Optional[Dict] = None,
    is_orc: bool = False,
    exclude_orc: bool = True
) -> List[Dict]:
    """Process a single image and extract patches.
    
    Args:
        image_path: Path to image file.
        mask_path: Optional path to mask file.
        output_dir: Directory to save patches.
        image_id: Unique identifier for the image.
        class_label: Class label for the image.
        config_path: Optional path to config YAML.
        artifact_info: Optional dict with 'artifact_mask' and 'artifact_ratio'.
        is_orc: Whether image is marked as orc.
        exclude_orc: Whether to skip orc images.
        
    Returns:
        List of patch metadata dictionaries.
    """
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Skip orc images if configured
    if is_orc and exclude_orc:
        logger.debug(f"Skipping orc image: {image_id}")
        return []
    
    # Config parameters
    patch_size = config.get('patch_size', 256)
    stride = config.get('stride', 256)
    min_tissue_ratio = config.get('min_tissue_ratio', 0.1)
    max_artifact_ratio = config.get('max_artifact_ratio', 0.5)
    crop_margin = config.get('crop_margin', 50)
    crop_min_size = tuple(config.get('crop_min_size', [256, 256]))
    normalize_method = config.get('normalize_method', 'none')
    mask_artifacts = config.get('mask_artifacts_in_patches', False)
    
    # Load image and mask
    image, mask = load_image_and_mask(image_path, mask_path)
    
    # Crop to mask if available
    if mask is not None:
        image, mask = crop_to_mask(image, mask, crop_margin, crop_min_size)
    
    # Handle artifacts
    artifact_mask = None
    if artifact_info and artifact_info.get('artifact_mask') is not None:
        artifact_mask = artifact_info['artifact_mask']
        if mask_artifacts:
            # Import here to avoid circular dependency
            try:
                from .artifact_detection import mask_artifacts_in_image
            except ImportError:
                from src.preprocessing.artifact_detection import mask_artifacts_in_image
            image = mask_artifacts_in_image(image, artifact_mask, method='black')
    
    # Extract patches
    patches = extract_patches(
        image,
        mask,
        patch_size,
        stride,
        min_tissue_ratio,
        artifact_mask,
        max_artifact_ratio
    )
    
    # Save patches and collect metadata
    output_dir = Path(output_dir) / class_label
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patch_metadata = []
    for i, patch_data in enumerate(patches):
        if not patch_data['valid']:
            continue
        
        patch_id = f"{image_id}__patch_{patch_data['x']}_{patch_data['y']}"
        patch_path = output_dir / f"{patch_id}.png"
        
        # Save patch
        patch_img = Image.fromarray(patch_data['patch'].astype(np.uint8))
        patch_img.save(patch_path)
        
        patch_metadata.append({
            'patch_id': patch_id,
            'source_image_id': image_id,
            'class_label': class_label,
            'patch_path': str(patch_path),
            'x': patch_data['x'],
            'y': patch_data['y'],
            'is_orc': is_orc,
            'has_cartoon_artifacts': artifact_info.get('has_cartoon_artifacts', False) if artifact_info else False,
            'artifact_ratio': artifact_info.get('artifact_ratio', 0.0) if artifact_info else 0.0,
            'tissue_mask_ratio': patch_data['tissue_ratio']
        })
    
    return patch_metadata

