"""Dataset discovery and validation utilities.

This module provides functions to discover, list, and validate
image-mask-label triplets in the Grumpy Doctogres dataset.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def discover_images(data_dir: str) -> List[str]:
    """Discover all image files in the dataset.
    
    Args:
        data_dir: Path to the directory containing images (e.g., 'train_data').
        
    Returns:
        List of image file paths (relative to data_dir).
    """
    data_path = Path(data_dir)
    image_files = sorted([f.name for f in data_path.glob("img_*.png")])
    logger.info(f"Found {len(image_files)} image files in {data_dir}")
    return image_files


def discover_masks(data_dir: str) -> List[str]:
    """Discover all mask files in the dataset.
    
    Args:
        data_dir: Path to the directory containing masks (e.g., 'train_data').
        
    Returns:
        List of mask file paths (relative to data_dir).
    """
    data_path = Path(data_dir)
    mask_files = sorted([f.name for f in data_path.glob("mask_*.png")])
    logger.info(f"Found {len(mask_files)} mask files in {data_dir}")
    return mask_files


def load_labels(csv_path: str) -> pd.DataFrame:
    """Load labels from CSV file.
    
    Args:
        csv_path: Path to the labels CSV file.
        
    Returns:
        DataFrame with columns 'sample_index' and 'label'.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} labels from {csv_path}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    return df


def get_image_id_from_filename(filename: str) -> str:
    """Extract image ID from filename.
    
    Args:
        filename: Image filename (e.g., 'img_0000.png' or 'mask_0000.png').
        
    Returns:
        Image ID (e.g., 'img_0000').
    """
    # Remove extension and return
    base = Path(filename).stem
    # For masks, convert mask_0000 -> img_0000
    if base.startswith("mask_"):
        base = base.replace("mask_", "img_")
    return base


def match_images_masks_labels(
    data_dir: str,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Match images with their corresponding masks and labels.
    
    Args:
        data_dir: Path to directory containing images and masks.
        labels_df: DataFrame with labels (columns: sample_index, label).
        
    Returns:
        DataFrame with columns:
            - image_id: Image identifier (e.g., 'img_0000')
            - image_path: Path to image file
            - mask_path: Path to mask file (or None if missing)
            - label: Class label
            - has_mask: Boolean indicating if mask exists
    """
    images = discover_images(data_dir)
    masks = discover_masks(data_dir)
    
    # Create dictionaries for quick lookup
    mask_dict = {get_image_id_from_filename(m): m for m in masks}
    
    # Build matched dataframe
    matched_data = []
    missing_masks = []
    missing_labels = []
    
    for img_file in images:
        image_id = get_image_id_from_filename(img_file)
        mask_file = mask_dict.get(image_id)
        
        # Find label
        label_row = labels_df[labels_df['sample_index'] == img_file]
        if label_row.empty:
            missing_labels.append(img_file)
            label = None
        else:
            label = label_row.iloc[0]['label']
        
        matched_data.append({
            'image_id': image_id,
            'image_path': str(Path(data_dir) / img_file),
            'mask_path': str(Path(data_dir) / mask_file) if mask_file else None,
            'label': label,
            'has_mask': mask_file is not None
        })
        
        if not mask_file:
            missing_masks.append(img_file)
    
    df = pd.DataFrame(matched_data)
    
    # Log warnings
    if missing_masks:
        logger.warning(f"Found {len(missing_masks)} images without masks")
        logger.debug(f"Missing masks: {missing_masks[:10]}...")
    
    if missing_labels:
        logger.warning(f"Found {len(missing_labels)} images without labels")
        logger.debug(f"Missing labels: {missing_labels[:10]}...")
    
    return df


def validate_image_mask_pair(image_path: str, mask_path: Optional[str]) -> Tuple[bool, Dict]:
    """Validate that an image and its mask are properly aligned.
    
    Args:
        image_path: Path to image file.
        mask_path: Path to mask file (can be None).
        
    Returns:
        Tuple of (is_valid, info_dict) where info_dict contains:
            - width, height: Image dimensions
            - mask_width, mask_height: Mask dimensions (if mask exists)
            - sizes_match: Whether image and mask have same size
    """
    info = {}
    
    try:
        img = Image.open(image_path)
        info['width'] = img.width
        info['height'] = img.height
        info['mode'] = img.mode
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return False, info
    
    if mask_path and os.path.exists(mask_path):
        try:
            mask = Image.open(mask_path)
            info['mask_width'] = mask.width
            info['mask_height'] = mask.height
            info['mask_mode'] = mask.mode
            info['sizes_match'] = (img.width == mask.width and img.height == mask.height)
        except Exception as e:
            logger.error(f"Failed to open mask {mask_path}: {e}")
            info['sizes_match'] = False
            return False, info
    else:
        info['sizes_match'] = None
        info['mask_width'] = None
        info['mask_height'] = None
    
    is_valid = info.get('sizes_match', None) is not False
    return is_valid, info


def get_dataset_summary(data_dir: str, labels_csv: str) -> Dict:
    """Generate a comprehensive summary of the dataset.
    
    Args:
        data_dir: Path to directory containing images and masks.
        labels_csv: Path to labels CSV file.
        
    Returns:
        Dictionary with dataset statistics and information.
    """
    labels_df = load_labels(labels_csv)
    matched_df = match_images_masks_labels(data_dir, labels_df)
    
    # Get image dimensions
    dimensions = []
    for _, row in matched_df.iterrows():
        try:
            img = Image.open(row['image_path'])
            dimensions.append((img.width, img.height))
        except Exception as e:
            logger.warning(f"Could not read dimensions for {row['image_path']}: {e}")
    
    summary = {
        'total_images': len(matched_df),
        'images_with_masks': matched_df['has_mask'].sum(),
        'images_without_masks': (~matched_df['has_mask']).sum(),
        'images_with_labels': matched_df['label'].notna().sum(),
        'class_distribution': matched_df['label'].value_counts().to_dict() if 'label' in matched_df.columns else {},
        'unique_classes': matched_df['label'].unique().tolist() if 'label' in matched_df.columns else [],
        'image_dimensions': dimensions,
        'unique_dimensions': list(set(dimensions)) if dimensions else [],
    }
    
    return summary

