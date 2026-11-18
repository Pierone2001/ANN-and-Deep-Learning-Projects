"""Train/validation split utilities.

This module implements stratified splitting at the image level
to ensure no data leakage between train and validation sets.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_image_split(
    metadata_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify_column: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Perform stratified split at image level.
    
    Args:
        metadata_df: DataFrame with image metadata, must have 'image_id' and label column.
        test_size: Fraction for test set (if val_size=None, this becomes val_size).
        val_size: Fraction for validation set (if None, only train/val split).
        random_state: Random seed.
        stratify_column: Column name to stratify on (default: 'label').
        
    Returns:
        Tuple of (train_df, val_df, test_df) where test_df is None if val_size is None.
    """
    if stratify_column not in metadata_df.columns:
        logger.warning(
            f"Stratify column '{stratify_column}' not found, "
            "performing random split without stratification"
        )
        stratify = None
    else:
        stratify = metadata_df[stratify_column]
    
    # If val_size is None, do simple train/val split
    if val_size is None:
        train_df, val_df = train_test_split(
            metadata_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")
        return train_df, val_df, None
    
    # Otherwise, do train/val/test split
    # First split: train vs (val+test)
    train_size = 1.0 - test_size - val_size
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=stratify
    )
    
    # Second split: val vs test
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=temp_df[stratify_column] if stratify is not None else None
    )
    
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_df, val_df, test_df


def create_patch_splits_from_image_splits(
    train_images_df: pd.DataFrame,
    val_images_df: pd.DataFrame,
    test_images_df: Optional[pd.DataFrame],
    patches_metadata_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Create patch-level splits from image-level splits.
    
    Args:
        train_images_df: DataFrame with train image IDs.
        val_images_df: DataFrame with val image IDs.
        test_images_df: Optional DataFrame with test image IDs.
        patches_metadata_df: DataFrame with patch metadata (must have 'source_image_id').
        
    Returns:
        Tuple of (train_patches_df, val_patches_df, test_patches_df).
    """
    train_image_ids = set(train_images_df['image_id'].values)
    val_image_ids = set(val_images_df['image_id'].values)
    test_image_ids = set(test_images_df['image_id'].values) if test_images_df is not None else set()
    
    # Filter patches by source image
    train_patches = patches_metadata_df[
        patches_metadata_df['source_image_id'].isin(train_image_ids)
    ].copy()
    
    val_patches = patches_metadata_df[
        patches_metadata_df['source_image_id'].isin(val_image_ids)
    ].copy()
    
    test_patches = None
    if test_image_ids:
        test_patches = patches_metadata_df[
            patches_metadata_df['source_image_id'].isin(test_image_ids)
        ].copy()
    
    logger.info(
        f"Patch splits: {len(train_patches)} train, "
        f"{len(val_patches)} val, "
        f"{len(test_patches) if test_patches is not None else 0} test"
    )
    
    return train_patches, val_patches, test_patches


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    output_dir: str,
    prefix: str = ''
) -> None:
    """Save split DataFrames to CSV files.
    
    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Optional test set DataFrame.
        output_dir: Directory to save CSV files.
        prefix: Optional prefix for filenames (e.g., 'images_' or 'patches_').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"{prefix}train.csv"
    val_path = output_dir / f"{prefix}val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"Saved train split to {train_path}")
    logger.info(f"Saved val split to {val_path}")
    
    if test_df is not None:
        test_path = output_dir / f"{prefix}test.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved test split to {test_path}")

