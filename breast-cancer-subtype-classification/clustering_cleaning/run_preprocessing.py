#!/usr/bin/env python3
"""Main preprocessing pipeline script.

This script runs the complete preprocessing pipeline:
1. Dataset discovery
2. Embedding extraction and clustering
3. Artifact detection
4. Patch extraction
5. Train/val splitting
"""

import warnings
# Suppress OpenMP warning on macOS (common false positive)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threadpoolctl')

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add src to path (relative to this script)
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_utils.dataset_discovery import (
    match_images_masks_labels,
    get_dataset_summary,
    validate_image_mask_pair
)
from src.preprocessing.orc_filter import (
    extract_and_save_embeddings,
    load_embeddings,
    cluster_embeddings,
    generate_all_cluster_montages,
    mark_orc_images
)
from src.preprocessing.artifact_detection import process_image_artifacts
from src.preprocessing.patching_pipeline import process_image_to_patches
from src.preprocessing.splits import (
    stratified_image_split,
    create_patch_splits_from_image_splits,
    save_splits
)


def setup_logging(logs_dir: str, log_level: str = "INFO"):
    """Setup logging to both file and console."""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_path / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def run_dataset_discovery(config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Step 1: Discover and validate dataset."""
    logger.info("=" * 60)
    logger.info("Step 1: Dataset Discovery")
    logger.info("=" * 60)
    
    data_dir = config['data_dir']
    labels_csv = config['labels_csv']
    
    # Get summary
    summary = get_dataset_summary(data_dir, labels_csv)
    logger.info(f"Dataset summary: {summary}")
    
    # Match images, masks, labels
    metadata_df = match_images_masks_labels(data_dir, pd.read_csv(labels_csv))
    
    # Add image dimensions
    dimensions = []
    for _, row in metadata_df.iterrows():
        try:
            from PIL import Image
            img = Image.open(row['image_path'])
            dimensions.append({'width': img.width, 'height': img.height})
        except Exception as e:
            logger.warning(f"Could not read {row['image_path']}: {e}")
            dimensions.append({'width': None, 'height': None})
    
    dims_df = pd.DataFrame(dimensions)
    metadata_df = pd.concat([metadata_df, dims_df], axis=1)
    
    # Save initial metadata
    metadata_path = Path(config['metadata_dir']) / config.get('images_metadata_path', 'images_metadata.csv').split('/')[-1]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved initial metadata to {metadata_path}")
    
    return metadata_df


def run_embedding_extraction(config: Dict, metadata_df: pd.DataFrame, logger: logging.Logger) -> None:
    """Step 2: Extract image embeddings."""
    logger.info("=" * 60)
    logger.info("Step 2: Embedding Extraction")
    logger.info("=" * 60)
    
    embeddings_path = config['embeddings_path']
    
    # Check if embeddings already exist
    if Path(embeddings_path).exists():
        logger.info(f"Embeddings already exist at {embeddings_path}, skipping extraction")
        return
    
    image_paths = metadata_df['image_path'].tolist()
    batch_size = config.get('batch_size', 32)
    
    extract_and_save_embeddings(image_paths, embeddings_path, batch_size)


def run_clustering(config: Dict, metadata_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Step 3: Cluster embeddings and mark orc images."""
    logger.info("=" * 60)
    logger.info("Step 3: Clustering and Orc Detection")
    logger.info("=" * 60)
    
    embeddings_path = config['embeddings_path']
    orc_config_path = config['orc_filter_config']
    
    # Load orc config
    with open(orc_config_path, 'r') as f:
        orc_config = yaml.safe_load(f)
    
    # Load embeddings
    embeddings, image_paths = load_embeddings(embeddings_path)
    
    # Cluster
    n_clusters = orc_config.get('n_clusters', 10)
    cluster_labels = cluster_embeddings(embeddings, n_clusters, orc_config.get('random_state', 42))
    
    # Add cluster IDs to metadata
    # Create mapping from image_path to cluster_id
    path_to_cluster = dict(zip(image_paths, cluster_labels))
    metadata_df['cluster_id'] = metadata_df['image_path'].map(path_to_cluster)
    
    # Generate montages
    montages_dir = config['cluster_montages_dir']
    generate_all_cluster_montages(
        embeddings_path,
        montages_dir,
        n_clusters,
        n_samples_per_cluster=30
    )
    logger.info(f"Generated cluster montages in {montages_dir}")
    logger.info("Please inspect montages and update orc_clusters in configs/orc_filter.yaml")
    
    # Mark orc images
    metadata_df = mark_orc_images(metadata_df, orc_config_path)
    
    return metadata_df


def run_artifact_detection(config: Dict, metadata_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Step 4: Detect artifacts in images."""
    logger.info("=" * 60)
    logger.info("Step 4: Artifact Detection")
    logger.info("=" * 60)
    
    artifacts_config_path = config['artifacts_config']
    
    artifact_ratios = []
    has_artifacts = []
    
    for idx, row in metadata_df.iterrows():
        if idx % 40 == 0 or idx == len(metadata_df) - 1:
            logger.info(f"Processing artifacts: {idx + 1}/{len(metadata_df)} ({(idx + 1)*100//len(metadata_df)}%)")
        
        artifact_info = process_image_artifacts(
            row['image_path'],
            row.get('mask_path'),
            artifacts_config_path
        )
        
        artifact_ratios.append(artifact_info['artifact_ratio'])
        has_artifacts.append(artifact_info['has_cartoon_artifacts'])
    
    metadata_df['artifact_ratio'] = artifact_ratios
    metadata_df['has_cartoon_artifacts'] = has_artifacts
    
    logger.info(f"Found {sum(has_artifacts)} images with cartoon artifacts")
    
    return metadata_df


def run_patch_extraction(config: Dict, metadata_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Step 5: Extract patches from images."""
    logger.info("=" * 60)
    logger.info("Step 5: Patch Extraction")
    logger.info("=" * 60)
    
    patching_config_path = config['patching_config']
    artifacts_config_path = config['artifacts_config']
    output_dir = Path(config['output_dir']) / 'patches'
    
    # Load orc config to check exclude_orc flag
    with open(config['orc_filter_config'], 'r') as f:
        orc_config = yaml.safe_load(f)
    exclude_orc = orc_config.get('exclude_orc_images', True)
    
    all_patches_metadata = []
    
    for idx, row in metadata_df.iterrows():
        if idx % 50 == 0:
            logger.info(f"Extracting patches: {idx}/{len(metadata_df)}")
        
        # Prepare artifact info
        artifact_info = {
            'artifact_ratio': row.get('artifact_ratio', 0.0),
            'has_cartoon_artifacts': row.get('has_cartoon_artifacts', False),
            'artifact_mask': None  # Could load if needed, but memory intensive
        }
        
        patch_metadata = process_image_to_patches(
            row['image_path'],
            row.get('mask_path'),
            output_dir,
            row['image_id'],
            row['label'],
            patching_config_path,
            artifact_info,
            row.get('is_orc', False),
            exclude_orc
        )
        
        all_patches_metadata.extend(patch_metadata)
    
    patches_df = pd.DataFrame(all_patches_metadata)
    
    # Save patches metadata
    patches_metadata_path = Path(config['metadata_dir']) / config.get('patches_metadata_path', 'patches_metadata.csv').split('/')[-1]
    patches_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    patches_df.to_csv(patches_metadata_path, index=False)
    logger.info(f"Extracted {len(patches_df)} patches")
    logger.info(f"Saved patches metadata to {patches_metadata_path}")
    
    return patches_df


def run_splits(config: Dict, metadata_df: pd.DataFrame, patches_df: pd.DataFrame, logger: logging.Logger) -> None:
    """Step 6: Create train/val splits."""
    logger.info("=" * 60)
    logger.info("Step 6: Train/Val Splitting")
    logger.info("=" * 60)
    
    val_size = config.get('val_size', 0.2)
    test_size = config.get('test_size', 0.0)
    random_state = config.get('random_state', 42)
    
    # Image-level split
    train_images, val_images, test_images = stratified_image_split(
        metadata_df,
        test_size=val_size if test_size == 0 else test_size,
        val_size=test_size if test_size > 0 else None,
        random_state=random_state
    )
    
    # Save image splits
    save_splits(
        train_images,
        val_images,
        test_images,
        config['metadata_dir'],
        prefix='images_'
    )
    
    # Patch-level splits
    train_patches, val_patches, test_patches = create_patch_splits_from_image_splits(
        train_images,
        val_images,
        test_images,
        patches_df
    )
    
    # Save patch splits
    save_splits(
        train_patches,
        val_patches,
        test_patches,
        config['metadata_dir'],
        prefix='patches_'
    )


def main():
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/main_preprocessing.yaml',
        help='Path to main config YAML file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config['logs_dir'], args.log_level)
    logger.info("Starting preprocessing pipeline")
    logger.info(f"Config: {args.config}")
    
    try:
        # Step 1: Dataset discovery
        if config.get('run_dataset_discovery', True):
            metadata_df = run_dataset_discovery(config, logger)
        else:
            # Load existing metadata
            metadata_path = Path(config['metadata_dir']) / config.get('images_metadata_path', 'images_metadata.csv').split('/')[-1]
            metadata_df = pd.read_csv(metadata_path)
            logger.info(f"Loaded existing metadata from {metadata_path}")
        
        # Step 2: Embedding extraction
        if config.get('run_embedding_extraction', True):
            run_embedding_extraction(config, metadata_df, logger)
        
        # Step 3: Clustering
        if config.get('run_clustering', True):
            metadata_df = run_clustering(config, metadata_df, logger)
            # Save updated metadata
            metadata_path = Path(config['metadata_dir']) / config.get('images_metadata_path', 'images_metadata.csv').split('/')[-1]
            metadata_df.to_csv(metadata_path, index=False)
        
        # Step 4: Artifact detection
        if config.get('run_artifact_detection', True):
            metadata_df = run_artifact_detection(config, metadata_df, logger)
            # Save updated metadata
            metadata_path = Path(config['metadata_dir']) / config.get('images_metadata_path', 'images_metadata.csv').split('/')[-1]
            metadata_df.to_csv(metadata_path, index=False)
        
        # Step 5: Patch extraction
        patches_df = None
        if config.get('run_patch_extraction', True):
            patches_df = run_patch_extraction(config, metadata_df, logger)
        else:
            logger.info("Patch extraction skipped (run_patch_extraction=False)")
        
        # Step 6: Splits (only if patches were extracted)
        if config.get('run_splits', True) and patches_df is not None:
            run_splits(config, metadata_df, patches_df, logger)
        elif config.get('run_splits', True):
            logger.info("Splits skipped (no patches available)")
        
        logger.info("=" * 60)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

