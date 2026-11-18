#!/usr/bin/env python3
"""Move orc images to a separate folder for inspection.

This script identifies images from specified clusters (orc images)
and moves them (along with their masks) to a separate folder.
"""

import argparse
import shutil
import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def move_orc_images(
    metadata_path: str,
    source_dir: str,
    dest_dir: str,
    orc_clusters: list = None,
    orc_config_path: str = None,
    dry_run: bool = False
):
    """Move images from orc clusters to a separate folder.
    
    Args:
        metadata_path: Path to images_metadata.csv
        source_dir: Source directory (dataset)
        dest_dir: Destination directory for orc images
        orc_clusters: List of cluster IDs to move (e.g., [2, 7, 9]). If None, reads from config.
        orc_config_path: Path to orc_filter.yaml (used if orc_clusters is None)
        dry_run: If True, only print what would be moved without actually moving
    """
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    if 'cluster_id' not in df.columns:
        logger.error("No 'cluster_id' column found in metadata. Run clustering first.")
        return
    
    # Get orc clusters from config if not provided
    if orc_clusters is None:
        if orc_config_path and Path(orc_config_path).exists():
            with open(orc_config_path, 'r') as f:
                orc_config = yaml.safe_load(f)
            orc_clusters = orc_config.get('orc_clusters', [])
        else:
            logger.error("No orc_clusters provided and no config file found!")
            return
    
    if not orc_clusters:
        logger.warning("No orc clusters specified!")
        return
    
    # Find images in orc clusters
    orc_images = df[df['cluster_id'].isin(orc_clusters)].copy()
    logger.info(f"Found {len(orc_images)} images in clusters {orc_clusters}")
    
    if len(orc_images) == 0:
        logger.warning("No images found in specified clusters!")
        return
    
    # Create destination directory
    dest_path = Path(dest_dir)
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created destination directory: {dest_path}")
    
    source_path = Path(source_dir)
    
    # Statistics
    moved_images = 0
    moved_masks = 0
    missing_images = 0
    missing_masks = 0
    
    # Move each image and its mask
    for idx, row in orc_images.iterrows():
        image_id = row['image_id']
        cluster_id = row['cluster_id']
        
        # Image file - extract just filename from full path
        image_filename = Path(row['image_path']).name
        image_source = source_path / image_filename
        image_dest = dest_path / image_filename
        
        # Mask file (if exists)
        mask_source = None
        mask_dest = None
        if pd.notna(row.get('mask_path')):
            mask_filename = Path(row['mask_path']).name
            mask_source = source_path / mask_filename
            mask_dest = dest_path / mask_filename
        
        if dry_run:
            logger.info(f"[DRY RUN] Would move: {image_filename} (cluster {cluster_id})")
            if mask_source:
                logger.info(f"[DRY RUN] Would move: {mask_filename}")
        else:
            # Move image
            if image_source.exists():
                shutil.move(str(image_source), str(image_dest))
                moved_images += 1
                logger.debug(f"Moved image: {image_filename} -> {image_dest}")
            else:
                missing_images += 1
                logger.warning(f"Image not found: {image_source}")
            
            # Move mask
            if mask_source and mask_source.exists():
                shutil.move(str(mask_source), str(mask_dest))
                moved_masks += 1
                logger.debug(f"Moved mask: {mask_filename} -> {mask_dest}")
            elif mask_source:
                missing_masks += 1
                logger.warning(f"Mask not found: {mask_source}")
    
    # Summary
    logger.info("=" * 60)
    if dry_run:
        logger.info("DRY RUN - No files were actually moved")
    else:
        logger.info(f"Moved {moved_images} images and {moved_masks} masks to {dest_path}")
        if missing_images > 0:
            logger.warning(f"{missing_images} images were not found")
        if missing_masks > 0:
            logger.warning(f"{missing_masks} masks were not found")
    
    # Save list of moved images
    if not dry_run:
        moved_list_path = dest_path / "moved_images_list.csv"
        orc_images[['image_id', 'cluster_id', 'label', 'image_path', 'mask_path']].to_csv(
            moved_list_path, index=False
        )
        logger.info(f"Saved list of moved images to {moved_list_path}")
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Move orc images to a separate folder for inspection'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='metadata/images_metadata.csv',
        help='Path to images_metadata.csv'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='dataset',
        help='Source directory (dataset)'
    )
    parser.add_argument(
        '--dest-dir',
        type=str,
        default='dataset_orc_removed',
        help='Destination directory for orc images'
    )
    parser.add_argument(
        '--orc-config',
        type=str,
        default='configs/orc_filter.yaml',
        help='Path to orc_filter.yaml (clusters read from here if --clusters not provided)'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        nargs='+',
        default=None,
        help='Cluster IDs to move (e.g., 2 7 9). If not provided, reads from orc_config.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - show what would be moved without actually moving'
    )
    
    args = parser.parse_args()
    
    logger.info("Moving orc images to separate folder")
    if args.clusters:
        logger.info(f"Clusters to move: {args.clusters}")
    else:
        logger.info(f"Clusters will be read from: {args.orc_config}")
    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Destination: {args.dest_dir}")
    
    move_orc_images(
        args.metadata,
        args.source_dir,
        args.dest_dir,
        args.clusters,
        args.orc_config,
        args.dry_run
    )


if __name__ == '__main__':
    main()

