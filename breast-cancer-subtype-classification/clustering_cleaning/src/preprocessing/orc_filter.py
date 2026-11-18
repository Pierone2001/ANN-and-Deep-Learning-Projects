"""Orc image filtering using embedding-based clustering.

This module extracts image embeddings, performs clustering, and allows
manual annotation of orc-like clusters for filtering.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.cluster import KMeans
import yaml

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """Extract embeddings from images using a pretrained CNN."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the embedder with a pretrained ResNet18.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pretrained ResNet18
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove final classification layer, keep features
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding for a single image.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Flattened embedding vector (512-dim for ResNet18).
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                embedding = embedding.flatten()
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding from {image_path}: {e}")
            # Return zero vector as fallback
            return np.zeros(512)
    
    def extract_embeddings_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths.
            batch_size: Batch size for processing.
            
        Returns:
            Array of shape (n_images, embedding_dim).
        """
        embeddings = []
        log_interval = 40  # Log every 40 images
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = [self.extract_embedding(path) for path in batch_paths]
            embeddings.extend(batch_embeddings)
            
            processed_count = i + len(batch_paths)
            # Log every 40 images
            if processed_count % log_interval == 0 or processed_count == len(image_paths):
                logger.info(f"Processed {processed_count} / {len(image_paths)} images ({processed_count*100//len(image_paths)}%)")
        
        return np.array(embeddings)


def extract_and_save_embeddings(
    image_paths: List[str],
    output_path: str,
    batch_size: int = 32
) -> None:
    """Extract embeddings for all images and save to file.
    
    Args:
        image_paths: List of image file paths.
        output_path: Path to save embeddings (.npz file).
        batch_size: Batch size for processing.
    """
    logger.info(f"Extracting embeddings for {len(image_paths)} images...")
    embedder = ImageEmbedder()
    embeddings = embedder.extract_embeddings_batch(image_paths, batch_size)
    
    # Save embeddings with file paths
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        image_paths=np.array(image_paths, dtype=object)
    )
    
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"Embedding shape: {embeddings.shape}")


def load_embeddings(embeddings_path: str) -> tuple:
    """Load embeddings from file.
    
    Args:
        embeddings_path: Path to .npz file containing embeddings.
        
    Returns:
        Tuple of (embeddings, image_paths).
    """
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    return embeddings, image_paths


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """Perform KMeans clustering on embeddings.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.
        
    Returns:
        Cluster labels for each sample.
    """
    logger.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    logger.info(f"Cluster distribution: {np.bincount(cluster_labels)}")
    return cluster_labels


def create_cluster_montage(
    image_paths: List[str],
    cluster_labels: np.ndarray,
    cluster_id: int,
    output_path: str,
    n_samples: int = 30,
    grid_size: tuple = (5, 6)
) -> None:
    """Create a montage image showing samples from a cluster.
    
    Args:
        image_paths: List of all image paths.
        cluster_labels: Cluster assignment for each image.
        cluster_id: ID of cluster to visualize.
        output_path: Path to save montage image.
        n_samples: Maximum number of samples to show.
        grid_size: (rows, cols) for montage grid.
    """
    from PIL import Image
    
    # Get images in this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    n_samples = min(n_samples, len(cluster_indices))
    selected_indices = np.random.choice(cluster_indices, n_samples, replace=False)
    
    # Load and resize images
    thumbnails = []
    for idx in selected_indices:
        try:
            img = Image.open(image_paths[idx])
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            thumbnails.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {image_paths[idx]}: {e}")
    
    if not thumbnails:
        logger.warning(f"No valid images for cluster {cluster_id}")
        return
    
    # Create montage
    rows, cols = grid_size
    montage_width = thumbnails[0].width * cols
    montage_height = thumbnails[0].height * rows
    
    montage = Image.new('RGB', (montage_width, montage_height), color='white')
    
    for i, thumb in enumerate(thumbnails[:rows * cols]):
        row = i // cols
        col = i % cols
        x = col * thumb.width
        y = row * thumb.height
        montage.paste(thumb, (x, y))
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)
    logger.info(f"Saved cluster {cluster_id} montage to {output_path}")


def generate_all_cluster_montages(
    embeddings_path: str,
    output_dir: str,
    n_clusters: int = 10,
    n_samples_per_cluster: int = 30
) -> None:
    """Generate montages for all clusters.
    
    Args:
        embeddings_path: Path to embeddings .npz file.
        output_dir: Directory to save montage images.
        n_clusters: Number of clusters.
        n_samples_per_cluster: Number of samples per montage.
    """
    embeddings, image_paths = load_embeddings(embeddings_path)
    cluster_labels = cluster_embeddings(embeddings, n_clusters)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_id in range(n_clusters):
        output_path = output_dir / f"cluster_{cluster_id:02d}.png"
        create_cluster_montage(
            image_paths,
            cluster_labels,
            cluster_id,
            str(output_path),
            n_samples_per_cluster
        )


def mark_orc_images(
    metadata_df: pd.DataFrame,
    config_path: str
) -> pd.DataFrame:
    """Mark images as orc based on cluster IDs in config.
    
    Args:
        metadata_df: DataFrame with image metadata (must have 'cluster_id' column).
        config_path: Path to config YAML with 'orc_clusters' list.
        
    Returns:
        DataFrame with added 'is_orc' column.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    orc_clusters = config.get('orc_clusters', [])
    exclude_orc = config.get('exclude_orc_images', False)
    
    logger.info(f"Marking images in clusters {orc_clusters} as orc")
    
    # Mark orc images
    if 'cluster_id' in metadata_df.columns:
        metadata_df['is_orc'] = metadata_df['cluster_id'].isin(orc_clusters)
    else:
        logger.warning("No 'cluster_id' column found, setting all is_orc=False")
        metadata_df['is_orc'] = False
    
    n_orc = metadata_df['is_orc'].sum()
    logger.info(f"Marked {n_orc} images as orc (exclude_orc={exclude_orc})")
    
    return metadata_df

