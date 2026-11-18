# Clustering Process - Preprocessing Pipeline

This directory contains all necessary components to run the complete preprocessing pipeline for dataset clustering, orc image identification, and removal.

## Directory Structure

```
Clustering_Process/
├── run_preprocessing.py          # Main script to run the preprocessing pipeline
├── move_orc_images.py            # Script to move orc images to a separate folder
├── configs/                      # Configuration files
│   ├── main_preprocessing.yaml   # Main preprocessing configuration
│   ├── orc_filter.yaml           # Orc cluster identification configuration
│   ├── artifacts.yaml            # Artifact detection configuration
│   ├── patching.yaml             # Patch extraction configuration (optional)
│   └── augmentations.yaml        # Augmentation configuration (optional)
├── src/                          # Source code
│   ├── dataset_utils/
│   │   └── dataset_discovery.py
│   └── preprocessing/
│       ├── orc_filter.py
│       ├── artifact_detection.py
│       ├── patching_pipeline.py
│       └── splits.py
├── dataset/                      # Dataset directory (place your data here)
│   ├── img_*.png                 # Image files
│   ├── mask_*.png                # Mask files
│   └── train_labels.csv          # CSV file with labels
├── metadata/                     # Output: Generated metadata files
├── artifacts/                    # Output: Saved embeddings
├── reports/                      # Output: Cluster montages for visual inspection
└── logs/                         # Output: Preprocessing logs
```

## Quick Start

### 1. Prepare the Dataset

Place your dataset in the `dataset/` directory with the following structure:

```
dataset/
├── img_0001.png
├── img_0002.png
├── ...
├── img_1000.png
├── mask_0001.png
├── mask_0002.png
├── ...
├── mask_1000.png
└── train_labels.csv
```

The `train_labels.csv` file must have the following structure:
```csv
sample_index,label
img_0000.png,Luminal A
img_0001.png,Luminal B
...
```

**Note**: Images and masks can be in the same directory, even if ordered separately (all images first, then all masks). Matching is performed by filename, not by file order.

### 2. Run Clustering and Orc Identification

From the `Clustering_Process/` directory, execute:

```bash
python run_preprocessing.py
```

This will execute:
- Dataset discovery
- Embedding extraction
- Clustering
- Artifact detection (optional)

**Patch extraction is disabled** by default in the configuration.

### 3. Inspect Clusters and Identify Orcs

After clustering completes, inspect the montages in `reports/orc_clusters/`:
- Open `reports/orc_clusters/cluster_00.png`, `cluster_01.png`, etc.
- Identify which clusters contain "orc" images (non-useful images)

### 4. Configure Orc Clusters

Update `configs/orc_filter.yaml` with the identified orc clusters:

```yaml
orc_clusters: [2, 7, 9]  # Insert the cluster IDs identified as orcs
```

### 5. Move Orc Images

Execute the script to move orc images to a separate directory:

```bash
# First verify what would be moved (dry-run)
python move_orc_images.py --dry-run

# Then move the images
python move_orc_images.py
```

Orc images (and their corresponding masks) will be moved to `dataset_orc_removed/`.

## Detailed Process Description

The preprocessing pipeline executes the following steps sequentially:

### Step 1: Dataset Discovery
- Discovers all images, masks, and labels
- Validates image-mask-label triplets
- Generates `metadata/images_metadata.csv`

### Step 2: Embedding Extraction
- Extracts embeddings using pretrained ResNet18 (ImageNet)
- Saves embeddings to `artifacts/image_embeddings.npz`
- Embeddings are reused if already present

### Step 3: Clustering and Orc Detection
- Applies KMeans clustering (default: 10 clusters)
- Generates montages for each cluster in `reports/orc_clusters/`
- **IMPORTANT**: Inspect the montages and update `configs/orc_filter.yaml` with clusters identified as orcs
- Marks images as `is_orc=True` based on cluster membership

### Step 4: Artifact Detection (Optional)
- Detects green cartoon-like artifacts using HSV color space analysis
- Calculates artifact-to-tissue ratio
- Marks images with `has_cartoon_artifacts=True`
- This step can be disabled if not needed (set `run_artifact_detection: false`)

### Step 5: Patch Extraction (Disabled by Default)
- Extracts patches from images (default: 256x256)
- Automatically excludes orc images if `exclude_orc_images: true` in `orc_filter.yaml`
- Saves patches to `processed/patches/`
- Generates `metadata/patches_metadata.csv`

### Step 6: Train/Val Splits (Disabled by Default)
- Creates stratified splits at image level
- Creates corresponding patch-level splits
- Saves to `metadata/images_train.csv`, `images_val.csv`, `patches_train.csv`, `patches_val.csv`

## Configuration

### Configuring Orc Clusters

After running clustering (Step 3), inspect the montages in `reports/orc_clusters/` and update `configs/orc_filter.yaml`:

```yaml
orc_clusters: [2, 7, 9]  # Insert cluster IDs identified as orcs here
exclude_orc_images: true  # If true, orc images are excluded from patch extraction
```

### Configuring Patch Parameters

Modify `configs/patching.yaml` to change:
- `patch_size`: Patch size (default: 256)
- `stride`: Stride for extraction (default: 256, non-overlapping)
- `min_tissue_ratio`: Minimum tissue ratio to keep a patch (default: 0.1)
- `max_artifact_ratio`: Maximum artifact ratio allowed (default: 0.5)

### Skipping Specific Steps

Modify `configs/main_preprocessing.yaml` to skip already completed steps:

```yaml
run_dataset_discovery: true
run_embedding_extraction: false  # Skip if already done
run_clustering: true
run_artifact_detection: false   # Skip if not needed
run_patch_extraction: false     # Disabled by default
run_splits: false               # Disabled by default
```

## Generated Outputs

### Metadata Files
- `metadata/images_metadata.csv`: Complete image metadata with columns:
  - `cluster_id`: Assigned cluster
  - `is_orc`: Orc flag
  - `artifact_ratio`: Artifact ratio
  - `has_cartoon_artifacts`: Artifact flag

- `metadata/patches_metadata.csv`: Patch metadata (only if patches are extracted)

### Cluster Montages
- `reports/orc_clusters/cluster_00.png` ... `cluster_09.png`: Montages for visual inspection

### Moved Images
- `dataset_orc_removed/`: Directory containing moved orc images and masks
- `dataset_orc_removed/moved_images_list.csv`: List of moved images with metadata

## Requirements

Ensure the following libraries are installed:

```bash
pip install pandas numpy scikit-learn torch torchvision pillow opencv-python pyyaml
```

## Notes

- The process is **incremental**: if a step is already completed, it can be skipped by modifying flags in the config
- Embeddings are saved and reused if already present
- Logs are saved to `logs/preprocessing_YYYYMMDD_HHMMSS.log`
- The system uses GPU if available, otherwise CPU

## Troubleshooting

### Error: "No 'cluster_id' column found"
- Run Step 3 (Clustering) first

### Error: "No orc clusters configured"
- Update `configs/orc_filter.yaml` with identified clusters

### Error: "Patches metadata file not found"
- This is expected if patch extraction is disabled
- Use `move_orc_images.py` instead to move orc images

## Recommended Workflow

1. **First complete run**:
   ```bash
   python run_preprocessing.py
   ```

2. **Inspect montages** in `reports/orc_clusters/`

3. **Update `configs/orc_filter.yaml`** with identified orc clusters

4. **Move orc images**:
   ```bash
   python move_orc_images.py --dry-run  # Verify
   python move_orc_images.py            # Execute
   ```

5. **Verify results** in the metadata files and `dataset_orc_removed/` directory
