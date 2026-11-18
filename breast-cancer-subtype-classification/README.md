# Breast Cancer Molecular Subtype Classification

Classify diseased tissue morphology in high-resolution **H&E histology images** into four breast-cancer molecular subtypes:

**Luminal A · Luminal B · HER2(+) · Triple Negative**

Each case provides one large H&E image plus a binary mask highlighting the regions most likely to contain diseased tissue.

## Pipeline

1. **Dataset cleaning** — the raw training set contains non-histological samples (illustrations and visual artifacts). They are removed before training:
   - `dataset_cleaning.ipynb`: color-based filter — flags images with a large fraction of pixels in a "forbidden" green HSV range, while a pink/purple (H&E tissue) threshold protects valid slides. Produces the lists of noisy images/masks.
   - `clustering_cleaning/`: alternative offline cleaning with clustering and pre-trained classifiers (see its own README).
   - `noisy_sample_lists/`: the resulting lists of image and mask paths to discard.
2. **Patch extraction** — 224×224 sliding window over each slide. A patch is labelled with the slide subtype if its mask tumor fraction is ≥ 0.05; otherwise, if it contains enough tissue (≥ 0.10), it becomes an extra **Healthy** class, which reduces spurious correlations.
3. **Model** (`main_dual_stream_resnet.ipynb`) — **dual-stream ResNet18** (transfer learning):
   - RGB stream on the image patch
   - mask stream on the 1-channel binary mask, used as a learnable spatial prior
4. **Training** — class-balanced loss, label smoothing (ε = 0.05), AdamW, cosine LR schedule, early stopping on validation macro-F1.
5. **Inference** — test-time augmentation (original + rotated patch), Healthy patches filtered out, slide label obtained by majority voting over the remaining patches.

`contrastive_learning.ipynb` contains a SimCLR contrastive pre-training variant explored during the challenge (it did not improve the leaderboard score).

## Files

```
├── main_dual_stream_resnet.ipynb   # final training + inference pipeline
├── dataset_cleaning.ipynb          # HSV-based removal of non-histological samples
├── contrastive_learning.ipynb      # experimental variant
├── clustering_cleaning/            # clustering-based offline cleaning (scripts + configs)
└── noisy_sample_lists/             # lists of discarded images / masks
```
