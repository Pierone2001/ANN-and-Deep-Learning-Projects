# Breast Cancer Subtyping & Pain Prediction

Two deep-learning challenges developed for the **Artificial Neural Networks and Deep Learning (AN2DL)** course at Politecnico di Milano (A.Y. 2025/2026), both evaluated on Kaggle.

| Project | Task | Data | Approach |
|---|---|---|---|
| [`breast-cancer-subtype-classification/`](breast-cancer-subtype-classification/) | Classify H&E histology slides into 4 breast-cancer molecular subtypes | Images + tumor masks | Dataset cleaning, patch extraction, dual-stream ResNet18 (RGB + mask) |
| [`pain-prediction/`](pain-prediction/) | Predict pain level (no / low / high) from multivariate time series | Tabular time series + static features | GRU with categorical embeddings and temporal attention |

Each folder has its own README with details and results.

## Running the notebooks

The notebooks were developed on **Google Colab** (GPU) with PyTorch. The challenge datasets are **not included** in this repository: download them from the course Kaggle competitions and update the data paths at the top of each notebook (they currently point to Google Drive folders).

## Team — *Backpropagandisti*

Piero Burigana · Gabriele Proverbio · Matteo Trombetta · Giovanni Passuello
