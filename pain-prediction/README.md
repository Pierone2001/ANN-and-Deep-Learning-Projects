# Pain Level Prediction from Time Series

Predict the overall **pain level** of a subject — *no pain*, *low pain*, *high pain* — from a multivariate time series plus static information:

- **dynamic features** over time: joint-related variables and four pain-survey answers
- **static features**: categorical descriptors (e.g. limb information), encoded with learned embeddings

## Approach (final model: `gru_mono.ipynb`)

- Categorical features cleaned and mapped to integers for `nn.Embedding`
- Each series split into overlapping windows (W = 32)
- **GRU** encoder + learned **embeddings** for categorical features + **temporal attention**
- **Subject-level 5-fold cross-validation** (no subject shared between train and validation)
- Window-level training; validation aggregates softmax probabilities over all windows of a subject and computes subject-level F1
- Early stopping on validation F1 (up to 500 epochs)

## Results

| Notebook | Description | F1 |
|---|---|---|
| `highest_scores.ipynb` | experiments that reached the best leaderboard score | **0.9554** (Kaggle) |
| `gru_bi.ipynb` | bidirectional GRU variant | 0.931 (Kaggle) |
| `gru_mono.ipynb` | unidirectional GRU + embeddings + attention — **selected final model** | 0.9463 ± 0.0104 (5-fold CV) |

`gru_mono.ipynb` was chosen as the final model for its stability and generalization across folds, even though it is not the single highest leaderboard submission. `highest_scores.ipynb` also documents other explored ideas (focal loss, feature augmentation, oversampling, convolutional layers).
