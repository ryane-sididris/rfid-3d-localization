# Tx/Rx Delta Regression

## Setup

- Dataset: `data/ds_Tx_Rx.csv`
- Samples: 4411
- Features: 276
- Outputs: 24 deltas = `(dx, dy, dz) x 8 antennes`
- Evaluation: CV 5-fold

Metrics:
- `rmse_avg`: RMSE 3D after reconstructing one position per antenna and averaging the 8 positions
- `rmse_sklearn`: `sqrt(mean_squared_error(y_true_xyz, y_pred_xyz))`
- `acc@1m/2m/3m`: percentage of samples below the 3D error threshold

## Direct Global Regression

One regressor takes all Tx/Rx features and predicts the 24 deltas at once.

### ExtraTrees

| Model | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|-------|-------------|------------------|--------|--------|--------|
| Global multi-output | **1.247** | **0.720** | **49.6%** | **93.3%** | **99.0%** |

Command:

```bash
python -m experiments.run_tx_rx_global_regression extratrees
```

## Per-Antenna Regression

One regressor is trained per antenna, each with 3 outputs `(dx, dy, dz)`.

### Model Comparison

| Model | Feature mode | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|-------|--------------|-------------|------------------|--------|--------|--------|
| ExtraTrees | All features | **1.245** | **0.719** | **48.9%** | **93.3%** | **99.0%** |
| ExtraTrees | Per-Tx features | 1.824 | 1.053 | 27.0% | 71.1% | 92.5% |
| RandomForest | All features | 1.306 | 0.754 | 47.6% | 91.6% | 98.4% |
| RandomForest | Per-Tx features | 1.828 | 1.056 | 26.8% | 71.1% | 92.3% |
| MLP | All features | 1.479 | 0.854 | 38.7% | 86.8% | 97.2% |
| MLP | Per-Tx features | **1.796** | **1.037** | **28.2%** | **72.8%** | **92.7%** |
| XGBoost | All features | 1.278 | 0.738 | 48.4% | 92.1% | 98.8% |
| XGBoost | Per-Tx features | 1.808 | 1.044 | 28.0% | 71.9% | 92.5% |

Command:

```bash
python -m experiments.run_tx_rx_per_antenna_regression extratrees randomforest mlp xgboost
```

## Interpretation

- `all_features` is essentially identical to the direct global model, which matches the expected hypothesis.
- `per_tx_features` is much worse across all tested models, so the full Tx/Rx interaction set carries useful information that is lost when slicing too aggressively by Tx.
- In `all_features`, `extratrees` remains the best overall, with `xgboost` very close and `randomforest` slightly behind.
- In `per_tx_features`, all models collapse to a narrower band; `mlp` is the best of the tested models on both RMSE and threshold accuracies.
- With tree-based models, the reconstructed RMSE is almost identical for every antenna, which suggests the model learns a common position estimate and mostly carries antenna-dependent offsets through the delta targets.
