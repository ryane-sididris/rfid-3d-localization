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

### ExtraTrees

| Feature mode | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|-------------|-------------|------------------|--------|--------|--------|
| All features | **1.245** | **0.719** | 48.9% | **93.3%** | **99.0%** |
| Per-Tx features | 1.824 | 1.053 | 27.0% | 71.1% | 92.5% |

Command:

```bash
python -m experiments.run_tx_rx_per_antenna_regression extratrees
```

## Interpretation

- `all_features` is essentially identical to the direct global model, which matches the expected hypothesis.
- `per_tx_features` is much worse with `extratrees`, so the full Tx/Rx interaction set carries useful information that is lost when slicing too aggressively by Tx.
- With tree-based models, the reconstructed RMSE is almost identical for every antenna, which suggests the model learns a common position estimate and mostly carries antenna-dependent offsets through the delta targets.

## Models To Prioritize Next For Per-Antenna

Suggested order:
1. `extratrees`: fastest strong baseline, already validated.
2. `randomforest`: close family, useful sanity check against ExtraTrees.
3. `xgboost`: stronger boosting baseline than plain sklearn GB in many tabular cases.
4. `mlp`: worth testing because it may exploit feature interactions differently.
5. `gradientboosting`: useful reference, but usually slower/weaker than ExtraTrees/XGBoost here.
6. `svm_rbf`: likely expensive on 4411 samples; keep it as a targeted benchmark, not the first sweep.
