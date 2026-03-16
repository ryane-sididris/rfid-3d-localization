"""
Direct position regression: RSSI features -> XYZ position.

Usage:
    python -m experiments.run_position extratrees
    python -m experiments.run_position randomforest
    python -m experiments.run_position xgboost
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.loaders.loader_elemen_ds import ElemenDsLoader
from src.loaders.loader_tx_ds import TxDsLoader
from src.loaders.loader_tx_rx_ds import TxRxDsLoader
from src.loaders.loader_advanced_ds import AdvancedDsLoader

from src.models.train_position import train_position
from src.models.regressors import available_regressors, get_default_params
from src.evaluation.metrics import rmse_3d, threshold_accuracy
from src.validation.cross_validation import cross_validate


# CLI

if len(sys.argv) < 2:
    print(f"Usage: python -m experiments.run_position <{'|'.join(available_regressors())}>")
    sys.exit(1)

MODEL_NAME = sys.argv[1].lower()

# Dataset

DATA_DIR = Path("data")
DS_NAME = "ds_Tx_Rx.csv"
#DS_NAME = "ds_Tx.csv"
#DS_NAME  = "ds_elemen.csv"
#DS_NAME = "ds_advanced_17Feb26.csv"

print(f"Loading dataset: {DS_NAME}")
loader = TxRxDsLoader()
#loader = TxDsLoader()
#loader = ElemenDsLoader()
#loader = AdvancedDsLoader()

X, y_xyz, _, _ = loader.load(DATA_DIR / DS_NAME)

# Helpers

def train_fn(X_train, y_train):
    return train_position(MODEL_NAME, X_train, y_train)

def predict_fn(model, X_test):
    return model.predict(X_test)

def evaluate_position(y_true, y_pred, _):
    errs = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "rmse":   rmse_3d(errs),
        "acc_1m": threshold_accuracy(errs, 1),
        "acc_2m": threshold_accuracy(errs, 2),
        "acc_3m": threshold_accuracy(errs, 3),
    }

def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for f in folds:
        for k, v in f.items():
            metrics[k].append(v)
    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}

# Run

print(f"Training & validation  [{MODEL_NAME}]")
result = cross_validate(
    X=X, y=y_xyz,
    train_fn=train_fn, predict_fn=predict_fn,
    eval_fn=evaluate_position, aggregate_fn=aggregate_mean_std,
    k=5, random_state=42,
)

# Print results

print(f"\n=== {MODEL_NAME} - {DS_NAME} ===")
params = get_default_params(MODEL_NAME)
print("Params : " + ", ".join(f"{k}={v}" for k, v in params.items()))
summary = pd.DataFrame(result["summary"]).T
print(summary)

#folds_results = pd.DataFrame(result["folds"])
#print(folds_results)