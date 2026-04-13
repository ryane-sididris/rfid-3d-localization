"""
Conv1D frequency-tensor position regression on ds_power_tx_rx_freq.csv.

Usage:
    uv run python -m experiments.run_keras_freq_position
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from pathlib import Path

from src.evaluation.metrics import rmse_3d, rmse_xyz, threshold_accuracy
from src.loaders.loader_power_tx_rx_freq_tensor import PowerTxRxFreqTensorLoader
from src.models.keras_tensor_models import build_freq_conv_xyz, default_callbacks_freq_xyz
from src.validation.cross_validation import cross_validate


DATA_PATH = Path("data") / "ds_power_tx_rx_freq.csv"
SEED = 42


def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for fold in folds:
        for key, value in fold.items():
            metrics[key].append(value)
    return {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics.items()}


def evaluate_position(y_true, y_pred, _):
    errs = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "rmse": rmse_3d(errs),
        "rmse_xyz": rmse_xyz(y_true, y_pred),
        "acc_1m": threshold_accuracy(errs, 1),
        "acc_2m": threshold_accuracy(errs, 2),
        "acc_3m": threshold_accuracy(errs, 3),
    }


def train_fn(X_train, y_train):
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    model = build_freq_conv_xyz(input_shape=X_train.shape[1:])
    batch_size = int(min(128, max(16, len(X_train) // 10)))
    model.fit(
        X_train,
        y_train.astype("float32"),
        validation_split=0.2,
        epochs=500,
        batch_size=batch_size,
        callbacks=default_callbacks_freq_xyz(),
        verbose=1,
    )
    return model


def predict_fn(model, X_test):
    return model.predict(X_test, verbose=0)


def main():
    print(f"Loading dataset: {DATA_PATH.name}")
    loader = PowerTxRxFreqTensorLoader()
    X_tensor, y_xyz = loader.load(DATA_PATH)

    print(f"Flat values shape:     {loader.flat_values_shape}")
    print(f"Semantic tensor shape: {loader.semantic_values_shape}")
    print(f"Model input shape:     {loader.model_input_shape}")

    result = cross_validate(
        X=X_tensor,
        y=y_xyz,
        train_fn=train_fn,
        predict_fn=predict_fn,
        eval_fn=evaluate_position,
        aggregate_fn=aggregate_mean_std,
        k=5,
        random_state=42,
        verbose=True,
    )

    print(f"\n=== keras_freq_conv_xyz - {DATA_PATH.name} ===")
    summary = pd.DataFrame(result["summary"]).T
    print(summary)


if __name__ == "__main__":
    main()
