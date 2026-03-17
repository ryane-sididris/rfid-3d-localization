"""
Direct global delta regression on ds_Tx_Rx.csv.

One regressor takes all Tx/Rx features and predicts all antenna deltas at once:
[dx, dy, dz] x 8 antennas.

Usage:
    python -m experiments.run_tx_rx_global_regression
    python -m experiments.run_tx_rx_global_regression extratrees randomforest
"""

import sys
from pathlib import Path

from sklearn.base import clone

from src.config.antennas import ANTENNA_NAMES
from src.evaluation.delta_regression import (
    aggregate_mean_std,
    evaluate_delta_regression_fold,
    get_antenna_positions_array,
)
from src.loaders.loader_tx_rx_ds_with_deltas import TxRxDsWithDeltasLoader
from src.models.regressors import get_regressor
from src.validation.cross_validation import cross_validate


DATA_PATH = Path("data") / "ds_Tx_Rx.csv"
K_FOLDS = 5
RANDOM_STATE = 42


def get_models():
    names = ["extratrees", "randomforest", "gradientboosting", "xgboost", "svm_rbf", "mlp"]
    return {name: get_regressor(name, multioutput=True) for name in names}


def print_summary(summary):
    print("\n    Per-antenna delta RMSE (mean ± std):")
    for ant_name in ANTENNA_NAMES:
        key = f"delta_rmse_{ant_name}"
        print(f"      {ant_name:>10s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")

    print("\n    Per-antenna position RMSE (mean ± std):")
    for ant_name in ANTENNA_NAMES:
        key = f"pos_rmse_{ant_name}"
        print(f"      {ant_name:>10s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")

    print("\n    Aggregated (mean ± std):")
    for key in ["rmse_avg", "rmse_sklearn", "acc_1m", "acc_2m", "acc_3m"]:
        print(f"      {key:>12s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")


def run(model_names=None):
    all_models = get_models()

    if model_names:
        selected = {}
        for name in model_names:
            key = name.lower()
            if key not in all_models:
                print(f"Unknown model: {name}")
                print(f"Available: {', '.join(all_models.keys())}")
                return
            selected[key] = all_models[key]
    else:
        selected = all_models

    print(f"Loading dataset: {DATA_PATH}")
    loader = TxRxDsWithDeltasLoader()
    X, y_deltas, y_xyz, _ = loader.load(DATA_PATH)
    antenna_positions = get_antenna_positions_array(loader.antenna_names)

    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Outputs: {y_deltas.shape[1]}")
    print(f"Models to run: {', '.join(selected.keys())}")

    for model_name, reg_template in selected.items():
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        def train_fn(X_train, y_train):
            model = clone(reg_template)
            model.fit(X_train, y_train)
            return model

        def predict_fn(model, X_test):
            return model.predict(X_test)

        def eval_fn(y_test, y_pred, test_index):
            return evaluate_delta_regression_fold(
                y_xyz_true=y_xyz[test_index],
                y_deltas_pred=y_pred,
                y_deltas_true=y_test,
                antenna_positions=antenna_positions,
                antenna_names=loader.antenna_names,
            )

        result = cross_validate(
            X=X,
            y=y_deltas,
            train_fn=train_fn,
            predict_fn=predict_fn,
            eval_fn=eval_fn,
            aggregate_fn=aggregate_mean_std,
            k=K_FOLDS,
            random_state=RANDOM_STATE,
            verbose=True,
        )

        print_summary(result["summary"])


if __name__ == "__main__":
    args = sys.argv[1:]
    run(model_names=args if args else None)
