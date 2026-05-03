"""
Tx-only barycenter heuristics on ds_power_tx_rx_freq.csv.

Usage:
    uv run python -m experiments.run_tx_barycenter_position
    uv run python -m experiments.run_tx_barycenter_position --score-mode raw_sum --top-k 8 --z-model linear
    uv run python -m experiments.run_tx_barycenter_position --score-mode raw_median --top-k 8 --z-model linear
    uv run python -m experiments.run_tx_barycenter_position --physical-sweep
    uv run python -m experiments.run_tx_barycenter_position --all
"""

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.antennas import ANTENNA_POSITIONS
from src.evaluation.metrics import mae_xy, rmse_xy, rmse_xy_peraxis, rmse_xyz, rmse_z, threshold_accuracy, xy_errors
from src.loaders.loader_power_tx_rx_freq_semantic import PowerTxRxFreqSemanticLoader
from src.models.tx_barycenter_heuristic import TxBarycenterHeuristic
from src.validation.cross_validation import cross_validate


DATA_PATH = Path("data") / "ds_power_tx_rx_freq.csv"
SPATIAL_MARGIN = 1.5  # metres beyond antenna bounding box, same as Christophe's notebook
K_FOLDS = 5
RANDOM_STATE = 42
CLASSIC_SCORE_MODES = ["raw_sum", "raw_median", "balanced_per_power"]
RSSI_MAX_SCORE_MODES = ["rssi_max", "rssi_max_power_corrected", "rssi_max_div_power_dbm", "rssi_max_div_power_linear"]
WEIGHT_EXPONENTS = [1.0, 0.5, 1.0 / 3.0, 0.25]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Tx-only barycenter heuristics on ds_power_tx_rx_freq.csv")
    parser.add_argument(
        "--score-mode",
        choices=CLASSIC_SCORE_MODES + RSSI_MAX_SCORE_MODES,
        default="balanced_per_power",
    )
    parser.add_argument("--top-k", choices=[4, 8], type=int, default=4)
    parser.add_argument("--z-model", choices=["constant", "linear", "extratrees"], default="extratrees")
    parser.add_argument("--weight-exponent", type=float, default=1.0)
    parser.add_argument("--physical-sweep", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--spatial-filter", action="store_true",
                        help="Restrict to samples within SPATIAL_MARGIN of antenna bounding box")
    return parser.parse_args(argv)


def spatial_filter_mask(y_xyz: np.ndarray, margin: float = SPATIAL_MARGIN) -> np.ndarray:
    """Return boolean mask keeping samples within margin of the antenna bounding box."""
    positions = np.array([v[:2] for v in ANTENNA_POSITIONS.values()])
    x_min, y_min = positions.min(axis=0) - margin
    x_max, y_max = positions.max(axis=0) + margin
    x, y = y_xyz[:, 0], y_xyz[:, 1]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for fold in folds:
        for key, value in fold.items():
            metrics[key].append(float(value))
    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in metrics.items()
    }


def evaluate_position(y_true, y_pred, _):
    xy_errs = xy_errors(y_true, y_pred)
    return {
        "acc_xy@2m": threshold_accuracy(xy_errs, 2.0),
        "mae_xy": mae_xy(y_true, y_pred),
        "rmse_xy_peraxis": rmse_xy_peraxis(y_true, y_pred),
        "rmse_xy": rmse_xy(y_true, y_pred),
        "rmse_z": rmse_z(y_true, y_pred),
        "rmse_xyz": rmse_xyz(y_true, y_pred),
    }


def build_experiment_configs(args):
    physical_configs = [
        (score_mode, 8, "constant", weight_exponent)
        for score_mode, weight_exponent in product(RSSI_MAX_SCORE_MODES, WEIGHT_EXPONENTS)
    ]

    if args.all:
        score_modes = CLASSIC_SCORE_MODES
        top_ks = [4, 8]
        z_models = ["constant", "linear", "extratrees"]
        classic_configs = [
            (score_mode, top_k, z_model, 1.0)
            for score_mode, top_k, z_model in product(score_modes, top_ks, z_models)
        ]
        return classic_configs + physical_configs

    if args.physical_sweep:
        return physical_configs

    return [(args.score_mode, args.top_k, args.z_model, args.weight_exponent)]


def run_one_config(
    values_by_power_tx_rx_freq,
    y_xyz,
    tx_names,
    powers,
    score_mode,
    top_k,
    z_model,
    weight_exponent,
):
    def train_fn(X_train, y_train):
        model = TxBarycenterHeuristic(
            tx_names=tx_names,
            powers=powers,
            score_mode=score_mode,
            top_k=top_k,
            z_model=z_model,
            weight_exponent=weight_exponent,
        )
        model.fit(X_train, y_train)
        return model

    def predict_fn(model, X_test):
        return model.predict(X_test)

    result = cross_validate(
        X=values_by_power_tx_rx_freq,
        y=y_xyz,
        train_fn=train_fn,
        predict_fn=predict_fn,
        eval_fn=evaluate_position,
        aggregate_fn=aggregate_mean_std,
        k=K_FOLDS,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    summary = result["summary"]
    return {
        "score_mode": score_mode,
        "top_k": top_k,
        "weight_exponent": f"{weight_exponent:.3f}",
        "z_model": z_model,
        "acc_xy@2m": f"{summary['acc_xy@2m']['mean']:.2f} +/- {summary['acc_xy@2m']['std']:.2f}",
        "mae_xy": f"{summary['mae_xy']['mean']:.3f} +/- {summary['mae_xy']['std']:.3f}",
        "rmse_xy_peraxis": f"{summary['rmse_xy_peraxis']['mean']:.3f} +/- {summary['rmse_xy_peraxis']['std']:.3f}",
        "rmse_xy": f"{summary['rmse_xy']['mean']:.3f} +/- {summary['rmse_xy']['std']:.3f}",
        "rmse_z": f"{summary['rmse_z']['mean']:.3f} +/- {summary['rmse_z']['std']:.3f}",
        "rmse_xyz": f"{summary['rmse_xyz']['mean']:.3f} +/- {summary['rmse_xyz']['std']:.3f}",
    }


def main(argv=None):
    args = parse_args(argv or [])

    print(f"Loading dataset: {DATA_PATH.name}")
    loader = PowerTxRxFreqSemanticLoader()
    dataset = loader.load(DATA_PATH)

    print(f"Semantic values shape: {dataset.values_by_power_tx_rx_freq.shape}")
    print(f"Mask shape:            {dataset.mask_by_power_tx_rx_freq.shape}")
    print(f"Targets shape:         {dataset.y_xyz.shape}")
    print(f"Powers:                {dataset.powers}")
    print(f"Tx names:              {dataset.tx_names}")
    print(f"Rx names:              {dataset.rx_names}")

    X, y = dataset.values_by_power_tx_rx_freq, dataset.y_xyz
    if args.spatial_filter:
        mask = spatial_filter_mask(y)
        X, y = X[mask], y[mask]
        print(f"Spatial filter:        {mask.sum()}/{len(mask)} samples kept")

    configs = build_experiment_configs(args)
    rows = []

    for score_mode, top_k, z_model, weight_exponent in configs:
        print(
            f"Running: score_mode={score_mode}, top_k={top_k}, "
            f"weight_exponent={weight_exponent:.3f}, z_model={z_model}"
        )
        row = run_one_config(
            values_by_power_tx_rx_freq=X,
            y_xyz=y,
            tx_names=dataset.tx_names,
            powers=dataset.powers,
            score_mode=score_mode,
            top_k=top_k,
            z_model=z_model,
            weight_exponent=weight_exponent,
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    print("\n=== tx_barycenter_heuristic - ds_power_tx_rx_freq.csv ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
