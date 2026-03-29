"""
Zone-specialized regression on ds_Tx_Rx.csv.

One global regressor for coarse zone routing, plus one regressor per antenna zone.
Square XY zones of configurable radius centered on each antenna.

Usage:
    uv run -m experiments.run_zone_regression
    uv run -m experiments.run_zone_regression --mode top_k --radius 3.0
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS
from src.evaluation.metrics import rmse_3d, rmse_xyz, threshold_accuracy
from src.loaders.loader_tx_rx_ds import TxRxDsLoader
from src.models.regressors import get_regressor
from src.models.zone_specialized_regressor import ZoneSpecializedRegressor
from src.validation.cross_validation import cross_validate


DATA_PATH = Path("data") / "ds_Tx_Rx.csv"
K_FOLDS = 5
RANDOM_STATE = 42


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Zone-specialized regression")
    parser.add_argument("--mode", choices=["single", "top_k"], default="single")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="extratrees")
    return parser.parse_args(argv)


def plot_zones(y_xyz, antenna_names, radius, save_path):
    """Plot XY view with antenna positions, square zones, and sample distribution."""
    antenna_xy = {name: ANTENNA_POSITIONS[name][:2] for name in antenna_names}
    colors = plt.cm.tab10(np.linspace(0, 1, len(antenna_names)))

    fig, ax = plt.subplots(figsize=(12, 6))

    # plot samples as gray dots
    ax.scatter(y_xyz[:, 0], y_xyz[:, 1], c="lightgray", s=5, alpha=0.5, label="Samples")

    # plot zones and antennas
    for i, name in enumerate(antenna_names):
        ax_x, ax_y = antenna_xy[name]
        rect = patches.Rectangle(
            (ax_x - radius, ax_y - radius),
            2 * radius, 2 * radius,
            linewidth=1.5, edgecolor=colors[i], facecolor=colors[i], alpha=0.1,
        )
        ax.add_patch(rect)
        ax.plot(ax_x, ax_y, "^", color=colors[i], markersize=10, markeredgecolor="black")
        ax.annotate(name, (ax_x, ax_y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color=colors[i])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Antenna zones (radius={radius}m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Zone plot saved to {save_path}")
    plt.close(fig)


def print_summary(summary):
    print("\n    Aggregated (mean +/- std):")
    for key in ["rmse", "rmse_xyz", "acc_0.5m", "acc_1m", "acc_2m", "acc_3m", "zone_routing_acc"]:
        if key in summary:
            print(f"      {key:>18s}: {summary[key]['mean']:.3f} +/- {summary[key]['std']:.3f}")


def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for fold in folds:
        for key, value in fold.items():
            if isinstance(value, (int, float, np.floating)):
                metrics[key].append(float(value))
    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in metrics.items()
    }


def run(argv=None):
    args = parse_args(argv or [])

    print(f"Loading dataset: {DATA_PATH}")
    loader = TxRxDsLoader()
    X, y_xyz, _, _ = loader.load(DATA_PATH)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    reg_template = get_regressor(args.model)

    # coverage report
    reporter = ZoneSpecializedRegressor(
        regressor_template=reg_template,
        radius=args.radius,
        inference_mode=args.mode,
        top_k=args.top_k,
    )
    reporter.zone_coverage_report(y_xyz)

    # zone plot
    results_dir = Path("experiments") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_zones(y_xyz, ANTENNA_NAMES, args.radius, results_dir / "zone_regression_zones.png")

    print(f"\nRunning CV: mode={args.mode}, radius={args.radius}m, model={args.model}")

    def train_fn(X_train, y_train):
        model = ZoneSpecializedRegressor(
            regressor_template=reg_template,
            radius=args.radius,
            inference_mode=args.mode,
            top_k=args.top_k,
        )
        model.train(X_train, y_train)
        return model

    def predict_fn(model, X_test):
        return model.predict(X_test)

    def eval_fn(y_test, y_pred, test_index):
        errs = np.linalg.norm(y_test - y_pred, axis=1)

        # zone routing accuracy: compare predicted zone to true zone
        true_zones = _nearest_antenna(y_test)
        pred_zones = _nearest_antenna(y_pred)
        zone_acc = float((true_zones == pred_zones).mean() * 100)

        return {
            "rmse": rmse_3d(errs),
            "rmse_xyz": rmse_xyz(y_test, y_pred),
            "acc_0.5m": threshold_accuracy(errs, 0.5),
            "acc_1m": threshold_accuracy(errs, 1.0),
            "acc_2m": threshold_accuracy(errs, 2.0),
            "acc_3m": threshold_accuracy(errs, 3.0),
            "zone_routing_acc": zone_acc,
        }

    result = cross_validate(
        X=X,
        y=y_xyz,
        train_fn=train_fn,
        predict_fn=predict_fn,
        eval_fn=eval_fn,
        aggregate_fn=aggregate_mean_std,
        k=K_FOLDS,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    print_summary(result["summary"])


def _nearest_antenna(y_xyz):
    """Assign each sample to the nearest antenna by XY distance."""
    antenna_xy = np.array(
        [ANTENNA_POSITIONS[name][:2] for name in ANTENNA_NAMES], dtype=np.float32
    )
    from scipy.spatial.distance import cdist
    dists = cdist(np.asarray(y_xyz)[:, :2], antenna_xy)
    return np.argmin(dists, axis=1)


if __name__ == "__main__":
    run(sys.argv[1:])
