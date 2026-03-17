"""
Classification then regression pipeline.

1. Classifier: RSSI -> zone (voxel)
2. Regressor: RSSI + zone_id -> deltas (dx, dy, dz) x 8 antennas
3. Reconstruct 8 positions (antenna_pos + delta), then average -> final position

Metrics: per-antenna RMSE + overall RMSE, acc@1m/2m/3m

Usage:
    python -m experiments.run_classif_then_regression                  # run all models
    python -m experiments.run_classif_then_regression extratrees mlp   # run specific models
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold

from src.loaders.loader_advanced_ds_with_deltas import AdvancedDsWithDeltasLoader
from src.models.classif_then_regressor import ClassifThenRegressor
from src.config.antennas import ANTENNA_NAMES
from src.evaluation.delta_regression import (
    aggregate_mean_std,
    compute_per_antenna_position_errors,
    evaluate_delta_regression_fold,
    get_antenna_positions_array,
)
from src.evaluation.metrics import rmse_3d
from src.models.regressors import get_regressor
from src.models.classifiers import get_classifier

# ─── config ───────────────────────────────────────────────────────────

DATA_PATH = Path("data") / "ds_advanced_17Feb26.csv"
VOXEL_SIZE = 2.0
K_FOLDS = 5
RANDOM_STATE = 42


def get_models():
    """
    Returns dict of {model_name: (classifier, regressor)}.
    Regressor is wrapped in MultiOutputRegressor when needed (GBR, SVM).
    """
    names = ["extratrees", "randomforest", "gradientboosting", "svm_rbf", "mlp"]
    return {
        name: (
            get_classifier(name),
            get_regressor(name, multioutput=True),
        )
        for name in names
    }

def evaluate_fold(y_xyz_test, y_deltas_pred, y_deltas_test, antenna_positions_arr, zone_ids_test):
    """
    Compute:
    - per-antenna delta RMSE (pred deltas vs true deltas, before reconstruction)
    - per-antenna reconstructed position RMSE
    - aggregated position RMSE + accuracy thresholds
    - per-zone breakdown
    """
    metrics = evaluate_delta_regression_fold(
        y_xyz_true=y_xyz_test,
        y_deltas_pred=y_deltas_pred,
        y_deltas_true=y_deltas_test,
        antenna_positions=antenna_positions_arr,
        antenna_names=ANTENNA_NAMES,
    )
    per_ant_errs, _, _ = compute_per_antenna_position_errors(
        y_xyz_true=y_xyz_test,
        y_deltas_pred=y_deltas_pred,
        antenna_positions=antenna_positions_arr,
        antenna_names=ANTENNA_NAMES,
    )

    # per-zone breakdown
    zone_ant_rmse = {}
    unique_zones = np.unique(zone_ids_test)
    for zone in unique_zones:
        mask = zone_ids_test == zone
        zone_ant_rmse[zone] = {}
        for ant_name in ANTENNA_NAMES:
            zone_errs = per_ant_errs[ant_name][mask]
            zone_ant_rmse[zone][ant_name] = rmse_3d(zone_errs)
    metrics["zone_ant_rmse"] = zone_ant_rmse

    return metrics

def aggregate_zone_ant_rmse(folds):
    """Average zone x antenna RMSE across folds (zones present in multiple folds)."""
    zone_ant_values = defaultdict(lambda: defaultdict(list))
    for f in folds:
        for zone, ant_dict in f["zone_ant_rmse"].items():
            for ant_name, rmse_val in ant_dict.items():
                zone_ant_values[zone][ant_name].append(rmse_val)

    # compute mean per zone x antenna
    result = {}
    for zone in sorted(zone_ant_values.keys()):
        result[zone] = {}
        for ant_name in ANTENNA_NAMES:
            vals = zone_ant_values[zone][ant_name]
            result[zone][ant_name] = np.mean(vals)
    return result


# ─── main ─────────────────────────────────────────────────────────────

def run(model_names=None):
    all_models = get_models()

    if model_names:
        for name in model_names:
            if name not in all_models:
                print(f"Unknown model: {name}")
                print(f"Available: {', '.join(all_models.keys())}")
                return
        selected = {name: all_models[name] for name in model_names}
    else:
        selected = all_models

    print(f"Loading dataset: {DATA_PATH}")
    loader = AdvancedDsWithDeltasLoader(voxel_size=VOXEL_SIZE)
    X, y_deltas, y_xyz, zone_ids, df = loader.load(DATA_PATH)

    antenna_positions_arr = get_antenna_positions_array(ANTENNA_NAMES)

    n_zones = len(np.unique(zone_ids))
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Zones: {n_zones}")
    print(f"Models to run: {', '.join(selected.keys())}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(X))

    from sklearn.base import clone

    for model_name, (clf_template, reg_template) in selected.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        # run two variants: without zone feature, then with zone feature
        for use_zone in [False, True]:
            variant = "WITH zone" if use_zone else "WITHOUT zone"
            print(f"\n  --- {variant} as regressor feature ---")

            folds_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_deltas_train, y_deltas_test = y_deltas[train_idx], y_deltas[test_idx]
                y_xyz_test = y_xyz[test_idx]
                zones_train = zone_ids[train_idx]
                zones_test = zone_ids[test_idx]

                clf = clone(clf_template)
                reg = clone(reg_template)

                pipeline = ClassifThenRegressor(clf, reg, use_zone_feature=use_zone)
                pipeline.train(X_train, y_deltas_train, zones_train)
                y_deltas_pred, _ = pipeline.predict(X_test)

                fold_metrics = evaluate_fold(y_xyz_test, y_deltas_pred, y_deltas_test, antenna_positions_arr, zones_test)
                folds_results.append(fold_metrics)

                print(f"    Fold {fold_idx+1}: rmse_avg={fold_metrics['rmse_avg']:.3f}  "
                      f"acc@1m={fold_metrics['acc_1m']:.1f}%")

            summary = aggregate_mean_std(folds_results, skip_keys={"zone_ant_rmse"})

            print(f"\n    Per-antenna delta RMSE (pred vs true deltas, mean ± std):")
            for ant_name in ANTENNA_NAMES:
                key = f"delta_rmse_{ant_name}"
                print(f"      {ant_name:>10s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")

            print(f"\n    Per-antenna position RMSE (reconstructed, mean ± std):")
            for ant_name in ANTENNA_NAMES:
                key = f"pos_rmse_{ant_name}"
                print(f"      {ant_name:>10s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")

            print(f"\n    Aggregated (mean ± std):")
            for key in ["rmse_avg", "rmse_sklearn", "acc_1m", "acc_2m", "acc_3m"]:
                print(f"      {key:>10s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")

            zone_ant = aggregate_zone_ant_rmse(folds_results)
            print(f"\n    RMSE per zone x antenna:")
            header = "        zone  | " + " | ".join(f"{a:>10s}" for a in ANTENNA_NAMES)
            print(header)
            print("        " + "-" * (len(header) - 8))
            for zone, ant_dict in zone_ant.items():
                vals = " | ".join(f"{ant_dict[a]:10.3f}" for a in ANTENNA_NAMES)
                print(f"        {zone:>5d} | {vals}")


if __name__ == "__main__":
    # pass model names as CLI args, or nothing to run all
    args = sys.argv[1:]
    run(model_names=args if args else None)
