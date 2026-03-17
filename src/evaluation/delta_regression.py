from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error

from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS
from src.evaluation.metrics import rmse_3d, threshold_accuracy


def get_antenna_positions_array(antenna_names=None):
    names = antenna_names if antenna_names is not None else ANTENNA_NAMES
    return np.array([ANTENNA_POSITIONS[name] for name in names], dtype=np.float32)


def deltas_to_positions(y_deltas, antenna_positions):
    """
    Reconstruct absolute positions from predicted deltas.

    y_deltas: (n_samples, 3 * n_antennas)
    antenna_positions: (n_antennas, 3)
    """
    n_samples = y_deltas.shape[0]
    n_antennas = len(antenna_positions)
    positions = np.zeros((n_samples, n_antennas, 3), dtype=np.float32)

    for i in range(n_antennas):
        positions[:, i, :] = antenna_positions[i] + y_deltas[:, i * 3:(i + 1) * 3]

    return positions


def compute_per_antenna_position_errors(y_xyz_true, y_deltas_pred, antenna_positions, antenna_names=None):
    names = antenna_names if antenna_names is not None else ANTENNA_NAMES
    pos_per_ant = deltas_to_positions(y_deltas_pred, antenna_positions)

    per_ant_errs = {}
    for i, ant_name in enumerate(names):
        per_ant_errs[ant_name] = np.linalg.norm(pos_per_ant[:, i, :] - y_xyz_true, axis=1)

    pos_mean = pos_per_ant.mean(axis=1)
    errs_mean = np.linalg.norm(pos_mean - y_xyz_true, axis=1)

    return per_ant_errs, pos_mean, errs_mean


def evaluate_delta_regression_fold(
    y_xyz_true,
    y_deltas_pred,
    y_deltas_true,
    antenna_positions,
    antenna_names=None,
):
    names = antenna_names if antenna_names is not None else ANTENNA_NAMES
    metrics = {}

    for i, ant_name in enumerate(names):
        delta_pred_i = y_deltas_pred[:, i * 3:(i + 1) * 3]
        delta_true_i = y_deltas_true[:, i * 3:(i + 1) * 3]
        delta_errs = np.linalg.norm(delta_pred_i - delta_true_i, axis=1)
        metrics[f"delta_rmse_{ant_name}"] = rmse_3d(delta_errs)

    per_ant_errs, pos_mean, errs_mean = compute_per_antenna_position_errors(
        y_xyz_true=y_xyz_true,
        y_deltas_pred=y_deltas_pred,
        antenna_positions=antenna_positions,
        antenna_names=names,
    )

    for ant_name in names:
        metrics[f"pos_rmse_{ant_name}"] = rmse_3d(per_ant_errs[ant_name])

    metrics["rmse_avg"] = rmse_3d(errs_mean)
    metrics["rmse_sklearn"] = float(np.sqrt(mean_squared_error(y_xyz_true, pos_mean)))
    metrics["acc_1m"] = threshold_accuracy(errs_mean, 1)
    metrics["acc_2m"] = threshold_accuracy(errs_mean, 2)
    metrics["acc_3m"] = threshold_accuracy(errs_mean, 3)

    return metrics


def aggregate_mean_std(folds, skip_keys=None):
    skip = set(skip_keys or [])
    metrics = defaultdict(list)

    for fold in folds:
        for key, value in fold.items():
            if key in skip:
                continue
            if isinstance(value, (int, float, np.floating)):
                metrics[key].append(float(value))

    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in metrics.items()
    }
