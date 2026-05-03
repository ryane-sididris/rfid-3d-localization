from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm


ANTENNA_POSITIONS = {
    "PORT_1": (5.7, 4.0, 3.9),
    "PORT_2": (8.5, 3.8, 3.9),
    "PORT_3": (5.6, 1.7, 3.9),
    "PORT_4": (8.4, 2.1, 3.9),
}

RX_ORDER = ["PORT_1__NONE", "PORT_2__NONE", "PORT_3__NONE", "PORT_4__NONE"]
RATIO_PREFIX = "mean__sqrt_rssi1/rssi2__"
PRED_COLS = ["x_ratio_lsq_no_fusion_linear", "y_ratio_lsq_no_fusion_linear", "z_ratio_lsq_no_fusion_linear"]


def solve_ratio_positions(
    df: pd.DataFrame,
    *,
    x_bounds=(1.5, 12.0),
    y_bounds=(1.0, 5.5),
    z_min=0.0,
    loss="soft_l1",
    f_scale=0.2,
    max_nfev=120,
    min_constraints=3,
    show_progress=True,
) -> pd.DataFrame:
    ratio_cols = [col for col in df.columns if col.startswith(RATIO_PREFIX)]
    if not ratio_cols:
        raise ValueError("No ratio feature columns found in dataframe.")

    rx_order = [rx for rx in RX_ORDER if any(rx in col for col in ratio_cols)]
    ratio_lookup = {
        tuple(col[len(RATIO_PREFIX):].split("|", maxsplit=1)): col
        for col in ratio_cols
    }
    anchors = {
        rx: np.asarray(ANTENNA_POSITIONS[rx.split("__", maxsplit=1)[0]], dtype=float)
        for rx in rx_order
    }

    z_plane = float(np.mean([anchors[rx][2] for rx in rx_order]))
    lower = np.array([x_bounds[0], y_bounds[0], z_min], dtype=float)
    upper = np.array([x_bounds[1], y_bounds[1], z_plane - 1e-6], dtype=float)
    x0 = 0.5 * (lower + upper)

    def build_direct_constraints(row: pd.Series):
        constraints = []
        for (rx_i, rx_j), col in ratio_lookup.items():
            value = row[col]
            if pd.notna(value) and value > 0:
                constraints.append((rx_i, rx_j, float(value)))
        return constraints

    def residuals(theta: np.ndarray, constraints):
        point = np.asarray(theta, dtype=float)
        distances = {
            rx: max(float(np.linalg.norm(point - anchor)), 1e-9)
            for rx, anchor in anchors.items()
        }
        return np.asarray(
            [(distances[rx_j] / distances[rx_i]) - ratio for rx_i, rx_j, ratio in constraints],
            dtype=float,
        )

    preds = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Solving") if show_progress else df.iterrows()
    for _, row in iterator:
        constraints = build_direct_constraints(row)
        if len(constraints) < min_constraints:
            preds.append((np.nan, np.nan, np.nan))
            continue

        result = least_squares(
            residuals,
            x0=x0,
            bounds=(lower, upper),
            args=(constraints,),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        preds.append(tuple(result.x))

    out = df.copy()
    out[PRED_COLS] = pd.DataFrame(preds, index=out.index)
    return out


def compute_ratio_position_metrics(
    df: pd.DataFrame,
    *,
    true_cols=("x", "y", "z"),
    pred_cols=PRED_COLS,
) -> dict[str, float]:
    if not set(true_cols).issubset(df.columns) or not set(pred_cols).issubset(df.columns):
        return {}

    y_true = df[list(true_cols)].to_numpy(dtype=float)
    y_pred = df[list(pred_cols)].to_numpy(dtype=float)
    valid = np.isfinite(y_pred).all(axis=1)
    if not np.any(valid):
        return {}

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    errs = np.linalg.norm(y_true - y_pred, axis=1)
    diff = y_true - y_pred
    errs_xy = np.linalg.norm(diff[:, :2], axis=1)

    return {
        "rows_solved": float(len(errs)),
        "mae_3d": float(np.mean(errs)),
        "rmse_3d": float(np.sqrt(np.mean(errs**2))),
        "rmse_xyz": float(np.sqrt(np.mean(diff**2))),
        "rmse_xy": float(np.sqrt(np.mean(errs_xy**2))),
        "rmse_z": float(np.sqrt(np.mean(diff[:, 2] ** 2))),
        "acc_1m": float((errs <= 1.0).mean() * 100),
        "acc_2m": float((errs <= 2.0).mean() * 100),
        "acc_3m": float((errs <= 3.0).mean() * 100),
    }


def print_ratio_position_metrics(metrics: dict[str, float]) -> None:
    if not metrics:
        return
    print("\nMetrics")
    print(f"  rows solved : {int(metrics['rows_solved'])}")
    print(f"  mae_3d      : {metrics['mae_3d']:.3f} m")
    print(f"  rmse_3d     : {metrics['rmse_3d']:.3f} m")
    print(f"  rmse_xyz    : {metrics['rmse_xyz']:.3f} m")
    print(f"  rmse_xy     : {metrics['rmse_xy']:.3f} m")
    print(f"  rmse_z      : {metrics['rmse_z']:.3f} m")
    print(f"  acc <= 1m   : {metrics['acc_1m']:.2f} %")
    print(f"  acc <= 2m   : {metrics['acc_2m']:.2f} %")
    print(f"  acc <= 3m   : {metrics['acc_3m']:.2f} %")


if __name__ == "__main__":
    dataset_path = Path("data/ds_analytical26April9h00.csv")
    output_path = dataset_path.with_name(f"{dataset_path.stem}_ratio_solver_no_fusion_linear.csv")

    df = pd.read_csv(dataset_path)
    df = solve_ratio_positions(df)
    print_ratio_position_metrics(compute_ratio_position_metrics(df))
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions : {output_path}")
