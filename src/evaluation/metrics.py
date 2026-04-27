import numpy as np
import pandas as pd
# euclidian error = √((x_true-x_pred)²+(y_true-y_pred)²+(z_true-z_pred)²)
# errs = np.linalg.norm(preds - true_positions, axis=1)

def mae_3d(errs: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) from an array of errors.

    This function works for:
    - distance errors between predicted and true antenna distances
    - Euclidean position errors in 3D space

    Args:
        errs: Array of errors (absolute distance errors or Euclidean errors)

    Returns:
        Mean error.
    """
    return float(np.mean(errs))

def rmse_3d(errs: np.ndarray) -> float:
    """Calculate the 3D Root Mean Squared Error (RMSE) between true and predicted positions.

    The 3D RMSE is the square root of the average squared Euclidean distances between each pair
    of true and predicted 3D coordinates.

    Args: 
        errs: Euclidean distance errors for each sample
    Returns:
        Root Mean Squared Error of the Euclidean distances between predicted and true 3D positions
    """
    return np.sqrt(np.mean(errs ** 2))




def rmse_xyz(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-axis average RMSE over x, y, z coordinates.

    Equivalent to sklearn's sqrt(mean_squared_error(y_true, y_pred)) on (n, 3) arrays,
    which averages the squared errors over all samples AND all 3 axes before taking sqrt.

    This differs from rmse_3d by a factor of 1/√3:
        rmse_xyz = rmse_3d(errs) / √3

    Args:
        y_true: True positions, shape (n, 3)
        y_pred: Predicted positions, shape (n, 3)
    Returns:
        Per-axis RMSE (in meters). Smaller than rmse_3d for the same predictions.
    """
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))


def xy_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Euclidean errors in the XY plane."""
    y_true_xy = np.asarray(y_true)[:, :2]
    y_pred_xy = np.asarray(y_pred)[:, :2]
    return np.linalg.norm(y_true_xy - y_pred_xy, axis=1)


def rmse_xy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared Euclidean error in the XY plane."""
    errs_xy = xy_errors(y_true, y_pred)
    return float(np.sqrt(np.mean(errs_xy ** 2)))


def rmse_z(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error on the Z axis only."""
    z_true = np.asarray(y_true)[:, 2]
    z_pred = np.asarray(y_pred)[:, 2]
    return float(np.sqrt(np.mean((z_true - z_pred) ** 2)))


def threshold_accuracy(errs: np.ndarray, threshold: float) -> float:
    """Calculate the percentage of predictions with error below a given threshold.

    Args:
        errs: Euclidean distance errors for each sample
    Returns:
        Percentage of predictions with error ≤ threshold
    """
    return (errs <= threshold).mean()*100

# print(f"  % ≤ {threshold}m : {threshold_accuracy(errs, threshold):.1f}%")
# ex: "% ≤ 1.0m : 49.3%"

def evaluate_model(y_true, y_pred, model_name="Model"):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    errs = np.linalg.norm(y_t - y_p, axis=1)
    
    rmse = rmse_3d(errs)
    pct_1m = threshold_accuracy(errs, 1.0)
    pct_2m = threshold_accuracy(errs, 2.0)
    pct_3m = threshold_accuracy(errs, 3.0)
    
    describe_stats = pd.Series(errs).describe()
    
    print(f"========== {model_name} ==========")
    print(f"RMSE 3D         : {rmse:.3f} m")
    print(f"Erreurs < 1m    : {pct_1m:.2f} %")
    print(f"Erreurs < 2m    : {pct_2m:.2f} %")
    print(f"Erreurs < 3m    : {pct_3m:.2f} %")
    print("\n--- 3D Errors Describe ---")
    print(describe_stats.to_string(float_format="{:.3f}".format))
    print("==================================\n")
