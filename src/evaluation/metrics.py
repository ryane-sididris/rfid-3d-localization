import numpy as np
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

