import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.loaders.loader_advanced_ds_with_distances import AdvancedDsWithDistancesLoader
from src.models.train_distance_regressor import train_distance_regressor as _train_fn
from src.utils.trilaterate_numerical import trilaterate
from src.evaluation.metrics import mae_3d, rmse_3d, threshold_accuracy
from src.validation.cross_validation import cross_validate
from src.config.antennas import ANTENNA_POSITIONS

print("loading dataset")

DATA_DIR = Path("data")
DS_NAME  = "ds_advanced_18Feb26.csv"
loader = AdvancedDsWithDistancesLoader()

X, y_distances, df = loader.load(DATA_DIR / DS_NAME)
y_xyz = df[["x", "y", "z"]].values.astype(np.float32)

antenna_names = loader.antenna_names
antenna_positions = np.array([ANTENNA_POSITIONS[a] for a in antenna_names])

_params = dict(n_estimators=200, random_state=42, n_jobs=-1)
def train_fn(X_train, y_train):
    params = _params
    return _train_fn(params, X_train, y_train)

def predict_fn(model, X_test):
    return model.predict(X_test)


def evaluate_position(y_dist_test, y_dist_pred, test_index):
    """
    y_dist_test: True distances to the antennas (n_samples, n_antennas).
    y_dist_pred: Predicted distances to the antennas (n_samples, n_antennas).
    test_index: Indexes of the test samples used to extract y_xyz_test from the closure.
    """
    y_xyz_test = y_xyz[test_index]

    # trilateration -> predicted xyz
    y_xyz_pred = np.array([
        trilaterate(row, antenna_positions)
        for row in y_dist_pred
    ])

    # distance error and Euclidean error
    dist_errs = np.abs(y_dist_pred - y_dist_test)
    pos_errs = np.linalg.norm(y_xyz_pred - y_xyz_test, axis=1)

    return {
        "dist_mae":  mae_3d(dist_errs),
        "rmse":      rmse_3d(pos_errs),
        "acc_1m":    threshold_accuracy(pos_errs, 1),
        "acc_2m":    threshold_accuracy(pos_errs, 2),
        "acc_3m":    threshold_accuracy(pos_errs, 3),
    }

def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for f in folds:
        for k, v in f.items():
            metrics[k].append(v)
    return {
        k: {"mean": np.mean(v), "std": np.std(v)}
        for k, v in metrics.items()
    }


print("training & validation")

result = cross_validate(
    X=X,
    y=y_distances,
    train_fn=train_fn,
    predict_fn=predict_fn,
    eval_fn=evaluate_position,
    aggregate_fn=aggregate_mean_std,
    k=5,
    random_state=42,
)



print("=== extratrees distances ===")
print("  Params : " + ", ".join(f"{k}={v}" for k, v in _params.items()))

df_folds   = pd.DataFrame(result["folds"])
df_summary = pd.DataFrame(result["summary"]).T

print(df_folds)
print(df_summary)

# python -m experiments.run_extratrees_distance

# # with advanced ds : 
# === extratrees distances ===
#   Params : n_estimators=200, random_state=42, n_jobs=-1
#    dist_mae      rmse     acc_1m     acc_2m     acc_3m
# 0  0.543347  1.309534  45.300113  91.279728  98.414496
# 1  0.533761  1.288297  46.938776  92.630385  98.866213
# 2  0.552805  1.337929  44.331066  92.063492  98.299320
# 3  0.539510  1.277583  47.392290  92.403628  99.206349
# 4  0.539253  1.338791  44.671202  91.609977  98.526077
#                mean       std
# dist_mae   0.541735  0.006322
# rmse       1.310426  0.025021
# acc_1m    45.726689  1.223696
# acc_2m    91.997442  0.497033
# acc_3m    98.662491  0.331448