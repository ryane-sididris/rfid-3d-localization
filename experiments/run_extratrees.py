import numpy as np
import pandas as pd
from pathlib import Path

from collections import defaultdict
from src.loaders.loader_advanced_ds import AdvancedDsLoader
from src.loaders.loader_elemen_ds import ElemenDsLoader
from src.models.train_extratrees import train_extratrees as train_extratrees_model
from src.evaluation.metrics import rmse_3d, threshold_accuracy
from src.validation.cross_validation import cross_validate

print("loading dataset")

DATA_DIR = Path("data")

DS_NAME = "ds_advanced_18Feb26.csv"
loader = AdvancedDsLoader()

# DS_NAME = "data/ds_elemen.csv"
# loader = ElemenDsLoader()

X, y, _ = loader.load(DATA_DIR / DS_NAME)

_params = dict(n_estimators=100, random_state=42, n_jobs=-1)
def train_extratrees(X_train, y_train):
    params = _params
    return train_extratrees_model(params, X_train, y_train)

def predict_extratrees(model, X_test):
    return model.predict(X_test)

def evaluate_position(y_true, y_pred, _):
    errs = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "rmse": rmse_3d(errs),
        "acc_1m": threshold_accuracy(errs, 1),
        "acc_2m": threshold_accuracy(errs, 2),
        "acc_3m": threshold_accuracy(errs, 3)
    }

def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for f in folds:
        for k,v in f.items():
            metrics[k].append(v)
    return {
        k:{
            "mean": np.mean(v),
            "std": np.std(v)
        }
        for k,v in metrics.items()
    }


print("training & validation")

result = cross_validate(
    X=X,
    y=y,
    train_fn = train_extratrees,
    predict_fn = predict_extratrees,
    eval_fn = evaluate_position,
    aggregate_fn = aggregate_mean_std,
    k=5,
    random_state=42,
)

print("=== extratrees ===")
print("  Params : " + ", ".join(f"{k}={v}" for k, v in _params.items()))

df = pd.DataFrame(result["folds"])
print(df)
summary = pd.DataFrame(result["summary"]).T
print(summary)

# python -m experiments.run_extratrees




# with advanced ds:
# === extratrees ===
#   Params : n_estimators=100, random_state=42, n_jobs=-1
#        rmse     acc_1m     acc_2m     acc_3m
# 0  1.247622  49.150623  92.751982  98.640997
# 1  1.244354  51.473923  93.537415  98.866213
# 2  1.290734  47.959184  91.836735  98.639456
# 3  1.227800  49.773243  93.424036  99.319728
# 4  1.274679  48.072562  92.857143  98.639456
#              mean       std
# rmse     1.257038  0.022588
# acc_1m  49.285907  1.286516
# acc_2m  92.881462  0.605553
# acc_3m  98.821170  0.264231