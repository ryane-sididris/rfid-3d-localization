import numpy as np
import pandas as pd
from pathlib import Path

from collections import defaultdict
from src.loaders.loader_advanced_ds import AdvancedDsLoader
from src.loaders.loader_elemen_ds import ElemenDsLoader
from src.loaders.loader_tx_rx_ds import TxRxDsLoader
from src.loaders.loader_tx_ds import TxDsLoader
from src.models.train_extratrees import train_extratrees as train_extratrees_model
from src.evaluation.metrics import rmse_3d, threshold_accuracy
from src.validation.cross_validation import cross_validate

print("loading dataset")

DATA_DIR = Path("data")

#DS_NAME = "ds_advanced_17Feb26.csv"
#loader = AdvancedDsLoader()

DS_NAME = "ds_elemen.csv"
loader = ElemenDsLoader()

# DS_NAME = "ds_Tx_Rx.csv"
# loader = TxRxDsLoader()

#DS_NAME = "ds_Tx.csv"
#loader = TxDsLoader()

X, y_xyz, _, _ = loader.load(DATA_DIR / DS_NAME)

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
    y=y_xyz,
    train_fn = train_extratrees,
    predict_fn = predict_extratrees,
    eval_fn = evaluate_position,
    aggregate_fn = aggregate_mean_std,
    k=5,
    random_state=42,
)

print("=== extratrees -", DS_NAME, "===")
print("Params : " + ", ".join(f"{k}={v}" for k, v in _params.items()))

#folds_results = pd.DataFrame(result["folds"])
#print(folds_results)
summary = pd.DataFrame(result["summary"]).T
print(summary)

# python -m experiments.run_extratrees