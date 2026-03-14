import numpy as np
import pandas as pd
from collections import defaultdict
from src.loaders.loader_advanced_ds import AdvancedDsLoader
from src.loaders.loader_elemen_ds import ElemenDsLoader
from src.models.train_extratrees import train_extratrees as train_extratrees_model
from src.evaluation.metrics import rmse_3d, threshold_accuracy
from src.validation.cross_validation import cross_validate

print("loading dataset")
#X, y, _ = AdvancedDsLoader().load("data/ds_advanced_18Feb26.csv")
X, y, _ = ElemenDsLoader().load("data/ds_elemen.csv")

def train_extratrees(X_train, y_train):
    params = dict(n_estimators=200, random_state=42, n_jobs=-1)
    return train_extratrees_model(params, X_train, y_train)

def predict_extratrees(model, X_test):
    return model.predict(X_test)

def evaluate_position(y_true, y_pred):
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

df = pd.DataFrame(result["folds"])
print(df)
summary = pd.DataFrame(result["summary"]).T
print(summary)

# python -m experiments.run_extratrees