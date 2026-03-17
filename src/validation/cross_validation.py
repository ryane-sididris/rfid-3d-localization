from collections import defaultdict
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold


def aggregate_mean_std(folds_results):
    """
    Agregation simple des metriques numeriques par fold.
    Retour: {metric: {"mean": ..., "std": ...}}
    """
    metrics = defaultdict(list)
    for fold in folds_results:
        for key, value in fold.items():
            metrics[key].append(value)

    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in metrics.items()
    }


def _format_fold_metrics(fold_result: dict) -> str:
    preferred = ["rmse", "dist_mae", "acc_1m", "acc_2m", "acc_3m"]
    items = []

    for key in preferred:
        value = fold_result.get(key)
        if isinstance(value, (int, float, np.floating)):
            if key.startswith("acc_"):
                items.append(f"{key}={float(value):.1f}%")
            else:
                items.append(f"{key}={float(value):.4f}")

    if items:
        return ", ".join(items)

    fallback_items = []
    for key, value in fold_result.items():
        if isinstance(value, (int, float, np.floating)):
            fallback_items.append(f"{key}={float(value):.4f}")
    return ", ".join(fallback_items)


def cross_validate(
    X,
    y,
    train_fn,
    predict_fn,
    eval_fn,
    aggregate_fn,
    k=5,
    random_state=42,
    verbose=False,
):

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    folds_results = []

    total_start = perf_counter()

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold_start = perf_counter()
        if verbose:
            print(f"[CV] Fold {fold_idx}/{k} - training...", flush=True)

        model = train_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)
        fold_result = eval_fn(y_test, y_pred, test_index)
        folds_results.append(fold_result)

        if verbose:
            elapsed = perf_counter() - fold_start
            metrics_txt = _format_fold_metrics(fold_result)
            if metrics_txt:
                print(
                    f"[CV] Fold {fold_idx}/{k} done in {elapsed:.2f}s | {metrics_txt}",
                    flush=True,
                )
            else:
                print(f"[CV] Fold {fold_idx}/{k} done in {elapsed:.2f}s", flush=True)

    summary_result = aggregate_fn(folds_results)
    if verbose:
        total_elapsed = perf_counter() - total_start
        print(f"[CV] Completed {k} folds in {total_elapsed:.2f}s", flush=True)

    return {"folds": folds_results, "summary": summary_result}
