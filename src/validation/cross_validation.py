from sklearn.model_selection import KFold
#from typing import Dict

def cross_validate(X, y, train_fn, predict_fn, eval_fn, aggregate_fn, k=5, random_state=42):
    """
    Perform k-fold cross-validation and return detailed results per fold and aggregated summary.
    
    Args:
        X: Input features
        y: Target values
        train_fn: Function to train a model ((X_train, y_train) -> model)
        predict_fn: Function to make predictions ((model, X_test) -> y_pred))
        eval_fn: Function to evaluate predictions ((y_test, y_pred) -> dict[str, float])
        aggregate_fn: Function to aggregate fold results into summary statistics
            ((List[Dict[str, float]]) -> Dict[str, Dict[str, float]])
        k: Number of folds
        random_state: Random seed for reproductibility
    Returns:
        Dict with two keys:
        - "folds": List of dictionaries, each containing metrics for a fold.
        - "summary": Aggregated summary of metrics across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    folds_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = train_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)

        fold_result = eval_fn(y_test, y_pred)
        folds_results.append(fold_result)
        summary_result = aggregate_fn(folds_results)
        
    return {
        "folds": folds_results,
        "summary": summary_result
    }
    

# examples :

# fold_result = {"rmse_3d":1.23,"acc_1m":45}

# folds_results
#  [
#   {"rmse_3d":1.23,"acc_1m":45},
#   {"rmse_3d":1.11,"acc_1m":66},
# ] 

# summary_result :
# {
#   "rmse_3d":{"mean":0.40,"std":0.01},
#   "acc_1m":{"mean":83,"std":1}
# } 
