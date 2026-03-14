from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
# _params = dict(n_estimators=200, random_state=0, n_jobs=-1)

def train_extratrees(_params, X_train, y_train):
    """
    Train a MultiOutput ExtraTrees regressor model.

    Args:
        _params (Dict[str, Any]): Dictionary of hyperparameters for the
            ExtraTreesRegressor base estimator. Common parameters include:
            n_estimators: Number of trees in the forest
            random_state: Seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            max_depth: Maximum depth of the trees
        X_train (np.ndarray): Training input features
        y_rain (np.ndarray): Training target values
    Returns:
        MultiOutputRegressor: Trained multi-output regressor model, wrapping an
            ExtraTreesRegressor for each output dimension.
    """
    model = MultiOutputRegressor(ExtraTreesRegressor(**_params))
    model.fit(X_train, y_train)
    return model