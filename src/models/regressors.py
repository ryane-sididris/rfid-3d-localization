from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

REGRESSOR_REGISTRY = {
    "extratrees":       ExtraTreesRegressor,
    "randomforest":     RandomForestRegressor,
    "gradientboosting": GradientBoostingRegressor,
    "xgboost":          XGBRegressor,
    "svm_rbf":          SVR,
    "mlp":              MLPRegressor,
}

# Models that do not natively support multi-output regression
MULTIOUTPUT_WRAP = {"gradientboosting", "svm_rbf"}

DEFAULT_PARAMS = {
    "extratrees":       dict(n_estimators=100, random_state=42, n_jobs=-1),
    "randomforest":     dict(n_estimators=100, random_state=42, n_jobs=-1),
    "gradientboosting": dict(n_estimators=100, random_state=42),
    "xgboost":          dict(n_estimators=400, max_depth=6, learning_rate=0.05,subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0),
    "svm_rbf":          dict(kernel="rbf"),
    "mlp":              dict(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}


def get_regressor(name: str, params: dict = None, multioutput: bool = False):
    """
    Instantiate a regressor by name.

    Args:
        name:        Key from REGRESSOR_REGISTRY (case-insensitive).
        params:      Hyperparameters. Defaults to DEFAULT_PARAMS[name].
        multioutput: If True, wrap in MultiOutputRegressor when needed.
    Returns:
        Fitted-ready sklearn estimator.
    """
    key = name.lower()
    if key not in REGRESSOR_REGISTRY:
        raise ValueError(f"Unknown regressor '{name}'. Choose from: {list(REGRESSOR_REGISTRY)}")
    resolved_params = params if params is not None else DEFAULT_PARAMS[key]
    estimator = REGRESSOR_REGISTRY[key](**resolved_params)
    if multioutput and key in MULTIOUTPUT_WRAP:
        return MultiOutputRegressor(estimator)
    return estimator


def available_regressors():
    """Return the list of available regressor names."""
    return sorted(REGRESSOR_REGISTRY.keys())


def get_default_params(name: str) -> dict:
    """Return the default hyperparameters for a given regressor."""
    key = name.lower()
    if key not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown regressor '{name}'. Choose from: {list(REGRESSOR_REGISTRY)}")
    return DEFAULT_PARAMS[key]