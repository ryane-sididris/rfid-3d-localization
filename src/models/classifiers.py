from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

CLASSIFIER_REGISTRY = {
    "extratrees":       ExtraTreesClassifier,
    "randomforest":     RandomForestClassifier,
    "gradientboosting": GradientBoostingClassifier,
    "svm_rbf":          SVC,
    "mlp":              MLPClassifier,
}

DEFAULT_PARAMS = {
    "extratrees":       dict(n_estimators=100, random_state=42, n_jobs=-1),
    "randomforest":     dict(n_estimators=100, random_state=42, n_jobs=-1),
    "gradientboosting": dict(n_estimators=100, random_state=42),
    "svm_rbf":          dict(kernel="rbf", random_state=42),
    "mlp":              dict(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}


def get_classifier(name: str, params: dict = None):
    """
    Instantiate a classifier by name.

    Args:
        name:   Key from CLASSIFIER_REGISTRY (case-insensitive).
        params: Hyperparameters. Defaults to DEFAULT_PARAMS[name].
    Returns:
        Fitted-ready sklearn estimator.
    """
    key = name.lower()
    if key not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Choose from: {list(CLASSIFIER_REGISTRY)}")
    resolved_params = params if params is not None else DEFAULT_PARAMS[key]
    return CLASSIFIER_REGISTRY[key](**resolved_params)


def available_classifiers():
    """Return the list of available classifier names."""
    return sorted(CLASSIFIER_REGISTRY.keys())