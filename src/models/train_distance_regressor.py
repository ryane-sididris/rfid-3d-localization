from sklearn.ensemble import ExtraTreesRegressor

def train_distance_regressor(params, X_train, y_train):
    """
    Trains a multi-output ExtraTreesRegressor:
    RSSI features -> distances to antennas.
    Unlike train_extratrees, we use ExtraTreesRegressor
    directly (without MultiOutputRegressor) because all outputs
    are homogeneous (distances in meters).

    Args:
        params  : hyperparameters for ExtraTreesRegressor
        X_train : RSSI features  (n_samples, n_features)
        y_train : target distances (n_samples, n_antennas)
    Returns:
        Trained ExtraTreesRegressor
    """
    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)
    return model