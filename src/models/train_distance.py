from .regressors import get_regressor

def train_distance(model_name: str, X_train, y_train, params: dict = None):
    """
    Predicts distances to antennas, without MultiOutputRegressor
    since all outputs are homogeneous (meters).
    """
    model = get_regressor(model_name, params, multioutput=True)
    model.fit(X_train, y_train)
    return model