from sklearn.multioutput import MultiOutputRegressor
from .regressors import get_regressor

def train_position(model_name: str, X_train, y_train, params: dict = None):
    """
    Predicted XYZ - wrapped in MultiOutputRegressor
    because the 3 outputs are heterogeneous (x, y, z).
    """
    model = MultiOutputRegressor(get_regressor(model_name, params))
    model.fit(X_train, y_train)
    return model