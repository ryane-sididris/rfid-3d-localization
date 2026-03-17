import numpy as np
from sklearn.base import clone

from src.config.antennas import ANTENNA_NAMES


class PerAntennaRegressor:
    """
    Train one regressor per antenna.

    Each regressor predicts the 3 deltas [dx, dy, dz] for its antenna.
    Feature selection is optional; when omitted, every antenna model uses all features.
    """

    def __init__(self, regressor, feature_indices_by_antenna=None, antenna_names=None):
        self.regressor = regressor
        self.feature_indices_by_antenna = feature_indices_by_antenna or {}
        self.antenna_names = antenna_names if antenna_names is not None else ANTENNA_NAMES
        self.models = {}

    def _select_features(self, X, antenna_name):
        indices = self.feature_indices_by_antenna.get(antenna_name)
        if indices is None:
            return X
        return X[:, indices]

    def train(self, X, y_deltas):
        self.models = {}

        for ant_idx, ant_name in enumerate(self.antenna_names):
            model = clone(self.regressor)
            y_ant = y_deltas[:, ant_idx * 3:(ant_idx + 1) * 3]
            model.fit(self._select_features(X, ant_name), y_ant)
            self.models[ant_name] = model

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.antenna_names) * 3), dtype=np.float32)

        for ant_idx, ant_name in enumerate(self.antenna_names):
            pred = np.asarray(self.models[ant_name].predict(self._select_features(X, ant_name)))
            if pred.ndim == 1:
                pred = pred[:, None]
            y_pred[:, ant_idx * 3:(ant_idx + 1) * 3] = pred

        return y_pred
