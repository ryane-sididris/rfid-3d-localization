import numpy as np
from sklearn.preprocessing import LabelEncoder


class ClassifThenRegressor:
    """
    Two-stage pipeline:
    1. Classifier predicts the zone (voxel) from RSSI features
    2. Regressor predicts deltas (dx, dy, dz per antenna) from RSSI + zone_id

    Both classifier and regressor are passed as already-instantiated sklearn estimators.

    If use_zone_feature=False, the classifier is still trained (for zone metrics)
    but zone_id is NOT added to the regressor features. This allows comparing
    "with zone" vs "without zone" to measure the classification's contribution.
    """

    def __init__(self, classifier, regressor, use_zone_feature=True):
        self.classifier = classifier
        self.regressor = regressor
        self.use_zone_feature = use_zone_feature
        self.label_encoder = LabelEncoder()

    def _augment(self, X, zone_encoded):
        if self.use_zone_feature:
            return np.column_stack([X, zone_encoded])
        return X

    def train(self, X, y_deltas, zone_ids):
        """
        X: RSSI features (n_samples, n_features)
        y_deltas: delta targets (n_samples, 24)
        zone_ids: integer zone labels (n_samples,)
        """
        zone_encoded = self.label_encoder.fit_transform(zone_ids)

        # train classifier (always, for zone metrics)
        self.classifier.fit(X, zone_encoded)

        # train regressor
        X_reg = self._augment(X, zone_encoded)
        self.regressor.fit(X_reg, y_deltas)

    def predict(self, X):
        """
        Returns:
            y_deltas_pred: (n_samples, 24)
            zone_pred: predicted zone ids (original labels, not encoded)
        """
        zone_encoded_pred = self.classifier.predict(X)

        X_reg = self._augment(X, zone_encoded_pred)
        y_deltas_pred = self.regressor.predict(X_reg)

        zone_pred = self.label_encoder.inverse_transform(zone_encoded_pred)
        return y_deltas_pred, zone_pred
