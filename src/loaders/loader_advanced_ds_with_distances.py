import numpy as np
import pandas as pd

class AdvancedDsWithDistancesLoader:
    """
    Loads the advanced dataset by extracting:
    - X: RSSI features (everything except position and distance)
    - y: distances to antennas (columns D_* except D_nearestAnt)
    - df: complete dataframe (also contains x, y, z for evaluation)

    Also returns the antenna names via the antenna_names attribute,
    populated after calling .load().
    """

    def __init__(self):
        self.antenna_names: list[str] = []
        self.distance_cols: list[str] = []

    def load(self, path):
        df = pd.read_csv(path)

        distance_cols = [c for c in df.columns if c.startswith("D_") and c != "D_nearestAnt"]
        self.distance_cols = distance_cols
        self.antenna_names = [c.replace("D_", "") for c in distance_cols]

        not_features = {"EPC", "xyz", "x", "y", "z", "polar",
                        "nearestAnt", "D_nearestAnt"}
        feature_cols = [
            c for c in df.columns
            if c not in not_features and c not in distance_cols
        ]

        X = df[feature_cols].values.astype(np.float32)
        y_distances = df[distance_cols].values.astype(np.float32)

        return X, y_distances, df