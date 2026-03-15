import numpy as np
import pandas as pd
from src.config.antennas import ANTENNA_NAMES

class AdvancedDsLoader:
    def __init__(self):
        self.antenna_names: list[str] = ANTENNA_NAMES

    def load(self, path):
        df = pd.read_csv(path)
        meta_cols = ["EPC", "xyz", "x", "y", "z", "polar", "nearestAnt"]
        dist_cols = [c for c in df.columns if c.startswith("D_")]
        feat_cols = [c for c in df.columns if c not in meta_cols + dist_cols]
        Xcols = [c for c in feat_cols if 'dBm' not in c]

        ds = df.copy()
        for col in Xcols:
            ds[col] = ds[col].fillna(-130)

        df_clean = ds[Xcols + ["x", "y", "z"] + ["D_" + a for a in ANTENNA_NAMES]].reset_index(drop=True)

        X           = df_clean[Xcols].values.astype(np.float32)
        y_xyz       = df_clean[["x", "y", "z"]].values.astype(np.float32)
        y_distances = df_clean[["D_" + a for a in ANTENNA_NAMES]].values.astype(np.float32)

        return X, y_xyz, y_distances, df_clean