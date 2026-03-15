import numpy as np
import pandas as pd

META_COLS = ["EPC", "xyz", "x", "y", "z", "polar", "nearestAnt", "nearestD"]

class TxRxDsLoader:
    """
    Loader for ds_Tx_Rx.csv.
    
    Features columns: max__rssi__* and max__Delta_rssi__*
    Distances columns: <antenna>__NONE  (e.g., PORT_1__NONE, 041A9F60__NONE)
    Missing RSSI values replaced with -130 dBm.
    """
    def __init__(self):
        self.feature_cols:  list[str] = []
        self.antenna_names: list[str] = []

    def load(self, path):
        df = pd.read_csv(path)

        dist_cols = [
            c for c in df.columns
            if c not in META_COLS
            and c.endswith("__NONE")
            and not c.startswith("max__")
        ]
        self.antenna_names = [c.replace("__NONE", "") for c in dist_cols]

        feat_cols = [
            c for c in df.columns
            if c not in META_COLS and c not in dist_cols
        ]
        self.feature_cols = feat_cols

        ds = df.copy()
        ds[feat_cols] = ds[feat_cols].fillna(-130)

        df_clean = ds[feat_cols + ["x", "y", "z"] + dist_cols].reset_index(drop=True)

        X           = df_clean[feat_cols].values.astype(np.float32)
        y_xyz       = df_clean[["x", "y", "z"]].values.astype(np.float32)
        y_distances = df_clean[dist_cols].values.astype(np.float32)

        return X, y_xyz, y_distances, df_clean