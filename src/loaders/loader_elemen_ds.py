import numpy as np
import pandas as pd

RSSI_RX_COLS = [
    "max_rssi_PORT_1__NONE",
    "max_rssi_PORT_2__NONE",
    "max_rssi_PORT_3__NONE",
    "max_rssi_PORT_4__NONE",
]

class ElemenDsLoader:
    def __init__(self):
        self.antenna_names: list[str] = []

    def load(self, path):
        df = pd.read_csv(path)

        tx_dummies    = pd.get_dummies(df["Tx"],           prefix="Tx")
        power_dummies = pd.get_dummies(df["txPower_dBm"],  prefix="power")

        X_df = pd.concat([
            df[RSSI_RX_COLS].fillna(0),
            tx_dummies,
            power_dummies,
        ], axis=1)

        X     = X_df.values.astype(np.float32)
        y_xyz = df[["x", "y", "z"]].values.astype(np.float32)

        y_distances = None

        return X, y_xyz, y_distances, df