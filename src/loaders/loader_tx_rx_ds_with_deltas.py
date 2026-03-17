import numpy as np

from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS
from src.loaders.loader_tx_rx_ds import TxRxDsLoader


class TxRxDsWithDeltasLoader:
    """
    Loader for ds_Tx_Rx.csv with per-antenna delta targets.

    Outputs:
    - X: all Tx/Rx features
    - y_deltas: [dx_ant0, dy_ant0, dz_ant0, dx_ant1, ...]
    - y_xyz: absolute positions

    Also exposes feature subsets grouped by Tx antenna for the per-antenna setup.
    """

    def __init__(self):
        self.base_loader = TxRxDsLoader()
        self.antenna_names: list[str] = list(ANTENNA_NAMES)
        self.feature_cols: list[str] = []
        self.delta_cols: list[str] = []
        self.feature_cols_by_tx: dict[str, list[str]] = {}
        self.feature_indices_by_tx: dict[str, np.ndarray] = {}

    def load(self, path):
        X, y_xyz, _, df_clean = self.base_loader.load(path)

        self.feature_cols = list(self.base_loader.feature_cols)
        self.antenna_names = [name for name in ANTENNA_NAMES if name in self.base_loader.antenna_names]
        self.feature_cols_by_tx, self.feature_indices_by_tx = self._build_feature_groups()

        delta_list = []
        self.delta_cols = []
        for ant_name in self.antenna_names:
            ant_pos = ANTENNA_POSITIONS[ant_name]
            dx = y_xyz[:, 0] - ant_pos[0]
            dy = y_xyz[:, 1] - ant_pos[1]
            dz = y_xyz[:, 2] - ant_pos[2]
            delta_list.extend([dx, dy, dz])
            self.delta_cols.extend([f"dx_{ant_name}", f"dy_{ant_name}", f"dz_{ant_name}"])

        y_deltas = np.column_stack(delta_list).astype(np.float32)
        return X, y_deltas, y_xyz, df_clean

    def _build_feature_groups(self):
        feature_cols_by_tx = {ant_name: [] for ant_name in self.antenna_names}
        feature_indices_by_tx = {ant_name: [] for ant_name in self.antenna_names}

        for idx, col in enumerate(self.feature_cols):
            tx_name = self._extract_tx_name(col)
            if tx_name in feature_cols_by_tx:
                feature_cols_by_tx[tx_name].append(col)
                feature_indices_by_tx[tx_name].append(idx)

        return (
            feature_cols_by_tx,
            {name: np.asarray(indices, dtype=int) for name, indices in feature_indices_by_tx.items()},
        )

    @staticmethod
    def _extract_tx_name(feature_col):
        parts = feature_col.split("__")
        if len(parts) < 4:
            return None
        return parts[3]
