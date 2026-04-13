import numpy as np
import pandas as pd


class PowerTxRxFreqTensorLoader:
    def __init__(self):
        self.feature_cols: list[str] = []
        self.powers: list[str] = []
        self.txs: list[str] = []
        self.rxs: list[str] = []
        self.freqs: list[str] = []
        self.flat_values_shape: tuple[int, int] | None = None
        self.semantic_values_shape: tuple[int, int, int, int, int] | None = None
        self.model_input_shape: tuple[int, int, int, int] | None = None

    def load(self, path):
        df = pd.read_csv(path)

        self.feature_cols = [col for col in df.columns if col.startswith("max__rssi__")]

        y_xyz = df[["x", "y", "z"]].values.astype(np.float32)
        X_values_flat = df[self.feature_cols].values.astype(np.float32)
        X_mask_flat = (X_values_flat != 0).astype(np.float32)

        n_samples = X_values_flat.shape[0]
        self.flat_values_shape = X_values_flat.shape

        feature_specs = [self._parse_feature_col(col) for col in self.feature_cols]
        self.powers = self._ordered_unique(spec[0] for spec in feature_specs)
        self.txs = self._ordered_unique(spec[1] for spec in feature_specs)
        self.rxs = self._ordered_unique(spec[2] for spec in feature_specs)
        self.freqs = self._ordered_unique(spec[3] for spec in feature_specs)

        power_to_idx = {value: idx for idx, value in enumerate(self.powers)}
        tx_to_idx = {value: idx for idx, value in enumerate(self.txs)}
        rx_to_idx = {value: idx for idx, value in enumerate(self.rxs)}
        freq_to_idx = {value: idx for idx, value in enumerate(self.freqs)}

        X_values_5d = np.zeros(
            (n_samples, len(self.powers), len(self.txs), len(self.rxs), len(self.freqs)),
            dtype=np.float32,
        )
        X_mask_5d = np.zeros_like(X_values_5d)
        for feature_idx, (power, tx, rx, freq) in enumerate(feature_specs):
            power_idx = power_to_idx[power]
            tx_idx = tx_to_idx[tx]
            rx_idx = rx_to_idx[rx]
            freq_idx = freq_to_idx[freq]
            X_values_5d[:, power_idx, tx_idx, rx_idx, freq_idx] = X_values_flat[:, feature_idx]
            X_mask_5d[:, power_idx, tx_idx, rx_idx, freq_idx] = X_mask_flat[:, feature_idx]

        self.semantic_values_shape = X_values_5d.shape

        # Fold the (power, tx, rx) axes into 96 paths while keeping 50 frequency bins per path.
        X_values_paths = X_values_5d.reshape(n_samples, 96, 50)
        X_mask_paths = X_mask_5d.reshape(n_samples, 96, 50)

        X_tensor = np.stack([X_values_paths, X_mask_paths], axis=-1).astype(np.float32)
        self.model_input_shape = X_tensor.shape

        return X_tensor, y_xyz

    @staticmethod
    def _ordered_unique(items):
        return list(dict.fromkeys(items))

    @staticmethod
    def _parse_feature_col(col):
        parts = col.split("__")
        power = parts[2]
        tx = "__".join(parts[3:5])
        rx = "__".join(parts[5:7])
        freq = parts[7]
        return power, tx, rx, freq
