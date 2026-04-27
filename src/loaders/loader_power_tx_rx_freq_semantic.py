from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PowerTxRxFreqSemanticDataset:
    values_by_power_tx_rx_freq: np.ndarray
    mask_by_power_tx_rx_freq: np.ndarray
    y_xyz: np.ndarray
    powers: list[str]
    tx_names: list[str]
    rx_names: list[str]
    freqs: list[str]
    feature_cols: list[str]
    flat_values_shape: tuple[int, int]


class PowerTxRxFreqSemanticLoader:
    def load(self, path) -> PowerTxRxFreqSemanticDataset:
        df = pd.read_csv(path)

        feature_cols = [col for col in df.columns if col.startswith("max__rssi__")]
        y_xyz = df[["x", "y", "z"]].values.astype(np.float32)
        values_flat = df[feature_cols].values.astype(np.float32)
        mask_flat = (values_flat != 0).astype(np.float32)

        feature_specs = [self._parse_feature_col(col) for col in feature_cols]
        powers = self._ordered_unique(spec[0] for spec in feature_specs)
        tx_names = self._ordered_unique(spec[1] for spec in feature_specs)
        rx_names = self._ordered_unique(spec[2] for spec in feature_specs)
        freqs = self._ordered_unique(spec[3] for spec in feature_specs)

        power_to_idx = {value: idx for idx, value in enumerate(powers)}
        tx_to_idx = {value: idx for idx, value in enumerate(tx_names)}
        rx_to_idx = {value: idx for idx, value in enumerate(rx_names)}
        freq_to_idx = {value: idx for idx, value in enumerate(freqs)}

        n_samples = values_flat.shape[0]
        values_by_power_tx_rx_freq = np.zeros(
            (n_samples, len(powers), len(tx_names), len(rx_names), len(freqs)),
            dtype=np.float32,
        )
        mask_by_power_tx_rx_freq = np.zeros_like(values_by_power_tx_rx_freq)

        for feature_idx, (power, tx_name, rx_name, freq) in enumerate(feature_specs):
            power_idx = power_to_idx[power]
            tx_idx = tx_to_idx[tx_name]
            rx_idx = rx_to_idx[rx_name]
            freq_idx = freq_to_idx[freq]

            values_by_power_tx_rx_freq[:, power_idx, tx_idx, rx_idx, freq_idx] = values_flat[:, feature_idx]
            mask_by_power_tx_rx_freq[:, power_idx, tx_idx, rx_idx, freq_idx] = mask_flat[:, feature_idx]

        return PowerTxRxFreqSemanticDataset(
            values_by_power_tx_rx_freq=values_by_power_tx_rx_freq,
            mask_by_power_tx_rx_freq=mask_by_power_tx_rx_freq,
            y_xyz=y_xyz,
            powers=powers,
            tx_names=tx_names,
            rx_names=rx_names,
            freqs=freqs,
            feature_cols=feature_cols,
            flat_values_shape=values_flat.shape,
        )

    @staticmethod
    def _ordered_unique(items):
        return list(dict.fromkeys(items))

    @staticmethod
    def _parse_feature_col(col):
        parts = col.split("__")
        power = parts[2]
        tx_name = "__".join(parts[3:5])
        rx_name = "__".join(parts[5:7])
        freq = parts[7]
        return power, tx_name, rx_name, freq
