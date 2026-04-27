import numpy as np

from src.loaders.loader_power_tx_rx_freq_semantic import PowerTxRxFreqSemanticLoader


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
        semantic_loader = PowerTxRxFreqSemanticLoader()
        dataset = semantic_loader.load(path)

        self.feature_cols = dataset.feature_cols
        self.powers = dataset.powers
        self.txs = dataset.tx_names
        self.rxs = dataset.rx_names
        self.freqs = dataset.freqs
        self.flat_values_shape = dataset.flat_values_shape

        X_values_5d = dataset.values_by_power_tx_rx_freq
        X_mask_5d = dataset.mask_by_power_tx_rx_freq
        n_samples = X_values_5d.shape[0]
        self.semantic_values_shape = X_values_5d.shape

        # Fold the (power, tx, rx) axes into 96 paths while keeping 50 frequency bins per path.
        n_paths = len(self.powers) * len(self.txs) * len(self.rxs)
        X_values_paths = X_values_5d.reshape(n_samples, n_paths, len(self.freqs))
        X_mask_paths = X_mask_5d.reshape(n_samples, n_paths, len(self.freqs))

        X_tensor = np.stack([X_values_paths, X_mask_paths], axis=-1).astype(np.float32)
        self.model_input_shape = X_tensor.shape

        return X_tensor, dataset.y_xyz
