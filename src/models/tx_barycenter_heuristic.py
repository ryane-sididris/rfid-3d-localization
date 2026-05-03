import numpy as np
from sklearn.linear_model import LinearRegression

from src.config.antennas import ANTENNA_POSITIONS
from src.models.regressors import get_regressor


class TxBarycenterHeuristic:
    def __init__(
        self,
        tx_names,
        powers=None,
        score_mode="balanced_per_power",
        top_k=4,
        z_model="extratrees",
        weight_exponent=1.0,
    ):
        self.tx_names = list(tx_names)
        self.powers = list(powers or [])
        self.score_mode = score_mode
        self.top_k = top_k
        self.z_model = z_model
        self.weight_exponent = weight_exponent

        self.tx_position_names = [tx_name.replace("__NONE", "") for tx_name in self.tx_names]
        self.tx_xy_positions = np.array(
            [ANTENNA_POSITIONS[tx_name][:2] for tx_name in self.tx_position_names],
            dtype=np.float32,
        )

        self.z_predictor = None
        self.z_constant = 0.0

    def fit(self, values_by_power_tx_rx_freq, y_xyz):
        tx_scores = self._build_tx_scores(values_by_power_tx_rx_freq)
        self._fit_z_model(tx_scores, y_xyz[:, 2])
        return self

    def predict(self, values_by_power_tx_rx_freq):
        tx_scores = self._build_tx_scores(values_by_power_tx_rx_freq)
        xy_pred = self._predict_xy_from_scores(tx_scores)
        z_pred = self._predict_z_from_scores(tx_scores)
        return np.column_stack([xy_pred, z_pred]).astype(np.float32)

    def _build_tx_scores(self, values_by_power_tx_rx_freq):
        if self.score_mode == "raw_sum":
            return self._raw_sum_scores(values_by_power_tx_rx_freq)

        if self.score_mode == "raw_median":
            return self._raw_median_scores(values_by_power_tx_rx_freq)

        if self.score_mode == "balanced_per_power":
            tx_energy_by_power = values_by_power_tx_rx_freq.sum(axis=(3, 4))
            total_energy_per_power = tx_energy_by_power.sum(axis=2, keepdims=True)
            tx_share_by_power = tx_energy_by_power / np.maximum(total_energy_per_power, 1e-12)
            return tx_share_by_power.mean(axis=1)

        if self.score_mode == "rssi_max":
            rssi_max_by_power_and_tx = values_by_power_tx_rx_freq.max(axis=(3, 4))
            return rssi_max_by_power_and_tx.max(axis=1)

        if self.score_mode == "rssi_max_power_corrected":
            rssi_max_by_power_and_tx = values_by_power_tx_rx_freq.max(axis=(3, 4))
            power_relative_factors = self._power_relative_factors()
            corrected_rssi = rssi_max_by_power_and_tx / power_relative_factors[None, :, None]
            return corrected_rssi.mean(axis=1)

        if self.score_mode == "rssi_max_div_power_dbm":
            # weight = (rssi_linear / P_dBm)^exponent — Christophe's notebook cell 209
            rssi_max_by_power_and_tx = values_by_power_tx_rx_freq.max(axis=(3, 4))
            power_dbm = np.array([float(p) for p in self.powers], dtype=np.float32)
            corrected_rssi = rssi_max_by_power_and_tx / power_dbm[None, :, None]
            return corrected_rssi.sum(axis=1)

        if self.score_mode == "rssi_max_div_power_linear":
            # weight = (rssi_linear / P_linear)^exponent, P_linear = 10^(P_dBm/10)
            rssi_max_by_power_and_tx = values_by_power_tx_rx_freq.max(axis=(3, 4))
            power_dbm = np.array([float(p) for p in self.powers], dtype=np.float32)
            power_linear = 10 ** (power_dbm / 10)
            corrected_rssi = rssi_max_by_power_and_tx / power_linear[None, :, None]
            return corrected_rssi.sum(axis=1)

        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def _raw_sum_scores(self, values_by_power_tx_rx_freq):
        return values_by_power_tx_rx_freq.sum(axis=(1, 3, 4))

    def _raw_median_scores(self, values_by_power_tx_rx_freq):
        return np.median(values_by_power_tx_rx_freq, axis=(1, 3, 4))

    def _predict_xy_from_scores(self, tx_scores):
        xy_pred = np.zeros((len(tx_scores), 2), dtype=np.float32)

        for sample_idx, sample_scores in enumerate(tx_scores):
            top_tx_indices = np.argsort(sample_scores)[-self.top_k:]
            top_tx_scores = sample_scores[top_tx_indices] ** self.weight_exponent
            top_tx_positions = self.tx_xy_positions[top_tx_indices]
            weight_sum = top_tx_scores.sum()

            if weight_sum == 0:
                xy_pred[sample_idx] = top_tx_positions.mean(axis=0)
                continue

            xy_pred[sample_idx] = (
                (top_tx_scores[:, None] * top_tx_positions).sum(axis=0) / weight_sum
            )

        return xy_pred

    def _power_relative_factors(self):
        power_dbm = np.array([float(power) for power in self.powers], dtype=np.float32)
        reference_power_dbm = power_dbm.min()
        return 10 ** ((power_dbm - reference_power_dbm) / 10)

    def _fit_z_model(self, tx_scores, z_true):
        if self.z_model == "constant":
            self.z_constant = float(np.mean(z_true))
            self.z_predictor = None
            return

        if self.z_model == "linear":
            self.z_predictor = LinearRegression()
            self.z_predictor.fit(tx_scores, z_true)
            return

        if self.z_model == "extratrees":
            self.z_predictor = get_regressor("extratrees")
            self.z_predictor.fit(tx_scores, z_true)
            return

        raise ValueError(f"Unknown z_model: {self.z_model}")

    def _predict_z_from_scores(self, tx_scores):
        if self.z_model == "constant":
            return np.full(len(tx_scores), self.z_constant, dtype=np.float32)

        return self.z_predictor.predict(tx_scores)
