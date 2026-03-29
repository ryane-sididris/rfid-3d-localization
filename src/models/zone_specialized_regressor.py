import numpy as np
from sklearn.base import clone
from scipy.spatial.distance import cdist

from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS


class ZoneSpecializedRegressor:
    """
    Train one regressor per antenna zone, plus a global regressor for zone routing.

    Zones are square regions in XY centered on each antenna's XY projection.
    A sample belongs to a zone if |x - ant_x| <= radius AND |y - ant_y| <= radius.
    Zones can overlap — samples near multiple antennas train multiple regressors.

    At inference, a global regressor produces a coarse (x, y, z) estimate,
    which is used to determine which zone regressor(s) to query.
    """

    def __init__(
        self,
        regressor_template,
        antenna_names=None,
        radius=3.0,
        inference_mode="single",
        top_k=3,
    ):
        self.regressor_template = regressor_template
        self.antenna_names = antenna_names if antenna_names is not None else ANTENNA_NAMES
        self.radius = radius
        self.inference_mode = inference_mode
        self.top_k = top_k

        self.antenna_xy = np.array(
            [ANTENNA_POSITIONS[name][:2] for name in self.antenna_names],
            dtype=np.float32,
        )

        self.global_regressor = None
        self.zone_regressors = {}

    def _assign_zones(self, y_xyz):
        """Assign samples to square XY zones around each antenna.

        Returns:
            dict {antenna_index: bool_mask} — a sample can appear in multiple zones.
        """
        xy = np.asarray(y_xyz)[:, :2]
        masks = {}
        for i, ant_xy in enumerate(self.antenna_xy):
            in_x = np.abs(xy[:, 0] - ant_xy[0]) <= self.radius
            in_y = np.abs(xy[:, 1] - ant_xy[1]) <= self.radius
            masks[i] = in_x & in_y
        return masks

    def zone_coverage_report(self, y_xyz):
        """Print how many tags fall in 0, 1, 2, ... zones."""
        masks = self._assign_zones(y_xyz)
        n_samples = len(y_xyz)
        zone_counts = np.zeros(n_samples, dtype=int)
        for mask in masks.values():
            zone_counts += mask.astype(int)

        not_covered = int((zone_counts == 0).sum())
        print(f"Zone coverage (radius={self.radius}m):")
        print(f"  Total samples: {n_samples}")
        print(f"  Not in any zone: {not_covered} ({not_covered / n_samples * 100:.1f}%)")
        for k in range(1, len(self.antenna_names) + 1):
            count = int((zone_counts == k).sum())
            if count > 0:
                print(f"  In {k} zone(s): {count} ({count / n_samples * 100:.1f}%)")
        print(f"  Samples per zone:")
        for i, name in enumerate(self.antenna_names):
            print(f"    {name:>10s}: {int(masks[i].sum())}")

    def train(self, X, y_xyz):
        zone_masks = self._assign_zones(y_xyz)

        self.global_regressor = clone(self.regressor_template)
        self.global_regressor.fit(X, y_xyz)

        self.zone_regressors = {}
        for i, name in enumerate(self.antenna_names):
            mask = zone_masks[i]
            if mask.sum() == 0:
                continue
            reg = clone(self.regressor_template)
            reg.fit(X[mask], y_xyz[mask])
            self.zone_regressors[i] = reg

    def predict(self, X):
        coarse_xyz = self.global_regressor.predict(X)
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 3), dtype=np.float64)

        if self.inference_mode == "single":
            dists_xy = cdist(
                np.asarray(coarse_xyz)[:, :2], self.antenna_xy
            )
            nearest = np.argmin(dists_xy, axis=1)
            for i in range(n_samples):
                zone_idx = nearest[i]
                if zone_idx in self.zone_regressors:
                    y_pred[i] = self.zone_regressors[zone_idx].predict(X[i:i+1])[0]
                else:
                    y_pred[i] = coarse_xyz[i]

        elif self.inference_mode == "top_k":
            dists_xy = cdist(
                np.asarray(coarse_xyz)[:, :2], self.antenna_xy
            )
            for i in range(n_samples):
                sorted_indices = np.argsort(dists_xy[i])
                top_indices = []
                for idx in sorted_indices:
                    if idx in self.zone_regressors:
                        top_indices.append(idx)
                    if len(top_indices) == self.top_k:
                        break

                if not top_indices:
                    y_pred[i] = coarse_xyz[i]
                    continue

                inv_dists = 1.0 / np.maximum(dists_xy[i, top_indices], 1e-6)
                weights = inv_dists / inv_dists.sum()

                weighted_pred = np.zeros(3)
                for j, idx in enumerate(top_indices):
                    weighted_pred += weights[j] * self.zone_regressors[idx].predict(X[i:i+1])[0]
                y_pred[i] = weighted_pred

        return y_pred
