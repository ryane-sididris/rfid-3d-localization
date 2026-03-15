import numpy as np
import pandas as pd
from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS


class AdvancedDsWithDeltasLoader:
    """
    Loads the advanced dataset and computes delta targets:
    For each antenna, delta = (x - ant_x, y - ant_y, z - ant_z)
    -> 24 outputs (3 * 8 antennas)

    Also computes zone labels (voxel classification) from (x, y, z).
    """

    def __init__(self, voxel_size=2.0):
        self.voxel_size = voxel_size
        self.antenna_names = ANTENNA_NAMES
        self.delta_cols = []

    def load(self, path):
        df = pd.read_csv(path)

        # --- features (same logic as other loaders) ---
        not_features = {"EPC", "xyz", "x", "y", "z", "polar",
                        "nearestAnt", "D_nearestAnt"}
        dist_cols = [c for c in df.columns if c.startswith("D_")]
        feature_cols = [
            c for c in df.columns
            if c not in not_features and c not in dist_cols
        ]

        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=-130.0)

        # --- positions ---
        y_xyz = df[["x", "y", "z"]].values.astype(np.float32)

        # --- deltas per antenna ---
        # shape: (n_samples, n_antennas * 3)
        # order: [dx_ant0, dy_ant0, dz_ant0, dx_ant1, dy_ant1, dz_ant1, ...]
        delta_list = []
        self.delta_cols = []
        for ant_name in self.antenna_names:
            ant_pos = ANTENNA_POSITIONS[ant_name]
            dx = df["x"].values - ant_pos[0]
            dy = df["y"].values - ant_pos[1]
            dz = df["z"].values - ant_pos[2]
            delta_list.extend([dx, dy, dz])
            self.delta_cols.extend([
                f"dx_{ant_name}", f"dy_{ant_name}", f"dz_{ant_name}"
            ])

        y_deltas = np.column_stack(delta_list).astype(np.float32)

        # --- zone labels (voxels) ---
        zone_ids = self._compute_zones(y_xyz)

        return X, y_deltas, y_xyz, zone_ids, df

    def _compute_zones(self, y_xyz):
        """Assign each sample to a voxel based on (x, y, z) and voxel_size."""
        voxel_indices = (y_xyz / self.voxel_size).astype(int)
        # encode (ix, iy, iz) -> unique integer
        # use a large multiplier to avoid collisions
        zone_ids = (voxel_indices[:, 0] * 10000
                    + voxel_indices[:, 1] * 100
                    + voxel_indices[:, 2])
        return zone_ids
