"""
Keras-based position regression on the advanced dataset.

Three models ported from RTLS_SF.ipynb (cells 190-191):
  - keras_trilat_strict     : dual-head (xyz + distances), strict bottleneck
  - keras_trilat_non_strict : dual-head, xyz head also sees hidden features
  - keras_xyz_nl            : xyz-only with residual blocks & trig augmentation

Usage:
    python -m experiments.run_keras_position keras_trilat_strict
    python -m experiments.run_keras_position keras_xyz_nl
    python -m experiments.run_keras_position all
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.config.antennas import ANTENNA_NAMES, ANTENNA_POSITIONS
from src.evaluation.metrics import rmse_3d, rmse_xyz, threshold_accuracy
from src.validation.cross_validation import cross_validate
from src.models.keras_models import (
    build_learned_trilat,
    build_xyz_aggressive_nl,
    default_callbacks_trilat,
    default_callbacks_xyz,
    available_keras_models,
    KERAS_MODELS,
)


# ── CLI ──────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print(f"Usage: python -m experiments.run_keras_position "
          f"<{'|'.join(available_keras_models())}|all>")
    sys.exit(1)

requested = sys.argv[1].lower()
if requested == "all":
    model_names = available_keras_models()
else:
    if requested not in KERAS_MODELS:
        print(f"Unknown model '{requested}'. "
              f"Choose from: {available_keras_models()} or 'all'")
        sys.exit(1)
    model_names = [requested]

SEED = 42

# ── Dataset ──────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
DS_NAME = "ds_advanced_17Feb26.csv"
print(f"Loading dataset: {DS_NAME}")

df = pd.read_csv(DATA_DIR / DS_NAME)
# Fix antenna name typos in CSV columns (0F17B312 -> 0F173B12, 0F17B313 -> 0F173B13)
df.columns = [c.replace("0F17B312", "0F173B12").replace("0F17B313", "0F173B13")
               for c in df.columns]

meta_cols = ["EPC", "xyz", "x", "y", "z", "polar", "nearestAnt", "D_nearestAnt"]
dist_cols = [c for c in df.columns if c.startswith("D_")]
feat_cols = [c for c in df.columns if c not in meta_cols + dist_cols]
for col in feat_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-130)

X = df[feat_cols].values.astype(np.float32)
y_xyz = df[["x", "y", "z"]].values.astype(np.float32)
print(f"Samples={X.shape[0]}, Features={X.shape[1]}")

# Antenna positions as numpy array (same order as ANTENNA_NAMES)
ant_xyz = np.array(
    [ANTENNA_POSITIONS[a] for a in ANTENNA_NAMES], dtype="float32"
)
n_anchors = ant_xyz.shape[0]


def dist_to_anchors(xyz, anchors):
    """Compute Euclidean distances from each sample to each anchor."""
    return np.linalg.norm(
        xyz[:, None, :] - anchors[None, :, :], axis=2
    ).astype("float32")


# ── Helpers ──────────────────────────────────────────────────────────────

def aggregate_mean_std(folds):
    metrics = defaultdict(list)
    for f in folds:
        for k, v in f.items():
            metrics[k].append(v)
    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}


def evaluate_position(y_true, y_pred, _):
    errs = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "rmse":     rmse_3d(errs),
        "rmse_xyz": rmse_xyz(y_true, y_pred),
        "acc_1m":   threshold_accuracy(errs, 1),
        "acc_2m":   threshold_accuracy(errs, 2),
        "acc_3m":   threshold_accuracy(errs, 3),
    }


# ── Run each model ──────────────────────────────────────────────────────

all_results = {}

for model_name in model_names:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    cfg = KERAS_MODELS[model_name]
    is_trilat = model_name.startswith("keras_trilat")

    # ── build train / predict closures ──

    if is_trilat:
        strict = cfg["strict"]

        def make_train_fn(strict_flag):
            def train_fn(X_train, y_train):
                tf.keras.utils.set_random_seed(SEED)
                np.random.seed(SEED)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train).astype("float32")
                D_train = dist_to_anchors(y_train.astype("float32"), ant_xyz)

                model = build_learned_trilat(
                    X_scaled.shape[1], n_anchors, strict=strict_flag
                )
                bs = int(min(256, max(32, len(X_scaled) // 12)))
                model.fit(
                    X_scaled,
                    {"xyz": y_train.astype("float32"),
                     "d_hat": D_train},
                    validation_split=0.2,
                    epochs=500,
                    batch_size=bs,
                    callbacks=default_callbacks_trilat(),
                    verbose=0,
                )
                return (model, scaler)
            return train_fn

        train_fn = make_train_fn(strict)

        def predict_fn(model_scaler, X_test):
            model, scaler = model_scaler
            X_scaled = scaler.transform(X_test).astype("float32")
            pred_xyz, _ = model.predict(X_scaled, verbose=0)
            return pred_xyz

    else:
        # keras_xyz_nl
        def train_fn(X_train, y_train):
            tf.keras.utils.set_random_seed(SEED)
            np.random.seed(SEED)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train).astype("float32")

            model = build_xyz_aggressive_nl(X_scaled.shape[1])
            bs = int(min(128, max(16, len(X_scaled) // 10)))
            model.fit(
                X_scaled,
                y_train.astype("float32"),
                validation_split=0.25,
                epochs=900,
                batch_size=bs,
                callbacks=default_callbacks_xyz(),
                verbose=0,
            )
            return (model, scaler)

        def predict_fn(model_scaler, X_test):
            model, scaler = model_scaler
            X_scaled = scaler.transform(X_test).astype("float32")
            return model.predict(X_scaled, verbose=0)

    # ── cross-validate ──

    result = cross_validate(
        X=X, y=y_xyz,
        train_fn=train_fn,
        predict_fn=predict_fn,
        eval_fn=evaluate_position,
        aggregate_fn=aggregate_mean_std,
        k=5, random_state=42, verbose=True,
    )

    all_results[model_name] = result

    summary = pd.DataFrame(result["summary"]).T
    print(f"\n--- {model_name} ({DS_NAME}) ---")
    print(summary)

# ── Final comparison ─────────────────────────────────────────────────────

if len(all_results) > 1:
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    rows = {}
    for name, res in all_results.items():
        rows[name] = {k: v["mean"] for k, v in res["summary"].items()}
    print(pd.DataFrame(rows).T.to_string(float_format="%.3f"))
