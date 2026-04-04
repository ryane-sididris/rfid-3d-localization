"""
Keras model builders for RFID position regression.

Two architectures from the original RTLS_SF.ipynb notebook:
  1. KerasLearnedTrilat  — dual-head (xyz + distance) with optional strict mode
  2. KerasXYZOnly_AggressiveNL — residual blocks with trigonometric feature augmentation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks


# ---------------------------------------------------------------------------
# 1) Learned-trilateration dual-head model
# ---------------------------------------------------------------------------

def build_learned_trilat(n_features: int, n_anchors: int, strict: bool = True):
    """
    Dual-head model: predicts distances to anchors (d_hat) then xyz.

    Args:
        n_features: number of input RSSI features.
        n_anchors:  number of anchors (size of distance head output).
        strict:     if True  -> xyz head sees only d_hat (bottleneck).
                    if False -> xyz head sees [d_hat, hidden].
    """
    inp = keras.Input(shape=(n_features,), name="X")

    h = layers.Dense(192, activation="swish",
                     kernel_regularizer=regularizers.l2(2e-4))(inp)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.30)(h)
    h = layers.Dense(96, activation="swish",
                     kernel_regularizer=regularizers.l2(2e-4))(h)
    h = layers.Dropout(0.20)(h)

    d_hat = layers.Dense(64, activation="swish",
                         kernel_regularizer=regularizers.l2(1e-4))(h)
    d_hat = layers.Dropout(0.15)(d_hat)
    d_hat = layers.Dense(n_anchors, activation="softplus", name="d_hat")(d_hat)

    t = d_hat if strict else layers.Concatenate()([d_hat, h])

    t = layers.Dense(96, activation="swish",
                     kernel_regularizer=regularizers.l2(1e-4))(t)
    t = layers.Dropout(0.20)(t)
    t = layers.Dense(48, activation="swish",
                     kernel_regularizer=regularizers.l2(1e-4))(t)
    xyz = layers.Dense(3, name="xyz")(t)

    tag = "strict" if strict else "non_strict"
    model = keras.Model(inp, [xyz, d_hat],
                        name=f"KerasLearnedTrilat_{tag}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "xyz":   keras.losses.Huber(delta=1.0),
            "d_hat": keras.losses.Huber(delta=0.5),
        },
        loss_weights={"xyz": 1.0, "d_hat": 0.25},
        metrics={"xyz": [keras.metrics.MeanAbsoluteError(name="mae")]},
    )
    return model


# ---------------------------------------------------------------------------
# 2) Aggressively non-linear xyz-only model with residual blocks
# ---------------------------------------------------------------------------

def _res_block(x, units, drop=0.18, wd=2e-4):
    s = x
    x = layers.Dense(units, activation="gelu",
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(units, activation="gelu",
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    if s.shape[-1] != units:
        s = layers.Dense(units, kernel_regularizer=regularizers.l2(wd))(s)
    x = layers.Add()([x, s])
    x = layers.Activation("gelu")(x)
    return x


def build_xyz_aggressive_nl(n_features: int):
    """
    XYZ-only model with GaussianNoise, trig feature augmentation,
    and residual blocks.
    """
    wd = 2e-4
    inp = keras.Input(shape=(n_features,), name="X")

    x = layers.GaussianNoise(0.03)(inp)
    x2   = layers.Lambda(lambda t: tf.square(t))(x)
    xsin = layers.Lambda(lambda t: tf.sin(np.pi * t))(x)
    xcos = layers.Lambda(lambda t: tf.cos(np.pi * t))(x)
    x = layers.Concatenate()([x, x2, xsin, xcos])

    x = layers.Dense(384, activation="gelu",
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.22)(x)

    x = _res_block(x, 384, drop=0.20, wd=wd)
    x = _res_block(x, 256, drop=0.18, wd=wd)
    x = _res_block(x, 128, drop=0.15, wd=wd)

    x = layers.Dense(64, activation="gelu",
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.Dropout(0.12)(x)
    out = layers.Dense(3, name="xyz")(x)

    model = keras.Model(inp, out, name="KerasXYZOnly_AggressiveNL")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ---------------------------------------------------------------------------
# Training helpers (used by the experiment script)
# ---------------------------------------------------------------------------

def default_callbacks_trilat():
    return [
        callbacks.EarlyStopping(
            monitor="val_xyz_loss", mode="min",
            patience=35, restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_xyz_loss", mode="min",
            factor=0.5, patience=12, min_lr=1e-5, verbose=0,
        ),
    ]


def default_callbacks_xyz():
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=60, restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=18, min_lr=1e-5, verbose=0,
        ),
    ]


KERAS_MODELS = {
    "keras_trilat_strict":     {"build": build_learned_trilat, "strict": True},
    "keras_trilat_non_strict": {"build": build_learned_trilat, "strict": False},
    "keras_xyz_nl":            {"build": build_xyz_aggressive_nl},
}


def available_keras_models():
    return sorted(KERAS_MODELS.keys())
