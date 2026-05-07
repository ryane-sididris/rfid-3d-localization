# Keras frequency tensor position

This experiment predicts `(x, y, z)` from the frequency-based tensor dataset with a Keras CNN.

The runner is [`experiments/run_keras_freq_position.py`](../run_keras_freq_position.py). The model itself is built in [`src/models/keras_tensor_models.py`](../../src/models/keras_tensor_models.py).

## Input shape

The loader reshapes the data into `96` paths with `50` frequency bins per path.

Each path has `2` channels:

- the RSSI values
- a mask channel that says whether a value is present

So the model sees one tensor where each sample is organized as many small paths instead of one flat feature vector.

## Model overview

The model applies the same small 1D convolution block to each path separately with `TimeDistributed`. This lets it learn local frequency patterns inside one path before mixing information from all paths together.

For each path, the encoder uses:

- `Conv1D(32, kernel_size=5, padding="same", activation="swish")`
- `BatchNormalization`
- `Conv1D(32, kernel_size=3, strides=2, padding="same", activation="swish")`
- `BatchNormalization`
- `Conv1D(64, kernel_size=3, strides=2, padding="same", activation="swish")`
- `BatchNormalization`
- `Flatten`
- `Dense(32, activation="swish")`

After that, the model combines all path embeddings with:

- `TimeDistributed(path_encoder)` over the `96` paths
- `Flatten`
- `Dense(256, activation="swish")`
- `Dropout(0.30)`
- `Dense(128, activation="swish")`
- `Dropout(0.20)`
- final `Dense(3)` output for `(x, y, z)`

In short, it first learns a small representation for each path, then joins everything to predict one final position.

## Training setup

The model uses `Adam` with learning rate `8e-4` and gradient clipping with `clipnorm=1.0`.

The loss is `Huber(delta=1.0)`, and the tracked metric is `MeanAbsoluteError`.

Training also uses `EarlyStopping` and `ReduceLROnPlateau` to stop when validation loss stops improving and to lower the learning rate when needed.
