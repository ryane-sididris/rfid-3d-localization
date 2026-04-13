from tensorflow import keras
from tensorflow.keras import callbacks, layers


def build_freq_conv_xyz(input_shape=(96, 50, 2)):
    path_input = keras.Input(shape=(50, 2), name="path_input")

    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="swish")(path_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, strides=2, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, strides=2, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    path_output = layers.Dense(32, activation="swish")(x)

    path_encoder = keras.Model(path_input, path_output, name="freq_path_encoder")

    model_input = keras.Input(shape=input_shape, name="X_tensor")
    x = layers.TimeDistributed(path_encoder, name="path_embeddings")(model_input)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.20)(x)
    xyz = layers.Dense(3, name="xyz")(x)

    model = keras.Model(model_input, xyz, name="KerasFreqConvXYZ")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def default_callbacks_freq_xyz():
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
