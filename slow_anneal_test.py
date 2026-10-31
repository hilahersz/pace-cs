"""One-off exploratory test: does a slower (longer) annealing schedule, combined
with an adaptive learning rate, get closer to full band cardinality for the
symmetric k=64 CAE config? Not part of the official v9/v10 sweep — results are
not uploaded to S3 or committed.

Tracks, per epoch: validation MSE, Gumbel temperature, number of distinct bands
selected, and the optimizer's learning rate (via ReduceLROnPlateau) — the three
plots Steve asked to see plus the LR question he raised.
"""
import tensorflow as tf  # noqa: F401  MUST be first: binds abseil symbols before
# pandas loads its own copy — otherwise TF's fit() deadlocks on its first step
# (see sweep_driver.py's note on the same issue).

import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

X = pd.read_pickle("X_cache.pkl")
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_scaled = X_train.dropna().values
input_dim = X_scaled.shape[1]
k = 64
columns = list(X.columns)
pickle.dump(columns, open("x_columns.pkl", "wb"))


class ConcreteSelect(layers.Layer):
    def __init__(self, k, input_dim, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.input_dim = input_dim
        self.temperature = tf.Variable(
            temperature, trainable=False, dtype=tf.float32, name="temperature"
        )

    def build(self, input_shape):
        self.logits = self.add_weight(
            shape=(self.k, self.input_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='logits'
        )

    def call(self, inputs, training=None):
        if training:
            uniform = tf.random.uniform(tf.shape(self.logits), minval=0, maxval=1)
            gumbel = -tf.math.log(-tf.math.log(uniform + 1e-20) + 1e-20)
            noisy_logits = (self.logits + gumbel) / self.temperature
            scores = tf.nn.softmax(noisy_logits, axis=-1)
        else:
            scores = tf.one_hot(tf.argmax(self.logits, axis=-1), depth=self.input_dim)
        return tf.matmul(inputs, tf.transpose(scores))


class TemperatureAnnealer(keras.callbacks.Callback):
    def __init__(self, layer, t_start=5.0, t_end=0.1, total_epochs=60):
        super().__init__()
        self.layer = layer
        self.t_start = t_start
        self.t_end = t_end
        self.total_epochs = total_epochs
        self.temperatures = []

    def on_epoch_begin(self, epoch, logs=None):
        frac = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        temperature = self.t_start * (self.t_end / self.t_start) ** frac
        self.layer.temperature.assign(temperature)

    def on_epoch_end(self, epoch, logs=None):
        self.temperatures.append(float(self.layer.temperature.numpy()))


class CardinalityTracker(keras.callbacks.Callback):
    """Records the number of distinct bands the layer would select in eval mode."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.cardinalities = []

    def on_epoch_end(self, epoch, logs=None):
        selected = tf.argmax(self.layer.logits, axis=-1).numpy()
        self.cardinalities.append(int(len(np.unique(selected))))


class LRTracker(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        self.lrs.append(float(lr))


TOTAL_EPOCHS = 200

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)
decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

annealer = TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=TOTAL_EPOCHS)
cardinality_tracker = CardinalityTracker(concrete_layer)
lr_tracker = LRTracker()
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

hist = cae.fit(
    X_scaled, X_scaled,
    epochs=TOTAL_EPOCHS,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=[annealer, reduce_lr, cardinality_tracker, lr_tracker],
    verbose=2,
)

selected = np.argmax(concrete_layer.logits.numpy(), axis=-1)
counts = Counter(selected.tolist())
dupes = sorted(
    [{"wavelength_nm": columns[b], "slots": n} for b, n in counts.items() if n > 1],
    key=lambda d: -d["slots"],
)

result = {
    "k": k,
    "total_epochs": TOTAL_EPOCHS,
    "distinct_bands": len(counts),
    "wasted_slots": k - len(counts),
    "duplicated_bands": dupes,
    "val_loss": [float(v) for v in hist.history["val_loss"]],
    "temperature": annealer.temperatures,
    "cardinality_per_epoch": cardinality_tracker.cardinalities,
    "learning_rate_per_epoch": lr_tracker.lrs,
}
os.makedirs("experiments", exist_ok=True)
json.dump(result, open("experiments/slow_anneal_symmetric_k64.json", "w"), indent=2)

print()
print(f"k={k}  total_epochs={TOTAL_EPOCHS}  distinct_bands={len(counts)}/{k}  wasted_slots={k - len(counts)}")
print(f"final val_loss={hist.history['val_loss'][-1]:.6f}")
print(f"final learning_rate={lr_tracker.lrs[-1]:.2e}")
print("duplicated bands:", dupes)
