"""Re-run all 10 CAE configs (5 asymmetric, 5 symmetric architectures x k)
from the paper's Table tab:architecture_results, at the ORIGINAL fixed
temperature=0.1 (matching that table's existing Val/Test numbers), with
cardinality tracking added so the new t<=k column can be filled in. One
process, reused data, sequential configs. Not part of the official sweep.
"""
import tensorflow as tf  # noqa: F401  MUST be first: binds abseil symbols before
# pandas loads its own copy — otherwise TF's fit() deadlocks on its first step
# (see sweep_driver.py's note on the same issue).

import json
import os
import time
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
columns = list(X.columns)

FIXED_TEMPERATURE = 0.1
FIXED_LR = 1e-3
TOTAL_EPOCHS = 60

# (architecture, k, [decoder layer sizes]); symmetric = single output layer
ASYM_LAYERS = {
    16: [32, 64, 128],
    24: [32, 64, 128],
    32: [64, 128, 256],
    48: [64, 128, 256],
    64: [64, 128, 256],
}
CONFIGS = [("asymmetric", k) for k in [16, 24, 32, 48, 64]] + [("symmetric", k) for k in [16, 24, 32, 48, 64]]


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


results = {}
os.makedirs("experiments", exist_ok=True)

for arch, k in CONFIGS:
    t0 = time.time()
    input_layer = keras.Input(shape=(input_dim,))
    concrete_layer = ConcreteSelect(k=k, input_dim=input_dim, temperature=FIXED_TEMPERATURE)
    encoded = concrete_layer(input_layer)

    if arch == "asymmetric":
        decoded = encoded
        for units in ASYM_LAYERS[k]:
            decoded = layers.Dense(units, activation="relu")(decoded)
        decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)
    else:
        decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

    cae = keras.Model(inputs=input_layer, outputs=decoded)
    cae.compile(optimizer=keras.optimizers.Adam(FIXED_LR), loss="mse")

    hist = cae.fit(
        X_scaled, X_scaled,
        epochs=TOTAL_EPOCHS,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        verbose=0,
    )

    selected = np.argmax(concrete_layer.logits.numpy(), axis=-1)
    counts = Counter(selected.tolist())
    t = len(counts)

    key = f"{arch}_{k}"
    results[key] = {
        "arch": arch,
        "k": k,
        "final_val_loss": float(hist.history["val_loss"][-1]),
        "distinct_bands": t,
        "wasted_slots": k - t,
    }
    elapsed = time.time() - t0
    print(f"[{elapsed:5.0f}s] {arch:10s} k={k:3d}  val_loss={hist.history['val_loss'][-1]:.6f}  t={t}/{k}", flush=True)

json.dump(results, open("experiments/v9_full_cardinality_sweep.json", "w"), indent=2)
print("\nsaved experiments/v9_full_cardinality_sweep.json")
