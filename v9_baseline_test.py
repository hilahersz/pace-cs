"""Re-run of the original v9 baseline (fixed temperature=0.1, fixed LR=1e-3,
60 epochs, symmetric k=64) with cardinality tracking added, so it can be
compared apples-to-apples against the annealed runs. The original v9 run
predates the cardinality tracker and its local weights were long since
overwritten by later runs. Not part of the official sweep.
"""
import tensorflow as tf  # noqa: F401  MUST be first: binds abseil symbols before
# pandas loads its own copy — otherwise TF's fit() deadlocks on its first step
# (see sweep_driver.py's note on the same issue).

import json
import os
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


class CardinalityTracker(keras.callbacks.Callback):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.cardinalities = []

    def on_epoch_end(self, epoch, logs=None):
        selected = tf.argmax(self.layer.logits, axis=-1).numpy()
        self.cardinalities.append(int(len(np.unique(selected))))


TOTAL_EPOCHS = 60
FIXED_TEMPERATURE = 0.1  # matches the original v9 default, no annealing

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim, temperature=FIXED_TEMPERATURE)
encoded = concrete_layer(input_layer)
decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

cardinality_tracker = CardinalityTracker(concrete_layer)

hist = cae.fit(
    X_scaled, X_scaled,
    epochs=TOTAL_EPOCHS,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=[cardinality_tracker],
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
    "temperature": FIXED_TEMPERATURE,
    "learning_rate": 1e-3,
    "distinct_bands": len(counts),
    "wasted_slots": k - len(counts),
    "duplicated_bands": dupes,
    "val_loss": [float(v) for v in hist.history["val_loss"]],
    "cardinality_per_epoch": cardinality_tracker.cardinalities,
}
os.makedirs("experiments", exist_ok=True)
json.dump(result, open("experiments/v9_baseline_symmetric_k64.json", "w"), indent=2)

print()
print(f"k={k}  fixed_T={FIXED_TEMPERATURE}  fixed_lr=1e-3  distinct_bands={len(counts)}/{k}  wasted_slots={k - len(counts)}")
print(f"final val_loss={hist.history['val_loss'][-1]:.6f}")
print("duplicated bands:", dupes)
