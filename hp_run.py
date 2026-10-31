import os
import shutil
import time

import pandas as pd
import boto3
import numpy as np
from dotenv import load_dotenv
import pyarrow.dataset as ds

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from boruta import BorutaPy

load_dotenv()

def get_results(hist, autoencoder):
  # Validation loss
  val_loss = hist.history["val_loss"][-1]

  # Test loss
  X_reconstructed = autoencoder.predict(X_test.values)
  test_loss = float(np.mean(np.square(X_test.values - X_reconstructed)))

  # Per feature
  errors = np.square(X_test - X_reconstructed)
  feature_mse = np.mean(errors, axis=0)
  features_95_percentile = np.percentile(feature_mse, 95)

  # Per pixel
  errors = np.square(X_test - X_reconstructed)
  pixel_mse = np.mean(errors, axis=1)
  pixel_95_percentile = np.percentile(pixel_mse, 95)

  # Per Critical Wavelengths
  critical_wavelengths = [c for c in X.columns if c > 630 and c < 650]
  critical_wavelengths = [c for c in X.columns if c > 630 and c < 650]
  critical_wavelength_mse = feature_mse[critical_wavelengths]
  critical_99_percentile = np.percentile(critical_wavelength_mse, 99)


  performance = {
      "val_loss": val_loss,
      "test_loss": test_loss,
      "features_95_percentile": features_95_percentile,
      "pixel_95_percentile": pixel_95_percentile,
      "critical_wavelength_99_percentile": critical_99_percentile
  }

  return pd.Series(performance)

bucket = os.getenv("BUCKET_NAME")
results_bucket = os.getenv("HYPERPARAMETERS_RESULTS_BUCKET_NAME")

column_path = f"s3://{bucket}/headers.parquet"
columns =  pd.read_parquet(column_path)['0'].values

s3_folder = f"s3://{bucket}/samples/"
dataset = ds.dataset(s3_folder, format="parquet")
toscore = dataset.to_table().to_pandas()

toscore.columns = columns

bad_wavelengths = [c for c in toscore.columns if c < 320 or (c > 590 and c < 610)]

X = toscore.drop(columns=bad_wavelengths).dropna()
X

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "k": self.k,
            "input_dim": self.input_dim,
            "temperature": float(self.temperature.numpy())
        })
        return config


class TemperatureAnnealer(keras.callbacks.Callback):
    """Exponentially decays a ConcreteSelect layer's Gumbel temperature over training."""

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


# False (default): fixed tau = 0.1, the setting behind the paper's results (v9).
# True: anneal tau from 5 to 0.1 over 60 epochs, the follow-up experiment (v10).
ANNEAL_TAU = False
CAE_VERSION = "v10" if ANNEAL_TAU else "v9"


class ColumnSelector(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = tf.constant(indices, dtype=tf.int32)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"indices": self.indices.numpy().tolist()})
        return cfg


def get_results(hist, autoencoder, layers, is_symmetric, tag=None):
  if tag:
      import json as _json
      os.makedirs("learning_curves", exist_ok=True)
      with open(os.path.join("learning_curves", tag.replace("/", "_") + ".json"), "w") as _fp:
          _json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, _fp)
  # Validation loss
  val_loss = hist.history["val_loss"][-1]

  # Test loss
  X_reconstructed = autoencoder.predict(X_test.values)
  test_loss = float(np.mean(np.square(X_test.values - X_reconstructed)))

  # Per feature
  errors = np.square(X_test - X_reconstructed)
  feature_mse = np.mean(errors, axis=0)
  features_95_percentile = np.percentile(feature_mse, 95)

  # Per pixel
  errors = np.square(X_test - X_reconstructed)
  pixel_mse = np.mean(errors, axis=1)
  pixel_95_percentile = np.percentile(pixel_mse, 95)

  # Per Critical Wavelengths
  critical_wavelengths = [c for c in X.columns if c > 630 and c < 650]
  critical_wavelengths = [c for c in X.columns if c > 630 and c < 650]
  critical_wavelength_mse = feature_mse[critical_wavelengths]
  critical_99_percentile = np.percentile(critical_wavelength_mse, 99)


  performance = {
      "layers": layers,
      "symmetric_architecture": is_symmetric,
      "val_loss": val_loss,
      "test_loss": test_loss,
      "features_95_percentile": features_95_percentile,
      "pixel_95_percentile": pixel_95_percentile,
      "critical_wavelength_99_percentile": critical_99_percentile
  }

  return pd.Series(performance)

num_features = 277
X_scaled = X_train.dropna().values

k = 16

input_dim = X_scaled.shape[1]

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)

decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(decoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer="adam", loss="mse")

annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
cae_hist = cae.fit(
    X_scaled, X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=annealers
)
if annealers:
    cae_hist.history["temperature"] = annealers[0].temperatures

encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Reconstruct
X_reconstructed = cae.predict(X_scaled)

version = CAE_VERSION
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(cae_hist, cae, encoder_size, 0, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

k = 24

input_dim = X_scaled.shape[1]

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)

decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(decoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer="adam", loss="mse")

annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
cae_hist = cae.fit(
    X_scaled, X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=annealers
)
if annealers:
    cae_hist.history["temperature"] = annealers[0].temperatures

encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Reconstruct
X_reconstructed = cae.predict(X_scaled)

version = CAE_VERSION
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(cae_hist, cae, encoder_size, 0, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

k = 32

input_dim = X_scaled.shape[1]

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)

decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(256, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer="adam", loss="mse")

annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
cae_hist = cae.fit(
    X_scaled, X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=annealers
)
if annealers:
    cae_hist.history["temperature"] = annealers[0].temperatures

encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Reconstruct
X_reconstructed = cae.predict(X_scaled)

version = CAE_VERSION
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(cae_hist, cae, encoder_size, 0, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

k = 48

input_dim = X_scaled.shape[1]

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)

decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(256, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer="adam", loss="mse")

annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
cae_hist = cae.fit(
    X_scaled, X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=annealers
)
if annealers:
    cae_hist.history["temperature"] = annealers[0].temperatures

encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Reconstruct
X_reconstructed = cae.predict(X_scaled)

version = CAE_VERSION
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(cae_hist, cae, encoder_size, 0, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

k = 64

input_dim = X_scaled.shape[1]

input_layer = keras.Input(shape=(input_dim,))
concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
encoded = concrete_layer(input_layer)

decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(256, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

cae = keras.Model(inputs=input_layer, outputs=decoded)
cae.compile(optimizer="adam", loss="mse")

annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
cae_hist = cae.fit(
    X_scaled, X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=annealers
)
if annealers:
    cae_hist.history["temperature"] = annealers[0].temperatures

encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Reconstruct
X_reconstructed = cae.predict(X_scaled)

version = CAE_VERSION
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(cae_hist, cae, encoder_size, 0, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

for k in [16, 24, 32, 48, 64]:
    num_features = 277
    X_scaled = X_train.dropna().values

    input_dim = X_scaled.shape[1]

    input_layer = keras.Input(shape=(input_dim,))
    concrete_layer = ConcreteSelect(k=k, input_dim=input_dim)
    encoded = concrete_layer(input_layer)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

    cae = keras.Model(inputs=input_layer, outputs=decoded)
    cae.compile(optimizer="adam", loss="mse")

    annealers = [TemperatureAnnealer(concrete_layer, t_start=5.0, t_end=0.1, total_epochs=60)] if ANNEAL_TAU else []
    cae_hist = cae.fit(
        X_scaled, X_scaled,
        epochs=60,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        callbacks=annealers
    )
    if annealers:
        cae_hist.history["temperature"] = annealers[0].temperatures

    encoder = keras.Model(inputs=input_layer, outputs=encoded)
    X_encoded = encoder.predict(X_scaled)

    # Reconstruct
    X_reconstructed = cae.predict(X_scaled)

    version = CAE_VERSION
    architecture = "symmetric_architecture"
    encoder_size = k

    performance = get_results(cae_hist, cae, encoder_size, 1, tag=f"cae/{architecture}/{encoder_size}/{version}").to_frame()

    validation_path = f"s3://{results_bucket}/concrete-auto-encoder/{architecture}/{encoder_size}/{version}/score.parquet"
    weights_path = f"concrete-auto-encoder/{architecture}/{str(encoder_size)}/{version}/cae_savedmodel.zip"

    performance.to_parquet(
        validation_path,
        index=True,
        engine="pyarrow",
    )

    artifact_dir = f"cae_artifacts/{architecture}_{encoder_size}_{version}"
    os.makedirs(artifact_dir, exist_ok=True)
    cae.save_weights(os.path.join(artifact_dir, "cae_weights.weights.h5"))

    zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
    s3 = boto3.client("s3")
    s3.upload_file(zip_path, results_bucket, weights_path)

class ColumnSelector(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = tf.constant(indices, dtype=tf.int32)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"indices": self.indices.numpy().tolist()})
        return cfg


y_train = None
y = None

k = 32
feat_names = X.columns.tolist()

X_scaled = X_train.sample(frac=0.001).dropna().values

n_clusters = 10
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
y_pseudo = km.fit_predict(X_scaled)

est = ExtraTreesClassifier(
    n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
)

boruta = BorutaPy(
    estimator=est,
    n_estimators="auto",
    two_step=True,
    max_iter=40,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
boruta.fit(X_scaled, y_pseudo)
t1 = time.perf_counter()
print(f"Boruta took {(t1 - t0):.1f} seconds")

confirmed_mask = boruta.support_
confirmed_idxs = np.where(confirmed_mask)[0]

if len(confirmed_idxs) < k:
    ranks = boruta.ranking_
    top_k_idxs = np.argsort(ranks)[:k]
    chosen_idxs = np.array(sorted(set(confirmed_idxs).union(set(top_k_idxs))))[:k]
else:
    ranks = boruta.ranking_
    confirmed_ranks = ranks[confirmed_idxs]
    order = np.argsort(confirmed_ranks)
    chosen_idxs = confirmed_idxs[order][:k]

chosen_idxs = np.sort(chosen_idxs)
selected_features = [feat_names[i] for i in chosen_idxs]

print(f"Selected {len(selected_features)} features (k={k}):")
print(selected_features)

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]
k = 32

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

x = layers.Dense(64, activation="relu")(encoded)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
out = layers.Dense(input_dim, activation="linear")(x)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 0, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]
k = 32

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)
out = layers.Dense(input_dim, activation="linear")(encoded)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "symmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 1, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

y_train = None
y = None

k = 16
feat_names = X.columns.tolist()

X_scaled = X_train.sample(frac=0.001).dropna().values

n_clusters = 10
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
y_pseudo = km.fit_predict(X_scaled)

est = ExtraTreesClassifier(
    n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
)

boruta = BorutaPy(
    estimator=est,
    n_estimators="auto",
    two_step=True,
    max_iter=40,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
boruta.fit(X_scaled, y_pseudo)
t1 = time.perf_counter()
print(f"Boruta took {(t1 - t0):.1f} seconds")

confirmed_mask = boruta.support_
confirmed_idxs = np.where(confirmed_mask)[0]

if len(confirmed_idxs) < k:
    ranks = boruta.ranking_
    top_k_idxs = np.argsort(ranks)[:k]
    chosen_idxs = np.array(sorted(set(confirmed_idxs).union(set(top_k_idxs))))[:k]
else:
    ranks = boruta.ranking_
    confirmed_ranks = ranks[confirmed_idxs]
    order = np.argsort(confirmed_ranks)
    chosen_idxs = confirmed_idxs[order][:k]

chosen_idxs = np.sort(chosen_idxs)
selected_features = [feat_names[i] for i in chosen_idxs]

print(f"Selected {len(selected_features)} features (k={k}):")
print(selected_features)

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

x = layers.Dense(32, activation="relu")(encoded)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(input_dim, activation="linear")(x)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 0, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)
out = layers.Dense(input_dim, activation="linear")(encoded)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "symmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 1, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

y_train = None
y = None

k = 24
feat_names = X.columns.tolist()

X_scaled = X_train.sample(frac=0.001).dropna().values

n_clusters = 10
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
y_pseudo = km.fit_predict(X_scaled)

est = ExtraTreesClassifier(
    n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
)

boruta = BorutaPy(
    estimator=est,
    n_estimators="auto",
    two_step=True,
    max_iter=40,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
boruta.fit(X_scaled, y_pseudo)
t1 = time.perf_counter()
print(f"Boruta took {(t1 - t0):.1f} seconds")

confirmed_mask = boruta.support_
confirmed_idxs = np.where(confirmed_mask)[0]

if len(confirmed_idxs) < k:
    ranks = boruta.ranking_
    top_k_idxs = np.argsort(ranks)[:k]
    chosen_idxs = np.array(sorted(set(confirmed_idxs).union(set(top_k_idxs))))[:k]
else:
    ranks = boruta.ranking_
    confirmed_ranks = ranks[confirmed_idxs]
    order = np.argsort(confirmed_ranks)
    chosen_idxs = confirmed_idxs[order][:k]

chosen_idxs = np.sort(chosen_idxs)
selected_features = [feat_names[i] for i in chosen_idxs]

print(f"Selected {len(selected_features)} features (k={k}):")
print(selected_features)

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

x = layers.Dense(32, activation="relu")(encoded)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(input_dim, activation="linear")(x)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 0, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

out = layers.Dense(input_dim, activation="linear")(encoded)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "symmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 1, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

y_train = None
y = None

k = 48
feat_names = X.columns.tolist()

X_scaled = X_train.sample(frac=0.001).dropna().values

n_clusters = 10
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
y_pseudo = km.fit_predict(X_scaled)

est = ExtraTreesClassifier(
    n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
)

boruta = BorutaPy(
    estimator=est,
    n_estimators="auto",
    two_step=True,
    max_iter=40,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
boruta.fit(X_scaled, y_pseudo)
t1 = time.perf_counter()
print(f"Boruta took {(t1 - t0):.1f} seconds")

confirmed_mask = boruta.support_
confirmed_idxs = np.where(confirmed_mask)[0]

if len(confirmed_idxs) < k:
    ranks = boruta.ranking_
    top_k_idxs = np.argsort(ranks)[:k]
    chosen_idxs = np.array(sorted(set(confirmed_idxs).union(set(top_k_idxs))))[:k]
else:
    ranks = boruta.ranking_
    confirmed_ranks = ranks[confirmed_idxs]
    order = np.argsort(confirmed_ranks)
    chosen_idxs = confirmed_idxs[order][:k]

chosen_idxs = np.sort(chosen_idxs)
selected_features = [feat_names[i] for i in chosen_idxs]

print(f"Selected {len(selected_features)} features (k={k}):")
print(selected_features)

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

x = layers.Dense(64, activation="relu")(encoded)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
out = layers.Dense(input_dim, activation="linear")(x)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 0, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

out = layers.Dense(input_dim, activation="linear")(encoded)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "symmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 1, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

y_train = None
y = None

k = 64
feat_names = X.columns.tolist()

X_scaled = X_train.sample(frac=0.001).dropna().values

n_clusters = 10
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
y_pseudo = km.fit_predict(X_scaled)

est = ExtraTreesClassifier(
    n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
)

boruta = BorutaPy(
    estimator=est,
    n_estimators="auto",
    two_step=True,
    max_iter=40,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
boruta.fit(X_scaled, y_pseudo)
t1 = time.perf_counter()
print(f"Boruta took {(t1 - t0):.1f} seconds")

confirmed_mask = boruta.support_
confirmed_idxs = np.where(confirmed_mask)[0]

if len(confirmed_idxs) < k:
    ranks = boruta.ranking_
    top_k_idxs = np.argsort(ranks)[:k]
    chosen_idxs = np.array(sorted(set(confirmed_idxs).union(set(top_k_idxs))))[:k]
else:
    ranks = boruta.ranking_
    confirmed_ranks = ranks[confirmed_idxs]
    order = np.argsort(confirmed_ranks)
    chosen_idxs = confirmed_idxs[order][:k]

chosen_idxs = np.sort(chosen_idxs)
selected_features = [feat_names[i] for i in chosen_idxs]

print(f"Selected {len(selected_features)} features (k={k}):")
print(selected_features)

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

x = layers.Dense(64, activation="relu")(encoded)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
out = layers.Dense(input_dim, activation="linear")(x)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "asymmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 0, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance

num_features = 277
X_scaled = X_train.dropna().values

input_dim = X_scaled.shape[1]

inp = keras.Input(shape=(input_dim,))
encoded = ColumnSelector(indices=chosen_idxs, name="boruta_encoder")(inp)

out = layers.Dense(input_dim, activation="linear")(encoded)

fsae = keras.Model(inputs=inp, outputs=out, name="boruta_shallow_encoder_ae")
fsae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

fsae_hist = fsae.fit(
    X_scaled,
    X_scaled,
    epochs=60,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
)

encoder = keras.Model(inputs=inp, outputs=encoded, name="boruta_encoder_only")

X_encoded = encoder.predict(X_scaled, batch_size=256)
X_reconstructed = fsae.predict(X_scaled, batch_size=256)


X_reconstructed = fsae.predict(X_scaled)

version = "v9"
architecture = "symmetric_architecture"
encoder_size = k

performance = get_results(fsae_hist, fsae, encoder_size, 1, tag=f"fsae/{architecture}/{encoder_size}/{version}").to_frame()

validation_path = f"s3://{results_bucket}/fsae/{architecture}/{encoder_size}/{version}/score.parquet"
weights_path = f"fsae/{architecture}/{str(encoder_size)}/{version}/fsae_savedmodel.zip"

performance.to_parquet(
      validation_path,
      index=True,
      engine="pyarrow",
  )

artifact_dir = f"fsae_artifacts/{architecture}_{encoder_size}_{version}"
os.makedirs(artifact_dir, exist_ok=True)
fsae.save_weights(os.path.join(artifact_dir, "fsae_weights.weights.h5"))

zip_path = shutil.make_archive(artifact_dir, "zip", artifact_dir)
s3 = boto3.client("s3")
s3.upload_file(zip_path, results_bucket, weights_path)


performance
