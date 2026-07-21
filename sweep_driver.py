#!/usr/bin/env python
"""Resumable driver for the hyperparameter sweep.

Executes the code cells of hyperparameter_tuning.ipynb in order in a shared
namespace. Training cells whose learning-curve files already exist
(learning_curves/<method>_<arch>_<k>_v9.json) are skipped, so the sweep can
be interrupted (laptop sleep) and relaunched at the cost of only the model
that was in flight.
"""
import json
import os
import pickle
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "hyperparameter_tuning.ipynb")
X_CACHE = os.path.join(HERE, "X_cache.pkl")
PENDING = os.path.join(HERE, "pending_scores")

# IMPORTANT: pyarrow must never be imported in this (training) process.
# TensorFlow and pyarrow each bundle abseil; on macOS the dynamic linker
# resolves TF's semaphore wait into libarrow's abseil copy, deadlocking
# model.fit() on its first step. All parquet/S3-dataset work therefore
# happens in short-lived subprocesses.

# Substitute for the S3 data-load cell once a local cache exists.
CACHED_LOAD = """
bucket = os.getenv("BUCKET_NAME")
results_bucket = os.getenv("HYPERPARAMETERS_RESULTS_BUCKET_NAME")
X = pd.read_pickle({cache!r})
print("loaded X from local cache:", X.shape, flush=True)
"""

BUILD_CACHE = """
import os
import pandas as pd
import pyarrow.dataset as ds
from dotenv import load_dotenv
load_dotenv(os.path.join({here!r}, ".env"))
bucket = os.getenv("BUCKET_NAME")
columns = pd.read_parquet(f"s3://{{bucket}}/headers.parquet")["0"].values
toscore = ds.dataset(f"s3://{{bucket}}/samples/", format="parquet").to_table().to_pandas()
toscore.columns = columns
bad = [c for c in toscore.columns if c < 320 or (c > 590 and c < 610)]
X = toscore.drop(columns=bad).dropna()
X.to_pickle({cache!r})
print("cached", X.shape)
"""

UPLOAD_PENDING = """
import os, pickle
import pandas as pd
for f in sorted(os.listdir({pending!r})):
    path, df = pickle.load(open(os.path.join({pending!r}, f), "rb"))
    df.to_parquet(path, index=True, engine="pyarrow")
    print("uploaded", path, flush=True)
"""


def save_pending(df, path):
    """Store a score frame locally; a pyarrow subprocess uploads it later."""
    os.makedirs(PENDING, exist_ok=True)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", path) + ".pkl"
    with open(os.path.join(PENDING, fname), "wb") as fp:
        pickle.dump((path, df), fp)
    print(f"[driver] queued score for {path}", flush=True)


def expected_curves(sec, ns):
    """Curve filenames this cell would produce, or None if not a training cell."""
    if "get_results(" not in sec or "_hist" not in sec:
        return None
    method = "cae" if "cae_hist" in sec else "fsae"
    m = re.search(r'architecture = "(\w+)"', sec)
    if not m:
        return None
    arch = m.group(1)
    loop = re.search(r"for k in \[([0-9, ]+)\]", sec)
    if loop:
        ks = [int(x) for x in loop.group(1).split(",")]
    else:
        km = re.search(r"^k = (\d+)", sec, re.M)
        ks = [int(km.group(1))] if km else [int(ns["k"])]
    return [f"{method}_{arch}_{k}_v9.json" for k in ks]


def main():
    cells = json.load(open(NB))["cells"]
    ns = {"__name__": "__main__", "save_pending": save_pending}
    for n, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        sec = "".join(cell["source"])
        if not sec.strip():
            continue
        # keep pyarrow out of this process (see note at top)
        sec = sec.replace("import pyarrow.dataset as ds\n", "")
        sec = re.sub(
            r"performance\.to_parquet\([^)]*\)",
            "save_pending(performance, validation_path)",
            sec,
        )
        try:
            curves = expected_curves(sec, ns)
        except Exception:
            curves = None
        if curves and all(
            os.path.exists(os.path.join(HERE, "learning_curves", c)) for c in curves
        ):
            print(f"[driver] cell {n}: SKIP (done: {', '.join(curves)})", flush=True)
            continue
        if "ds.dataset(" in sec:
            if not os.path.exists(X_CACHE):
                print(f"[driver] cell {n}: building X cache in subprocess", flush=True)
                subprocess.run(
                    [sys.executable, "-c", BUILD_CACHE.format(here=HERE, cache=X_CACHE)],
                    check=True,
                )
            print(f"[driver] cell {n}: loading X from local cache", flush=True)
            exec(compile(CACHED_LOAD.format(cache=X_CACHE), "cached_load", "exec"), ns)
            continue
        label = f"-> {', '.join(curves)}" if curves else ""
        print(f"[driver] cell {n}: run {label}", flush=True)
        t0 = time.time()
        exec(compile(sec, f"nb_cell_{n}", "exec"), ns)
        print(f"[driver] cell {n}: done in {time.time() - t0:.0f}s", flush=True)
    if os.path.isdir(PENDING) and os.listdir(PENDING):
        print("[driver] uploading queued scores via pyarrow subprocess", flush=True)
        subprocess.run(
            [sys.executable, "-c", UPLOAD_PENDING.format(pending=PENDING)], check=True
        )
    print("[driver] sweep complete", flush=True)


if __name__ == "__main__":
    main()
