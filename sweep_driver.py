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
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "hyperparameter_tuning.ipynb")
X_CACHE = os.path.join(HERE, "X_cache.pkl")

# Substitute for the S3 data-load cell once a local cache exists.
CACHED_LOAD = """
bucket = os.getenv("BUCKET_NAME")
results_bucket = os.getenv("HYPERPARAMETERS_RESULTS_BUCKET_NAME")
X = pd.read_pickle({cache!r})
print("loaded X from local cache:", X.shape, flush=True)
"""


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
    ns = {"__name__": "__main__"}
    for n, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        sec = "".join(cell["source"])
        if not sec.strip():
            continue
        try:
            curves = expected_curves(sec, ns)
        except Exception:
            curves = None
        if curves and all(
            os.path.exists(os.path.join(HERE, "learning_curves", c)) for c in curves
        ):
            print(f"[driver] cell {n}: SKIP (done: {', '.join(curves)})", flush=True)
            continue
        if "ds.dataset(" in sec and os.path.exists(X_CACHE):
            print(f"[driver] cell {n}: using local X cache", flush=True)
            exec(compile(CACHED_LOAD.format(cache=X_CACHE), "cached_load", "exec"), ns)
            continue
        label = f"-> {', '.join(curves)}" if curves else ""
        print(f"[driver] cell {n}: run {label}", flush=True)
        t0 = time.time()
        exec(compile(sec, f"nb_cell_{n}", "exec"), ns)
        print(f"[driver] cell {n}: done in {time.time() - t0:.0f}s", flush=True)
        if "ds.dataset(" in sec and not os.path.exists(X_CACHE):
            ns["X"].to_pickle(X_CACHE)
            print(f"[driver] cached X to {X_CACHE}", flush=True)
    print("[driver] sweep complete", flush=True)


if __name__ == "__main__":
    main()
