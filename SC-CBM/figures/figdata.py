"""Shared data loaders for the SC-CBM paper figures.

Reads the run JSONs in code/results/ and exposes per-run arrays (one row per
seed x fold) plus a percentile bootstrap-95%-CI helper.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
RES = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = [42, 7, 123]
OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "out"))


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def run_rows(doc):
    """Per-run (BAcc, cF1, spatial coherence) in %, one row per fold."""
    return [[r["diagnosis"]["balanced_accuracy"] * 100,
             r["concepts"]["concept_avg_f1"] * 100,
             r["spatial_coherence"] * 100] for r in doc["folds"]]


def load_runs(prefix, ds):
    """All runs of a configuration: 3 seeds x 5 folds (15 rows) when the
    _s{seed} multi-seed files exist, else the single-seed 5-fold file."""
    rows, got = [], False
    for s in SEEDS:
        doc = load(f"{prefix}_s{s}_{ds}")
        if doc:
            got, rows = True, rows + run_rows(doc)
    if not got:
        doc = load(f"{prefix}_{ds}")
        if doc:
            rows = run_rows(doc)
    return np.array(rows) if rows else None


def per_concept_runs(prefix, ds, k=8):
    """Array [n_runs, k] of per-concept F1 in %."""
    arrs, got = [], False
    for s in SEEDS:
        doc = load(f"{prefix}_s{s}_{ds}")
        if doc:
            got = True
            arrs.append([[r["concepts"][f"concept_{i}_f1"] * 100
                          for i in range(k)] for r in doc["folds"]])
    if not got:
        doc = load(f"{prefix}_{ds}")
        if doc:
            arrs.append([[r["concepts"][f"concept_{i}_f1"] * 100
                          for i in range(k)] for r in doc["folds"]])
    return np.vstack(arrs) if arrs else None


def bci(values, n=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean. Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    if len(v) < 2:
        m = float(np.mean(v))
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def declutter(vals, min_gap):
    """Return label y-positions nudged apart so text at nearby values doesn't
    overlap, while the true data points stay put. Greedy: sort ascending, push
    each label up until it clears the previous one by min_gap."""
    order = np.argsort(vals)
    ys = np.array(vals, dtype=float)
    for k in range(1, len(order)):
        i, prev = order[k], order[k - 1]
        if ys[i] - ys[prev] < min_gap:
            ys[i] = ys[prev] + min_gap
    return ys
