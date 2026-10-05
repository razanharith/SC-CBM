"""Aggregate multi-seed result JSONs (E1): pool fold scores across seeds.

Reads results/{tag}_s{seed}_{dataset}.json produced by run_multiseed.sh and prints
pooled mean +- std over all folds x seeds, plus per-seed means, plus a paired
Wilcoxon signed-rank test between two tags (matched on seed+fold).

Usage:
    python multiseed_agg.py sccbm ph2
    python multiseed_agg.py --wilcoxon ablate_concept sccbm ph2
"""

import argparse
import glob
import json
import os
import re

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results")

DIAG_KEYS = ["balanced_accuracy", "accuracy", "sensitivity", "specificity", "f1"]


def fold_metric(r, k):
    if k in DIAG_KEYS:
        return r["diagnosis"][k]
    if k == "concept_avg_f1":
        return r["concepts"]["concept_avg_f1"]
    return r[k]


def load_runs(tag, dataset):
    """Return {seed: [per-fold result dicts]}."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, f"{tag}_s*_{dataset}.json"))):
        m = re.search(rf"{re.escape(tag)}_s(\d+)_{re.escape(dataset)}\.json$", f)
        if not m:
            continue
        doc = json.load(open(f))
        runs[int(m.group(1))] = doc["folds"]
    return runs


def pool(tag, dataset):
    runs = load_runs(tag, dataset)
    if not runs:
        return None
    out = {"seeds": sorted(runs), "n_runs": sum(len(v) for v in runs.values())}
    for k in DIAG_KEYS + ["concept_avg_f1", "intervention_bacc"]:
        vals, per_seed = [], {}
        for s, folds in sorted(runs.items()):
            fv = [fold_metric(r, k) for r in folds]
            per_seed[s] = float(np.mean(fv))
            vals.extend(fv)
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "per_seed": per_seed}
    sc = []
    for folds in runs.values():
        sc.extend([r["spatial_coherence"] for r in folds if not np.isnan(r["spatial_coherence"])])
    out["spatial_coherence"] = {"mean": float(np.mean(sc)) if sc else float("nan"),
                                "std": float(np.std(sc)) if sc else float("nan")}
    return out


def print_pool(tag, dataset):
    p = pool(tag, dataset)
    if not p:
        print(f"{tag}/{dataset}: no multi-seed runs found")
        return
    print(f"== {tag}/{dataset}  (seeds {p['seeds']}, {p['n_runs']} folds) ==")
    for k in DIAG_KEYS + ["concept_avg_f1", "spatial_coherence", "intervention_bacc"]:
        v = p[k]
        if np.isnan(v["mean"]):
            print(f"  {k:20} n/a")
        else:
            per_seed = "   per-seed: " + " ".join(
                f"{m*100:.1f}" for m in v["per_seed"].values()) if "per_seed" in v else ""
            print(f"  {k:20} {v['mean']*100:5.1f} +- {v['std']*100:4.1f}{per_seed}")


def wilcoxon(tag_a, tag_b, dataset):
    from scipy.stats import wilcoxon
    a, b = load_runs(tag_a, dataset), load_runs(tag_b, dataset)
    common = sorted(set(a) & set(b))
    if not common:
        print("no common seeds")
        return
    for k in ["balanced_accuracy", "concept_avg_f1"]:
        va = [fold_metric(r, k) for s in common for r in a[s]]
        vb = [fold_metric(r, k) for s in common for r in b[s]]
        stat, p = wilcoxon(va, vb)
        print(f"  {k:20} {tag_a}={np.mean(va)*100:.1f} vs {tag_b}={np.mean(vb)*100:.1f} "
              f"  Wilcoxon W={stat}, p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("dataset", nargs="?", default="ph2")
    ap.add_argument("--wilcoxon", metavar="OTHER_TAG")
    args = ap.parse_args()
    print_pool(args.tag, args.dataset)
    if args.wilcoxon:
        wilcoxon(args.wilcoxon, args.tag, args.dataset)


if __name__ == "__main__":
    main()
