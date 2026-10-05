"""Turn the SC-CBM result JSONs into Chapter-3 tables (markdown + LaTeX booktabs).

Reads results/{tag}_{dataset}.json produced by train.py and prints:
  1. Main 5-fold results (SC-CBM) for PH2 and Derm7pt.
  2. The PH2 ablation ladder: plain -> +concept -> +spatial (full).
  3. Per-concept F1 (mean over folds) for the full model.
Run after run_all.sh finishes:  python make_report.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from concepts_meta import CONCEPT_CODES

RES = os.path.join(os.path.dirname(__file__), "results")


def load(tag, dataset):
    p = os.path.join(RES, f"{tag}_{dataset}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def m(agg, key, scale=100):
    a = agg[key]
    return f"{a['mean']*scale:.1f}", f"{a['std']*scale:.1f}"


def per_concept_f1(doc):
    """Mean per-concept F1 across folds -> dict code->f1(%)."""
    folds = doc["folds"]
    K = len(CONCEPT_CODES)
    vals = np.zeros(K)
    for r in folds:
        c = r["concepts"]
        vals += np.array([c[f"concept_{i}_f1"] for i in range(K)])
    vals /= len(folds)
    return {CONCEPT_CODES[i]: vals[i] * 100 for i in range(K)}


def main():
    print("=" * 70)
    print("MAIN RESULTS (5-fold SC-CBM)")
    print("=" * 70)
    header = f"{'Dataset':10} {'BAcc':>12} {'Sens':>12} {'Spec':>12} {'F1':>12} {'cF1':>12}"
    print(header)
    for ds in ["ph2", "derm7pt"]:
        doc = load("sccbm", ds)
        if not doc or not doc.get("aggregate"):
            print(f"{ds:10}  (pending)")
            continue
        a = doc["aggregate"]
        def cell(k):
            mn, sd = m(a, k)
            return f"{mn}+-{sd}"
        print(f"{ds:10} {cell('balanced_accuracy'):>12} {cell('sensitivity'):>12} "
              f"{cell('specificity'):>12} {cell('f1'):>12} {cell('concept_avg_f1'):>12}")

    print("\n" + "=" * 70)
    print("PH2 ABLATION LADDER")
    print("=" * 70)
    print(f"{'Config':22} {'BAcc':>10} {'cF1':>10} {'Spatial':>10} {'Interv':>10}")
    labels = [("ablate_plain", "diagnosis CE only"),
              ("ablate_concept", "+ concept BCE"),
              ("sccbm", "+ spatial (full)")]
    for tag, label in labels:
        doc = load(tag, "ph2")
        if not doc or not doc.get("aggregate"):
            print(f"{label:22}  (pending)")
            continue
        a = doc["aggregate"]
        bacc = f"{a['balanced_accuracy']['mean']*100:.1f}"
        cf1 = f"{a['concept_avg_f1']['mean']*100:.1f}"
        spc = a["spatial_coherence"]["mean"]
        spc = f"{spc*100:.1f}" if not np.isnan(spc) else "n/a"
        iv = f"{a['intervention_bacc']['mean']*100:.1f}"
        print(f"{label:22} {bacc:>10} {cf1:>10} {spc:>10} {iv:>10}")

    print("\n" + "=" * 70)
    print("PER-CONCEPT F1 (full SC-CBM, mean over folds)")
    print("=" * 70)
    for ds in ["ph2", "derm7pt"]:
        doc = load("sccbm", ds)
        if not doc:
            continue
        pcf = per_concept_f1(doc)
        print(f"[{ds}] " + "  ".join(f"{k}={v:.0f}" for k, v in pcf.items()))

    # LaTeX booktabs for the ablation table
    print("\n" + "=" * 70)
    print("LATEX (PH2 ablation, booktabs)")
    print("=" * 70)
    rows = []
    for tag, label in labels:
        doc = load(tag, "ph2")
        if not doc or not doc.get("aggregate"):
            continue
        a = doc["aggregate"]
        spc = a["spatial_coherence"]["mean"]
        spc = f"{spc*100:.1f}" if not np.isnan(spc) else "--"
        rows.append(f"    {label} & {a['balanced_accuracy']['mean']*100:.1f} & "
                    f"{a['concept_avg_f1']['mean']*100:.1f} & {spc} & "
                    f"{a['intervention_bacc']['mean']*100:.1f} \\\\")
    if rows:
        print("\\begin{tabular}{lcccc}\n\\toprule")
        print("    Configuration & BAcc & Concept F1 & Spatial Coh. & Interv. BAcc \\\\")
        print("\\midrule")
        print("\n".join(rows))
        print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
