"""Emit LaTeX booktabs tables for the enriched SC-CBM paper.

Reads the result JSONs and prints ready-to-paste tables:
  T1  backbone comparison (PH2 + Derm7pt)
  T2  spatial-coherence weight (lambda_s) sweep
  T3  per-concept F1 (8 concepts x 2 datasets)
  T4  pooling ablation (GAP vs attention)
  T5  comparison with prior concept-based methods on PH2
Run after the sweep finishes: python make_tables.py
"""

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from concepts_meta import CONCEPT_CODES

RES = os.path.join(os.path.dirname(__file__), "results")


def load(tag, ds):
    p = os.path.join(RES, f"{tag}_{ds}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def agg(doc, key):
    return doc["aggregate"][key]["mean"] * 100


def pcf1(doc):
    K = len(CONCEPT_CODES)
    v = np.zeros(K)
    for r in doc["folds"]:
        v += np.array([r["concepts"][f"concept_{i}_f1"] for i in range(K)])
    return v / len(doc["folds"]) * 100


def t_backbone():
    order = [("resnet18", "ResNet-18"), ("resnet50", "ResNet-50"),
             ("resnet101", "ResNet-101"), ("densenet201", "DenseNet-201")]
    print("% ── T1 backbone comparison ──")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Dataset & Backbone & BAcc & Concept F1 & Spatial coh. \\\\\n\\midrule")
    for ds, dsname in [("ph2", "\\textsc{PH$^2$}"), ("derm7pt", "\\textsc{Derm7pt}")]:
        rows = []
        for tag, name in order:
            doc = load("sccbm", ds) if tag == "resnet50" else load(f"bb_{tag}", ds)
            if not doc or not doc.get("aggregate"):
                continue
            sp = doc["aggregate"]["spatial_coherence"]["mean"]
            sp = f"{sp*100:.1f}" if not np.isnan(sp) else "--"
            rows.append((name, agg(doc, "balanced_accuracy"),
                         agg(doc, "concept_avg_f1"), sp))
        best = max(r[1] for r in rows) if rows else None
        for i, (name, ba, cf, sp) in enumerate(rows):
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{dsname}}} & " if i == 0 else " & "
            bstr = f"\\textbf{{{ba:.1f}}}" if ba == best else f"{ba:.1f}"
            print(f"{first}{name} & {bstr} & {cf:.1f} & {sp} \\\\")
        print("\\midrule")
    print("\\bottomrule\n\\end{tabular}\n")


def t_lambda():
    pts = {}
    p0 = load("ablate_concept", "ph2")
    if p0: pts[0.0] = p0
    p5 = load("sccbm", "ph2")
    if p5: pts[0.5] = p5
    for f in glob.glob(os.path.join(RES, "sccbm_ls*_ph2.json")):
        m = re.search(r"sccbm_ls([0-9.]+)_ph2", f)
        if m: pts[float(m.group(1))] = json.load(open(f))
    print("% ── T2 lambda_s sweep (PH2) ──")
    print("\\begin{tabular}{lccc}\n\\toprule")
    print("$\\lambda_s$ & BAcc & Concept F1 & Spatial coh. \\\\\n\\midrule")
    for ls in sorted(pts):
        d = pts[ls]
        sp = d["aggregate"]["spatial_coherence"]["mean"]
        sp = f"{sp*100:.1f}" if not np.isnan(sp) else "--"
        ba = agg(d, "balanced_accuracy")
        bstr = f"\\textbf{{{ba:.1f}}}" if abs(ls-0.5) < 1e-9 else f"{ba:.1f}"
        print(f"{ls:g} & {bstr} & {agg(d,'concept_avg_f1'):.1f} & {sp} \\\\")
    print("\\bottomrule\n\\end{tabular}\n")


def t_perconcept():
    ph2, d7 = load("sccbm", "ph2"), load("sccbm", "derm7pt")
    if not (ph2 and d7):
        return
    a, b = pcf1(ph2), pcf1(d7)
    print("% ── T3 per-concept F1 ──")
    print("\\begin{tabular}{l" + "c"*len(CONCEPT_CODES) + "}\n\\toprule")
    print("Dataset & " + " & ".join(CONCEPT_CODES) + " \\\\\n\\midrule")
    print("\\textsc{PH$^2$} & " + " & ".join(f"{v:.0f}" for v in a) + " \\\\")
    print("\\textsc{Derm7pt} & " + " & ".join(f"{v:.0f}" for v in b) + " \\\\")
    print("\\bottomrule\n\\end{tabular}\n")


def t_pooling():
    gap, att = load("sccbm", "ph2"), load("pool_attention", "ph2")
    if not (gap and att):
        print("% pooling: attention run missing\n")
        return
    print("% ── T4 pooling ablation (PH2) ──")
    print("\\begin{tabular}{lccc}\n\\toprule")
    print("Pooling & BAcc & Concept F1 & Spatial coh. \\\\\n\\midrule")
    for name, d in [("Global average", gap), ("Attention", att)]:
        sp = d["aggregate"]["spatial_coherence"]["mean"]
        print(f"{name} & {agg(d,'balanced_accuracy'):.1f} & "
              f"{agg(d,'concept_avg_f1'):.1f} & {sp*100:.1f} \\\\")
    print("\\bottomrule\n\\end{tabular}\n")


def t_comparison():
    ph2 = load("sccbm", "ph2")
    d7 = load("sccbm", "derm7pt")
    print("% ── T5 comparison with prior concept-based methods ──")
    print("\\begin{tabular}{llcc}\n\\toprule")
    print("Method & Training & PH$^2$ & Derm7pt \\\\\n\\midrule")
    print("Coherent-CBE~\\citep{patricio2023coherent} & Trained CBM & -- & -- \\\\")
    print("Two-Step~\\citep{patricio2025twostep} & Training-free & 85.05 & -- \\\\")
    ph = f"\\textbf{{{agg(ph2,'balanced_accuracy'):.1f}}}" if ph2 else "--"
    d = f"{agg(d7,'balanced_accuracy'):.1f}" if d7 else "--"
    print(f"\\method (ours) & Trained CBM & {ph} & {d} \\\\")
    print("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    t_comparison()
    t_backbone()
    t_lambda()
    t_perconcept()
    t_pooling()
