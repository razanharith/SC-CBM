"""sccbm_lambda_tradeoff.pdf — spatial-regulariser weight (lambda_s) trade-off.

Single-seed PH2 sweep. Three series with direct end-of-line labels (no legend),
the lambda_s = 0.5 operating point ringed, light band over lambda_s in [0.25,
0.75]. Vector PDF; paper/SC-CBM.tex includes the .pdf.
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import load, RES, OUT

F.set_style()

points = {}
p0 = os.path.join(RES, "ablate_concept_ph2.json")        # lambda_s = 0
if os.path.exists(p0):
    points[0.0] = load("ablate_concept_ph2")
points[0.5] = load("sccbm_ph2")
import glob, json
for f in glob.glob(os.path.join(RES, "sccbm_ls*_ph2.json")):
    m = re.search(r"sccbm_ls([0-9.]+)_ph2", f)
    if m:
        points[float(m.group(1))] = json.load(open(f))

def agg(doc):
    a = doc["aggregate"]
    return (a["balanced_accuracy"]["mean"] * 100,
            a["concept_avg_f1"]["mean"] * 100,
            a["spatial_coherence"]["mean"] * 100)

xs = sorted(points)
bacc = [agg(points[x])[0] for x in xs]
cf1 = [agg(points[x])[1] for x in xs]
spa = [agg(points[x])[2] for x in xs]

fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
ax.axvspan(0.25, 0.75, color=F.COL["pale"], alpha=0.35, zorder=1)
# three well-separated hues (blue / green / brick-red) so the lines stay
# distinct; the teal 2nd-series colour would sit too close to the green here
series = [(spa, F.COL["hero"], "Spatial coherence", (4, -2)),
          (bacc, F.COL["good"], "Balanced accuracy", (4, 7)),
          (cf1, F.COL["mel"], "Concept F1", (4, -4))]
for ys, col, label, off in series:
    ax.plot(xs, ys, "-", color=col, lw=1.6, zorder=3,
            solid_capstyle="round")
    ax.scatter(xs, ys, s=26, color=col, zorder=4)
    ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                xytext=off, fontsize=8, color=col, fontweight="bold",
                va="center", zorder=5)
i5 = xs.index(0.5)
ax.scatter([0.5], [spa[i5]], s=200, facecolors="none", edgecolors="black",
           linewidths=1.2, zorder=6)
ax.annotate("$\\lambda_s$ = 0.5 (used)", (0.5, spa[i5]), textcoords="offset points",
            xytext=(-8, -16), fontsize=7.5, color="#333333", ha="right",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=6)
ax.set_xlabel(r"Spatial-coherence weight $\lambda_s$")
ax.set_ylabel("Score on PH$^2$ (%)")
ax.set_xlim(-0.06, 1.24)
ax.set_ylim(15, 102)
F.save(fig, "sccbm_lambda_tradeoff", outdir=OUT)
