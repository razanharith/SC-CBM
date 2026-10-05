"""fig_calibration.pdf — before/after dumbbell of per-concept F1 (PH2).

One row per concept: open dot = raw (tau=0.5), filled dot = calibrated (per-
concept tau_k), connected by an arrow whose colour shows the direction (sage
green = F1 recovered, brick-red = dropped). A dumbbell avoids the line-crossing
of a slope chart. Pooled cF1 / ECE annotated; RSTR has no validation positives.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import RES, OUT

F.set_style()

per_concept = json.load(open(os.path.join(RES, "calibration_per_concept_ph2.json")))
cal = json.load(open(os.path.join(RES, "calibrated_sccbm_ph2.json")))["pooled"]

codes = list(per_concept.keys())
raw = np.array([per_concept[c]["raw"] for c in codes]) * 100
calib = np.array([per_concept[c]["calibrated"] for c in codes]) * 100

# order rows by calibrated F1 so the strong concepts sit on top
order = np.argsort(calib)
ys = np.arange(len(codes))

fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
for y, i in zip(ys, order):
    r, c = raw[i], calib[i]
    if r == 0 and c == 0:
        col = F.COL["muted"]
        ax.scatter([0], [y], s=30, color=col, zorder=4)
    else:
        col = F.COL["good"] if c >= r else F.COL["mel"]
        ax.annotate("", xy=(c, y), xytext=(r, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=3)
        ax.scatter([r], [y], s=34, facecolors="white", edgecolors="#888888",
                   linewidths=1.2, zorder=4)              # raw = open
        ax.scatter([c], [y], s=40, color=col, zorder=5)    # calibrated = filled
        ax.text(max(r, c) + 2.0, y, f"{c:.0f}", va="center", ha="left",
                fontsize=7.5, fontweight="bold", color=col)
ax.set_yticks(ys)
ax.set_yticklabels([codes[i] for i in order], fontsize=8.5)
ax.set_xlim(-2, 90)
ax.set_ylim(-0.6, len(codes) - 0.4)
ax.set_xlabel("Per-concept F1 on PH$^2$ (%)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
# legend: raw vs calibrated marker + pooled summary
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white",
                  markeredgecolor="#888888", markersize=7, label="raw ($\\tau=0.5$)"),
           Line2D([], [], marker="o", ls="", color=F.COL["good"], markersize=7,
                  label="calibrated ($\\tau_k$)")]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, handletextpad=0.4)
ax.annotate(f"pooled cF1 {cal['cf1_raw_mean']*100:.1f} $\\rightarrow$ "
            f"{cal['cf1_calibrated_mean']*100:.1f}   |   "
            f"ECE {cal['ece_raw']:.3f} $\\rightarrow$ {cal['ece_temperature']:.3f}",
            xy=(0.5, 1.01), xycoords="axes fraction", ha="center", va="bottom",
            fontsize=7.8, color=F.COL["ink"])
F.save(fig, "fig_calibration", outdir=OUT)
