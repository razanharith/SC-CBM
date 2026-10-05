"""per_concept_f1.pdf — paired Cleveland dots, per-concept F1 on PH2 vs Derm7pt.

Color = dataset (PH2 hero, Derm7pt amber); filled marker = malignant concept,
open marker = benign concept (redundant channel, not color alone). Connected
dots per concept; thin bootstrap-95%-CI whiskers from the pooled runs.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import figdata as D
from concepts_meta import CONCEPT_CODES, MALIGNANT_CONCEPTS
from figdata import bci, OUT

F.set_style()

ph2 = D.per_concept_runs("sccbm", "ph2")
d7 = D.per_concept_runs("sccbm", "derm7pt")

fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
ys = np.arange(len(CONCEPT_CODES))[::-1]
for y, code in zip(ys, CONCEPT_CODES):
    k = CONCEPT_CODES.index(code)
    mal = code in MALIGNANT_CONCEPTS
    for vals, col, side in [(ph2[:, k], F.COL["hero"], -1), (d7[:, k], F.COL["amber"], 1)]:
        m, lo, hi = bci(vals)
        ax.plot([lo, hi], [y, y], color=col, lw=1.4, alpha=0.6, zorder=3,
                solid_capstyle="round")
        if mal:
            ax.scatter([m], [y], s=42, color=col, zorder=4)
        else:
            ax.scatter([m], [y], s=46, facecolors="white", edgecolors=col,
                       linewidths=1.5, zorder=4)
        # clear the whisker's own end (hi on the right, lo on the left), not
        # just the mean, so the label never sits on top of the CI line; a
        # small vertical stagger (hero text above the row, Derm7pt below)
        # keeps the two labels from merging on rows where the two dots (and
        # their outward-facing labels) sit close together, e.g. BWV, RS
        edge = hi if side > 0 else lo
        ax.text(edge + 1.8 * side, y - 0.16 * side, f"{m:.0f}", va="center",
                ha="left" if side > 0 else "right", fontsize=7.5,
                fontweight="bold", color=col)
    ax.plot([ph2[:, k].mean(), d7[:, k].mean()], [y, y], color="#BBBBBB",
            lw=0.9, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels(CONCEPT_CODES, fontsize=8.5)
ax.set_xlim(-9, 78)
ax.set_xlabel("Per-concept F1 (%)  [95% bootstrap CI]")
handles = [Line2D([], [], marker="o", color=F.COL["hero"], ls="", markersize=7,
                  label="PH$^2$"),
           Line2D([], [], marker="o", color=F.COL["amber"], ls="", markersize=7,
                  label="Derm7pt"),
           Line2D([], [], marker="o", color="#555555", ls="", markersize=7,
                  markerfacecolor="#555555", label="malignant concept"),
           Line2D([], [], marker="o", color="#555555", ls="", markersize=7,
                  markerfacecolor="white", label="benign concept")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=4, fontsize=7.5, handletextpad=0.4, columnspacing=1.1)
F.save(fig, "per_concept_f1", outdir=OUT)
