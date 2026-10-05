"""backbone_comparison.pdf — grouped horizontal bars of BAcc per backbone.

Shape: grouped comparison bars (skill section 1). Balanced accuracy per frozen
backbone, PH2 (hero) vs Derm7pt (amber), bars from 0 so the reader can compare
absolute levels honestly; per-concept F1 is annotated per bar, and the
concept-layer collapse of ResNet-18 / DenseNet-201 (PH2 cF1 < 10%) is called out.
Horizontal bars keep the long backbone names off a rotated axis.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import load, OUT

F.set_style()

ORDER = [("resnet18", "ResNet-18"), ("resnet50", "ResNet-50"),
         ("resnet101", "ResNet-101"), ("densenet201", "DenseNet-201")]
H = 0.38  # bar thickness within a backbone group


def backbone_stats(tag):
    """(PH2 BAcc, PH2 cF1, Derm7pt BAcc, Derm7pt cF1) in %, averaged."""
    if tag == "resnet50":  # default backbone: pool the multi-seed runs
        def pool(ds):
            rows = []
            for s in (42, 7, 123):
                rows += [[r["diagnosis"]["balanced_accuracy"] * 100,
                          r["concepts"]["concept_avg_f1"] * 100]
                         for r in load(f"sccbm_s{s}_{ds}")["folds"]]
            return np.array(rows).mean(axis=0)
        b2, c2 = pool("ph2")
        b7, c7 = pool("derm7pt")
    else:
        a2 = load(f"bb_{tag}_ph2")["aggregate"]
        a7 = load(f"bb_{tag}_derm7pt")["aggregate"]
        b2 = a2["balanced_accuracy"]["mean"] * 100
        c2 = a2["concept_avg_f1"]["mean"] * 100
        b7 = a7["balanced_accuracy"]["mean"] * 100
        c7 = a7["concept_avg_f1"]["mean"] * 100
    return b2, c2, b7, c7


fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
ys = np.arange(len(ORDER))[::-1]
for y, (tag, name) in zip(ys, ORDER):
    b2, c2, b7, c7 = backbone_stats(tag)
    ax.barh(y + H / 2, b2, H, color=F.COL["hero"], zorder=3)
    ax.barh(y - H / 2, b7, H, color=F.COL["amber"], zorder=3)
    ax.text(b2 + 1.2, y + H / 2, f"{b2:.1f}", va="center", ha="left",
            fontsize=7.3, fontweight="bold", color=F.COL["hero"])
    ax.text(b7 + 1.2, y - H / 2, f"{b7:.1f}", va="center", ha="left",
            fontsize=7.3, fontweight="bold", color=F.COL["amber"])
    # per-concept F1 past the bars; a collapsed PH2 concept layer (cF1 < 10%)
    # is flagged by colouring the label brick-red instead of drawing an arrow
    collapsed = c2 < 10
    ax.text(101, y, f"cF1 {c2:.0f} / {c7:.0f}"
            + ("  (collapse)" if collapsed else ""),
            va="center", ha="left", fontsize=7.2,
            color=F.COL["mel"] if collapsed else F.COL["muted"],
            fontweight="bold" if collapsed else "normal")
ax.set_yticks(ys)
ax.set_yticklabels([n for _, n in ORDER], fontsize=8.5)
ax.set_xlim(0, 132)
ax.set_xticks(range(0, 101, 25))
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_xlabel("Balanced accuracy (%)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
handles = [Line2D([], [], marker="s", color=F.COL["hero"], ls="", markersize=8,
                  label="PH$^2$"),
           Line2D([], [], marker="s", color=F.COL["amber"], ls="", markersize=8,
                  label="Derm7pt")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, fontsize=8, handletextpad=0.4, columnspacing=1.4)
F.save(fig, "backbone_comparison", outdir=OUT)
