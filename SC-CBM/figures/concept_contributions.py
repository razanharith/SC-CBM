"""concept_contributions.pdf — diverging lollipop of the linear head's net
contribution of each concept toward the melanoma vote (fold-0 checkpoint, PH2).

Stems start at 0; malignant concepts in mel brick-red, benign in nevus
steel-blue; sorted by contribution; value labels at stem ends.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import RES, OUT
from concepts_meta import CONCEPT_CODES, MALIGNANT_CONCEPTS

F.set_style()

import torch
from model import SC_CBM

ck = torch.load(os.path.join(RES, "checkpoints", "sccbm_ph2_fold0.pth"),
                map_location="cpu")
m = SC_CBM(backbone=ck["args"].get("backbone", "resnet50"))
m.load_state_dict(ck["model_state_dict"])
W = m.classifier.weight.detach().numpy()          # [2, 8]: nevus, melanoma
contrib = W[1] - W[0]                             # net vote toward melanoma

order = np.argsort(contrib)
fig, ax = plt.subplots(figsize=(3.5, 2.9), constrained_layout=True)
for row, i in enumerate(order):
    v = contrib[i]
    col = F.COL["mel"] if CONCEPT_CODES[i] in MALIGNANT_CONCEPTS else F.COL["nevus"]
    ax.plot([0, v], [row, row], color=col, lw=1.6, zorder=3,
            solid_capstyle="round")
    ax.scatter([v], [row], s=42, color=col, zorder=4)
    side = 1 if v >= 0 else -1
    ax.text(v + 0.06 * side, row, f"{v:+.2f}", va="center",
            ha="left" if side > 0 else "right", fontsize=8,
            fontweight="bold", color=col)
ax.axvline(0, color="#555555", lw=1.0, zorder=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([CONCEPT_CODES[i] for i in order], fontsize=8.5)
ax.set_xlim(-2.6, 3.4)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("Net contribution toward melanoma (head weight)")
handles = [Line2D([], [], marker="o", color=F.COL["mel"], ls="", markersize=7,
                  label="malignant concept"),
           Line2D([], [], marker="o", color=F.COL["nevus"], ls="", markersize=7,
                  label="benign concept")]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, handletextpad=0.4)
F.save(fig, "concept_contributions", outdir=OUT)
