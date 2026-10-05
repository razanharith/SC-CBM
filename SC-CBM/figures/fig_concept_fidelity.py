"""fig_concept_fidelity.pdf — two-panel concept figure for Section 4.9.

(a) which concepts SC-CBM detects well: per-concept F1 on PH2 vs Derm7pt, as
    grouped horizontal bars from 0 with 95% bootstrap CI caps (skill section 1).
(b) which concepts drive the diagnosis: the linear head's learned net weight
    toward melanoma, as diverging bars around 0 (skill section 2), malignant in
    brick-red and benign in steel-blue.
Read together they connect detection quality to decision polarity concept-by-
concept, the paper's core interpretability claim.

Supersedes standalone per_concept_f1.pdf and concept_contributions.pdf.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import figdata as D
from figdata import RES, OUT, bci
from concepts_meta import CONCEPT_CODES, MALIGNANT_CONCEPTS
from model import SC_CBM

F.set_style()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(F.DOUBLE[0], 3.4),
                               constrained_layout=True)

# ---- (a) per-concept F1, PH2 vs Derm7pt (grouped bars + 95% CI) -----------
ph2 = D.per_concept_runs("sccbm", "ph2")
d7 = D.per_concept_runs("sccbm", "derm7pt")
ys = np.arange(len(CONCEPT_CODES))[::-1]
Hh = 0.38
for y, code in zip(ys, CONCEPT_CODES):
    k = CONCEPT_CODES.index(code)
    for vals, col, yy in [(ph2[:, k], F.COL["hero"], y + Hh / 2),
                          (d7[:, k], F.COL["amber"], y - Hh / 2)]:
        m, lo, hi = bci(vals)
        axL.barh(yy, m, Hh, color=col, zorder=3,
                 xerr=[[m - lo], [hi - m]], capsize=2,
                 error_kw=dict(lw=0.9, ecolor="#666666", zorder=4))
        axL.text(hi + 1.4, yy, f"{m:.0f}", va="center", ha="left", fontsize=7.2,
                 fontweight="bold", color=col)
axL.set_yticks(ys)
axL.set_yticklabels(CONCEPT_CODES, fontsize=8.5)
axL.set_xlim(0, 90)
axL.set_xticks(range(0, 81, 20))
axL.set_ylim(-0.6, len(CONCEPT_CODES) - 0.4)
axL.set_xlabel("Per-concept F1 (%)  [95% bootstrap CI]")
axL.set_title("(a) Detection", loc="left", fontsize=9.5)
axL.grid(axis="y", visible=False)
axL.grid(axis="x", visible=True)
handlesL = [Line2D([], [], marker="s", color=F.COL["hero"], ls="", markersize=8,
                   label="PH$^2$"),
            Line2D([], [], marker="s", color=F.COL["amber"], ls="", markersize=8,
                   label="Derm7pt")]
# above the axes (title sits left, legend right) so it never covers a bar/label
axL.legend(handles=handlesL, loc="lower right", bbox_to_anchor=(1.0, 1.0),
           ncol=2, fontsize=8, handletextpad=0.4, columnspacing=1.2)

# ---- (b) learned head weight per concept (diverging bars) -----------------
ck = torch.load(os.path.join(RES, "checkpoints", "sccbm_ph2_fold0.pth"),
                map_location="cpu")
m = SC_CBM(backbone=ck["args"].get("backbone", "resnet50"))
m.load_state_dict(ck["model_state_dict"])
W = m.classifier.weight.detach().numpy()          # [2, 8]: nevus, melanoma
contrib = W[1] - W[0]                             # net vote toward melanoma

order = np.argsort(contrib)                       # most benign at bottom
for row, i in enumerate(order):
    v = contrib[i]
    col = F.COL["mel"] if CONCEPT_CODES[i] in MALIGNANT_CONCEPTS else F.COL["nevus"]
    axR.barh(row, v, 0.62, color=col, zorder=3)
    side = 1 if v >= 0 else -1
    axR.text(v + 0.07 * side, row, f"{v:+.2f}", va="center",
             ha="left" if side > 0 else "right", fontsize=8,
             fontweight="bold", color=col)
axR.axvline(0, color="#555555", lw=1.0, zorder=4)
axR.set_yticks(range(len(order)))
axR.set_yticklabels([CONCEPT_CODES[i] for i in order], fontsize=8.5)
axR.set_xlim(-0.42, 0.40)
axR.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
axR.set_ylim(-0.6, len(order) - 0.4)
axR.grid(axis="y", visible=False)
axR.grid(axis="x", visible=True)
axR.set_xlabel("Net contribution toward melanoma (head weight)")
axR.set_title("(b) Polarity", loc="left", fontsize=9.5)
handlesR = [Line2D([], [], marker="s", color=F.COL["mel"], ls="", markersize=8,
                   label="malignant concept"),
            Line2D([], [], marker="s", color=F.COL["nevus"], ls="", markersize=8,
                   label="benign concept")]
axR.legend(handles=handlesR, loc="lower right", fontsize=7.5, handletextpad=0.4)

F.save(fig, "fig_concept_fidelity", outdir=OUT)
