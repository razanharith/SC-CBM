"""fig_multiseed.pdf — strip plot of per-run BAcc on PH2 across 4 configs.

Shape: strip / jitter with mean diamond + bootstrap-95%-CI bar (skill section 3).
One dot per run (3 seeds x 5 folds = 15); no boxes.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import load_runs, bci, OUT

F.set_style()

CONFIGS = [("Plain GAP\n(diagnosis only)", "ablate_plain", F.COL["baseline"], {}),
           ("GAP\n+ GT masks", "sccbm", F.COL["amber"], {}),
           ("Attention\n+ GT masks", "pool_attention", F.COL["hero"], {}),
           ("Attention\n+ SAM masks", "pool_attention_sam", F.COL["hero"],
            {"facecolors": "white", "edgecolors": F.COL["hero"], "linewidths": 1.2})]

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.5, 3.1), constrained_layout=True)
for i, (name, prefix, col, kw) in enumerate(CONFIGS):
    runs = load_runs(prefix, "ph2")
    pts = runs[:, 0]
    x = rng.uniform(-0.15, 0.15, len(pts)) + i
    ax.scatter(x, pts, s=13, alpha=0.6, color=col, zorder=3, **kw)
    m, lo, hi = bci(pts)
    ax.plot([i, i], [lo, hi], color="#333333", lw=2.6, zorder=4,
            solid_capstyle="round")
    ax.plot([i], [m], marker="D", color="#1a1a1a", markersize=7, zorder=5)
    ax.text(i + 0.24, m, f"{m:.1f}", va="center", ha="left", fontsize=8,
            fontweight="bold", color="#1a1a1a")
    ax.text(i + 0.24, m - 2.6, f"n={len(pts)}", va="center", ha="left",
            fontsize=7, color=F.COL["muted"])
ax.set_xticks(range(len(CONFIGS)))
ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=8)
ax.set_ylabel("Balanced accuracy on PH$^2$ (%)")
ax.set_ylim(55, 100)
F.save(fig, "fig_multiseed", outdir=OUT)
