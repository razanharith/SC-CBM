"""fig_maskfidelity.pdf — grouped bars of the three PH2 metrics across the four
mask x pooling configurations (skill section 1).

Four configs (GAP/attention pooling x GT/SAM masks), three metrics each
(balanced accuracy, spatial coherence, concept F1) with 95% bootstrap CI caps
and value labels. Replaces the earlier scatter+dot-size encoding, which was hard
to read; the bars make every number directly comparable and show that attention
lifts coherence while SAM masks mainly cost concept F1.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import load_runs, bci, OUT

F.set_style()

# (label, runs-prefix); column order in run rows is [BAcc, cF1, coherence]
CONFIGS = [("GAP\n+ GT", "sccbm"),
           ("GAP\n+ SAM", "sccbm_sam"),
           ("Attention\n+ GT", "pool_attention"),
           ("Attention\n+ SAM", "pool_attention_sam")]
METRICS = [("Balanced accuracy", 0, F.COL["hero"]),
           ("Spatial coherence", 2, F.COL["good"]),
           ("Concept F1", 1, F.COL["purple"])]

x = np.arange(len(CONFIGS))
w = 0.26
fig, ax = plt.subplots(figsize=(F.DOUBLE[0] * 0.72, 3.3), constrained_layout=True)
for j, (mname, col_idx, colr) in enumerate(METRICS):
    means, los, his = [], [], []
    for _, prefix in CONFIGS:
        runs = load_runs(prefix, "ph2")
        m, lo, hi = bci(runs[:, col_idx])
        means.append(m); los.append(m - lo); his.append(hi - m)
    off = (j - 1) * w
    ax.bar(x + off, means, w, color=colr, label=mname, zorder=3,
           yerr=[los, his], capsize=2.5,
           error_kw=dict(lw=0.9, ecolor="#666666", zorder=4))
    for xi, m, hi in zip(x + off, means, his):
        ax.text(xi, m + hi + 1.4, f"{m:.0f}", ha="center", va="bottom",
                fontsize=7.0, fontweight="bold", color=colr)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=8.5)
ax.set_ylabel("Score on PH$^2$ (%)")
ax.set_ylim(0, 108)
ax.set_yticks(range(0, 101, 25))
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8,
          handletextpad=0.4, columnspacing=1.3)
F.save(fig, "fig_maskfidelity", outdir=OUT)
