"""main_comparison.pdf — forest/CI dot plot, SC-CBM vs baselines (PH2, Derm7pt).

Shape: forest / CI plot (skill section 6). Our runs get hero-blue dots with
bootstrap-95%-CI whiskers; the Two-Step number is a published value (dot only,
annotated "published"); the standard CBM ablation is slate.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import load_runs, bci, OUT

F.set_style()

PUB = {"ph2": 85.05, "derm7pt": 79.1}
# per panel: (label, runs-prefix or None for published, color, published value)
PANELS = {
    "ph2": [("SC-CBM (attn + GT)", "pool_attention", None, F.COL["hero"], "o"),
            ("SC-CBM (attn + auto)", "pool_attention_sam", None, F.COL["hero"], "o"),
            ("Two-Step CBMs [26]", None, PUB["ph2"], F.COL["baseline"], "o"),
            ("Standard CBM", "ablate_plain", None, F.COL["baseline"], "o")],
    "derm7pt": [("SC-CBM (attn + GT)", "pool_attention", None, F.COL["hero"], "o"),
                ("SC-CBM (GAP + auto)", "sccbm_sam", None, F.COL["amber"], "o"),
                ("Two-Step CBMs [26]", None, PUB["derm7pt"], F.COL["baseline"], "o"),
                ("Standard CBM", "ablate_plain", None, F.COL["baseline"], "o")],
}

fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE[0], 2.9), constrained_layout=True)
for ax, ds, title in [(axes[0], "ph2", "(a) PH$^2$"),
                      (axes[1], "derm7pt", "(b) Derm7pt")]:
    rows = []
    for label, prefix, pub, col, mk in PANELS[ds]:
        if prefix is None:
            rows.append((label, pub, pub, pub, col, True))
        else:
            runs = load_runs(prefix, ds)
            if runs is None:
                continue
            m, lo, hi = bci(runs[:, 0])
            rows.append((label, m, lo, hi, col, False))
    ys = np.arange(len(rows))[::-1]           # best on top
    for y, (label, m, lo, hi, col, published) in zip(ys, rows):
        if not published:
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round",
                    zorder=3, alpha=0.85)
        ax.scatter([m], [y], s=52, color=col, zorder=4,
                   facecolors="white" if col == F.COL["amber"] else col,
                   edgecolors=col, linewidths=1.4)
        # clear the whisker's own right end, not just the dot, so the label
        # never sits on top of the CI line
        ax.text(hi + 0.7, y, f"{m:.1f}", va="center", ha="left", fontsize=8.5,
                fontweight="bold", color=col, zorder=5)
        if published:
            ax.text(m - 0.7, y, "published", va="center", ha="right",
                    fontsize=7, color=F.COL["muted"], style="italic", zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(70, 99)
    ax.set_xticks(range(70, 96, 5))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Balanced accuracy (%)", fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9.5)
fig.text(0.5, -0.04, "Dot = mean, whisker = 95% bootstrap CI",
         ha="center", fontsize=7.5, color=F.COL["muted"])
F.save(fig, "main_comparison", outdir=OUT)
