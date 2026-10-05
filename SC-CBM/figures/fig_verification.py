"""fig_verification.pdf — ECDF of spatial coverage for correct vs incorrect
concept assertions (PH2). The gap between the curves is shaded; means annotated.
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

dist = json.load(open(os.path.join(RES, "verify_distributions_ph2.json")))
PANELS = [("(a) SC-CBM (spatial term)", "sccbm"),
          ("(b) Concept supervision only", "ablate_concept_ck")]

fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE[0], 2.9), sharey=True,
                         constrained_layout=True)
for ax, (title, tag) in zip(axes, PANELS):
    d = dist[tag]
    series = [(np.asarray(d["correct"]), F.COL["hero"], "Correct assertions"),
              (np.asarray(d["incorrect"]), F.COL["mel"], "Incorrect assertions")]
    xs_all = []
    for vals, col, _ in series:
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        xs_all.append(xs)
        ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]), color=col,
                lw=1.9, where="post", zorder=3)
        # rug ticks beneath the axis
        ax.plot([vals, vals], [-0.035, -0.005], color=col, lw=0.6, alpha=0.5,
                zorder=2, clip_on=False)
        m = vals.mean()
        y_m = np.interp(m, np.concatenate([[0], xs]), np.concatenate([[0], ys]))
        ax.scatter([m], [y_m], marker="D", s=30, color=col, zorder=5)
        # anchor at the diamond's own position and point the label away from
        # the other curve (up-right for correct, down-left for incorrect) so
        # the two labels diverge instead of colliding when the means are close
        mel = col == F.COL["mel"]
        dx, dy, ha = (-9, -11, "right") if mel else (9, 11, "left")
        ax.annotate(f"$\\mu$={m:.2f}  (n={len(vals)})", (m, y_m),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=8, color=col, fontweight="bold")
    # shade the gap between the two ECDFs
    xg = np.linspace(0, 1, 400)
    e0 = np.interp(xg, np.concatenate([[0], np.sort(series[0][0])]),
                   np.concatenate([[0], np.arange(1, len(series[0][0]) + 1) / len(series[0][0])]))
    e1 = np.interp(xg, np.concatenate([[0], np.sort(series[1][0])]),
                   np.concatenate([[0], np.arange(1, len(series[1][0]) + 1) / len(series[1][0])]))
    ax.fill_between(xg, e0, e1, color=F.COL["hero"], alpha=0.10, zorder=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Spatial coverage of asserted concept")
    ax.set_title(title, loc="left", fontsize=9.5)
axes[0].set_ylabel("Cumulative fraction")
F.save(fig, "fig_verification", outdir=OUT)
