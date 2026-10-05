"""figstyle.py — shared academic figure style for Q1 paper figures.

Import this from a standalone `fig_*.py` script so every figure in a paper shares
one flat, restrained, publication-grade look:

    import figstyle as F
    F.set_style()
    fig, ax = plt.subplots(figsize=F.SINGLE, constrained_layout=True)
    ax.bar(x, y, color=F.COL["hero"])
    F.save(fig, "fig_main", outdir="figures")

Design rationale (why it looks professional, not just "colorful"):
  * ONE hero color carries the eye to the method being sold; everything else is a
    neutral slate or a muted categorical. Bright, saturated primaries and heavy
    green read as "default matplotlib" / "business slide", not "journal".
  * The frame is minimal: only left+bottom spines, a faint y-grid, no top/right
    box. Ink goes to data, not chrome.
  * Value labels live ON the marks, so the reader never traces back to an axis.
  * White background, sans-serif, high-DPI raster — prints clean in a column.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette ---------------------------------------------------------------
# A restrained academic palette. Use `hero` for the method you are selling,
# `baseline` (neutral slate) for what you compare against, and the muted
# categoricals for ablation arms / classes. Reserve `good` (green) for rare
# positive accents — do not flood a figure with green.
# A muted, cohesive journal palette. The two dataset colours (hero / amber) are
# a deep blue and a muted teal rather than the old blue + saturated orange, which
# repeated across most figures and read as "default matplotlib". Low-saturation
# and mostly cool, with one warm brick-red reserved for the clinical
# malignant/negative semantic so it stands out.
COL = {
    "hero":     "#345E8B",   # PH2 / primary — deep muted blue
    "hero_dk":  "#22415F",   # darker blue: emphasis, heatmap-high
    "baseline": "#9AA3AE",   # neutral slate gray: the thing we beat
    "amber":    "#4C9B8F",   # Derm7pt / 2nd series — muted teal (was orange)
    "purple":   "#7E6BA8",   # 3rd category — muted violet
    "mel":      "#BC5A46",   # warm brick red: malignant / melanoma / negative
    "nevus":    "#6E92B6",   # soft steel blue: benign / secondary class
    "good":     "#5B9E77",   # muted sage green: positive accent (use sparingly)
    "pale":     "#CBD6DE",   # pale gray-blue fill
    "ink":      "#1a1a1a",   # text
    "muted":    "#777777",   # captions / footnotes inside a figure
    "grid":     "#E8E8E8",
}
# ordered list for N-category plots (hero first, then distinct muted hues)
CYCLE = [COL["hero"], COL["amber"], COL["purple"], COL["mel"],
         COL["nevus"], COL["baseline"]]

# ---- canonical figure sizes (inches) ---------------------------------------
# Author at the REAL print width so fonts land >=8pt at 100% scale.
SINGLE = (3.5, 3.0)      # one column (IEEE/Elsevier two-column layout)
SINGLE_WIDE = (3.5, 2.4)
DOUBLE = (7.16, 3.0)     # full text width (use \includegraphics in figure*)
PANEL3 = (7.16, 2.6)     # three panels across the full width


def set_style():
    """Flat, white, sans-serif academic rcParams. Call once at import time."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 500, "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue",
                            "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.5, "axes.unicode_minus": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 8,
        "axes.titlecolor": COL["ink"],
        "axes.labelsize": 9.5, "axes.labelcolor": "#333333",
        "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": COL["grid"], "grid.linewidth": 0.9,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 3,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def bar_labels(ax, bars, fmt="{:.0f}", inside=True, color=None, dy=None):
    """Write each bar's height on it. inside=True puts a white label near the top
    of the bar (best for tall bars); inside=False floats it just above."""
    for b in bars:
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        if inside:
            ax.text(x, h - (dy or 6), fmt.format(h), ha="center", va="top",
                    fontsize=8, fontweight="bold", color=color or "white")
        else:
            ax.text(x, h + (dy or 1), fmt.format(h), ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color or COL["ink"])


def save(fig, name, outdir="figures", also_png=True):
    """Save as PDF (vector, for LaTeX) plus a PNG preview. bbox tight."""
    os.makedirs(outdir, exist_ok=True)
    stem = name[:-4] if name.endswith((".pdf", ".png")) else name
    pdf = os.path.join(outdir, stem + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    if also_png:
        fig.savefig(os.path.join(outdir, stem + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", os.path.abspath(pdf))
    return pdf
