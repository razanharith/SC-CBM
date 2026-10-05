"""fig_mask_qc.pdf — curated SAM mask quality examples (2x2), grounding the
IoU and near-full-field-mask claims in Section 4.4's text with real cases.

Crops four cells straight out of the dev QC contact sheets that masks.py
already wrote (code/results/figures/sam_qc_{ph2,derm7pt}.png, a 4x4 grid of
450x450 cells: ~95px title band, then the case image with the SAM contour in
lime). Picking cells by (row, col) keeps this reproducible without rerunning
SAM; if masks.py's contact_sheet() grid or case order ever changes, update
CASES below to match.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
from figdata import RES, OUT

F.set_style()

CELL = 450
TITLE_H = 95  # crop below this to drop the "IMD003"-style dev title

# (dataset, row, col, panel letter, one-line caption)
CASES = [
    ("ph2", 0, 1, "(a)", "PH$^2$ -- clean segmentation"),
    ("ph2", 0, 0, "(b)", "PH$^2$ -- hair-artifact failure"),
    ("derm7pt", 1, 0, "(c)", "Derm7pt -- clean segmentation"),
    ("derm7pt", 0, 0, "(d)", "Derm7pt -- near-full-field mask\n(non-dermoscopic photo)"),
]


def crop_cell(dataset, row, col):
    sheet = os.path.join(RES, "figures", f"sam_qc_{dataset}.png")
    im = Image.open(sheet).convert("RGB")
    x0, y0 = col * CELL, row * CELL
    box = (x0 + 35, y0 + TITLE_H, x0 + CELL - 30, y0 + CELL - 2)
    return np.array(im.crop(box))


fig, axes = plt.subplots(2, 2, figsize=(F.DOUBLE[0], 5.6),
                         constrained_layout=True)
for ax, (dataset, row, col, letter, caption) in zip(axes.flat, CASES):
    ax.imshow(crop_cell(dataset, row, col))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{letter} {caption}", loc="left", fontsize=8.7, fontweight="bold",
                pad=4)
F.save(fig, "fig_mask_qc", outdir=OUT)
