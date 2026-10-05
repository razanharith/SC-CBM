"""SC-CBM main framework figure (Fig. 1).

Draws the full pipeline with REAL data thumbnails: a PH2 melanoma test image,
its lesion mask, the frozen ResNet-50 feature map, the 8 concept activation maps
(14x14, with the lesion contour overlaid) and the concept vector c, all taken
from the trained sccbm_ph2_fold0 checkpoint on this very image.

Run:  python figures/fig_framework.py
Out:  figures/out/framework.pdf (+ .png preview)
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as tF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import figstyle as F

F.set_style()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CKPT = os.path.join(HERE, "..", "results", "checkpoints", "sccbm_ph2_fold0.pth")

CONCEPT_CODES = ["TPN", "APN", "BWV", "ISTR", "RSTR", "RDG", "IDG", "RS"]
BENIGN = {"TPN", "RSTR", "RDG"}
POL = {c: ("+" if c in BENIGN else "−") for c in CONCEPT_CODES}
POL_COL = {c: (F.COL["good"] if c in BENIGN else F.COL["mel"]) for c in CONCEPT_CODES}

# warm highlight for the novel contribution (the project palette is cool-only)
AMBER = "#E2A63D"
AMBER_DK = "#B97C1E"

HEAT = LinearSegmentedColormap.from_list("sccbm", ["#F8FAFC", "#B9CFE0", F.COL["hero"], F.COL["hero_dk"]])

W, H = 7.16, 4.05  # full text width, inches


# ---------------------------------------------------------------- model pass
def load_example():
    import data as datamod
    from model import SC_CBM

    _, test_ids = datamod.load_fold("ph2", 0)
    ds = datamod.make_dataset("ph2", test_ids, train=False)
    order = sorted(range(len(ds)), key=lambda i: -int(ds[i]["label"]))
    pick = next(i for i in order if ds[i]["img_id"] == "IMD168")
    sample = ds[pick]  # melanoma case with a moderate-size lesion

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = SC_CBM(backbone=ckpt["args"].get("backbone", "resnet50"),
                   pool=ckpt["args"].get("pool", "gap"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img = sample["image"].unsqueeze(0)
    mask = sample["mask"].numpy()[0]                       # [224,224]
    with torch.no_grad():
        feat = model.backbone(img)                         # [1,D,14,14]
        out = model(img)
    cmaps = tF.relu(out["concept_maps"])[0].numpy()         # [8,14,14]
    cprobs = out["concept_probs"][0].numpy()               # [8]
    probs = torch.softmax(out["logits"], dim=-1)[0].numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.clip(sample["image"].numpy().transpose(1, 2, 0) * std + mean, 0, 1)
    fmap = feat[0].mean(0).numpy()
    fmap = (fmap - fmap.min()) / (np.ptp(fmap) + 1e-8)
    m14 = tF.adaptive_max_pool2d(torch.from_numpy(mask)[None, None].float(), 14)[0, 0].numpy()
    return im, mask, fmap, m14, cmaps, cprobs, probs, int(sample["label"])


# ---------------------------------------------------------------- drawing
def rbox(ax, x, y, w, h, fc, ec, lw=1.0, dashed=False, r=0.05, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw,
                       linestyle=(0, (4, 3)) if dashed else "solid", zorder=z)
    ax.add_patch(p)
    return p


def arr(ax, p0, p1, color, lw=1.2, dashed=False, rad=0.0, z=4):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                        lw=lw, color=color,
                        linestyle=(0, (4, 3)) if dashed else "solid",
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(a)


def inset(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def main():
    im, mask, fmap, m14, cmaps, cprobs, probs, label = load_example()

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=0))

    ink, muted, pale = F.COL["ink"], F.COL["muted"], F.COL["pale"]
    FLOW = 2.75          # vertical centre of the main pipeline
    box_ec = F.COL["hero_dk"]

    # ---- main pipeline -----------------------------------------------------
    # input image
    a = inset(fig, 0.10, 2.30, 0.85, 0.85)
    a.imshow(im)
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True)
        s.set_color("#B0B0B0")
        s.set_linewidth(0.8)
    ax.text(0.525, 3.34, "Dermoscopic image $x$", ha="center", fontsize=8.5,
            fontweight="bold", color=ink)
    ax.text(0.525, 3.235, "(224$\\times$224 RGB)", ha="center", fontsize=7, color=muted)

    # SAM box + mask
    rbox(ax, 0.97, 3.44, 0.72, 0.44, "#EAF3EC", F.COL["good"], lw=0.9)
    ax.text(1.33, 3.755, "SAM masks (ViT-B)", ha="center", fontsize=7.5,
            fontweight="bold", color=ink)
    ax.text(1.33, 3.615, "no manual annotation", ha="center", fontsize=6.8, color=muted)
    a = inset(fig, 0.99, 2.42, 0.66, 0.66)
    a.imshow(mask, cmap="gray", vmin=0, vmax=1)
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True)
        s.set_color("#B0B0B0")
        s.set_linewidth(0.8)
    arr(ax, (1.33, 3.44), (1.33, 3.10), muted, lw=0.9)
    ax.text(1.32, 2.335, "Lesion mask $M$", ha="center", fontsize=8, fontweight="bold", color=ink)
    ax.text(1.32, 2.225, "(SAM or PH$^2$ GT)", ha="center", fontsize=6.5, color=muted)

    # frozen backbone
    rbox(ax, 1.77, 2.05, 0.54, 1.40, F.COL["hero"], box_ec)
    ax.text(2.04, FLOW, "Frozen\nImageNet\nbackbone\n(ResNet-50)", ha="center",
            va="center", fontsize=7.6, fontweight="bold", color="white", linespacing=1.35)

    # feature map thumbnail
    a = inset(fig, 2.42, 2.53, 0.44, 0.44)
    a.imshow(fmap, cmap=HEAT, vmin=0, vmax=1)
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True)
        s.set_color("#B0B0B0")
        s.set_linewidth(0.8)
    ax.text(2.64, 2.40, "$F\\in\\mathbb{R}^{D\\times14\\times14}$",
            ha="center", fontsize=7, color=ink)

    # 1x1-conv concept layer
    rbox(ax, 3.00, 2.05, 0.50, 1.40, F.COL["hero"], box_ec)
    ax.text(3.25, FLOW, "1$\\times$1-conv\nconcept\nlayer\n(8 ch.)", ha="center",
            va="center", fontsize=8, fontweight="bold", color="white", linespacing=1.35)

    # concept activation maps (2 cols x 4 rows, reading order = order of c)
    ax.text(4.06, 3.53, "Concept maps $A\\in\\mathbb{R}^{8\\times14\\times14}$\n(reading order = $c$, top$\\to$bottom)",
            ha="center", fontsize=8, fontweight="bold", color=ink, linespacing=1.25)
    for k in range(8):
        row, col = divmod(k, 2)
        cx = 3.66 + col * 0.40
        cy = 3.45 - (row + 1) * 0.30 - row * 0.09
        m = cmaps[k]
        m = (m / (m.max() + 1e-8)) ** 0.6
        a = inset(fig, cx, cy, 0.30, 0.30)
        a.imshow(m, cmap=HEAT, vmin=0, vmax=1, interpolation="bilinear")
        a.contour(m14, levels=[0.5], colors=[F.COL["mel"]], linewidths=0.9)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color("#B0B0B0")
            s.set_linewidth(0.5)

    # pooling
    rbox(ax, 4.60, 2.15, 0.52, 1.20, "#E9EEF4", box_ec, lw=0.9)
    ax.text(4.86, FLOW, "Spatial\npooling $+$\nsigmoid\n(GAP / attn)", ha="center",
            va="center", fontsize=7.5, color=ink, linespacing=1.3)

    # concept vector (real probabilities)
    rbox(ax, 5.26, 1.98, 0.86, 1.50, "#F7F9FB", "#C9D2DC", lw=0.9)
    ax.text(5.69, 3.60, "$c=\\sigma(\\mathrm{pool}(A))\\in[0,1]^8$", ha="center",
            fontsize=8, fontweight="bold", color=ink)
    for k, code in enumerate(CONCEPT_CODES):
        ytop = 3.36 - k * 0.172
        v = float(cprobs[k])
        ax.text(5.50, ytop - 0.055, f"{code} {POL[code]}", ha="right", va="center",
                fontsize=7, color=POL_COL[code], fontweight="bold")
        ax.add_patch(Rectangle((5.55, ytop - 0.115), 0.40 * v, 0.115,
                               fc=POL_COL[code], ec="none", zorder=3))
        ax.add_patch(Rectangle((5.55, ytop - 0.115), 0.40, 0.115,
                               fc="none", ec="#C9D2DC", lw=0.6, zorder=3))
        ax.text(5.99, ytop - 0.055, f"{v:.2f}", ha="left", va="center",
                fontsize=6.2, color=muted)

    # interpretable linear head
    rbox(ax, 6.28, 2.15, 0.32, 1.20, F.COL["hero"], box_ec)
    ax.text(6.44, FLOW, "Interpretable linear head $W$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", rotation=90)

    # diagnosis chips (real softmax probs)
    pred = int(probs.argmax())
    chips = [(2.84, 3.20, "Nevus", probs[0], "#E9EEF4", F.COL["nevus"], ink),
             (2.30, 2.66, "Melanoma", probs[1], F.COL["mel"], F.COL["mel"], "white")]
    for y0, y1, name, p, fc, ec, tc in chips:
        rbox(ax, 6.72, y0, 0.42, y1 - y0, fc, ec, lw=1.0)
        ax.text(6.93, (y0 + y1) / 2 + 0.045, name, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=tc)
        ax.text(6.93, (y0 + y1) / 2 - 0.055, f"$p$={p:.2f}", ha="center", va="center",
                fontsize=6.5, color=tc)
    ax.text(6.93, 2.10, "Diagnosis $\\hat{y}$", ha="center", fontsize=8,
            fontweight="bold", color=ink)

    # forward arrows + flow labels
    for x0, x1 in [(0.97, 1.75), (2.33, 2.40), (2.88, 2.98), (3.52, 3.64),
                   (4.50, 4.58), (5.14, 5.24), (6.14, 6.26), (6.62, 6.70)]:
        arr(ax, (x0, FLOW), (x1, FLOW), ink, lw=1.3)
    for x, t in [(2.36, "$F$"), (3.58, "$A$"), (5.20, "$c$"), (6.68, "$\\hat{y}$")]:
        if t:
            ax.text(x, FLOW + 0.09, t, ha="center", fontsize=7.5, color=muted)

    # ---- training signals --------------------------------------------------
    rbox(ax, 1.79, 0.72, 1.26, 0.62, "#FBF6EC", "#B8B8B8", lw=0.9, dashed=True)
    ax.text(2.42, 1.19, "Concept labels", ha="center", fontsize=8, fontweight="bold", color=ink)
    ax.text(2.42, 1.00, "$c^*\\in\\{0,1\\}^8$\n(7-point checklist)", ha="center",
            fontsize=7, color=ink, linespacing=1.25)

    rbox(ax, 3.35, 0.78, 1.00, 0.50, "#E9EEF4", "#9AA3AE", lw=0.9)
    ax.text(3.85, 1.105, "$\\mathcal{L}_{con}$: weighted BCE", ha="center",
            fontsize=7.8, fontweight="bold", color=ink)
    ax.text(3.85, 0.935, "(per-concept $w_k$)", ha="center", fontsize=6.8, color=muted)

    arr(ax, (3.05, 1.03), (3.33, 1.03), muted, lw=1.0, dashed=True)
    arr(ax, (3.55, 1.28), (3.33, 2.03), muted, lw=1.0, dashed=True)
    ax.text(3.60, 1.66, "supervises\nconcepts", ha="left", fontsize=6.8, color=muted,
            linespacing=1.2)

    # spatial-coherence chip (the contribution)
    rbox(ax, 4.75, 0.72, 1.80, 0.62, AMBER, AMBER_DK, lw=1.6)
    ax.text(5.65, 1.19, "$\\mathcal{L}_{spa}$ — spatial coherence (novel)",
            ha="center", fontsize=7.8, fontweight="bold", color=ink)
    ax.text(5.65, 0.93,
            "$1-\\dfrac{\\sum\\,\\mathrm{ReLU}(A_k)\\odot M}"
            "{\\sum\\,\\mathrm{ReLU}(A_k)}$",
            ha="center", fontsize=8.2, color=ink)

    arr(ax, (1.32, 2.40), (1.32, 1.02), muted, lw=1.0, dashed=True)
    arr(ax, (1.32, 1.02), (4.73, 1.02), muted, lw=1.0, dashed=True)
    ax.text(1.235, 1.71, "$M$ downsampled to 14$\\times$14", ha="center", va="center",
            fontsize=6.8, color=muted, rotation=90)

    arr(ax, (5.50, 1.34), (4.30, 1.99), AMBER_DK, lw=1.3, dashed=True, rad=0.12)
    ax.text(5.44, 1.66, "pulls evidence inside lesion\n(present $c_k^*$ only)",
            ha="left", fontsize=6.8, color=AMBER_DK, linespacing=1.2)

    ax.text(6.90, 1.17, "$\\mathcal{L}=\\mathcal{L}_{cls}$", ha="center",
            fontsize=7.0, color=ink)
    ax.text(6.90, 1.03, "$+\\,\\lambda_c\\,\\mathcal{L}_{con}$", ha="center",
            fontsize=7.0, color=ink)
    ax.text(6.90, 0.89, "$+\\,\\lambda_s\\,\\mathcal{L}_{spa}$", ha="center",
            fontsize=7.0, color=ink)

    # ---- post-hoc strip ----------------------------------------------------
    rbox(ax, 0.08, 0.12, 7.00, 0.40, "#F2F5F8", "#C9D2DC", lw=0.8, r=0.06)
    ax.text(0.24, 0.32, "Post-hoc,\nno retraining", ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=ink, linespacing=1.25)
    rbox(ax, 1.55, 0.20, 2.85, 0.24, "white", "#C9D2DC", lw=0.8)
    ax.text(2.98, 0.365, "Per-concept calibration: F1 thresholds $+$ temperature",
            ha="center", fontsize=7, color=ink)
    ax.text(2.98, 0.27, "(cF1 41.6$\\%\\to$54.0$\\%$ on PH$^2$)",
            ha="center", fontsize=6.5, color=muted)
    rbox(ax, 4.55, 0.20, 2.40, 0.24, "white", "#C9D2DC", lw=0.8)
    ax.text(5.75, 0.365, "Per-case verification flag (in-lesion coverage)",
            ha="center", fontsize=7, color=ink)
    ax.text(5.75, 0.27, "(AUROC 0.85 vs 0.67 without $\\mathcal{L}_{spa}$)",
            ha="center", fontsize=6.5, color=muted)

    # ---- legend ------------------------------------------------------------
    ly = 3.975
    arr(ax, (1.30, ly), (1.55, ly), ink, lw=1.3)
    ax.text(1.60, ly, "forward pass", fontsize=7.2, va="center", color=ink)
    arr(ax, (2.55, ly), (2.80, ly), muted, lw=1.1, dashed=True)
    ax.text(2.85, ly, "training signal", fontsize=7.2, va="center", color=ink)
    arr(ax, (3.95, ly), (4.20, ly), AMBER_DK, lw=1.2, dashed=True)
    ax.text(4.25, ly, "spatial coherence (novel)", fontsize=7.2, va="center", color=ink)
    ax.add_patch(Rectangle((5.75, ly - 0.05), 0.14, 0.10, fc=F.COL["good"], ec="none"))
    ax.text(5.92, ly, "benign concept", fontsize=7.2, va="center", color=ink)
    ax.add_patch(Rectangle((6.75, ly - 0.05), 0.14, 0.10, fc=F.COL["mel"], ec="none"))
    ax.text(6.92, ly, "malignant concept", fontsize=7.2, va="center", color=ink)

    F.save(fig, "framework", outdir=OUT)
    print(f"example label={label} (1=melanoma), concept probs="
          + ",".join(f"{c}={p:.2f}" for c, p in zip(CONCEPT_CODES, cprobs)))


if __name__ == "__main__":
    main()
