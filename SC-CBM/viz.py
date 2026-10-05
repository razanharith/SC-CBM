"""Qualitative concept-map visualisation for SC-CBM (Chapter 3).

Loads a saved SC-CBM checkpoint, runs it on a few test images, and overlays each
concept's activation map on the image with the lesion boundary drawn on top. This
is the figure that shows the spatial-coherence regulariser pulling concept
evidence inside the lesion.

Usage:
    python viz.py --ckpt results/checkpoints/sccbm_ph2_fold0.pth --n 4
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as datamod
from model import SC_CBM
from concepts_meta import CONCEPT_CODES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="ph2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--concepts", nargs="+", default=["APN", "BWV", "RS"])
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = ckpt["args"].get("backbone", "resnet50")
    model = SC_CBM(backbone=backbone, pool=ckpt["args"].get("pool", "gap")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, test_ids = datamod.load_fold(args.dataset, args.fold)
    ds = datamod.make_dataset(args.dataset, test_ids, train=False)

    concept_idx = [CONCEPT_CODES.index(c) for c in args.concepts]

    # pick melanoma cases (label 1) first
    order = sorted(range(len(ds)), key=lambda i: -int(ds[i]["label"]))[:args.n]
    ncol = 1 + len(args.concepts)
    fig, axes = plt.subplots(args.n, ncol, figsize=(3 * ncol, 3 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for r, i in enumerate(order):
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy()[0]
        with torch.no_grad():
            out = model(img)
        cmaps = F.relu(out["concept_maps"])[0]  # [K,h,w]
        base = denorm(sample["image"])

        axes[r, 0].imshow(base)
        axes[r, 0].contour(mask, levels=[0.5], colors="white", linewidths=1.5)
        axes[r, 0].set_title(f"label={int(sample['label'])}", fontsize=9)
        axes[r, 0].axis("off")

        for c, cj in enumerate(concept_idx):
            hm = cmaps[cj].cpu().numpy()
            hm = np.array(torch.nn.functional.interpolate(
                torch.tensor(hm)[None, None], size=base.shape[:2], mode="bilinear",
                align_corners=False)[0, 0])
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            ax = axes[r, 1 + c]
            ax.imshow(base)
            ax.imshow(hm, cmap="viridis", alpha=0.5)
            ax.contour(mask, levels=[0.5], colors="white", linewidths=1.4)
            p = float(out["concept_probs"][0, cj])
            ax.set_title(f"{args.concepts[c]} (p={p:.2f})", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out, f"concept_maps_{args.dataset}_fold{args.fold}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()
