"""Automatic lesion masks with SAM (E2 experiment).

Dermoscopic images are lesion-centred, so a point prompt at the image centre
(plus a small central grid) drives SAM (ViT-B). Candidates outside a plausible
lesion-area range (2%-85% of the image) are rejected before ranking — the raw
best-by-predicted-IoU candidate is usually the dark dermoscope vignette on
these images — and the best predicted IoU among the rest is kept. If every
point-prompt candidate is degenerate, fall back to the largest centre-containing
mask from SamAutomaticMaskGenerator (on CPU; MPS cannot run it). Masks are
saved as 8-bit PNGs (255 = lesion) at 224x224 into the shared derived/ folder
so data.py can load them as a drop-in replacement for PH2's GT masks (Derm7pt
has none).

Usage:
    python masks.py                      # ph2 + derm7pt
    python masks.py --datasets ph2       # only PH2
    python masks.py --datasets derm7pt --limit 50

For PH2 the script also reports mean IoU between the SAM masks and the GT
masks so the prompt strategy can be sanity-checked (measured: 0.67 mean /
0.84 median over all 200 images).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as datamod  # noqa: E402

SAM_CKPT = os.path.join(datamod.DATASETS_ROOT, "derived", "sam",
                        "sam_vit_b_01ec64.pth")
SAM_MASKS_ROOT = os.path.join(datamod.DATASETS_ROOT, "derived", "sam_masks")
OUT_SIZE = 224
LONG_SIDE = 1024
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "figures")


# ---------------------------------------------------------------------------
# Image-id universe + original image paths
# ---------------------------------------------------------------------------
def universe_ids(dataset):
    """Union of the 5 folds' train+test image ids (deduped, stable order)."""
    ids = []
    seen = set()
    for fold in range(5):
        for kind in ("train", "test"):
            tr, te = datamod.load_fold(dataset, fold)
            for item in list(tr) + list(te):
                img_id = item[0] if isinstance(item, tuple) else item
                if img_id not in seen:
                    seen.add(img_id)
                    ids.append(img_id)
    return ids


def image_paths(dataset, ids):
    """Map img_id -> original image file path (via the GroundDerm loaders)."""
    paths = {}
    if dataset == "ph2":
        ds = datamod.PH2Dataset(root=datamod.PH2_ROOT, split="train",
                                transform=None, mask_transform=None,
                                image_ids=ids, segment_images=False)
        for img_id in ids:
            p = ds._find_image(img_id)
            if p is None:
                print(f"[WARN] no image file for {img_id}, skipping")
                continue
            paths[img_id] = p
    else:  # derm7pt
        rows = datamod.Derm7ptDataset(
            root=datamod.DERM7PT_ROOT, transform=None, mask_transform=None,
            image_ids=[(i, 0) for i in ids], segment_images=False)
        for img_id, _label, _concepts, meta_idx in rows.samples:
            p = rows._find_image(meta_idx)
            if p is None:
                print(f"[WARN] no image file for {img_id}, skipping")
                continue
            paths[img_id] = p
    return paths


# ---------------------------------------------------------------------------
# SAM helpers
# ---------------------------------------------------------------------------
def load_sam(device):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def central_point_prompts(h, w):
    """Centre point + 4 points offset by 5% of each dim (lesion-centred prior)."""
    cx, cy = w // 2, h // 2
    dx, dy = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    pts = [[cx, cy], [cx - dx, cy], [cx + dx, cy], [cx, cy - dy], [cx, cy + dy]]
    return np.array(pts, dtype=np.float32)


# Plausible lesion area range (fraction of the resized image). SAM's best-by-
# predicted-IoU candidate is often the dark dermoscope vignette (~0.94 FOV) on
# these images, and tiny blobs appear when a centre point lands on a hair, so
# candidates outside [MIN_FRAC, MAX_FRAC] are rejected before ranking (tuned on
# 40 PH2 images: mean SAM-vs-GT IoU 0.56 -> 0.69, median 0.84).
MIN_FRAC, MAX_FRAC = 0.02, 0.85


def predict_lesion(predictor, image_rgb, cpu_generator=None):
    """Return a bool mask (H, W) at the (resized) image scale."""
    h, w = image_rgb.shape[:2]
    n_pix = h * w
    predictor.set_image(image_rgb)
    pts = central_point_prompts(h, w)
    labels = np.ones(len(pts), dtype=np.int64)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=labels, multimask_output=True)

    def plausible(m):
        frac = m.sum() / n_pix
        return MIN_FRAC <= frac <= MAX_FRAC

    good = [i for i in range(len(masks)) if plausible(masks[i])]
    if good:
        return masks[max(good, key=lambda i: scores[i])]

    # All point-prompt candidates degenerate: fall back to the automatic mask
    # generator (on CPU — MPS cannot run it due to a float64 limitation) and
    # keep the largest mask that contains the image centre.
    if cpu_generator is not None:
        auto = cpu_generator.generate(image_rgb)
        cx, cy = w // 2, h // 2
        centric = [m for m in auto
                   if m["segmentation"][cy, cx]
                   and m["segmentation"].mean() <= MAX_FRAC]
        if not centric:
            centric = [m for m in auto if m["segmentation"].mean() <= MAX_FRAC]
        if centric:
            return (max(centric, key=lambda m: m["area"])["segmentation"]
                    .astype(bool))

    best = masks[int(np.argmax(scores))]
    if best.sum() == 0:
        best = np.ones((h, w), dtype=bool)
    return best


_cpu_generator = None


def get_cpu_generator():
    """Lazily build a CPU automatic mask generator (MPS can't run it)."""
    global _cpu_generator
    if _cpu_generator is None:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam_cpu = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
        sam_cpu.to("cpu")
        _cpu_generator = SamAutomaticMaskGenerator(
            sam_cpu, points_per_side=16, pred_iou_thresh=0.7,
            stability_score_thresh=0.7, min_mask_region_area=100)
    return _cpu_generator


def save_mask(mask_bool, out_path):
    img = Image.fromarray((mask_bool.astype(np.uint8)) * 255)
    img = img.resize((OUT_SIZE, OUT_SIZE), Image.NEAREST)
    img.save(out_path)


# ---------------------------------------------------------------------------
# QC contact sheet
# ---------------------------------------------------------------------------
def contact_sheet(dataset, samples, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(16, len(samples))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")
        if k >= n:
            continue
        img_id, img_path, mask = samples[k]
        im = Image.open(img_path).convert("RGB").resize((OUT_SIZE, OUT_SIZE))
        ax.imshow(im)
        ax.contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.5)
        ax.set_title(img_id, fontsize=8)
    fig.suptitle(f"SAM lesion masks — {dataset}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_dataset(dataset, predictor, args):
    ids = universe_ids(dataset)
    if args.limit:
        ids = ids[: args.limit]
    paths = image_paths(dataset, ids)
    out_dir = os.path.join(SAM_MASKS_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # PH2 GT masks (for IoU validation)
    gt_finder = None
    if dataset == "ph2":
        ds = datamod.PH2Dataset(root=datamod.PH2_ROOT, split="train",
                                transform=None, mask_transform=None,
                                image_ids=ids, segment_images=False)
        gt_finder = ds._find_mask

    frac_sum, frac_sq, n_done = 0.0, 0.0, 0
    frac_lo, frac_hi = 1.0, 0.0
    iou_sum, n_iou = 0.0, 0
    qc = []
    t0 = time.time()
    for k, img_id in enumerate(ids):
        out_path = os.path.join(out_dir, f"{img_id}.png")
        if os.path.exists(out_path) and not args.overwrite:
            mask = np.array(Image.open(out_path)) > 127
            frac = mask.mean()
        else:
            img_path = paths.get(img_id)
            if img_path is None:
                continue
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > LONG_SIDE:
                r = LONG_SIDE / max(image.size)
                image = image.resize(
                    (round(image.width * r), round(image.height * r)),
                    Image.BILINEAR)
            image_rgb = np.array(image)
            mask = predict_lesion(predictor, image_rgb, get_cpu_generator())
            save_mask(mask, out_path)
            frac = mask.mean()
            mask224 = np.array(Image.open(out_path)) > 127
            if len(qc) < 16:
                qc.append((img_id, img_path, mask224))

        frac_sum += frac
        frac_sq += frac * frac
        frac_lo, frac_hi = min(frac_lo, frac), max(frac_hi, frac)
        n_done += 1
        if gt_finder is not None:
            gt_path = gt_finder(img_id)
            if gt_path is not None:
                gt = np.array(Image.open(gt_path).convert("L").resize(
                    (OUT_SIZE, OUT_SIZE), Image.NEAREST)) > 127
                sam = np.array(Image.open(out_path)) > 127
                inter = (gt & sam).sum()
                union = (gt | sam).sum()
                iou_sum += inter / max(1, union)
                n_iou += 1
        if (k + 1) % 25 == 0 or k + 1 == len(ids):
            print(f"  [{dataset}] {k + 1}/{len(ids)} done "
                  f"({time.time() - t0:.0f}s)")

    mean = frac_sum / max(1, n_done)
    std = np.sqrt(max(0.0, frac_sq / max(1, n_done) - mean * mean))
    print(f"[{dataset}] {n_done} masks -> {out_dir}")
    print(f"[{dataset}] area fraction: mean={mean:.3f} std={std:.3f} "
          f"min={frac_lo:.3f} max={frac_hi:.3f}")
    if n_iou:
        print(f"[{dataset}] mean SAM-vs-GT IoU ({n_iou} images): "
              f"{iou_sum / n_iou:.3f}")
    if qc:
        sheet = os.path.join(FIG_DIR, f"sam_qc_{dataset}.png")
        contact_sheet(dataset, qc, sheet)
        print(f"[{dataset}] QC sheet: {sheet}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ph2", "derm7pt"],
                    choices=["ph2", "derm7pt"])
    ap.add_argument("--limit", type=int, default=0, help="debug: first N ids only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.exists(SAM_CKPT):
        sys.exit(f"SAM checkpoint not found: {SAM_CKPT}")

    device = torch.device("cpu")
    if args.device in ("auto", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device not in ("auto", "mps"):
        device = torch.device(args.device)
    print(f"[INFO] SAM vit_b on {device}")

    try:
        predictor = load_sam(device)
        # MPS can fail on some SAM ops; probe once and fall back to CPU.
        probe = np.zeros((64, 64, 3), dtype=np.uint8)
        probe[16:48, 16:48] = 128
        predict_lesion(predictor, probe)
    except RuntimeError as e:
        if device.type == "mps":
            print(f"[WARN] MPS failed ({e}); falling back to CPU")
            device = torch.device("cpu")
            predictor = load_sam(device)
        else:
            raise

    for dataset in args.datasets:
        print(f"\n=== SAM masks: {dataset} ===")
        run_dataset(dataset, predictor, args)


if __name__ == "__main__":
    main()
