"""E4: de-confounded robustness of the spatial regulariser (Chapter 3, PH2).

Two test-set corruptions are built as on-the-fly dataset wrappers, applied AFTER
the eval transforms, on the normalised tensors, using each sample's GT mask and
only touching pixels OUTSIDE the lesion:

  (a) "bg-swap": pixels outside the lesion are replaced by the outside-lesion
      pixels of a random OTHER image from the same fold's test set
      (composite via the sample's own mask, using both images' normalised
      tensors, so no denormalisation is needed);
  (b) "distractor": synthetic artefacts drawn only outside the lesion --
      3-7 dark quadratic-Bezier hair strokes (thickness 1-3 px, intensity
      ~ N(0.05, 0.02) in [0,1] image space) plus a thin black ruler band with
      white-ish tick marks along a random image edge.

Three checkpoint sets (sccbm, ablate_concept_ck, ablate_plain_ck) are evaluated
on all 5 PH2 folds under {clean, bg-swap, distractor}: diagnosis BAcc and
concept cF1 (cF1 is only meaningful for the two concept-supervised models).
Results go to results/deconfound_ph2.json.

Run:
    python deconfound.py
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

import data as datamod
from concepts_meta import CONCEPT_CODES
from model import SC_CBM

CKPT_DIR = os.path.join("results", "checkpoints")
CORRUPTIONS = ["clean", "bg-swap", "distractor"]
MODEL_TAGS = ["sccbm", "ablate_concept_ck", "ablate_plain_ck"]
CONCEPT_SUPERVISED = {"sccbm", "ablate_concept_ck"}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(tag, fold, device):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{tag}_ph2_fold{fold}.pth"),
                      map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    model = SC_CBM(backbone=cargs.get("backbone", "resnet50"),
                   pool=cargs.get("pool", "gap")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# corruption wrappers
# ---------------------------------------------------------------------------

class BGSwapDataset(Dataset):
    """Outside-lesion pixels replaced by those of a random other test image."""

    def __init__(self, base, seed=42):
        self.base = base
        n = len(base)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        self.partner = perm if (perm != np.arange(n)).any() else (perm + 1) % n
        self.partner = [int(p) if p != i else int((p + 1) % n)
                        for i, p in enumerate(self.partner)]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        o = self.base[self.partner[idx]]
        img, m = s["image"].clone(), s["mask"].clone()
        outside = (m < 0.5).float()                       # [1, H, W]
        img = img * (1 - outside) + o["image"] * outside
        return {**s, "image": img}


def _bezier_stroke(canvas, rng):
    """Draw one dark quadratic-Bezier hair stroke onto a [H, W] canvas in [0,1]."""
    h, w = canvas.shape
    p0 = np.array([rng.uniform(0, w), rng.uniform(0, h)])
    p2 = np.array([rng.uniform(0, w), rng.uniform(0, h)])
    p1 = (p0 + p2) / 2 + np.array([rng.uniform(-w, w), rng.uniform(-h, h)]) * 0.4
    t = np.linspace(0, 1, 64)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    thick = int(rng.integers(1, 4))                      # 1-3 px
    intensity = float(np.clip(rng.normal(0.05, 0.02), 0.0, 1.0))
    x0, x1 = int(max(0, pts[:, 0].min() - thick - 1)), int(min(w, pts[:, 0].max() + thick + 2))
    y0, y1 = int(max(0, pts[:, 1].min() - thick - 1)), int(min(h, pts[:, 1].max() + thick + 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = np.full((y1 - y0, x1 - x0), np.inf)
    for p in pts[::2]:                                    # subsample is plenty
        d2 = np.minimum(d2, (xx - p[0]) ** 2 + (yy - p[1]) ** 2)
    canvas[y0:y1, x0:x1] = np.where(d2 <= thick ** 2, intensity, canvas[y0:y1, x0:x1])


def _ruler_band(canvas, rng):
    """Thin black ruler band with white-ish tick marks along a random edge."""
    h, w = canvas.shape
    band = int(max(6, min(h, w) * 0.04))
    edge = int(rng.integers(0, 4))
    if edge == 0:   # top
        canvas[:band, :] = 0.0
        for x in range(0, w, 8):
            canvas[:band, x:x + 1] = 0.9
    elif edge == 1:  # bottom
        canvas[-band:, :] = 0.0
        for x in range(0, w, 8):
            canvas[-band:, x:x + 1] = 0.9
    elif edge == 2:  # left
        canvas[:, :band] = 0.0
        for y in range(0, h, 8):
            canvas[y:y + 1, :band] = 0.9
    else:            # right
        canvas[:, -band:] = 0.0
        for y in range(0, h, 8):
            canvas[y:y + 1, -band:] = 0.9


class DistractorDataset(Dataset):
    """Synthetic hair strokes + ruler band drawn only OUTSIDE the lesion.

    Artwork is done in [0,1] image space (unnormalise, draw, renormalise) so
    that "black" and "white-ish" have their usual pixel meanings; only pixels
    where the GT mask is background are modified.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __init__(self, base, seed=42):
        self.base = base
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        img = s["image"].numpy()
        outside = (s["mask"].numpy() < 0.5).all(axis=0)   # [H, W] bool
        img01 = np.clip(img * self.STD + self.MEAN, 0.0, 1.0)

        canvas = img01[0].copy()                          # 2-D artwork plane
        for _ in range(int(self.rng.integers(3, 8))):
            _bezier_stroke(canvas, self.rng)
        _ruler_band(canvas, self.rng)

        corrupted = np.where(outside[None, :, :], canvas[None, :, :], img01)
        new = (corrupted - self.MEAN) / self.STD
        return {**s, "image": torch.from_numpy(new.astype(np.float32))}


def build_eval_set(test_ids, corruption, fold):
    ds = datamod.make_dataset("ph2", test_ids, train=False)
    if corruption == "bg-swap":
        ds = BGSwapDataset(ds, seed=42 + fold)
    elif corruption == "distractor":
        ds = DistractorDataset(ds, seed=42 + fold)
    return ds


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, with_concepts):
    model.eval()
    y_true, y_pred, c_true, c_pred = [], [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        y_true.extend(batch["label"].numpy().tolist())
        y_pred.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())
        if with_concepts:
            c_true.append(batch["concepts"].numpy())
            c_pred.append((out["concept_probs"].cpu().numpy() >= 0.5).astype(np.float32))
    res = {"bacc": float(balanced_accuracy_score(y_true, y_pred))}
    if with_concepts:
        ct, cp = np.concatenate(c_true), np.concatenate(c_pred)
        res["cf1"] = float(np.mean([f1_score(ct[:, k], cp[:, k], zero_division=0)
                                    for k in range(ct.shape[1])]))
    else:
        res["cf1"] = None
    return res


def run(device, n_folds=5):
    results = {t: {c: {"bacc": [], "cf1": []} for c in CORRUPTIONS} for t in MODEL_TAGS}
    for fold in range(n_folds):
        _, test_ids = datamod.load_fold("ph2", fold)
        for tag in MODEL_TAGS:
            model = load_model(tag, fold, device)
            for corr in CORRUPTIONS:
                ld = DataLoader(build_eval_set(test_ids, corr, fold), batch_size=32,
                                shuffle=False, collate_fn=datamod.collate)
                r = evaluate(model, ld, device, with_concepts=tag in CONCEPT_SUPERVISED)
                results[tag][corr]["bacc"].append(r["bacc"])
                results[tag][corr]["cf1"].append(r["cf1"])
                cf1_str = f"{r['cf1']:.3f}" if r["cf1"] is not None else "  n/a "
                print(f"fold{fold} {tag:18s} {corr:10s} BAcc={r['bacc']:.3f} cF1={cf1_str}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    device = pick_device()
    print(f"[INFO] device={device}")
    results = run(device)

    out = {"dataset": "ph2", "folds": 5,
           "models": {t: {c: {"bacc_mean": float(np.mean(v["bacc"])),
                              "bacc_std": float(np.std(v["bacc"])),
                              "bacc_per_fold": v["bacc"],
                              "cf1_mean": (float(np.mean(v["cf1"]))
                                           if v["cf1"][0] is not None else None)}
                          for c, v in results[t].items()}
                      for t in MODEL_TAGS}}
    path = os.path.join(args.output, "deconfound_ph2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== De-confounded robustness (PH2, 5-fold mean +/- std) ===")
    hdr = f"{'model':18s} {'corruption':10s} {'BAcc':>15s} {'dBAcc':>7s} {'cF1':>7s}"
    print(hdr)
    for tag in MODEL_TAGS:
        clean_bacc = np.mean(results[tag]["clean"]["bacc"])
        for corr in CORRUPTIONS:
            b = results[tag][corr]["bacc"]
            c = results[tag][corr]["cf1"]
            cf1_str = f"{np.mean(c):.3f}" if c[0] is not None else "  n/a "
            d = np.mean(b) - clean_bacc if corr != "clean" else 0.0
            print(f"{tag:18s} {corr:10s} {np.mean(b):6.3f}+/-{np.std(b):<5.2f} "
                  f"{d:+7.3f} {cf1_str:>7s}")
    print(f"\n[INFO] saved {path}")


if __name__ == "__main__":
    main()
