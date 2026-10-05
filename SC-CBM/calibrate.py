"""E3: per-concept calibration of the SC-CBM concept layer (Chapter 3).

For each fold, the SAME internal validation split the trainer used
(datamod.stratified_val_split over the train fold, seed=42, val_frac=0.15) is
rebuilt WITHOUT augmentation, and two per-concept calibrators are fitted:

  (a) an F1-optimal decision threshold t_k in [0.05, 0.95] (grid search);
      concepts with <3 positives or <3 negatives in val fall back to 0.5;
  (b) a per-concept temperature tau_k by grid search over [0.5, 2.0]
      minimising BCE(sigmoid(clogits / tau), gt).

The fold's TEST split is then scored with raw 0.5 thresholds vs calibrated
thresholds (concept F1) and with raw vs temperature-scaled probabilities
(ECE over 10 equal-mass bins, pooled over calibrated concepts). Metrics are
pooled across all 5 folds and written to results/calibrated_{tag}_{dataset}.json.

Run:
    python calibrate.py                  # ph2 + derm7pt, sccbm checkpoints
    python calibrate.py --dataset ph2
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

import data as datamod
from concepts_meta import CONCEPT_CODES
from model import SC_CBM

CKPT_DIR = os.path.join("results", "checkpoints")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(tag, dataset, fold, device):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{tag}_{dataset}_fold{fold}.pth"),
                      map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    model = SC_CBM(backbone=cargs.get("backbone", "resnet50"),
                   pool=cargs.get("pool", "gap")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def internal_val_split(dataset, train_ids, val_frac=0.15, seed=42):
    """The trainer's internal val indices, rebuilt on the eval (un-augmented)
    dataset built from the same train ids, so val is scored without augmentation."""
    train_ds_aug = datamod.make_dataset(dataset, train_ids, train=True)
    _, va_idx = datamod.stratified_val_split(train_ds_aug, val_frac, seed)
    val_eval_ds = datamod.make_dataset(dataset, train_ids, train=False)
    return Subset(val_eval_ds, va_idx)


@torch.no_grad()
def concept_outputs(model, loader, device):
    """Return (clogits, cprobs, c_true) arrays for every sample in the loader."""
    logits, probs, gts = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out["concept_logits"].cpu())
        probs.append(out["concept_probs"].cpu())
        gts.append(batch["concepts"].float())
    return (torch.cat(logits).numpy(), torch.cat(probs).numpy(),
            torch.cat(gts).numpy())


def fit_threshold(probs, gt, grid=None):
    """F1-optimal threshold in [0.05, 0.95]; fall back to 0.5 on degenerate val."""
    pos, neg = gt > 0.5, gt <= 0.5
    if pos.sum() < 3 or neg.sum() < 3:
        return 0.5, False
    if grid is None:
        grid = np.arange(0.05, 0.9501, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(gt, (probs >= t).astype(float), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, True


def fit_temperature(clogits, gt, lo=0.5, hi=2.0, n=151):
    """Grid-search the per-concept temperature minimising concept BCE."""
    cl = torch.from_numpy(clogits)
    y = torch.from_numpy(gt)
    best_tau, best_bce = 1.0, float("inf")
    for tau in np.linspace(lo, hi, n):
        p = torch.sigmoid(cl / tau)
        bce = F.binary_cross_entropy(p, y).item()
        if bce < best_bce:
            best_tau, best_bce = float(tau), bce
    return best_tau, best_bce


def ece_equal_mass(probs, gt, n_bins=10):
    """Expected calibration error with equal-mass bins over [0, 1]."""
    order = np.argsort(probs)
    probs_s, gt_s = probs[order], gt[order]
    n = len(probs_s)
    ece, total = 0.0, 0
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        pb, gb = probs_s[lo:hi], gt_s[lo:hi]
        ece += len(pb) * abs(gb.mean() - pb.mean())
        total += len(pb)
    return float(ece / max(total, 1))


def concept_f1(cprobs, c_true, thresholds, valid):
    """Mean per-concept F1 over the concepts with fitted thresholds."""
    f1s = []
    for k in range(c_true.shape[1]):
        if not valid[k]:
            continue
        pred = (cprobs[:, k] >= thresholds[k]).astype(float)
        f1s.append(f1_score(c_true[:, k], pred, zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


def run_dataset(dataset, tag, device, n_folds=5):
    fold_params, fold_test = [], []
    cf1_raw, cf1_cal = [], []
    for fold in range(n_folds):
        train_ids, test_ids = datamod.load_fold(dataset, fold)
        model = load_model(tag, dataset, fold, device)
        val_ld = DataLoader(internal_val_split(dataset, train_ids),
                            batch_size=32, shuffle=False, collate_fn=datamod.collate)
        test_ld = DataLoader(datamod.make_dataset(dataset, test_ids, train=False),
                             batch_size=32, shuffle=False, collate_fn=datamod.collate)

        v_logits, v_probs, v_gt = concept_outputs(model, val_ld, device)
        t_logits, t_probs, t_gt = concept_outputs(model, test_ld, device)

        thresholds, temps, valid = [], [], []
        for k in range(v_gt.shape[1]):
            t_k, ok_t = fit_threshold(v_probs[:, k], v_gt[:, k])
            tau_k, _ = fit_temperature(v_logits[:, k], v_gt[:, k])
            thresholds.append(t_k)
            temps.append(tau_k)
            valid.append(ok_t)

        scaled_logits = t_logits / np.array(temps)[None, :]
        scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))

        cf1_raw.append(concept_f1(t_probs, t_gt, np.full(8, 0.5), valid))
        cf1_cal.append(concept_f1(t_probs, t_gt, np.array(thresholds), valid))

        calib_idx = np.where(valid)[0]
        flat = lambda a: a[:, calib_idx].ravel()  # noqa: E731
        fold_test.append({
            "raw_probs": flat(t_probs), "scaled_probs": flat(scaled_probs),
            "raw_gt": flat(t_gt),
        })
        fold_params.append({
            "fold": fold,
            "n_val": int(len(v_gt)),
            "n_test": int(len(t_gt)),
            "thresholds": {CONCEPT_CODES[k]: round(thresholds[k], 3) for k in range(8)},
            "temperatures": {CONCEPT_CODES[k]: round(temps[k], 3) for k in range(8)},
            "fitted": {CONCEPT_CODES[k]: bool(valid[k]) for k in range(8)},
            "n_val_pos": {CONCEPT_CODES[k]: int((v_gt[:, k] > 0.5).sum()) for k in range(8)},
        })
        print(f"[{dataset}] fold{fold}: fitted={sum(valid)}/8 concepts, "
              f"val n={len(v_gt)}, test n={len(t_gt)}")

    # pooled-across-folds metrics
    raw_probs = np.concatenate([f["raw_probs"] for f in fold_test])
    scaled_probs = np.concatenate([f["scaled_probs"] for f in fold_test])
    raw_gt = np.concatenate([f["raw_gt"] for f in fold_test])

    out = {
        "dataset": dataset, "tag": tag,
        "pooled": {
            "n_pairs": int(len(raw_gt)),
            "cf1_raw_mean": float(np.mean(cf1_raw)),
            "cf1_raw_per_fold": cf1_raw,
            "cf1_calibrated_mean": float(np.mean(cf1_cal)),
            "cf1_calibrated_per_fold": cf1_cal,
            "ece_raw": ece_equal_mass(raw_probs, raw_gt),
            "ece_temperature": ece_equal_mass(scaled_probs, raw_gt),
        },
        "folds": fold_params,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["ph2", "derm7pt", "both"])
    ap.add_argument("--tag", default="sccbm")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    device = pick_device()
    datasets = ["ph2", "derm7pt"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        res = run_dataset(dataset, args.tag, device)
        path = os.path.join(args.output, f"calibrated_{args.tag}_{dataset}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        p = res["pooled"]
        print(f"\n=== {dataset} ({args.tag}) ===")
        print(f"  cF1  raw={p['cf1_raw_mean']:.4f}  calibrated={p['cf1_calibrated_mean']:.4f}")
        print(f"  ECE  raw={p['ece_raw']:.4f}  temperature={p['ece_temperature']:.4f}")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
