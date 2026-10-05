"""Prepare the per-assertion data behind fig_calibration and fig_verification.

The published JSONs aggregate too early for those two figures:
  - calibrated_sccbm_ph2.json pools per-concept F1 across concepts per fold,
    so fig_calibration needs per-concept raw-vs-calibrated F1 recomputed here;
  - verify_case_ph2.json keeps only mean coverage per correctness group,
    so fig_verification needs the per-assertion coverage lists recomputed here.

Both recomputations reuse the exact machinery (and checkpoints) of
calibrate.py / verify_case.py, so the numbers match the paper's aggregates.
Writes:
  - results/calibration_per_concept_ph2.json
  - results/verify_distributions_ph2.json

Run: python prep_fig_data.py
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import calibrate
import verify_case
from concepts_meta import CONCEPT_CODES

CKPT_DIR = os.path.join("results", "checkpoints")


def prep_calibration(tag="sccbm", dataset="ph2", n_folds=5):
    device = torch.device("cpu")
    raw, cal = [], []
    for fold in range(n_folds):
        train_ids, test_ids = calibrate.datamod.load_fold(dataset, fold)
        model = calibrate.load_model(tag, dataset, fold, device)
        val_ld = DataLoader(calibrate.internal_val_split(dataset, train_ids),
                            batch_size=32, shuffle=False,
                            collate_fn=calibrate.datamod.collate)
        test_ld = DataLoader(calibrate.datamod.make_dataset(dataset, test_ids,
                                                            train=False),
                             batch_size=32, shuffle=False,
                             collate_fn=calibrate.datamod.collate)
        v_logits, v_probs, v_gt = calibrate.concept_outputs(model, val_ld, device)
        t_logits, t_probs, t_gt = calibrate.concept_outputs(model, test_ld, device)
        thresholds = [calibrate.fit_threshold(v_probs[:, k], v_gt[:, k])[0]
                      for k in range(v_gt.shape[1])]
        for k in range(t_gt.shape[1]):
            raw.append(f1_score(t_gt[:, k], (t_probs[:, k] >= 0.5).astype(float),
                                zero_division=0))
            cal.append(f1_score(t_gt[:, k],
                                (t_probs[:, k] >= thresholds[k]).astype(float),
                                zero_division=0))
    raw = np.array(raw).reshape(n_folds, -1).mean(axis=0)
    cal = np.array(cal).reshape(n_folds, -1).mean(axis=0)
    out = {CONCEPT_CODES[k]: {"raw": float(raw[k]), "calibrated": float(cal[k])}
           for k in range(len(CONCEPT_CODES))}
    path = os.path.join("results", f"calibration_per_concept_{dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")
    for k, c in enumerate(CONCEPT_CODES):
        print(f"  {c}: raw={raw[k]*100:.1f} calibrated={cal[k]*100:.1f}")


def prep_verification(dataset="ph2", n_folds=5):
    device = torch.device("cpu")
    out = {}
    for tag in verify_case.MODEL_TAGS:
        cov_c, cov_i = [], []
        for fold in range(n_folds):
            train_ids, test_ids = verify_case.datamod.load_fold(dataset, fold)
            model = verify_case.load_model(tag, fold, device)
            test_ld = DataLoader(verify_case.datamod.make_dataset(dataset, test_ids,
                                                                  train=False),
                                 batch_size=32, shuffle=False,
                                 collate_fn=verify_case.datamod.collate)
            cov, cprob, gt = verify_case.concept_pair_outputs(model, test_ld, device)
            st, a_cov, a_ok = verify_case.pair_stats(cov, cprob, gt,
                                                     np.full(8, 0.5))
            cov_c.append(a_cov[a_ok])
            cov_i.append(a_cov[~a_ok])
        out[tag] = {"correct": np.concatenate(cov_c).round(4).tolist(),
                    "incorrect": np.concatenate(cov_i).round(4).tolist()}
        print(f"{tag}: n_correct={len(out[tag]['correct'])} "
              f"mean={np.mean(out[tag]['correct']):.3f} | "
              f"n_incorrect={len(out[tag]['incorrect'])} "
              f"mean={np.mean(out[tag]['incorrect']):.3f}")
    path = os.path.join("results", "verify_distributions_ph2.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"saved {path}")


if __name__ == "__main__":
    prep_calibration()
    prep_verification()
