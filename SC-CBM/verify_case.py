"""E5: per-case verification flag (Chapter 3, PH2).

For each case, the model "asserts" the concepts with concept_probs >= 0.5. The
case-level flag score is the mean spatial coverage (fraction of ReLU concept-map
energy inside the GT lesion, mask interpolated to the map resolution) over the
asserted concepts -- 0 if none are asserted. Ground truth for verification is
"has concept error": any asserted/GT concept mismatch. The flag predicts an
error when the score is LOW (asserted evidence not coherently inside the lesion).

The flag threshold tau is fitted on each fold's internal val split (same
construction as calibrate.py) by maximising Youden's J over tau in [0, 1],
then reported on the fold's test set: AUROC of flag score vs has-error,
sensitivity/specificity at tau, and the fraction of cases flagged.

Per-case ground truth turned out to be saturated on PH2 (any-of-8 mismatch is
true for ~97% of cases), so a second, well-posed analysis is added
("per_concept" in the JSON): the unit is an ASSERTED (case, concept) pair,
the flag score is that pair's coverage, and the label is whether the assertion
matches GT. It is reported for raw 0.5 assertions and for assertions with the
E3 calibrated per-concept thresholds (results/calibrated_sccbm_ph2.json),
pooling pairs across folds and fitting tau on the pooled internal-val pairs.

Expected finding (verified, not forced): for ablate_concept (coverage never
trained) low coverage is a strong signal of a wrong assertion (AUROC >> 0.5);
for sccbm, asserted-concept coverage is high even when wrong, so the AUROC is
much closer to 0.5 -- the gate's value is for models WITHOUT baked-in
coherence, while SC-CBM's trained coherence is what makes its evidence
auditably in-lesion. Results: results/verify_case_ph2.json.

Run:
    python verify_case.py            # per-concept + per-case
    python verify_case.py --skip-case
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import data as datamod
from concepts_meta import CONCEPT_CODES
from model import SC_CBM

CKPT_DIR = os.path.join("results", "checkpoints")
MODEL_TAGS = ["sccbm", "ablate_concept_ck"]


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


def internal_val_split(train_ids, val_frac=0.15, seed=42):
    """The trainer's internal val indices, rebuilt without augmentation."""
    train_ds_aug = datamod.make_dataset("ph2", train_ids, train=True)
    _, va_idx = datamod.stratified_val_split(train_ds_aug, val_frac, seed)
    val_eval_ds = datamod.make_dataset("ph2", train_ids, train=False)
    return Subset(val_eval_ds, va_idx)


@torch.no_grad()
def case_flags(model, loader, device):
    """Per-case (flag_score, has_error, asserted, n_asserted, coverage[K])."""
    model.eval()
    scores, errors, asserted_all, coverages = [], [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        cprob = out["concept_probs"].cpu().numpy()               # [B, K]
        gt = batch["concepts"].numpy()                           # [B, K]
        asserted = cprob >= 0.5
        has_err = (asserted != (gt > 0.5)).any(axis=1)

        cmaps = out["concept_maps"]
        energy = F.relu(cmaps)
        b, k, h, w = energy.shape
        m = F.interpolate(batch["mask"].to(device), size=(h, w), mode="nearest")
        m = (m > 0.5).float()
        cov = ((energy * m).sum(dim=(2, 3)) / (energy.sum(dim=(2, 3)) + 1e-8))
        cov = cov.cpu().numpy()                                  # [B, K]

        for i in range(b):
            n = int(asserted[i].sum())
            score = float(cov[i][asserted[i]].mean()) if n > 0 else 0.0
            scores.append(score)
            errors.append(bool(has_err[i]))
            asserted_all.append([CONCEPT_CODES[j] for j in range(k) if asserted[i, j]])
            coverages.append(cov[i].tolist())
    return {
        "flag_score": np.array(scores), "has_error": np.array(errors),
        "asserted": asserted_all, "coverage": coverages,
    }


def youden_threshold(scores, errors, grid=None):
    """tau in [0,1] maximising J = sens + spec - 1, flagging score <= tau."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    best_tau, best_j = 0.5, -2.0
    for tau in grid:
        pred_err = scores <= tau
        sens = pred_err[errors].mean() if errors.any() else np.nan
        spec = (~pred_err)[~errors].mean() if (~errors).any() else np.nan
        j = np.nanmean([sens, spec]) * 2 - 1
        if j > best_j:
            best_tau, best_j = float(tau), float(j)
    return best_tau, best_j


@torch.no_grad()
def concept_pair_outputs(model, loader, device):
    """Per (case, concept) coverage [N,K], concept_probs [N,K], gt [N,K]."""
    model.eval()
    covs, cprobs, gts = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        cmaps = out["concept_maps"]
        energy = F.relu(cmaps)
        b, k, h, w = energy.shape
        m = F.interpolate(batch["mask"].to(device), size=(h, w), mode="nearest")
        m = (m > 0.5).float()
        cov = ((energy * m).sum(dim=(2, 3)) / (energy.sum(dim=(2, 3)) + 1e-8))
        covs.append(cov.cpu().numpy())
        cprobs.append(out["concept_probs"].cpu().numpy())
        gts.append(batch["concepts"].numpy())
    return np.concatenate(covs), np.concatenate(cprobs), np.concatenate(gts)


def load_calibrated_thresholds(path="results/calibrated_sccbm_ph2.json"):
    """Per-fold F1-optimal concept thresholds fitted in calibrate.py (E3)."""
    with open(path) as f:
        calib = json.load(f)
    return {row["fold"]: np.array([row["thresholds"][c] for c in CONCEPT_CODES])
            for row in calib["folds"]}


def pair_stats(cov, cprob, gt, thr):
    """Coverage-vs-assertion-correctness stats over asserted (case, concept) pairs.

    Returns a dict; fields are None when a statistic is undefined (single class).
    """
    asserted = cprob >= thr[None, :]
    correct = (gt > 0.5) == asserted
    a_cov, a_ok = cov[asserted], correct[asserted]
    n = len(a_cov)
    n_incorrect = int((~a_ok).sum())
    if n == 0 or n_incorrect == 0 or n_incorrect == n:
        auroc = None
    else:
        auroc = float(roc_auc_score(a_ok, a_cov))
    out = {
        "n_pairs": int(n), "n_incorrect": n_incorrect,
        "frac_incorrect": float((~a_ok).mean()) if n else None,
        "auroc": auroc,
        "mean_cov_correct": float(a_cov[a_ok].mean()) if n > n_incorrect else None,
        "mean_cov_incorrect": float(a_cov[~a_ok].mean()) if n_incorrect else None,
    }
    return out, a_cov, a_ok


def eval_flag_at_tau(a_cov, a_ok, tau):
    """Sensitivity/specificity of flagging coverage <= tau as an incorrect assertion."""
    pred_bad = a_cov <= tau
    n_bad = int((~a_ok).sum())
    n_good = int(a_ok.sum())
    sens = float(pred_bad[~a_ok].mean()) if n_bad else None
    spec = float((~pred_bad)[a_ok].mean()) if n_good else None
    return {"tau": float(tau), "sensitivity": sens, "specificity": spec,
            "frac_flagged": float(pred_bad.mean()) if len(a_ok) else None}


def run_model_per_concept(tag, device, calib_thr, n_folds=5):
    """Per-concept verification flag: unit = asserted (case, concept) pair,
    score = coverage, label = assertion correct. Pooled across folds; the
    threshold tau is fitted on the pooled internal-val pairs via Youden J."""
    val_rec, test_rec = [], []
    for fold in range(n_folds):
        train_ids, test_ids = datamod.load_fold("ph2", fold)
        model = load_model(tag, fold, device)
        val_ld = DataLoader(internal_val_split(train_ids), batch_size=32,
                            shuffle=False, collate_fn=datamod.collate)
        test_ld = DataLoader(datamod.make_dataset("ph2", test_ids, train=False),
                             batch_size=32, shuffle=False, collate_fn=datamod.collate)
        for ld, rec in [(val_ld, val_rec), (test_ld, test_rec)]:
            cov, cprob, gt = concept_pair_outputs(model, ld, device)
            rec.append((cov, cprob, gt))
        print(f"[{tag}] fold{fold}: per-concept records collected")

    variants = {}
    for name, get_thr in [("raw_0.5", lambda f: np.full(8, 0.5)),
                          ("calibrated", lambda f: calib_thr[f])]:
        v_val, v_test, per_fold_auroc = [], [], []
        for fold in range(n_folds):
            thr = get_thr(fold)
            st_v, a_cov_v, a_ok_v = pair_stats(*val_rec[fold], thr)
            st_t, a_cov_t, a_ok_t = pair_stats(*test_rec[fold], thr)
            v_val.append((a_cov_v, a_ok_v))
            v_test.append((a_cov_t, a_ok_t))
            per_fold_auroc.append(st_t["auroc"])
        val_cov = np.concatenate([v[0] for v in v_val])
        val_ok = np.concatenate([v[1] for v in v_val])
        test_cov = np.concatenate([v[0] for v in v_test])
        test_ok = np.concatenate([v[1] for v in v_test])

        n_bad_val = int((~val_ok).sum())
        tau = None
        if n_bad_val > 0 and n_bad_val < len(val_ok):
            tau, j = youden_threshold(val_cov, ~val_ok)
        flag = eval_flag_at_tau(test_cov, test_ok, tau) if tau is not None else None

        n_bad = int((~test_ok).sum())
        auroc = (float(roc_auc_score(test_ok, test_cov))
                 if 0 < n_bad < len(test_ok) else None)
        variants[name] = {
            "n_pairs": int(len(test_cov)),
            "n_incorrect": n_bad,
            "frac_incorrect": float((~test_ok).mean()) if len(test_ok) else None,
            "auroc": auroc,
            "auroc_per_fold": per_fold_auroc,
            "mean_cov_correct": float(test_cov[test_ok].mean()) if len(test_ok) > n_bad else None,
            "mean_cov_incorrect": float(test_cov[~test_ok].mean()) if n_bad else None,
            "val_youden_tau": tau,
            "test_at_tau": flag,
        }
        a = variants[name]
        print(f"[{tag}|{name}] pairs={a['n_pairs']} incorrect={a['n_incorrect']} "
              f"AUROC={a['auroc'] if a['auroc'] is None else round(a['auroc'], 3)} "
              f"cov_ok={a['mean_cov_correct'] if a['mean_cov_correct'] is None else round(a['mean_cov_correct'], 3)} "
              f"cov_bad={a['mean_cov_incorrect'] if a['mean_cov_incorrect'] is None else round(a['mean_cov_incorrect'], 3)}")
    return variants


def run_model(tag, device, n_folds=5):
    fold_rows = []
    for fold in range(n_folds):
        train_ids, test_ids = datamod.load_fold("ph2", fold)
        model = load_model(tag, fold, device)
        val_ld = DataLoader(internal_val_split(train_ids), batch_size=32,
                            shuffle=False, collate_fn=datamod.collate)
        test_ld = DataLoader(datamod.make_dataset("ph2", test_ids, train=False),
                             batch_size=32, shuffle=False, collate_fn=datamod.collate)

        v = case_flags(model, val_ld, device)
        t = case_flags(model, test_ld, device)
        tau, j = youden_threshold(v["flag_score"], v["has_error"])

        pred_err = t["flag_score"] <= tau
        if len(np.unique(t["has_error"])) < 2:  # single class -- AUROC undefined
            auroc = None
        else:
            auroc = float(roc_auc_score(t["has_error"], -t["flag_score"]))
        sens = float(pred_err[t["has_error"]].mean()) if t["has_error"].any() else None
        spec = float((~pred_err)[~t["has_error"]].mean()) if (~t["has_error"]).any() else None
        frac_flagged = float(pred_err.mean())
        err_rate = float(t["has_error"].mean())
        mean_flag = float(t["flag_score"].mean())
        mean_cov = float(np.mean([np.mean(c) for c in t["coverage"]]))

        fold_rows.append({
            "fold": fold, "tau": tau, "val_youden_j": j,
            "test_auroc": auroc, "test_sensitivity": sens,
            "test_specificity": spec, "test_frac_flagged": frac_flagged,
            "test_error_rate": err_rate,
            "test_mean_flag_score": mean_flag,
            "test_mean_coverage_all_concepts": mean_cov,
            "test_n": int(len(t["has_error"])),
            "test_n_errors": int(t["has_error"].sum()),
            "test_flag_scores": t["flag_score"].round(4).tolist(),
            "test_has_error": t["has_error"].astype(int).tolist(),
        })
        print(f"[{tag}] fold{fold}: tau={tau:.2f} (val J={j:.2f}) | "
              f"AUROC={auroc if auroc is None else round(auroc, 3)} "
              f"sens={sens if sens is None else round(sens, 3)} "
              f"spec={spec if spec is None else round(spec, 3)} "
              f"flagged={frac_flagged:.2f} err_rate={err_rate:.2f} "
              f"mean_flag={mean_flag:.3f} mean_cov={mean_cov:.3f}")

    def agg(key):
        vals = [r[key] for r in fold_rows
                if r[key] is not None and np.isfinite(r[key])]
        return {"mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "per_fold": [r[key] for r in fold_rows]}

    return {
        "tag": tag,
        "aggregate": {k: agg(k) for k in
                      ["test_auroc", "test_sensitivity", "test_specificity",
                       "test_frac_flagged", "test_error_rate",
                       "test_mean_flag_score", "test_mean_coverage_all_concepts"]},
        "folds": fold_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results")
    ap.add_argument("--skip-case", action="store_true",
                    help="only run the per-concept analysis (skip per-case flags)")
    args = ap.parse_args()
    device = pick_device()
    print(f"[INFO] device={device}")

    calib_thr = load_calibrated_thresholds(
        os.path.join(args.output, "calibrated_sccbm_ph2.json"))

    out = {"dataset": "ph2", "models": {}}
    for tag in MODEL_TAGS:
        per_concept = run_model_per_concept(tag, device, calib_thr)
        out["models"][tag] = {"per_concept": per_concept}
        if not args.skip_case:
            out["models"][tag].update(run_model(tag, device))

    path = os.path.join(args.output, "verify_case_ph2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Per-concept verification flag (PH2 test, asserted pairs pooled over 5 folds) ===")
    print(f"{'model':18s} {'assertion':10s} {'pairs':>6s} {'%wrong':>7s} {'AUROC':>7s} "
          f"{'cov_ok':>7s} {'cov_bad':>7s} {'tau':>5s} {'sens':>6s} {'spec':>6s}")
    for tag in MODEL_TAGS:
        for vname, a in out["models"][tag]["per_concept"].items():
            f2 = lambda x: f"{x:.3f}" if x is not None else "  n/a"  # noqa: E731
            tau = a["val_youden_tau"]
            fl = a["test_at_tau"] or {}
            print(f"{tag:18s} {vname:10s} {a['n_pairs']:6d} "
                  f"{f2(a['frac_incorrect']):>7s} {f2(a['auroc']):>7s} "
                  f"{f2(a['mean_cov_correct']):>7s} {f2(a['mean_cov_incorrect']):>7s} "
                  f"{f2(tau):>5s} {f2(fl.get('sensitivity')):>6s} {f2(fl.get('specificity')):>6s}")

    if not args.skip_case:
        print("\n=== Per-case verification flag (PH2 test, 5-fold) ===")
        print(f"{'model':18s} {'AUROC':>14s} {'sens':>14s} {'spec':>14s} {'flagged':>8s}")
        for tag in MODEL_TAGS:
            a = out["models"][tag]["aggregate"]
            fmt = lambda m: f"{m['mean']:.3f}+/-{m['std']:.2f}" if m["mean"] is not None else "  n/a "  # noqa: E731
            print(f"{tag:18s} {fmt(a['test_auroc']):>14s} {fmt(a['test_sensitivity']):>14s} "
                  f"{fmt(a['test_specificity']):>14s} {a['test_frac_flagged']['mean']:8.3f}")
    print(f"\n[INFO] saved {path}")


if __name__ == "__main__":
    main()
