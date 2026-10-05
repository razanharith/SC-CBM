"""Evaluation for SC-CBM (Chapter 3).

Produces, per test set:
  - diagnosis metrics (BAcc / accuracy / sensitivity / specificity / F1) via
    GroundDerm's compute_metrics (shared with Chapter 4),
  - per-concept F1 + mean per-concept F1 via compute_concept_metrics,
  - a spatial-coherence metric: mean fraction of concept activation energy that
    falls inside the lesion, over ground-truth-present concepts (the quantity the
    novel regulariser optimises),
  - a concept-intervention result: diagnosis BAcc when predicted concept
    probabilities are replaced by ground-truth concepts (faithfulness upper bound,
    and the empirical hook that motivates Chapter 4's grounding).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

GROUNDERM = os.environ.get("GROUNDERM_ROOT", "../GroundDerm/main-code")
if GROUNDERM not in sys.path:
    sys.path.insert(0, GROUNDERM)
from evaluation.metrics import compute_metrics, compute_concept_metrics  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    c_true, c_pred = [], []
    coverage_sum = np.zeros(model.classifier.in_features, dtype=np.float64)
    coverage_cnt = np.zeros(model.classifier.in_features, dtype=np.float64)
    gt_concepts_all, masks_present = [], []

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        y_true.extend(batch["label"].numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

        concepts = batch["concepts"].numpy()               # [B, K] gt
        cprob = out["concept_probs"].cpu().numpy()          # [B, K]
        c_true.append(concepts)
        c_pred.append((cprob >= 0.5).astype(np.float32))
        gt_concepts_all.append(out["concept_probs"].cpu())

        # spatial coverage over gt-present concepts
        cmaps = out["concept_maps"]
        energy = F.relu(cmaps)
        b, k, h, w = energy.shape
        m = F.interpolate(batch["mask"].to(device), size=(h, w), mode="nearest")
        m = (m > 0.5).float()
        inside = (energy * m).sum(dim=(2, 3))
        total = energy.sum(dim=(2, 3)) + 1e-8
        cov = (inside / total).cpu().numpy()                # [B, K]
        present = concepts > 0.5
        mask_is_real = (batch["mask"].view(batch["mask"].shape[0], -1).mean(dim=1) < 0.999).numpy()
        for i in range(b):
            if not mask_is_real[i]:
                continue
            for j in range(k):
                if present[i, j]:
                    coverage_sum[j] += cov[i, j]
                    coverage_cnt[j] += 1

    return {
        "y_true": np.array(y_true), "y_pred": np.array(y_pred),
        "y_score": np.array(y_score),
        "c_true": np.concatenate(c_true, axis=0),
        "c_pred": np.concatenate(c_pred, axis=0),
        "coverage_sum": coverage_sum, "coverage_cnt": coverage_cnt,
    }


@torch.no_grad()
def intervention_bacc(model, loader, device):
    """Diagnosis BAcc when concept probabilities are replaced by GT concepts."""
    from sklearn.metrics import balanced_accuracy_score
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        gt = batch["concepts"].to(device).float()
        logits = model.classify_from_concepts(gt)
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(batch["label"].numpy().tolist())
    return balanced_accuracy_score(y_true, y_pred)


def evaluate_model(model, loader, device):
    p = collect_predictions(model, loader, device)
    metrics = compute_metrics(p["y_true"], p["y_pred"], p["y_score"])
    cmetrics = compute_concept_metrics(p["c_true"], p["c_pred"])
    with np.errstate(invalid="ignore", divide="ignore"):
        per_concept_cov = np.where(p["coverage_cnt"] > 0,
                                   p["coverage_sum"] / np.maximum(p["coverage_cnt"], 1), np.nan)
    valid = p["coverage_cnt"] > 0
    spatial_coherence = float(np.nanmean(per_concept_cov[valid])) if valid.any() else float("nan")
    interv = intervention_bacc(model, loader, device)
    return {
        "diagnosis": metrics,
        "concepts": cmetrics,
        "spatial_coherence": spatial_coherence,
        "per_concept_coverage": per_concept_cov.tolist(),
        "intervention_bacc": interv,
    }
