"""Loss terms for SC-CBM (Chapter 3).

Total objective:

    L = L_cls  +  lambda_c * L_concept  +  lambda_s * L_spatial

  L_cls     : class-weighted cross-entropy on the diagnosis (handles the ~4:1
              nevus:melanoma imbalance).
  L_concept : per-concept BCE-with-logits on the 8 dermoscopic concepts, with a
              per-concept positive weight so rare concepts are not ignored.
  L_spatial : the novelty. For each ground-truth-present concept, the fraction of
              its (non-negative) activation energy that falls OUTSIDE the lesion
              mask is penalised, so concept evidence is spatially faithful to the
              lesion. Inert when no real mask is available (coverage == 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_labels(labels, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    labels = torch.as_tensor(labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def concept_pos_weights(concepts_train, clamp: float = 20.0) -> torch.Tensor:
    """Per-concept BCE pos_weight = (#neg / #pos). 0 for degenerate concepts."""
    c = torch.as_tensor(concepts_train, dtype=torch.float32)  # [N, K]
    pos = c.sum(dim=0)
    neg = c.shape[0] - pos
    pw = torch.where(pos > 0, (neg / pos.clamp(min=1.0)).clamp(max=clamp),
                     torch.zeros_like(pos))
    return pw


class SCCBMLoss(nn.Module):
    def __init__(self, class_weight=None, concept_pos_weight=None,
                 lambda_c: float = 1.0, lambda_s: float = 0.5,
                 use_concept: bool = True, use_spatial: bool = True):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.use_concept = use_concept
        self.use_spatial = use_spatial
        self.register_buffer("class_weight",
                             class_weight if class_weight is not None else None)
        # per-concept positive weight (may contain zeros for degenerate concepts)
        self.register_buffer("concept_pos_weight",
                             concept_pos_weight if concept_pos_weight is not None else None)

    def forward(self, out, batch):
        logits = out["logits"]
        clogits = out["concept_logits"]
        cmaps = out["concept_maps"]
        labels = batch["label"].to(logits.device)
        concepts = batch["concepts"].to(logits.device)     # [B, K] in {0,1}
        masks = batch["mask"].to(logits.device)             # [B, 1, H, W] in [0,1]

        # --- classification ---
        cw = self.class_weight.to(logits.device) if self.class_weight is not None else None
        l_cls = F.cross_entropy(logits, labels, weight=cw)

        # --- concept supervision (BCE with logits, per-concept pos_weight) ---
        l_concept = logits.new_zeros(())
        if self.use_concept:
            pw = self.concept_pos_weight
            if pw is not None:
                pw = pw.to(logits.device)
                bce = F.binary_cross_entropy_with_logits(
                    clogits, concepts, pos_weight=pw, reduction="none")
                # drop degenerate concepts (pos_weight == 0) from the mean
                keep = (pw > 0).float().unsqueeze(0)
                denom = keep.sum().clamp(min=1.0) * clogits.shape[0]
                l_concept = (bce * keep).sum() / denom
            else:
                l_concept = F.binary_cross_entropy_with_logits(clogits, concepts)

        # --- spatial-coherence regulariser (the novelty) ---
        l_spatial = logits.new_zeros(())
        if self.use_spatial:
            energy = F.relu(cmaps)                                   # [B, K, h, w]
            b, k, h, w = energy.shape
            m = F.interpolate(masks, size=(h, w), mode="nearest")   # [B, 1, h, w]
            m = (m > 0.5).float()
            inside = (energy * m).sum(dim=(2, 3))                   # [B, K]
            total = energy.sum(dim=(2, 3)) + 1e-8                   # [B, K]
            coverage = inside / total                               # [B, K]
            # penalise out-of-lesion energy only for concepts truly present
            weight = concepts                                       # [B, K]
            penalty = (1.0 - coverage) * weight                    # [B, K]
            denom = weight.sum().clamp(min=1.0)
            l_spatial = penalty.sum() / denom

        total_loss = l_cls + self.lambda_c * l_concept + self.lambda_s * l_spatial
        return total_loss, {
            "loss": float(total_loss.detach()),
            "l_cls": float(l_cls.detach()),
            "l_concept": float(l_concept.detach()) if self.use_concept else 0.0,
            "l_spatial": float(l_spatial.detach()) if self.use_spatial else 0.0,
        }
