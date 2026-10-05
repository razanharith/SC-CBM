"""SC-CBM: Spatially-Coherent Concept Bottleneck Model (Chapter 3).

A trained, interpretable concept-based diagnosis model with NO language model.
Architecture (in the coherent-cbe-skin lineage, Patricio et al. CVPR-W 2023):

    frozen CNN backbone  ->  1x1 conv concept layer  ->  concept-logit maps
                          ->  spatial pooling         ->  8-dim concept logits
                          ->  sigmoid                 ->  concept probabilities
                          ->  interpretable linear head -> diagnosis logits

Novelty over the baseline coherent CBM: the concept-logit maps are pushed to
concentrate INSIDE the lesion by a spatial-coherence regulariser (see losses.py),
using the Chapter-2 / PH2 segmentation mask. This bakes spatial faithfulness of
concept evidence into training; Chapter 4 (GroundDerm) later verifies the same
property at inference time, training-free, with an LLM reasoner.

The linear head is interpretable: its weight matrix gives each concept's signed
contribution to each class, and swapping predicted concept probabilities for
ground-truth concepts (test-time intervention) probes faithfulness.
"""

import torch
import torch.nn as nn
import torchvision

from concepts_meta import NUM_CONCEPTS, NUM_CLASSES, class_indicator_matrix


def _make_backbone(name: str):
    """Return (feature_extractor, out_channels). Backbone outputs a spatial map."""
    if name == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=w)
        feat = nn.Sequential(*list(net.children())[:-3])  # up to layer3 -> [B,1024,14,14]
        return feat, 1024
    if name == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=w)
        feat = nn.Sequential(*list(net.children())[:-3])  # -> [B,256,14,14]
        return feat, 256
    if name == "resnet101":
        w = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet101(weights=w)
        feat = nn.Sequential(*list(net.children())[:-3])  # -> [B,1024,14,14]
        return feat, 1024
    if name == "densenet201":
        w = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        net = torchvision.models.densenet201(weights=w)
        feat = nn.Sequential(
            *list(net.features.children())[:-3],
            *list(net.features.transition3.children())[:-1],
        )
        return feat, 896
    raise ValueError(f"Unknown backbone: {name}")


class AttentionPool(nn.Module):
    """Per-concept attention pooling over the spatial concept-logit maps.

    Each concept channel is pooled with softmax-attention weights derived from
    the channel itself, an interpretable alternative to global average pooling.
    """

    def forward(self, cmaps: torch.Tensor) -> torch.Tensor:
        b, k, h, w = cmaps.shape
        flat = cmaps.view(b, k, h * w)
        attn = torch.softmax(flat, dim=-1)
        return (flat * attn).sum(dim=-1)  # [B, K]


class SC_CBM(nn.Module):
    def __init__(self, backbone: str = "resnet50", pool: str = "gap",
                 dropout: float = 0.5, freeze_backbone: bool = True):
        super().__init__()
        self.backbone, feat_ch = _make_backbone(backbone)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Concept layer: 1x1 conv producing one logit map per concept (no ReLU,
        # so pooled logits can be negative for BCE concept supervision).
        self.dropout = nn.Dropout2d(p=dropout)
        self.concept_conv = nn.Conv2d(feat_ch, NUM_CONCEPTS, kernel_size=1)

        self.pool_kind = pool
        self.attn_pool = AttentionPool() if pool == "attention" else None

        # Interpretable linear head on concept probabilities.
        self.classifier = nn.Linear(NUM_CONCEPTS, NUM_CLASSES)
        self._init_head()

    def _init_head(self):
        # Initialise so each concept already votes for its clinical class.
        ind = torch.from_numpy(class_indicator_matrix())          # [K, C]
        self.classifier.weight.data = (0.1 * ind).T.contiguous()  # [C, K]
        self.classifier.bias.data.zero_()

    def _pool(self, cmaps: torch.Tensor) -> torch.Tensor:
        if self.pool_kind == "attention":
            return self.attn_pool(cmaps)
        return cmaps.mean(dim=(2, 3))  # global average pooling -> [B, K]

    def forward(self, x: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        cmaps = self.concept_conv(self.dropout(feat))  # [B, K, h, w] logit maps
        clogits = self._pool(cmaps)                    # [B, K]
        cprobs = torch.sigmoid(clogits)                # [B, K] interpretable
        logits = self.classifier(cprobs)               # [B, num_classes]
        return {
            "logits": logits, "concept_logits": clogits,
            "concept_probs": cprobs, "concept_maps": cmaps,
        }

    def classify_from_concepts(self, cprobs: torch.Tensor) -> torch.Tensor:
        """Diagnosis logits from arbitrary concept probabilities (intervention)."""
        return self.classifier(cprobs)

    def concept_contributions(self) -> torch.Tensor:
        """Signed per-concept contribution to each class = head weight, [K, C]."""
        return self.classifier.weight.data.T.contiguous()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
