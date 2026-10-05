"""Data pipeline for SC-CBM (Chapter 3).

Reuses GroundDerm's dataset loaders (no duplication): PH2Dataset and
Derm7ptDataset both return a dict with keys
    image, mask, label, concepts (8-dim), ...
and 5-fold split CSVs live in GroundDerm's data/splits/.

PH2 ships ground-truth lesion masks (used by the spatial-coherence regulariser);
Derm7pt has none, so its mask is a full-image placeholder and the spatial term is
inert there (coverage == 1 -> zero penalty).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --- locate GroundDerm (Chapter 4) code + the datasets on disk -------------
GROUNDERM = os.environ.get("GROUNDERM_ROOT", "../GroundDerm/main-code")
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "../datasets")
PH2_ROOT = os.path.join(DATASETS_ROOT, "PH2")
DERM7PT_ROOT = os.path.join(DATASETS_ROOT, "derm7pt")
SPLITS_DIR = os.path.join(GROUNDERM, "data", "splits")

if GROUNDERM not in sys.path:
    sys.path.insert(0, GROUNDERM)

from datasets.ph2 import PH2Dataset, load_fold_ids as ph2_fold_ids   # noqa: E402
from datasets.derm7pt import Derm7ptDataset, load_fold_ids as d7_fold_ids  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Mask source for the spatial-coherence term: "gt" (default) uses each
# dataset's own mask (PH2 GT; full-image placeholder for Derm7pt), "sam"
# loads automatic SAM masks from the shared derived/sam_masks folder
# (see masks.py). Set SCCBM_MASK_SOURCE or pass --mask-source to train.py.
MASK_SOURCE = os.environ.get("SCCBM_MASK_SOURCE", "gt")
SAM_MASKS_ROOT = os.path.join(DATASETS_ROOT, "derived", "sam_masks")


class SAMMaskDataset(torch.utils.data.Dataset):
    """Wrap a dataset so sample["mask"] comes from derived/sam_masks instead.

    Everything else (image, label, concepts, ...) is delegated to the
    underlying dataset unchanged.
    """

    def __init__(self, base, dataset: str):
        self.base = base
        self.dataset = dataset
        self.mask_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # -> [1, H, W] in [0, 1]
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]  # fresh dict per call: safe to override "mask"
        path = os.path.join(SAM_MASKS_ROOT, self.dataset,
                            f"{sample['img_id']}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SAM mask missing for {sample['img_id']}: {path} "
                f"(run masks.py first or use --mask-source gt)")
        mask_pil = Image.open(path).convert("L")
        sample["mask"] = self.mask_tf(mask_pil)
        return sample


def _apply_mask_source(ds, dataset: str):
    if MASK_SOURCE == "sam":
        return SAMMaskDataset(ds, dataset)
    return ds


def build_transforms(train: bool):
    """Image transform (ImageNet normalisation) and mask transform (to [0,1])."""
    if train:
        img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    mask_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),  # -> [1, H, W] in [0, 1]
    ])
    return img_tf, mask_tf


def ph2_split_csv(kind: str, fold: int) -> str:
    return os.path.join(SPLITS_DIR, f"PH2_{kind}_split_{fold}.csv")


def derm7pt_split_csv(kind: str, fold: int) -> str:
    return os.path.join(SPLITS_DIR, f"derm7pt_{kind}_split_{fold}.csv")


def make_ph2_dataset(image_ids, train: bool):
    img_tf, mask_tf = build_transforms(train)
    ds = PH2Dataset(
        root=PH2_ROOT, split="train", transform=img_tf,
        mask_transform=mask_tf, image_ids=image_ids, segment_images=False,
    )
    return _apply_mask_source(ds, "ph2")


def make_derm7pt_dataset(image_ids, train: bool):
    img_tf, mask_tf = build_transforms(train)
    ds = Derm7ptDataset(
        root=DERM7PT_ROOT, split="test", transform=img_tf,
        mask_transform=mask_tf, image_ids=image_ids, segment_images=False,
    )
    return _apply_mask_source(ds, "derm7pt")


def load_fold(dataset: str, fold: int):
    """Return (train_ids, test_ids) for a fold, in the form each loader expects."""
    if dataset == "ph2":
        train_ids = ph2_fold_ids(ph2_split_csv("train", fold))
        test_ids = ph2_fold_ids(ph2_split_csv("test", fold))
    elif dataset == "derm7pt":
        train_ids = d7_fold_ids(derm7pt_split_csv("train", fold))
        test_ids = d7_fold_ids(derm7pt_split_csv("test", fold))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_ids, test_ids


def make_dataset(dataset: str, image_ids, train: bool):
    if dataset == "ph2":
        return make_ph2_dataset(image_ids, train)
    if dataset == "derm7pt":
        return make_derm7pt_dataset(image_ids, train)
    raise ValueError(f"Unknown dataset: {dataset}")


def stratified_val_split(dataset, val_frac: float, seed: int = 42):
    """Split a torch Dataset's indices into (train_idx, val_idx), stratified by label."""
    rng = np.random.default_rng(seed)
    labels = np.array([int(dataset[i]["label"]) for i in range(len(dataset))])
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def collate(batch):
    """Collate the loader dicts into batched tensors we need for training."""
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    concepts = torch.stack([torch.as_tensor(np.asarray(b["concepts"]), dtype=torch.float32) for b in batch])
    img_ids = [b["img_id"] for b in batch]
    return {
        "image": images, "mask": masks, "label": labels,
        "concepts": concepts, "img_id": img_ids,
    }
