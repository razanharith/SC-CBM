"""Cross-dataset generalization for SC-CBM (Chapter 3 / JIIM paper).

Train on the full source dataset and evaluate, with no retraining, on the full
target dataset. Both benchmarks share the same eight-concept vocabulary and the
melanoma-versus-nevus task, so the trained model transfers directly. This probes
whether spatially-coherent concept training generalizes across cohorts and
acquisition protocols, the standard cross-dataset check JIIM reviewers expect.

Usage:
    python cross_dataset.py --source derm7pt --target ph2
    python cross_dataset.py --source ph2 --target derm7pt
"""

import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import data as datamod
from model import SC_CBM
from losses import SCCBMLoss, class_weights_from_labels, concept_pos_weights
from evaluate import evaluate_model
from train import pick_device, val_bacc


def full_ids(dataset):
    """Union of a fold's train + test split = the whole dataset."""
    tr, te = datamod.load_fold(dataset, 0)
    return list(tr) + list(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["ph2", "derm7pt"])
    ap.add_argument("--target", required=True, choices=["ph2", "derm7pt"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="results")
    ap.add_argument("--mask-source", default="gt", choices=["gt", "sam"],
                    help="mask source for the spatial term (sam requires masks.py)")
    ap.add_argument("--tag", default="",
                    help="optional suffix for the output JSON name")
    args = ap.parse_args()

    datamod.MASK_SOURCE = args.mask_source

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] cross-dataset {args.source} -> {args.target} on {device}")

    src_ids = full_ids(args.source)
    tgt_ids = full_ids(args.target)
    full_train = datamod.make_dataset(args.source, src_ids, train=True)
    val_eval = datamod.make_dataset(args.source, src_ids, train=False)
    tgt_test = datamod.make_dataset(args.target, tgt_ids, train=False)

    tr_idx, va_idx = datamod.stratified_val_split(full_train, args.val_frac, args.seed)
    train_ld = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=datamod.collate)
    val_ld = DataLoader(Subset(val_eval, va_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=datamod.collate)
    test_ld = DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=datamod.collate)

    tr_labels = np.array([int(full_train[i]["label"]) for i in tr_idx])
    tr_concepts = np.stack([np.asarray(full_train[i]["concepts"]) for i in tr_idx])
    cw = class_weights_from_labels(tr_labels)
    cpw = concept_pos_weights(tr_concepts)

    model = SC_CBM(backbone=args.backbone, freeze_backbone=True).to(device)
    criterion = SCCBMLoss(class_weight=cw, concept_pos_weight=cpw,
                          lambda_c=args.lambda_c, lambda_s=args.lambda_s).to(device)
    optim = torch.optim.Adam(model.trainable_parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    best, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        model.backbone.eval()
        for batch in train_ld:
            optim.zero_grad()
            out = model(batch["image"].to(device))
            loss, _ = criterion(out, batch)
            loss.backward()
            optim.step()
        vb = val_bacc(model, val_ld, device)
        print(f"  ep{ep:02d} val_bacc={vb:.3f}")
        if vb > best:
            best, best_state, patience = vb, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    res = evaluate_model(model, test_ld, device)
    d = res["diagnosis"]
    out = {"source": args.source, "target": args.target,
           "n_source": len(src_ids), "n_target": len(tgt_ids),
           "mask_source": args.mask_source,
           "best_val_bacc": best, "result": res}
    os.makedirs(args.output, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.output, f"cross_{args.source}_to_{args.target}{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[{args.source}->{args.target}] BAcc={d['balanced_accuracy']:.3f} "
          f"Acc={d['accuracy']:.3f} AUC={d['auc']:.3f} Sens={d['sensitivity']:.3f} "
          f"Spec={d['specificity']:.3f} F1={d['f1']:.3f} "
          f"cF1={res['concepts']['concept_avg_f1']:.3f}")
    print(f"[INFO] saved {path}")


if __name__ == "__main__":
    main()
