"""Train + evaluate SC-CBM with k-fold cross-validation (Chapter 3).

Examples
--------
Smoke test (1 fold, few epochs):
    python train.py --dataset ph2 --folds 0 --epochs 3 --tag smoke

Full PH2 5-fold (spatially-coherent SC-CBM):
    python train.py --dataset ph2 --epochs 40 --tag sccbm

Ablations:
    python train.py --dataset ph2 --no-spatial --tag ablate_nospatial
    python train.py --dataset ph2 --no-concept --no-spatial --tag ablate_plain
"""

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score

import data as datamod
from model import SC_CBM
from losses import SCCBMLoss, class_weights_from_labels, concept_pos_weights
from evaluate import evaluate_model


def pick_device(arg):
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def val_bacc(model, loader, device):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        yp.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())
        yt.extend(batch["label"].numpy().tolist())
    return balanced_accuracy_score(yt, yp)


def train_one_fold(args, fold, device):
    train_ids, test_ids = datamod.load_fold(args.dataset, fold)
    full_train = datamod.make_dataset(args.dataset, train_ids, train=True)
    test_ds = datamod.make_dataset(args.dataset, test_ids, train=False)

    # stratified internal val for early stopping (eval-time transforms)
    tr_idx, va_idx = datamod.stratified_val_split(full_train, args.val_frac, args.seed)
    val_eval_ds = datamod.make_dataset(args.dataset, train_ids, train=False)
    train_ld = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=datamod.collate)
    val_ld = DataLoader(Subset(val_eval_ds, va_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=datamod.collate)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=datamod.collate)

    # class + concept statistics from the training portion
    tr_labels = np.array([int(full_train[i]["label"]) for i in tr_idx])
    tr_concepts = np.stack([np.asarray(full_train[i]["concepts"]) for i in tr_idx])
    cw = class_weights_from_labels(tr_labels)
    cpw = concept_pos_weights(tr_concepts)

    model = SC_CBM(backbone=args.backbone, pool=args.pool, freeze_backbone=True).to(device)
    criterion = SCCBMLoss(
        class_weight=cw, concept_pos_weight=cpw,
        lambda_c=args.lambda_c, lambda_s=args.lambda_s,
        use_concept=not args.no_concept, use_spatial=not args.no_spatial,
    ).to(device)
    optim = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_bacc, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()  # keep frozen BN stats fixed
        running = {}
        for batch in train_ld:
            optim.zero_grad()
            out = model(batch["image"].to(device))
            loss, parts = criterion(out, batch)
            loss.backward()
            optim.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
        vb = val_bacc(model, val_ld, device)
        n = max(1, len(train_ld))
        print(f"  fold{fold} ep{epoch:02d} "
              f"loss={running.get('loss',0)/n:.3f} "
              f"cls={running.get('l_cls',0)/n:.3f} "
              f"con={running.get('l_concept',0)/n:.3f} "
              f"spa={running.get('l_spatial',0)/n:.3f} "
              f"val_bacc={vb:.3f}")
        if vb > best_bacc:
            best_bacc, best_state, patience = vb, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = evaluate_model(model, test_ld, device)
    result["best_val_bacc"] = best_bacc

    if args.save_ckpt:
        ckpt_dir = os.path.join(args.output, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                    "fold": fold},
                   os.path.join(ckpt_dir, f"{args.tag}_{args.dataset}_fold{fold}.pth"))
    return result


def aggregate(fold_results):
    keys = ["balanced_accuracy", "accuracy", "sensitivity", "specificity", "f1"]
    agg = {}
    for k in keys:
        vals = [r["diagnosis"][k] for r in fold_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "per_fold": [float(v) for v in vals]}
    cavg = [r["concepts"]["concept_avg_f1"] for r in fold_results]
    agg["concept_avg_f1"] = {"mean": float(np.mean(cavg)), "std": float(np.std(cavg)),
                             "per_fold": [float(v) for v in cavg]}
    sc = [r["spatial_coherence"] for r in fold_results if not np.isnan(r["spatial_coherence"])]
    agg["spatial_coherence"] = {"mean": float(np.mean(sc)) if sc else float("nan"),
                                "std": float(np.std(sc)) if sc else float("nan")}
    iv = [r["intervention_bacc"] for r in fold_results]
    agg["intervention_bacc"] = {"mean": float(np.mean(iv)), "std": float(np.std(iv)),
                                "per_fold": [float(v) for v in iv]}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ph2", choices=["ph2", "derm7pt"])
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet18", "resnet50", "resnet101", "densenet201"])
    ap.add_argument("--pool", default="gap", choices=["gap", "attention"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    ap.add_argument("--no-concept", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="results")
    ap.add_argument("--tag", default="sccbm")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--mask-source", default="gt", choices=["gt", "sam"],
                    help="mask source for the spatial term (sam requires masks.py)")
    args = ap.parse_args()

    datamod.MASK_SOURCE = args.mask_source  # recorded in config below

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] device={device} dataset={args.dataset} backbone={args.backbone} "
          f"concept={not args.no_concept} spatial={not args.no_spatial} "
          f"lambda_c={args.lambda_c} lambda_s={args.lambda_s} "
          f"mask_source={args.mask_source}")

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()
    fold_results = []
    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")
        res = train_one_fold(args, fold, device)
        print(f"  fold{fold} test: BAcc={res['diagnosis']['balanced_accuracy']:.3f} "
              f"Sens={res['diagnosis']['sensitivity']:.3f} "
              f"Spec={res['diagnosis']['specificity']:.3f} "
              f"cF1={res['concepts']['concept_avg_f1']:.3f} "
              f"spatial={res['spatial_coherence']:.3f} "
              f"interv={res['intervention_bacc']:.3f}")
        fold_results.append(res)

    out = {
        "config": vars(args),
        "aggregate": aggregate(fold_results) if len(fold_results) > 1 else None,
        "folds": fold_results,
        "runtime_sec": time.time() - t0,
    }
    out_path = os.path.join(args.output, f"{args.tag}_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] saved {out_path}")
    if out["aggregate"]:
        a = out["aggregate"]
        print(f"[SUMMARY] BAcc={a['balanced_accuracy']['mean']:.3f}"
              f"+/-{a['balanced_accuracy']['std']:.3f} | "
              f"cF1={a['concept_avg_f1']['mean']:.3f} | "
              f"spatial={a['spatial_coherence']['mean']:.3f} | "
              f"interv={a['intervention_bacc']['mean']:.3f}")


if __name__ == "__main__":
    main()
