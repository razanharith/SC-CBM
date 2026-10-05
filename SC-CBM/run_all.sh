#!/bin/bash
# Full experiment sweep for SC-CBM (Chapter 3).
# Ablation ladder isolates the two training signals:
#   plain    : diagnosis CE only         (no concept sup, no spatial)
#   concept  : + concept BCE supervision (no spatial)
#   sccbm    : + spatial-coherence regulariser  (the full model)
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate

EP=30
PAT=8

echo "########## PH2 ##########"
python train.py --dataset ph2 --epochs $EP --patience $PAT --no-concept --no-spatial --tag ablate_plain
python train.py --dataset ph2 --epochs $EP --patience $PAT --no-spatial              --tag ablate_concept
python train.py --dataset ph2 --epochs $EP --patience $PAT                            --tag sccbm --save-ckpt

echo "########## Derm7pt (no GT masks -> spatial term inert) ##########"
python train.py --dataset derm7pt --epochs $EP --patience $PAT --no-concept --no-spatial --tag ablate_plain
python train.py --dataset derm7pt --epochs $EP --patience $PAT                            --tag sccbm --save-ckpt

echo "ALL DONE"
