#!/bin/bash
# Multi-seed reruns of the core ablation ladder + full model (E1).
# Produces {tag}_s{seed}_{dataset}.json; aggregate with multiseed_agg.py.
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate

EP=30
PAT=8

for SEED in 42 7 123; do
  echo "########## seed $SEED ##########"
  python train.py --dataset ph2    --epochs $EP --patience $PAT --seed $SEED --no-concept --no-spatial --tag ablate_plain_s$SEED
  python train.py --dataset ph2    --epochs $EP --patience $PAT --seed $SEED --no-spatial              --tag ablate_concept_s$SEED
  python train.py --dataset ph2    --epochs $EP --patience $PAT --seed $SEED                            --tag sccbm_s$SEED --save-ckpt
  python train.py --dataset derm7pt --epochs $EP --patience $PAT --seed $SEED --no-concept --no-spatial --tag ablate_plain_s$SEED
  python train.py --dataset derm7pt --epochs $EP --patience $PAT --seed $SEED                            --tag sccbm_s$SEED --save-ckpt
done
echo "MULTISEED DONE"
