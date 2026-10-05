#!/bin/bash
# E2a: SC-CBM with SAM auto-masks on PH2 (mask-fidelity ablation vs GT), 3 seeds.
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate

EP=30
PAT=8

for SEED in 42 7 123; do
  echo "########## PH2 SAM seed $SEED ##########"
  EXTRA=""
  if [ "$SEED" = "42" ]; then EXTRA="--save-ckpt"; fi
  python train.py --dataset ph2 --epochs $EP --patience $PAT --seed $SEED \
      --mask-source sam --tag sccbm_sam_s$SEED $EXTRA
done
echo "SAM PH2 DONE"
