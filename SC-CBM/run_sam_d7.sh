#!/bin/bash
# E2b: SC-CBM with SAM auto-masks on Derm7pt (spatial term now active), 3 seeds,
# plus SAM-mask cross-dataset runs. Run AFTER the multi-seed job finishes.
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate

EP=30
PAT=8

for SEED in 42 7 123; do
  echo "########## Derm7pt SAM seed $SEED ##########"
  EXTRA=""
  if [ "$SEED" = "42" ]; then EXTRA="--save-ckpt"; fi
  python train.py --dataset derm7pt --epochs $EP --patience $PAT --seed $SEED \
      --mask-source sam --tag sccbm_sam_s$SEED $EXTRA
done

echo "########## cross-dataset with SAM masks ##########"
python cross_dataset.py --source derm7pt --target ph2     --mask-source sam --tag sam
python cross_dataset.py --source ph2 --target derm7pt     --mask-source sam --tag sam
echo "SAM D7 DONE"
