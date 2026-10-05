#!/bin/bash
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate
for LS in 0.1 0.25 0.75 1.0; do
  echo "########## lambda_s=$LS ##########"
  python train.py --dataset ph2 --epochs 30 --patience 8 --lambda-s $LS --tag sccbm_ls${LS}
done
echo "SWEEP DONE"
