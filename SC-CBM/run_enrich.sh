#!/bin/bash
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate
EP=30; PAT=8
echo "########## PH2 backbones ##########"
python train.py --dataset ph2 --epochs $EP --patience $PAT --backbone resnet18   --tag bb_resnet18   --save-ckpt
python train.py --dataset ph2 --epochs $EP --patience $PAT --backbone resnet101  --tag bb_resnet101
python train.py --dataset ph2 --epochs $EP --patience $PAT --backbone densenet201 --tag bb_densenet201
echo "########## PH2 pooling ##########"
python train.py --dataset ph2 --epochs $EP --patience $PAT --pool attention --tag pool_attention
echo "########## Derm7pt backbones ##########"
python train.py --dataset derm7pt --epochs $EP --patience $PAT --backbone resnet18   --tag bb_resnet18
python train.py --dataset derm7pt --epochs $EP --patience $PAT --backbone resnet101  --tag bb_resnet101
python train.py --dataset derm7pt --epochs $EP --patience $PAT --backbone densenet201 --tag bb_densenet201
echo "ENRICH DONE"
