#!/bin/bash
set -e
cd "$(dirname "$0")"
# Activate your Python environment before running, e.g.:
#   source /path/to/venv/bin/activate
echo "########## derm7pt -> ph2 ##########"
python cross_dataset.py --source derm7pt --target ph2
echo "########## ph2 -> derm7pt ##########"
python cross_dataset.py --source ph2 --target derm7pt
echo "CROSS DONE"
