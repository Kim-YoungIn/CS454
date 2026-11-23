#!/usr/bin/env bash
set -eu
export TORCH_HOME="$PWD/.runtime/torch-cache"
export MPLCONFIGDIR=/tmp/cs454-plot-cache
cs454_python="$PWD/.runtime/python-env/bin/python"
"$cs454_python" benchmark_generalization.py --dataset research/labelled_v2_preserved_plan \
  --split development --metric l2 --reuse-controls runs/labelled-v2-current512-controls/results.csv \
  --output runs/labelled-v2-l2-preserved-development > runs/labelled-v2-l2-preserved-development.log 2>&1
python3 compare_generalization.py runs/labelled-v2-l2-preserved-development \
  --output runs/labelled-v2-l2-preserved-development-summary
