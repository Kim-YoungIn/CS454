#!/usr/bin/env bash
set -eu
export TORCH_HOME=/tmp/cs454-torch
export MPLCONFIGDIR=/tmp/cs454-plot-cache
CS454_PYTHON=/tmp/cs454-baseline-env/bin/python
"$CS454_PYTHON" -m unittest -v research.test_fog_integrated research.test_support_integrated research.test_dense_radial_probe > runs/session3-radial-tests.txt 2>&1
"$CS454_PYTHON" benchmark_attacks.py --implementation research.dense_radial_probe \
  --images sock --norms l2 --queries 900 --seconds 165 --threads 4 --seed 454 \
  --output runs/session3-radial-sock900 --tag probe900 > /tmp/cs454-session3-radial-sock.txt
"$CS454_PYTHON" benchmark_attacks.py --implementation research.dense_radial_probe \
  --images knot --norms l2 --queries 1800 --seconds 165 --threads 4 --seed 454 \
  --output runs/session3-radial-knot1800 --tag probe1800 > /tmp/cs454-session3-radial-knot.txt
