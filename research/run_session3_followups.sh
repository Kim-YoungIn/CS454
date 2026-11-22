#!/usr/bin/env bash
# Run from the repository root, after the earlier isolated CPU queue completes.
set -eu
export TORCH_HOME=/tmp/cs454-torch
export MPLCONFIGDIR=/tmp/cs454-plot-cache
CS454_PYTHON=/tmp/cs454-baseline-env/bin/python
"$CS454_PYTHON" -m unittest -v \
  test_attacks test_fog_geometry test_structured_sparse \
  research.test_sparse_candidate research.test_cache_candidate research.test_cached_structured \
  research.test_prefix_cached_prune research.test_value_prune research.test_quota_prune \
  research.test_multistart_prune research.test_recombine_prune \
  research.test_dense_session3_boundary research.test_elite_session3 research.test_fog_octave_ga \
  research.test_fog_integrated research.test_support_integrated \
  > runs/session3-contract-tests.txt 2>&1
"$CS454_PYTHON" benchmark_attacks.py --implementation research.session3_baseline.epsilon_ball --images sock --norms l2 --queries 900 \
  --seconds 165 --threads 4 --seed 454 --output runs/session3-dense-sock-control --tag pair900 \
  > /tmp/cs454-session3-dense-sock-control.txt
"$CS454_PYTHON" benchmark_attacks.py --implementation research.dense_session3_fast \
  --images sock --norms l2 --queries 900 --seconds 165 --threads 4 --seed 454 \
  --output runs/session3-dense-sock-fast --tag pair900 \
  > /tmp/cs454-session3-dense-sock-fast.txt
"$CS454_PYTHON" benchmark_fog.py --implementation research.session3_baseline.perlin_fog --images images/sock.jpg images/tank.jpg \
  --generations 80 --seconds 90 --threads 4 --seed 454 \
  --output runs/session3-fog-other-control/results.csv --artifacts runs/session3-fog-other-control \
  > /tmp/cs454-session3-fog-other-control.txt
"$CS454_PYTHON" benchmark_fog.py --implementation research.fog_integrated \
  --images images/sock.jpg images/tank.jpg --generations 80 --seconds 90 --threads 4 --seed 454 \
  --output runs/session3-fog-other-integrated/results.csv --artifacts runs/session3-fog-other-integrated \
  > /tmp/cs454-session3-fog-other-integrated.txt
"$CS454_PYTHON" benchmark_attacks.py --implementation research.epsilon_support_integrated \
  --images mink --norms l0 --seconds 165 --threads 4 --seed 454 \
  --output runs/session3-support-integrated-default --tag default165 \
  > /tmp/cs454-session3-support-integrated-default.txt
"$CS454_PYTHON" benchmark_fog.py --implementation research.fog_integrated \
  --images images/mink.jpg --generations 80 --seconds 270 --threads 4 --seed 454 \
  --output runs/session3-fog-integrated-default/results.csv --artifacts runs/session3-fog-integrated-default \
  > /tmp/cs454-session3-fog-integrated-default.txt
