#!/usr/bin/env bash
# The two development IDs were fixed in labelled_v2_research_roadmap.json.
set -eu
export TORCH_HOME="$PWD/.runtime/torch-cache"
export MPLCONFIGDIR=/tmp/cs454-plot-cache
export CS454_L2_SEARCH=original
export CS454_L2_STYLE=energy
python3 - <<'PY'
import hashlib, json
from pathlib import Path
plan = Path('research/labelled_v2_original_wall_plan')
for name, record in json.loads((plan/'source_freeze.json').read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'frozen original-wall source changed: {name}')
PY
"$PWD/.runtime/python-env/bin/python" benchmark_attacks.py \
  --implementation research.session3_promoted.epsilon_ball \
  --images research/generalization_imagenette_v2/images/development_n02979186_00.png \
           research/generalization_imagenette_v2/images/development_n03425413_00.png \
  --norms l2 --seconds 165 --threads 4 --seed 456 \
  --output runs/labelled-v2-l2-original-development-wall --tag labelled_wall
