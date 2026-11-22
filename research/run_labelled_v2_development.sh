#!/usr/bin/env bash
# Execute after the initial20-row labelled audit has completed.
set -eu
export TORCH_HOME=/tmp/cs454-torch
export MPLCONFIGDIR=/tmp/cs454-plot-cache
CS454_PYTHON=/tmp/cs454-baseline-env/bin/python
python3 - <<'PY'
import csv,hashlib,json
from pathlib import Path
source=Path('runs/labelled-v2-l2-baseline-development/results.csv')
rows=list(csv.DictReader(source.open()))
assert len(rows)==20 and all(int(row['queries'])==900 for row in rows)
manifest={str(Path(r['path']).resolve()):int(r['expected_class']) for r in csv.DictReader(Path('research/generalization_imagenette_v2/manifest.csv').open()) if r['split']=='development'}
assert all(int(row['reference_class'])==manifest[row['image']] for row in rows)
controls=[dict(row) for row in rows if row['method']=='candidate']
assert len(controls)==10
for row in controls:
 assert row['implementation']=='research.generalization_session_promoted.epsilon_ball'
 row['method']='control'
out=Path('runs/labelled-v2-current512-controls');out.mkdir()
with (out/'results.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(controls[0]));writer.writeheader();writer.writerows(controls)
(out/'provenance.json').write_text(json.dumps(dict(source_results=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),action='Relabel previous comparison candidate as the new comparison control only; preserve configuration/raw paths, implementation, scores and actual work. No new VGG call or selection.'),indent=2)+'\n')
PY
"$CS454_PYTHON" benchmark_generalization.py --dataset research/labelled_v2_reversible_plan \
  --split development --metric l2 --reuse-controls runs/labelled-v2-current512-controls/results.csv \
  --output runs/labelled-v2-l2-reversible-development > /tmp/cs454-labelled-v2-reversible-development.txt 2>&1
"$CS454_PYTHON" benchmark_generalization.py --dataset research/labelled_v2_reversible_wall_development_plan \
  --split development --metric l2 --output runs/labelled-v2-l2-reversible-development-wall \
  > /tmp/cs454-labelled-v2-reversible-development-wall.txt 2>&1
