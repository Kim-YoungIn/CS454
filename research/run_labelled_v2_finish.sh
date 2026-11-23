#!/usr/bin/env bash
# Run only after the frozen900-call preserved-state screen has completed.
# No confirmation attacks or production changes are performed here.
set -eu
export TORCH_HOME="$PWD/.runtime/torch-cache"
export MPLCONFIGDIR=/tmp/cs454-plot-cache
cs454_python="$PWD/.runtime/python-env/bin/python"
if python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('runs/labelled-v2-l2-preserved-development-summary/summary.json').read_text())[0]
roadmap = json.loads(Path('research/labelled_v2_research_roadmap.json').read_text())
gate = roadmap['preconfirmation_gate']
eligible = (report['pairs'] == len(roadmap['development_screen']['ids'])
            and report['success_regressions'] == 0
            and report['common_successes'] == report['pairs']
            and report['geometric_mean_ratio'] <= gate['development_screen_max_geomean']
            and report['worst_ratio'] <= gate['development_screen_max_worst_ratio'])
print('Frozen development-screen eligibility:', eligible, flush=True)
raise SystemExit(0 if eligible else 1)
PY
then
    "$cs454_python" benchmark_generalization.py \
        --dataset research/labelled_v2_preserved_wall_plan --split development --metric l2 \
        --output runs/labelled-v2-l2-preserved-development-wall \
        > runs/labelled-v2-l2-preserved-development-wall.log 2>&1
    python3 compare_generalization.py runs/labelled-v2-l2-preserved-development-wall \
        --output runs/labelled-v2-l2-preserved-development-wall-summary
    python3 research/analyze_labelled_windows.py runs/labelled-v2-l2-preserved-development-wall \
        --output runs/labelled-v2-l2-preserved-development-wall-summary/integrity.json
fi
bash research/run_labelled_v2_original_wall.sh \
    > runs/labelled-v2-l2-original-development-wall.log 2>&1
python3 research/summarize_labelled_v2.py --output runs/labelled-v2-completion-study
"$cs454_python" validate_generalization.py runs/labelled-v2-completion-study \
    --output runs/labelled-v2-completion-validation/artifacts.csv
