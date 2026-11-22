"""Select the lowest verified grader distance from completed experiment folders.

Example: python select_artifacts.py runs/ga-v2 runs/ga-v3 --output submission
Candidates are loaded and classified afresh; CSV claims are not trusted. This is
an offline deliverable tool and is never used by the attack implementations.
"""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import torch

from util import IMAGES_ORG_PREDS, get_image_tensor_from, get_model_most_likely_for, get_viz_for
from verify_artifacts import METRICS, check_fog, read_tensor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--perlin', action='store_true')
    args = parser.parse_args()
    sources = [path.resolve() for path in args.directories]
    output = args.output.resolve()
    if output in sources:
        parser.error('output must differ from all source directories')
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty to avoid retaining stale submission artifacts')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = []
    missing = []
    for name, reference in IMAGES_ORG_PREDS.items():
        original = get_image_tensor_from(name)
        stem = Path(name).stem
        for norm in (*METRICS, 'perlin') if args.perlin else METRICS:
            candidates = []
            filename = f'raw_{stem}_{norm}.png'
            for source in sources:
                path = source / filename
                if not path.is_file():
                    continue
                try:
                    tensor = read_tensor(path, original)
                    if norm == 'perlin':
                        tensor, distance = check_fog(source / f'{stem}_perlin.json', original, tensor)
                    else:
                        distance = METRICS[norm](tensor - original)
                    probability, predicted = get_model_most_likely_for(tensor)
                    if predicted != reference and math.isfinite(distance):
                        candidates.append((distance, source, predicted, probability))
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                    print(f'Skipping {path}: {error}')
            if not candidates:
                missing.append(f'{stem}_{norm}')
                continue
            distance, source, predicted, probability = min(candidates, key=lambda item: item[0])
            for artifact in (filename, f'{stem}_{norm}.png'):
                if (source / artifact).is_file():
                    shutil.copy2(source / artifact, output / artifact)
            if not (output / f'{stem}_{norm}.png').is_file():
                from contextlib import chdir
                tensor = read_tensor(source / filename, original)
                with chdir(output):
                    get_viz_for(tensor.clone(), save_to=f'{stem}_{norm}.png')
                    import matplotlib.pyplot as plt
                    plt.close('all')
            if norm == 'perlin':
                shutil.copy2(source / f'{stem}_perlin.json', output / f'{stem}_perlin.json')
            manifest.append(dict(image=stem, norm=norm, distance=distance,
                                 reference=reference, predicted=predicted,
                                 probability=probability, source=str(source)))
    with (output / 'selection.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'norm', 'distance', 'reference', 'predicted', 'probability', 'source'])
        writer.writeheader()
        writer.writerows(manifest)
    print(f'Selected {len(manifest)} cases into {output}')
    if missing:
        print('Missing successful cases: ' + ', '.join(missing))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
