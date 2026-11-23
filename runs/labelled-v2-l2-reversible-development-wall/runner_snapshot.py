"""Paired experiments on a fixed, versioned attack-development image split.

Every algorithm/source is frozen before attack results. Expected search failures
are recorded and do not abort the rest of the paired matrix. Case order is fixed;
method order alternates to reduce timing drift. Confirmation is a one-time check.
"""
import argparse
import csv
import hashlib
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import benchmark_attacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('research/generalization_v1'))
    parser.add_argument('--split', choices=('development', 'confirmation'), required=True)
    parser.add_argument('--metric', choices=('l2', 'l0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-controls', type=Path, help='Completed results.csv with matching frozen control cases/budgets')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve completed measurements')
    protocol = json.loads((args.dataset/'protocol.json').read_text())
    freeze = json.loads((args.dataset/'source_freeze.json').read_text())
    for name, record in freeze.items():
        if args.metric not in record.get('metrics', ('l2', 'l0')):
            continue
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'frozen experiment source changed: {name}')
    with (args.dataset/'manifest.csv').open() as stream:
        images = [row for row in csv.DictReader(stream) if row['split'] == args.split]
    for image in images:
        if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest() != image['prepared_sha256']:
            raise ValueError(f"dataset image changed: {image['id']}")
    settings = protocol['primary' if args.metric == 'l2' else 'secondary']
    seed = settings[f'{args.split}_seed']
    os.environ.update(CS454_L0_STYLE='parental', CS454_L2_STYLE='energy',
                      CS454_DISCOVERY_REFERENCE='900', CS454_DISCOVERY_FRACTION='.55',
                      CS454_BOUNDARY_SUCCESSES='3', CS454_BOUNDARY_FRACTION='.99',
                      CS454_BOUNDARY_ALPHA='.25')
    os.environ.pop('CS454_BOUNDARY_MIN_GENERATIONS', None)
    for key, value in settings.get('environment', {}).items():
        if not key.startswith('CS454_'):
            raise ValueError('experiment settings must use task-specific CS454_ variables')
        os.environ[key] = str(value)
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_bytes((args.dataset/'protocol.json').read_bytes())
    (args.output/'manifest.csv').write_bytes((args.dataset/'manifest.csv').read_bytes())
    (args.output/'source_freeze.json').write_bytes((args.dataset/'source_freeze.json').read_bytes())
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records = []
    reused = {}
    if args.reuse_controls:
        with args.reuse_controls.open() as stream:
            controls = [row for row in csv.DictReader(stream) if row['method'] == 'control']
        for row in controls:
            if (row['split'] != args.split or row['norm'] != args.metric or int(row['seed']) != seed
                or int(row['queries_limit']) != settings['queries'] or float(row['seconds_limit']) != settings['seconds']
                or int(row['threads']) != settings['threads'] or row['implementation'] != settings['control']):
                raise ValueError('reused control does not match the frozen experiment')
            reused[Path(row['image']).resolve()] = row
        if set(reused) != {Path(image['path']).resolve() for image in images}:
            raise ValueError('reused controls do not cover exactly this split')
        (args.output/'reused_controls.json').write_text(json.dumps({'results': str(args.reuse_controls),
            'sha256': hashlib.sha256(args.reuse_controls.read_bytes()).hexdigest(),
            'timing': 'Controls reused at fixed query/time ceilings; no claim of paired current-window speed.'},indent=2)+'\n')
    for index, image in enumerate(images):
        if reused:
            records.append(reused[Path(image['path']).resolve()])
            order = ('candidate',)
        else:
            order = ('control', 'candidate') if index % 2 == 0 else ('candidate', 'control')
        for kind in order:
            output = args.output/f"{image['id']}_{kind}"
            options = SimpleNamespace(images=[image['path']], norms=[args.metric], seed=seed,
                                      seconds=settings['seconds'], queries=settings['queries'],
                                      threads=settings['threads'], implementation=settings[kind],
                                      output=output, tag=f'{args.split}_{kind}')
            with (args.output/f"{image['id']}_{kind}.log").open('w') as stream:
                with redirect_stdout(stream):
                    exit_code = benchmark_attacks.run(options)
            with (output/'results.csv').open() as stream:
                result = next(csv.DictReader(stream))
            result.update(split=args.split, method=kind, implementation=settings[kind],
                          experiment_exit=exit_code, configuration=str(output/f'config_{options.tag}.json'))
            records.append(result)
            with (args.output/'results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            print(json.dumps({key: result[key] for key in ('image','norm','method','seed','distance','queries','runtime_seconds','status')}), flush=True)
    print(f'Completed {len(records)} paired trial rows; failures remain in results.csv.', flush=True)


if __name__ == '__main__':
    main()
