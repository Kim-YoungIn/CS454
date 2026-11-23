"""Close the labelled development study without consuming confirmation data.

Reads immutable trial outputs, applies the predeclared screen gate and writes a
deduplicated artifact-verification input. Original-global wall trials are an
explicitly separate timing diagnostic, not a third balanced treatment arm.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous summaries; choose a fresh output')
    roadmap = json.loads(Path('research/labelled_v2_research_roadmap.json').read_text())
    gates = roadmap['preconfirmation_gate']
    directories = {
        'baseline_screen': Path('runs/labelled-v2-l2-baseline-development'),
        'reversible_screen': Path('runs/labelled-v2-l2-reversible-development'),
        'preserved_screen': Path('runs/labelled-v2-l2-preserved-development'),
        'reversible_wall': Path('runs/labelled-v2-l2-reversible-development-wall'),
    }
    preserved_wall = Path('runs/labelled-v2-l2-preserved-development-wall')
    if preserved_wall.exists():
        directories['preserved_wall'] = preserved_wall
    reports, unique, provenance = {}, {}, []
    for name, directory in directories.items():
        report = json.loads(Path(f'{directory}-summary/summary.json').read_text())
        if len(report) != 1:
            raise ValueError(f'expected one comparison group: {name}')
        reports[name] = report[0]
        for row in rows(directory/'results.csv'):
            if row['split'] != 'development':
                raise ValueError('confirmation exposure in development study')
            key = (row['configuration'], row['image'], row['norm'])
            if key in unique:
                for field in ('distance', 'queries', 'runtime_seconds', 'status', 'predicted_class'):
                    if unique[key][field] != row[field]:
                        raise ValueError(f'conflicting reused trial: {key}, {field}')
            else:
                unique[key] = row
        provenance.append({'directory': str(directory), 'results_sha256':
            hashlib.sha256((directory/'results.csv').read_bytes()).hexdigest()})
    decisions = {}
    for candidate in ('reversible', 'preserved'):
        screen = reports[f'{candidate}_screen']
        screen_pass = (screen['pairs'] == len(roadmap['development_screen']['ids'])
                       and screen['success_regressions'] == 0
                       and screen['common_successes'] == screen['pairs']
                       and screen['geometric_mean_ratio'] <= gates['development_screen_max_geomean']
                       and screen['worst_ratio'] <= gates['development_screen_max_worst_ratio'])
        wall = reports.get(f'{candidate}_wall')
        wall_pass = bool(wall and wall['pairs'] == len(roadmap['development_wall']['ids'])
                         and wall['success_regressions'] == 0
                         and wall['common_successes'] == wall['pairs']
                         and wall['geometric_mean_ratio'] <= gates['development_wall_max_geomean']
                         and wall['worst_ratio'] <= gates['development_wall_max_worst_ratio'])
        decisions[candidate] = dict(screen_gate_pass=screen_pass, wall_gate_pass=wall_pass,
                                   eligible_for_confirmation=screen_pass and wall_pass)
    original = Path('runs/labelled-v2-l2-original-development-wall')
    diagnostics = []
    if original.exists():
        current = {row['image']: row for row in rows(directories['reversible_wall']/'results.csv')
                   if row['method'] == 'control'}
        current_later = ({row['image']: row for row in rows(preserved_wall/'results.csv')
                          if row['method'] == 'control'} if 'preserved_wall' in reports else {})
        manifest = {str(Path(row['path']).resolve()): row for row in
                    rows(Path('research/labelled_v2_original_wall_plan/manifest.csv'))}
        seen_original = set()
        for row in rows(original/'results.csv'):
            if row['image'] in seen_original or row['image'] not in manifest:
                raise ValueError('duplicate or unexpected original-global diagnostic')
            seen_original.add(row['image'])
            reference = current[row['image']]
            if (row['status'] != 'pass' or row['success'] != 'True'
                    or row['round_trip'] != 'True' or row['queries_limit']
                    or int(row['reference_class']) != int(manifest[row['image']]['expected_class'])
                    or int(row['seed']) != roadmap['additional_original_wall']['seed']
                    or float(row['seconds_limit']) != roadmap['additional_original_wall']['seconds']
                    or int(row['threads']) != 4):
                raise ValueError('invalid original-global diagnostic')
            record = dict(row, split='development', method='original_diagnostic',
                          implementation=roadmap['additional_original_wall']['implementation'],
                          configuration=str(original/'config_labelled_wall.json'))
            unique[(record['configuration'], record['image'], record['norm'])] = record
            diagnostics.append(dict(image=Path(row['image']).stem,
                original_distance=float(row['distance']), current_distance=float(reference['distance']),
                original_over_current=float(row['distance'])/float(reference['distance']),
                original_queries=int(row['queries']), current_queries=int(reference['queries']),
                original_seconds=float(row['runtime_seconds']), current_seconds=float(reference['runtime_seconds'])))
            if row['image'] in current_later:
                later = current_later[row['image']]
                diagnostics[-1].update(current_later_distance=float(later['distance']),
                    original_over_current_later=float(row['distance'])/float(later['distance']),
                    current_later_queries=int(later['queries']),
                    current_later_seconds=float(later['runtime_seconds']))
        if seen_original != set(manifest):
            raise ValueError('incomplete original-global wall diagnostic')
        provenance.append({'directory': str(original), 'results_sha256':
            hashlib.sha256((original/'results.csv').read_bytes()).hexdigest()})
    usage = json.loads(Path('research/labelled_v2_usage.json').read_text())
    if usage['confirmation_attack_used']:
        raise ValueError('confirmation is no longer untouched; revise study scope')
    args.output.mkdir(parents=True)
    records = list(unique.values())
    fields = list(dict.fromkeys(field for row in records for field in row))
    with (args.output/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = dict(comparisons=reports, frozen_gates=gates, decisions=decisions,
        original_wall_diagnostic=diagnostics,
        original_wall_geomean=(math.exp(sum(math.log(r['original_over_current']) for r in diagnostics)
                                      / len(diagnostics)) if diagnostics else None),
        original_wall_geomean_later_control=(math.exp(sum(math.log(r['original_over_current_later'])
            for r in diagnostics)/len(diagnostics))
            if diagnostics and all('original_over_current_later' in r for r in diagnostics) else None),
        interpretation='Development outcomes only; separate-window original timings are descriptive. '
                       'All completed current-wall blocks are reported; neither is selected for a favorable ratio. '
                       'Fixed-query controls are reused, not fresh trials. No algorithm was selected on confirmation.',
        unique_trials=len(records), confirmation_attack_used=[],
        reserved_confirmation_count=len(usage['confirmation_reserved_ids']), provenance=provenance)
    (args.output/'study.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(decisions=decisions, unique_trials=len(records),
                         original_wall=diagnostics), indent=2))


if __name__ == '__main__':
    main()
