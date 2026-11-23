"""Report mixed serial/parallel execution without changing frozen evaluation.

Standard-library only. Final reports require every source case and independent
verification row, preserving failed outcomes and all forty case denominators.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


NORMS = ('linf', 'l2', 'l0', 'perlin')
SERIAL = 'serial_unpinned_four_threads'
PARALLEL = 'parallel_pinned_four_cores_two_workers'
LIMITATIONS = (
    'The first11 terminal trials used unrestricted scheduling and four Torch threads; '
    'the remaining29 used two concurrent workers, each pinned to four distinct physical '
    'cores with one SMT thread per core. The groups do not overlap physical cores. '
    'Memory bandwidth, CPU package power, frequency and host activity remain shared. '
    'These strata contain different cases and were not randomized between execution modes. '
    'Their timing, call counts, distances and success rates are descriptive; no same-case '
    'counterfactual establishes speedup, slowdown or an algorithmic gain. '
    'This is one seed and one confirmation photograph per each of ten classes. '
    'Imagenette may overlap VGG pretraining. All40 cases, including failures, remain '
    'in the total success denominator; every norm has ten cases. '
    'Distance units differ across norms and must not be compared across attacks. '
    'The confirmation bank is now consumed and must not be relabeled fresh after tuning.'
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def numeric_stats(values):
    values = [result for value in values if (result := number(value)) is not None]
    return dict(count=len(values), median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                total=sum(values))


def unique(rows, field):
    indexed = {row[field]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'duplicate {field}')
    return indexed


def collect(run, verification_path, amendment_path):
    protocol_path, manifest_path = run/'protocol.json', run/'manifest.csv'
    protocol = json.loads(protocol_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    cases = unique(protocol['cases'], 'case_id')
    manifest = unique(csv_rows(manifest_path), 'id')
    results = unique(csv_rows(run/'results.csv'), 'case_id')
    verified = unique(csv_rows(verification_path), 'case_id')
    if len(cases) != 40 or len(manifest) != 10 or set(results) != set(cases) or set(verified) != set(cases):
        raise ValueError('require all40 frozen results and independent verification rows for10 images')
    if {(case['image_id'], case['norm']) for case in cases.values()} != {
            (image_id, norm) for image_id in manifest for norm in NORMS}:
        raise ValueError('frozen cases are not a complete10x4 matrix')
    if any(row['split'] != 'confirmation' for row in manifest.values()):
        raise ValueError('manifest contains non-confirmation data')
    serial = set(amendment['serial_cases_preserved'])
    if len(serial) != 11 or not serial <= set(cases) or amendment['remaining_cases'] != 29:
        raise ValueError('execution amendment must preserve11 serial cases and schedule29 remaining')
    groups = [sorted(map(int, value.split(','))) for value in amendment['cpu_groups']]
    if len(groups) != 2 or any(len(set(group)) != 4 for group in groups) or set(groups[0]) & set(groups[1]):
        raise ValueError('invalid amended CPU groups')
    records, execution_hashes = [], {}
    for case_id, case in cases.items():
        row, check = results[case_id], verified[case_id]
        if check['verification_status'] == 'pending':
            raise ValueError('independent validation is incomplete')
        if (row['image_id'] != case['image_id'] or row['norm'] != case['norm']
                or check['image_id'] != case['image_id'] or check['norm'] != case['norm']
                or int(row['seed']) != case['seed'] or int(row['threads']) != case['threads']
                or float(row['seconds_limit']) != case['seconds_limit']
                or float(row['grading_seconds_limit']) != case['grading_seconds_limit']):
            raise ValueError(f'context differs from frozen protocol: {case_id}')
        execution_path = run/'cases'/case_id/'execution.json'
        if case_id in serial:
            if execution_path.exists():
                raise ValueError(f'preserved serial case has parallel metadata: {case_id}')
            mode, affinity, slot = SERIAL, 'unrestricted; four Torch threads', ''
            started = finished = ''
        else:
            execution = json.loads(execution_path.read_text())
            slot = execution['worker_slot']
            if (execution['mode'] != 'parallel_two_disjoint_core_groups' or slot not in (0, 1)
                    or sorted(execution['cpu_affinity']) != groups[slot]
                    or execution['torch_threads'] != case['threads']
                    or not execution.get('finished_utc')):
                raise ValueError(f'invalid or incomplete parallel metadata: {case_id}')
            mode = PARALLEL
            affinity = ','.join(map(str, groups[slot]))
            started, finished = execution['started_utc'], execution['finished_utc']
            execution_hashes[str(execution_path)] = digest(execution_path)
        success = check['verification_status'] == 'pass'
        measured = number(check['measured_distance'])
        if success and (measured is None or measured <= 0):
            raise ValueError(f'verified success lacks a finite positive distance: {case_id}')
        if check['reported_distance'] != row['distance']:
            raise ValueError(f'verification does not refer to this reported score: {case_id}')
        records.append(dict(case_id=case_id, image_id=case['image_id'],
            expected_name=manifest[case['image_id']]['expected_name'],
            expected_class=manifest[case['image_id']]['expected_class'], norm=case['norm'],
            mode=mode, worker_slot=slot, cpu_group=affinity, seed=case['seed'], threads=case['threads'],
            source_status=row['status'], verification_status=check['verification_status'],
            independent_success=success, reported_distance=row['distance'],
            measured_distance=check['measured_distance'],
            grader_score=measured if success else 999999999,
            runtime_seconds=row['runtime_seconds'], grading_runtime_seconds=row['grading_runtime_seconds'],
            planned_budget_seconds=case['seconds_limit'], grading_budget_seconds=case['grading_seconds_limit'],
            planned_budget_exceeded=check['planned_budget_exceeded'], grading_timeout=check['grading_timeout'],
            queries=row['queries'], started_utc=started, finished_utc=finished,
            error=check.get('detail') or row.get('error', '')))
    source_paths = [protocol_path, manifest_path, run/'source_freeze.json', run/'environment.json',
                    run/'results.csv', verification_path, amendment_path, Path(__file__)]
    provenance = {str(path): digest(path) for path in source_paths}
    provenance.update(execution_hashes)
    # The independent report pins the aggregate it verified, when available.
    verification_summary = verification_path.parent/'summary.json'
    if verification_summary.exists():
        summary = json.loads(verification_summary.read_text())
        if summary.get('results_sha256') != provenance[str(run/'results.csv')]:
            raise ValueError('verification summary refers to a different aggregate')
        provenance[str(verification_summary)] = digest(verification_summary)
    return records, manifest, amendment, provenance


def group_summary(records):
    return dict(cases=len(records), independently_verified_successes=sum(row['independent_success'] for row in records),
        success_rate=sum(row['independent_success'] for row in records)/len(records) if records else None,
        verification_statuses=dict(Counter(row['verification_status'] for row in records)),
        source_statuses=dict(Counter(row['source_status'] for row in records)),
        entry_seconds=numeric_stats(row['runtime_seconds'] for row in records),
        grading_seconds=numeric_stats(row['grading_runtime_seconds'] for row in records),
        actual_queries=numeric_stats(row['queries'] for row in records),
        failure_inclusive_grader_score_mean=statistics.mean(row['grader_score'] for row in records)
            if records and len({row['norm'] for row in records}) == 1 else None)


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pretty(value):
    result = number(value)
    return '—' if result is None else f'{result:.6g}'


def report(run, verification_path, amendment_path, output):
    if output.exists():
        raise ValueError('output exists; preserve reports and choose a fresh directory')
    records, manifest, amendment, provenance = collect(run, verification_path, amendment_path)
    strata = {mode: dict(overall=group_summary([row for row in records if row['mode'] == mode]),
                        per_norm={norm: group_summary([row for row in records
                                  if row['mode'] == mode and row['norm'] == norm]) for norm in NORMS})
              for mode in (SERIAL, PARALLEL)}
    summary = dict(overall=group_summary(records),
        per_norm={norm: group_summary([row for row in records if row['norm'] == norm]) for norm in NORMS},
        execution_strata=strata, cpu_groups=amendment['cpu_groups'], limitations=LIMITATIONS,
        grading='Mandatory attack+distance/render+classification uses180/300 seconds. '
                'Frozen optimizer165/270 targets are separate diagnostics. Failed cases retain999999999.',
        score_statistics='Scores have different units; aggregate means are computed within one norm only. '
                         'Mixed-norm overall/stratum score means are deliberately null.',
        timing_statistics='Runtime totals are sums of recorded per-case wall times, not elapsed makespan '
                          'or CPU seconds. Parallel case intervals overlap. Query totals are actual '
                          'model-evaluation examples as recorded by each attack.',
        provenance_sha256=provenance)
    output.mkdir(parents=True)
    write_csv(output/'execution_cases.csv', records)
    by_pair = {(row['image_id'], row['norm']): row for row in records}
    wide = []
    fields = ('reported_distance', 'measured_distance', 'verification_status', 'mode', 'cpu_group', 'runtime_seconds',
              'grading_runtime_seconds', 'queries', 'grader_score', 'source_status')
    for image_id, image in manifest.items():
        row = dict(image_id=image_id, expected_name=image['expected_name'], expected_class=image['expected_class'])
        for norm in NORMS:
            source = by_pair[image_id, norm]
            row.update({norm+'_'+field: source[field] for field in fields})
        wide.append(row)
    write_csv(output/'image_norm_wide.csv', wide)
    (output/'execution_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    success = summary['overall']['independently_verified_successes']
    lines = ['# All-norm confirmation with explicit execution strata', '',
             f'Independent one-thread successes: **{success}/40**. '
             'All forty cases remain in the denominator; each norm has ten cases.', '',
             '| Norm | Verified successes / 10 | Failed cases | Failure-inclusive mean grade score |',
             '| --- | ---: | ---: | ---: |']
    for norm in NORMS:
        group = summary['per_norm'][norm]
        lines.append(f"| {norm} | {group['independently_verified_successes']}/10 | "
                     f"{10-group['independently_verified_successes']} | "
                     f"{pretty(group['failure_inclusive_grader_score_mean'])} |")
    lines += ['', '## Ten-image × four-attack outcomes', '',
              'Cells show independently measured distance and execution mode: '
              '**S** = serial, unrestricted affinity/four Torch threads; '
              '**P0/P1** = parallel worker pinned to the corresponding CPU group. '
              'Failed cases retain their failure label; they are not omitted.', '',
              '| Image (class) | Linf | Squared L2 | L0 elements | Fog magnitude |',
              '| --- | ---: | ---: | ---: | ---: |']
    for image_id, image in manifest.items():
        cells = []
        for norm in NORMS:
            row = by_pair[image_id, norm]
            mode = 'S' if row['mode'] == SERIAL else 'P'+str(row['worker_slot'])
            value = pretty(row['measured_distance']) if row['independent_success'] else row['verification_status']
            cells.append(value+' ['+mode+']')
        label = image['expected_name'].split(',')[0]
        lines.append('| '+image_id+' ('+label+') | '+' | '.join(cells)+' |')
    lines += ['', '## Execution metadata for every case', '',
              'CPU groups: P0 = `'+amendment['cpu_groups'][0]+'`; P1 = `'+amendment['cpu_groups'][1]+'`. '
              'Entry and grading seconds are the originally recorded times, not remeasured validation times.', '',
              '| Image | Attack | Mode | CPU affinity | Entry seconds | Grading seconds | Actual queries | Verification |',
              '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in records:
        mode = 'S' if row['mode'] == SERIAL else 'P'+str(row['worker_slot'])
        lines.append('| '+row['image_id']+' | '+row['norm']+' | '+mode+' | '+row['cpu_group']+' | '
                     +pretty(row['runtime_seconds'])+' | '+pretty(row['grading_runtime_seconds'])+' | '
                     +pretty(row['queries'])+' | '+row['verification_status']+' |')
    lines += ['', '## Descriptive strata', '',
              '| Execution | Norm | Verified successes / cases in stratum | Median entry seconds | Median actual queries |',
              '| --- | --- | ---: | ---: | ---: |']
    for mode, stratum in strata.items():
        for norm, group in stratum['per_norm'].items():
            lines.append(f"| {'S' if mode == SERIAL else 'P0/P1'} | {norm} | "
                         f"{group['independently_verified_successes']}/{group['cases']} | "
                         f"{pretty(group['entry_seconds']['median'])} | {pretty(group['actual_queries']['median'])} |")
    lines += ['', LIMITATIONS, '', summary['grading'], '', summary['timing_statistics'], '',
              'Full forty-case details: `execution_cases.csv`. Wide ten-image table: `image_norm_wide.csv`. '
              'Stratum counts and input/source hashes: `execution_summary.json`.', '']
    (output/'execution_results.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--amendment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = report(args.run, args.verification,
                     args.amendment or args.run/'execution_amendment.json', args.output)
    print(json.dumps(dict(overall=summary['overall'],
        stratum_counts={mode: row['overall']['cases'] for mode, row in summary['execution_strata'].items()}), indent=2))


if __name__ == '__main__':
    main()
