"""Independently validate a frozen Imagenette all-norm confirmation run.

Report-only mode imports only the standard library and labels successes as
reported, not independently verified. Every frozen case remains in the output,
including missing, failed, invalid and timed-out attacks.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


NORMS = ('linf', 'l2', 'l0', 'perlin')
LIMITATIONS = (
    'One fixed seed and one confirmation photograph per each of ten classes; '
    'the clean-correct eligibility rule is not an estimate of classifier accuracy. '
    'Imagenette is ImageNet-derived and may overlap VGG pretraining. '
    'These are absolute performance measurements of one frozen implementation; '
    'there is no paired comparator and no claim of an algorithmic improvement. '
    'All failures and incomplete cases remain in the denominator. '
    'Distances across different norms are not comparable. '
    'A consumed confirmation bank must not be reused as fresh independent validation '
    'after tuning on these outcomes.'
)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def true(value):
    return str(value).lower() in ('true', '1')


def finite_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def load_run(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    selected_rows = read_csv(directory/'manifest.csv')
    manifest = {row['id']: row for row in selected_rows}
    if len(manifest) != len(selected_rows):
        raise ValueError('duplicate image in frozen manifest')
    if not manifest or any(row['split'] != 'confirmation' for row in manifest.values()):
        raise ValueError('expected a confirmation-only manifest')
    cases = protocol['cases']
    expected = {}
    for case in cases:
        case_id, image_id, norm = case['case_id'], case['image_id'], case['norm']
        if (case_id in expected or image_id not in manifest or norm not in NORMS
                or case_id != image_id+'_'+norm):
            raise ValueError('duplicate or invalid frozen case')
        if (finite_number(case['seconds_limit']) is None or case['seconds_limit'] <= 0
                or case['grading_seconds_limit'] != (300 if norm == 'perlin' else 180)
                or case['seconds_limit'] > case['grading_seconds_limit']):
            raise ValueError('invalid frozen time budget')
        expected[case_id] = case
    if len(cases) != protocol.get('expected_cases', len(cases)) or {
            (case['image_id'], case['norm']) for case in cases} != {
            (image_id, norm) for image_id in manifest for norm in NORMS}:
        raise ValueError('protocol must cover every image and all four attacks')
    records = {}
    path = directory/'results.csv'
    if path.exists():
        for row in read_csv(path):
            case_id = row['case_id']
            if case_id in records or case_id not in expected:
                raise ValueError('unexpected or duplicate result case')
            case = expected[case_id]
            if (row['image_id'] != case['image_id'] or row['norm'] != case['norm']
                    or int(row['seed']) != case['seed']
                    or int(row['threads']) != case['threads']
                    or float(row['seconds_limit']) != case['seconds_limit']
                    or float(row['grading_seconds_limit']) != case['grading_seconds_limit']
                    or Path(row['image']).resolve() != Path(manifest[case['image_id']]['path']).resolve()):
                raise ValueError(f'result context differs from protocol: {case_id}')
            records[case_id] = row
    return protocol, manifest, expected, records


def source_success(row):
    return row.get('status') == 'pass' and true(row.get('success')) and true(row.get('round_trip'))


def base_record(case, row):
    return dict(case_id=case['case_id'], image_id=case['image_id'], norm=case['norm'],
                source_status=row.get('status', '') if row else 'missing',
                reported_distance=row.get('distance', '') if row else '',
                measured_distance='', reference_class=row.get('reference_class', '') if row else '',
                expected_class='', predicted_class='', same_predicted_class='',
                regenerated_predicted_class='', runtime_seconds=row.get('runtime_seconds', '') if row else '',
                grading_runtime_seconds=row.get('grading_runtime_seconds', '') if row else '',
                seconds_limit=case['seconds_limit'], grading_seconds_limit=case['grading_seconds_limit'],
                queries=row.get('queries', '') if row else '',
                planned_budget_exceeded=False, grading_timeout=False,
                verification_status='pending', detail='')


def validate(directory, manifest, cases, records):
    # Deliberately defer all model/framework imports until independent validation.
    import torch
    from util import get_image_tensor_from, get_model_most_likely_for
    from verify_artifacts import METRICS, check_fog, read_tensor

    torch.set_num_threads(1)
    references = {}
    verification = []
    with torch.inference_mode():
        for case_id, case in cases.items():
            row = records.get(case_id)
            result = base_record(case, row)
            selected = manifest[case['image_id']]
            result['expected_class'] = int(selected['expected_class'])
            if row is None:
                result['detail'] = 'Frozen case has no result row.'
                verification.append(result)
                continue
            elapsed = finite_number(row['runtime_seconds'])
            grading_elapsed = finite_number(row.get('grading_runtime_seconds'))
            result['planned_budget_exceeded'] = elapsed is not None and elapsed > case['seconds_limit']
            result['grading_timeout'] = grading_elapsed is not None and grading_elapsed > case['grading_seconds_limit']
            if not source_success(row):
                result.update(verification_status='source_failure',
                              detail=row.get('error', '') or 'Source did not report success with exact reload.')
                verification.append(result)
                continue
            try:
                if elapsed is None or elapsed < 0 or grading_elapsed is None or grading_elapsed < elapsed:
                    raise ValueError('missing or invalid entry-point/grading runtime')
                if case['image_id'] not in references:
                    path = Path(selected['path'])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != selected['prepared_sha256']:
                        raise ValueError('prepared image checksum differs from frozen manifest')
                    original = get_image_tensor_from(str(path)).contiguous()
                    _, reference = get_model_most_likely_for(original)
                    if reference != int(selected['expected_class']):
                        raise ValueError('one-thread clean prediction no longer matches expected class')
                    references[case['image_id']] = original, reference
                original, reference = references[case['image_id']]
                if int(row['reference_class']) != reference:
                    raise ValueError('reported clean reference differs from independent reference')
                folder = Path(row['artifact_directory'])
                saved = read_tensor(folder/f"raw_{case['image_id']}_{case['norm']}.png", original)
                _, predicted = get_model_most_likely_for(saved)
                result.update(reference_class=reference, predicted_class=predicted,
                              same_predicted_class=predicted == int(row['predicted_class']))
                if predicted == reference:
                    raise ValueError('saved tensor does not change the clean class at one thread')
                if case['norm'] == 'perlin':
                    rebuilt, measured = check_fog(folder/f"{case['image_id']}_perlin.json", original, saved)
                    _, rebuilt_class = get_model_most_likely_for(rebuilt)
                    result['regenerated_predicted_class'] = rebuilt_class
                    if rebuilt_class == reference:
                        raise ValueError('instructor-regenerated fog does not change the clean class')
                    close = math.isclose(measured, float(row['distance']), rel_tol=1e-12, abs_tol=1e-12)
                else:
                    measured = METRICS[case['norm']](saved-original)
                    close = (math.isclose(measured, float(row['distance']), rel_tol=1e-6, abs_tol=1e-7)
                             if case['norm'] == 'l2' else measured == float(row['distance']))
                result['measured_distance'] = measured
                if not math.isfinite(measured) or measured <= 0 or not close:
                    raise ValueError('authoritative distance differs from report or is nonpositive/nonfinite')
                if result['grading_timeout']:
                    result.update(verification_status='grading_timeout',
                                  detail='Valid attack exceeds the instructor grading time limit.')
                else:
                    result['verification_status'] = 'pass'
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, OverflowError) as error:
                result.update(verification_status='verification_failure', detail=f'{type(error).__name__}: {error}')
            verification.append(result)
    return verification


def summarize(cases, records, verification=None):
    checked = {row['case_id']: row for row in verification} if verification is not None else None
    summaries = {}
    for norm in NORMS:
        selected = [case for case in cases.values() if case['norm'] == norm]
        present = [records[case['case_id']] for case in selected if case['case_id'] in records]
        successful = [row for row in present if source_success(row)]
        if checked is not None:
            accepted = [checked[case['case_id']] for case in selected
                        if checked[case['case_id']]['verification_status'] == 'pass']
            distances = [float(row['measured_distance']) for row in accepted]
            statuses = Counter(checked[case['case_id']]['verification_status'] for case in selected)
            success_count = len(accepted)
        else:
            accepted = [row for row in successful
                        if finite_number(row['runtime_seconds']) is not None
                        and finite_number(row.get('distance')) is not None and float(row['distance']) > 0
                        and finite_number(row.get('grading_runtime_seconds')) is not None
                        and 0 <= float(row['runtime_seconds'])
                        and float(row['runtime_seconds']) <= float(row['grading_runtime_seconds'])
                        <= (300 if norm == 'perlin' else 180)]
            distances = [float(row['distance']) for row in accepted
                         if finite_number(row['distance']) is not None and float(row['distance']) > 0]
            statuses = Counter(row['status'] for row in present)
            statuses['pending'] += len(selected)-len(present)
            success_count = len(accepted)
        runtimes = [value for row in present if (value := finite_number(row['runtime_seconds'])) is not None]
        grading_runtimes = [value for row in present
                           if (value := finite_number(row.get('grading_runtime_seconds'))) is not None]
        queries = [value for row in present if (value := finite_number(row.get('queries'))) is not None]
        accepted_ids = {row['case_id'] for row in accepted}
        # A failure-inclusive score is useful only when the entire frozen group
        # has finished. Pending trials must not be silently assigned outcomes.
        graded_scores = ([float(checked[case['case_id']]['measured_distance'])
                          if case['case_id'] in accepted_ids else 999999999
                          for case in selected] if checked is not None else
                         [float(records[case['case_id']]['distance'])
                          if case['case_id'] in accepted_ids else 999999999
                          for case in selected]) if len(present) == len(selected) else None
        summaries[norm] = dict(expected_cases=len(selected), completed_cases=len(present),
            reported_successes=len(successful),
            independently_verified_successes=success_count if checked is not None else None,
            success_count=success_count, success_rate=success_count/len(selected),
            success_basis='independent_one_thread' if checked is not None else 'reported_only_unverified',
            statuses=dict(statuses),
            grading_timeouts=sum(value > (300 if norm == 'perlin' else 180) for value in grading_runtimes),
            planned_budget_overruns=sum(float(row['runtime_seconds']) > case['seconds_limit']
                for case in selected if (row := records.get(case['case_id']))
                and finite_number(row['runtime_seconds']) is not None),
            successful_distance_count=len(distances),
            successful_distance_min=min(distances) if distances else None,
            successful_distance_median=statistics.median(distances) if distances else None,
            successful_distance_geomean=math.exp(statistics.mean(map(math.log, distances))) if distances else None,
            successful_distance_max=max(distances) if distances else None,
            runtime_median=statistics.median(runtimes) if runtimes else None,
            runtime_max=max(runtimes) if runtimes else None,
            grading_runtime_median=statistics.median(grading_runtimes) if grading_runtimes else None,
            grading_runtime_max=max(grading_runtimes) if grading_runtimes else None,
            failure_inclusive_grader_score_mean=statistics.mean(graded_scores) if graded_scores is not None else None,
            queries_total=sum(queries), queries_median=statistics.median(queries) if queries else None)
    return summaries


def write_outputs(output, directory, protocol, manifest, cases, records, verification):
    summaries = summarize(cases, records, verification)
    report = dict(run=str(directory), independently_verified=verification is not None,
                  expected_cases=len(cases), completed_cases=len(records),
                  results_sha256=hashlib.sha256((directory/'results.csv').read_bytes()).hexdigest()
                      if (directory/'results.csv').exists() else None,
                  per_norm=summaries, limitations=LIMITATIONS,
                  distance_statistics='Conditional on successful cases only; failures remain in success-rate denominator.',
                  seed=protocol.get('seed'), threads=protocol.get('threads'))
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if verification is not None:
        with (output/'verification.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(verification[0]))
            writer.writeheader()
            writer.writerows(verification)
    per_image = []
    checked = {row['case_id']: row for row in verification} if verification is not None else {}
    for case_id, case in cases.items():
        row, check = records.get(case_id, {}), checked.get(case_id, {})
        per_image.append(dict(case_id=case_id, image_id=case['image_id'],
            expected_name=manifest[case['image_id']]['expected_name'], norm=case['norm'],
            source_status=row.get('status', 'pending'),
            verification_status=check.get('verification_status', 'not_run'),
            reported_distance=row.get('distance', ''), measured_distance=check.get('measured_distance', ''),
            runtime_seconds=row.get('runtime_seconds', ''), queries=row.get('queries', ''),
            grading_runtime_seconds=row.get('grading_runtime_seconds', ''),
            detail=check.get('detail', row.get('error', ''))))
    with (output/'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    def number(value):
        return '—' if value is None else f'{value:.6g}'
    lines = ['# Imagenette all-norm confirmation results', '',
             'Success basis: '+('independent one-thread validation.' if verification is not None
                                else '**reported only; independent validation has not run.**'), '',
             '| Attack | Success / frozen cases | Median distance* | Geometric mean* | Failure-inclusive mean score | Max grading seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for norm, summary in summaries.items():
        lines.append(f"| {norm} | {summary['success_count']} / {summary['expected_cases']} | "
                     f"{number(summary['successful_distance_median'])} | "
                     f"{number(summary['successful_distance_geomean'])} | "
                     f"{number(summary['failure_inclusive_grader_score_mean'])} | "
                     f"{number(summary['grading_runtime_max'])} |")
    lines += ['', '*Distance statistics condition on successful cases. L2 is the sum of squared differences; '
              'L0 counts changed channel elements; fog uses the sum of positive magnitudes.', '', LIMITATIONS, '',
              'The failure-inclusive mean assigns the grader penalty 999999999 to failed cases; '
              'it is omitted until all cases in a norm have finished. Frozen optimizer ceilings are '
              '165/270 seconds; grading ceilings are 180/300 seconds and include mandatory final '
              'scoring/rendering. Optimizer overruns are recorded separately; only the authoritative '
              'grading ceiling determines the time-related grade penalty.', '',
              'Full per-image outcomes, including failures: `per_image.csv`. '
              'Independent verification details: `verification.csv` (validation mode only).', '']
    (output/'results.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous validation and choose a fresh directory')
    protocol, manifest, cases, records = load_run(args.run)
    verification = None if args.report_only else validate(args.run, manifest, cases, records)
    report = write_outputs(args.output, args.run, protocol, manifest, cases, records, verification)
    print(json.dumps(dict(expected_cases=len(cases), completed_cases=len(records),
                         independently_verified=verification is not None,
                         per_norm={norm: {'successes': row['success_count'], 'expected': row['expected_cases'],
                                          'statuses': row['statuses']} for norm, row in report['per_norm'].items()}), indent=2))
    return 0 if args.report_only or all(row['verification_status'] == 'pass' for row in verification) else 1


if __name__ == '__main__':
    raise SystemExit(main())
