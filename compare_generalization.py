"""Summarize complete paired attack matrices without dropping failed cases."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def passed(row):
    return row['status'] == 'pass' and row['success'] == 'True' and row['round_trip'] == 'True'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; use a fresh summary directory')
    trials = defaultdict(dict)
    for directory in args.directories:
        with (directory/'results.csv').open() as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[name] for name in ('split','image','norm','seed','queries_limit','seconds_limit'))
                if row['method'] in trials[key]:
                    raise ValueError(f'duplicate method for pair: {key}')
                trials[key][row['method']] = row
    pairs = []
    for key, methods in trials.items():
        if set(methods) != {'control','candidate'}:
            raise ValueError(f'incomplete pair: {key}')
        control, candidate = methods['control'], methods['candidate']
        if control['reference_class'] != candidate['reference_class']:
            raise ValueError(f'clean reference changed: {key}')
        left, right = passed(control), passed(candidate)
        cdist = float(control['distance']) if left else None
        ndist = float(candidate['distance']) if right else None
        ratio = ndist/cdist if left and right and cdist > 0 else None
        verdict = ('win' if right else 'loss') if left != right else 'both_fail' if not left else (
            'tie' if math.isclose(cdist, ndist, rel_tol=1e-6, abs_tol=1e-9) else 'win' if ndist < cdist else 'loss')
        engine = json.loads(candidate['engine_metadata'] or '{}')
        activated = (any(event.get('event') == 'boundary_start' for event in engine.get('dense_trace', []))
                     if candidate['norm'] == 'l2' else any(chunk.get('origin',{}).get('kind') == 'local_support_tail' for chunk in engine.get('chunks', [])))
        pairs.append(dict(split=key[0],image=Path(key[1]).stem,norm=key[2],seed=key[3],queries_limit=key[4],seconds_limit=key[5],
                          control_success=left,candidate_success=right,control_distance=cdist,candidate_distance=ndist,
                          ratio=ratio,verdict=verdict,activated=activated,control_queries=control['queries'],
                          candidate_queries=candidate['queries'],control_seconds=control['runtime_seconds'],
                          candidate_seconds=candidate['runtime_seconds']))
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair['norm'],pair['split'],pair['queries_limit'],pair['seconds_limit'])].append(pair)
    summaries = []
    for (norm,split,queries,seconds), group in groups.items():
        ratios = [row['ratio'] for row in group if row['ratio'] is not None]
        summaries.append(dict(norm=norm,split=split,queries_limit=queries,seconds_limit=seconds,pairs=len(group),
                              control_successes=sum(row['control_success'] for row in group),
                              candidate_successes=sum(row['candidate_success'] for row in group),
                              success_regressions=sum(row['control_success'] and not row['candidate_success'] for row in group),
                              wins=sum(row['verdict']=='win' for row in group),ties=sum(row['verdict']=='tie' for row in group),
                              losses=sum(row['verdict']=='loss' for row in group),both_failures=sum(row['verdict']=='both_fail' for row in group),
                              activated_pairs=sum(row['activated'] for row in group),common_successes=len(ratios),
                              geometric_mean_ratio=math.exp(sum(math.log(value) for value in ratios)/len(ratios)) if ratios else None,
                              worst_ratio=max(ratios) if ratios else None))
    args.output.mkdir(parents=True)
    with (args.output/'pairs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
