"""Frozen, resumable, sequential Imagenette confirmation for all four attacks.

The manager imports no ML frameworks. A fresh worker persists each terminal
result atomically; recovery never repeats a completed trial or selects a seed.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIELDS = ('case_id image_id image norm seed seconds_limit grading_seconds_limit threads '
          'runtime_seconds grading_runtime_seconds distance grader_score reference_class '
          'predicted_class success status queries round_trip configuration artifact_directory '
          'engine_metadata error failure_stage').split()
PACKAGES = ('torch', 'torchvision', 'numpy', 'Pillow', 'matplotlib', 'perlin-noise')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def read_manifest(plan):
    with (plan/'manifest.csv').open() as stream:
        return {row['id']: row for row in csv.DictReader(stream)}


def freeze(plan):
    if (plan/'source_freeze.json').exists():
        raise ValueError('plan already frozen; preserve it')
    sources = ['epsilon_ball.py', 'perlin_fog.py', 'util.py', 'grade.py', 'README.md',
               'benchmark_attacks.py', 'verify_artifacts.py',
               'research/run_allnorm_confirmation.py', 'research/validate_allnorm_confirmation.py',
               str(plan/'protocol.json'), str(plan/'manifest.csv')]
    records = {}
    for name in sources:
        target = plan/'frozen_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        records[name] = dict(sha256=digest(name), snapshot=str(target))
    checkpoint = ROOT/'.runtime/torch-cache/hub/checkpoints/vgg11-8a719046.pth'
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        packages={name: version(name) for name in PACKAGES}, checkpoint_sha256=digest(checkpoint),
        cpu=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), 'unknown'),
        frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(plan/'environment.json', environment)
    records[str(plan/'environment.json')] = dict(sha256=digest(plan/'environment.json'))
    write_json(plan/'source_freeze.json', records)
    verify_plan(plan, checkpoint=True)
    print('Frozen40-case confirmation before attack outcomes.', flush=True)


def verify_plan(plan, checkpoint=False):
    for name, record in json.loads((plan/'source_freeze.json').read_text()).items():
        if digest(name) != record['sha256']:
            raise ValueError(f'frozen source changed: {name}')
    protocol = json.loads((plan/'protocol.json').read_text())
    manifest = read_manifest(plan)
    expected = {(image, norm) for image in manifest for norm in ('linf', 'l2', 'l0', 'perlin')}
    actual = {(case['image_id'], case['norm']) for case in protocol['cases']}
    if len(protocol['cases']) != 40 or actual != expected or len(manifest) != 10:
        raise ValueError('confirmation matrix is not exactly10images x4attacks')
    if len({case['case_id'] for case in protocol['cases']}) != 40:
        raise ValueError('duplicate case IDs')
    for row in manifest.values():
        if row['split'] != 'confirmation' or digest(row['path']) != row['prepared_sha256']:
            raise ValueError('confirmation manifest or image changed')
    environment = json.loads((plan/'environment.json').read_text())
    if environment['python'] != platform.python_version() or environment['packages'] != {n: version(n) for n in PACKAGES}:
        raise ValueError('runtime differs from frozen versions')
    if checkpoint and digest(ROOT/'.runtime/torch-cache/hub/checkpoints/vgg11-8a719046.pth') != environment['checkpoint_sha256']:
        raise ValueError('checkpoint differs from frozen source')
    return protocol, manifest


def case_environment(protocol, case):
    env = {k: v for k, v in os.environ.items() if not k.startswith('CS454_')}
    env.update(protocol['environment'])
    env.update(CS454_THREADS=str(case['threads']), CS454_SEED=str(case['seed']),
               CS454_SECONDS=str(case['seconds_limit']), MPLBACKEND='Agg',
               MPLCONFIGDIR='/tmp/cs454-plot-cache', TORCH_HOME=str(ROOT/'.runtime/torch-cache'))
    return env


def blank_record(case, image, folder):
    record = dict.fromkeys(FIELDS, '')
    record.update(case, image=str(Path(image['path']).resolve()), success=False, round_trip=False,
                  grader_score=999999999, status='error', artifact_directory=str(folder.resolve()))
    return record


def worker(plan, output, case_id):
    protocol, manifest = verify_plan(plan)
    case = next(c for c in protocol['cases'] if c['case_id'] == case_id)
    image_record = manifest[case['image_id']]
    folder = output/'cases'/case_id
    record = blank_record(case, image_record, folder)
    started = None
    module = None
    try:
        record['failure_stage'] = 'import_and_reference'
        import torch
        import epsilon_ball
        import perlin_fog
        import benchmark_attacks
        from util import get_image_tensor_from, get_model_most_likely_for
        from verify_artifacts import check_fog, read_tensor
        torch.set_num_threads(case['threads'])
        with torch.inference_mode():
            original = get_image_tensor_from(record['image']).contiguous()
            _, reference = get_model_most_likely_for(original)
            record['reference_class'] = reference
            if reference != int(image_record['expected_class']):
                raise ValueError('clean reference differs from the official eligibility label')
            fog = case['norm'] == 'perlin'
            module = perlin_fog if fog else epsilon_ball
            options = SimpleNamespace(images=[record['image']], norms=[case['norm']], seed=case['seed'],
                seconds=case['seconds_limit'], queries=None, threads=case['threads'],
                implementation=module.__name__, output=folder, tag='confirmation')
            benchmark_attacks.make_metadata(options, [Path(record['image'])], folder, module)
            record['configuration'] = str(folder/'config_confirmation.json')
            if fog:
                shutil.copyfile(folder/'config_confirmation.json', folder/'fog.metadata.json')
                record['configuration'] = str(folder/'fog.metadata.json')
            module.LAST_RUN.clear()
            torch.manual_seed(case['seed'])
            record['failure_stage'] = 'attack'
            function = module.adv_func if fog else getattr(module, {
                'linf':'make_Linf_adversarial_from', 'l2':'make_L2_adversarial_from',
                'l0':'make_L0_adversarial_from'}[case['norm']])
            started = time.perf_counter()
            result = function(record['image'])
            record['runtime_seconds'] = time.perf_counter()-started
            record['failure_stage'] = 'mandatory_grading'
            if fog:
                if (not isinstance(result, (list, tuple)) or not 1 <= len(result) <= 50
                        or any(not isinstance(g, (list, tuple)) or len(g) != 2
                               or type(g[0]) is not int or g[0] <= 0
                               or type(g[1]) not in (int, float) or not math.isfinite(g[1])
                               or g[1] <= 0 for g in result)):
                    raise ValueError('attack returned an invalid fog genotype')
                genes = [[o, m] for o, m in result]
                image = perlin_fog.add_noise_to_image(perlin_fog.genotype2phenotype(genes), original)
                distance = sum(m for _, m in genes)
                metric = lambda delta: distance
            else:
                image = result
                metric = benchmark_attacks.METRICS[case['norm']]
                distance = metric(image-original)
            _, predicted = get_model_most_likely_for(image)
            record['grading_runtime_seconds'] = time.perf_counter()-started
            record.update(distance=distance, predicted_class=predicted)
            if image.shape != original.shape or image.dtype != original.dtype or not torch.isfinite(image).all():
                raise ValueError('invalid returned tensor')
            record['failure_stage'] = 'artifact_export'
            if fog:
                (folder/f"{case['image_id']}_perlin.json").write_text(json.dumps(genes)+'\n')
            exported = benchmark_attacks.export_and_verify(image, original, reference, metric, folder,
                                                           f"{case['image_id']}_{case['norm']}.png")
            if exported[0] != distance or exported[2] != predicted:
                raise ValueError('export changed authoritative distance/prediction')
            if fog:
                raw = read_tensor(folder/f"raw_{case['image_id']}_perlin.png", original)
                rebuilt, cost = check_fog(folder/f"{case['image_id']}_perlin.json", original, raw)
                _, rebuilt_class = get_model_most_likely_for(rebuilt)
                if cost != distance or rebuilt_class != predicted:
                    raise ValueError('exact instructor fog regeneration changed result')
            record['round_trip'] = True
            timed_out = record['grading_runtime_seconds'] > case['grading_seconds_limit']
            record['success'] = predicted != reference and not timed_out
            record['status'] = ('grading_timeout' if timed_out else 'pass' if predicted != reference
                                else 'class_unchanged')
            record['grader_score'] = distance if record['success'] else 999999999
            record['failure_stage'] = ''
    except Exception as error:
        if started is not None and record['runtime_seconds'] == '':
            record['runtime_seconds'] = time.perf_counter()-started
        record.update(success=False, status='error', error=f'{type(error).__name__}: {error}')
    finally:
        if module is not None:
            metadata = dict(module.LAST_RUN)
            record.update(queries=metadata.get('queries', ''), engine_metadata=json.dumps(metadata, sort_keys=True))
        write_json(folder/'result.json', record)
    print(json.dumps({k: record[k] for k in ('case_id', 'status', 'distance', 'runtime_seconds')}), flush=True)


def write_aggregate(output, protocol):
    records = []
    for case in protocol['cases']:
        path = output/'cases'/case['case_id']/'result.json'
        if path.exists():
            row = json.loads(path.read_text())
            if any(row[k] != case[k] for k in case):
                raise ValueError(f'completed trial settings changed: {case["case_id"]}')
            records.append(row)
    temporary = output/'results.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output/'results.csv')
    counts = {norm: dict(completed=sum(r['norm']==norm for r in records),
                        passed=sum(r['norm']==norm and r['status']=='pass' for r in records))
              for norm in ('linf', 'l2', 'l0', 'perlin')}
    write_json(output/'progress.json', dict(completed=len(records), total=40, by_norm=counts,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        next_case=next((c['case_id'] for c in protocol['cases']
                       if c['case_id'] not in {r['case_id'] for r in records}), None)))
    return records


def record_exposure(protocol, case, output, complete=False):
    name = protocol.get('consumption_record')
    if not name:
        return
    path = Path(name)
    usage = json.loads(path.read_text())
    used = usage.setdefault('confirmation_attack_used', [])
    if case['image_id'] not in used:
        used.append(case['image_id'])
    usage['confirmation_reserved_ids'] = [image for image in usage.get('confirmation_reserved_ids', [])
                                          if image not in used]
    evaluation = usage.setdefault('allnorm_confirmation', {})
    attempted = evaluation.setdefault('attempted_cases', [])
    if case['case_id'] not in attempted:
        attempted.append(case['case_id'])
    if complete:
        completed = evaluation.setdefault('completed_cases', [])
        if case['case_id'] not in completed:
            completed.append(case['case_id'])
    evaluation.update(run=str(output), expected_cases=40,
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      interpretation='Consumed confirmation; no tuning on these outcomes or fresh-holdout reuse.')
    write_json(path, usage)


def manager(plan, output):
    protocol, manifest = verify_plan(plan, checkpoint=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return manager_locked(plan, output, protocol, manifest)


def manager_locked(plan, output, protocol, manifest):
    for name in ('protocol.json', 'manifest.csv', 'source_freeze.json', 'environment.json'):
        target = output/name
        if target.exists() and target.read_bytes() != (plan/name).read_bytes():
            raise ValueError(f'resumed experiment differs: {name}')
        if not target.exists():
            shutil.copyfile(plan/name, target)
    (output/'cases').mkdir(exist_ok=True)
    records = write_aggregate(output, protocol)
    completed = {r['case_id'] for r in records}
    for case in protocol['cases']:
        if case['case_id'] in completed:
            continue
        verify_plan(plan)
        folder = output/'cases'/case['case_id']
        if folder.exists():
            recovery = output/'interrupted'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            recovery.mkdir(parents=True)
            folder.rename(recovery/folder.name)
            with (output/'recovery.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(case_id=case['case_id'], action='archive_nonterminal_then_resume',
                                             archive=str(recovery)))+'\n')
        folder.mkdir()
        write_json(folder/'case.json', case)
        record_exposure(protocol, case, output)
        print(f'START {case["case_id"]}', flush=True)
        with (folder/'worker.log').open('w') as stream:
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--plan', str(plan),
                    '--output', str(output), '--case-id', case['case_id']], cwd=ROOT,
                    env=case_environment(protocol, case), stdout=stream, stderr=subprocess.STDOUT,
                    timeout=case['grading_seconds_limit']+240)
                error = f'worker exited{child.returncode} without terminal record'
            except subprocess.TimeoutExpired:
                error = 'worker exceeded outer watchdog (includes import/export/validation)'
        if not (folder/'result.json').exists():
            record = blank_record(case, manifest[case['image_id']], folder)
            record.update(status='worker_error', error=error, failure_stage='worker')
            write_json(folder/'result.json', record)
        records = write_aggregate(output, protocol)
        record_exposure(protocol, case, output, complete=True)
        row = next(r for r in records if r['case_id']==case['case_id'])
        print(json.dumps({k:row[k] for k in ('case_id','status','distance','queries','runtime_seconds','grading_runtime_seconds')}), flush=True)
    print('Completed all40 confirmation cases; terminal failures retained.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--case-id')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.freeze:
        freeze(args.plan)
    elif args.output is None:
        parser.error('--output required for execution')
    elif args.case_id:
        worker(args.plan, args.output.resolve(), args.case_id)
    else:
        manager(args.plan, args.output.resolve())


if __name__ == '__main__':
    main()
