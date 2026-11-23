"""Two disjoint four-physical-core workers; frozen attack worker is unchanged.

Scheduling amendment is explicit because the user requested parallel execution
after serial confirmation had begun. Existing terminal results are preserved.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import run_allnorm_confirmation as base


def cpu_groups(groups):
    sets = [set(map(int, group.split(','))) for group in groups]
    if len(sets) != 2 or any(len(group) != 4 for group in sets) or sets[0] & sets[1]:
        raise ValueError('need two disjoint groups of four CPUs')
    if not set.union(*sets) <= os.sched_getaffinity(0):
        raise ValueError('requested CPU is unavailable')
    physical = []
    for group in sets:
        ids = {(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip(),
                Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip())
               for cpu in group}
        if len(ids) != 4:
            raise ValueError('each worker must have four distinct physical cores')
        physical.append(ids)
    if physical[0] & physical[1]:
        raise ValueError('workers share SMT siblings of a physical core')
    return sets


def prepare_case(output, case):
    folder = output/'cases'/case['case_id']
    if folder.exists():
        if (folder/'result.json').exists():
            raise ValueError('never rerun an existing terminal result')
        recovery = output/'interrupted'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        recovery.mkdir(parents=True)
        folder.rename(recovery/folder.name)
        with (output/'recovery.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(case_id=case['case_id'],
                action='archive_nonterminal_then_resume_parallel', archive=str(recovery)))+'\n')
    folder.mkdir()
    base.write_json(folder/'case.json', case)
    return folder


def run_case(plan, output, protocol, manifest, case, cpus, slot):
    folder = output/'cases'/case['case_id']
    execution = dict(mode='parallel_two_disjoint_core_groups', worker_slot=slot,
        cpu_affinity=sorted(cpus), torch_threads=case['threads'],
        started_utc=datetime.now(timezone.utc).isoformat(),
        note='CPU cores disjoint; memory bandwidth, package power and host activity remain shared.')
    base.write_json(folder/'execution.json', execution)
    env = base.case_environment(protocol, case)
    env.update(CS454_EXECUTION_MODE='parallel_two_disjoint_core_groups',
               CS454_CPU_AFFINITY=','.join(map(str, sorted(cpus))),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    command = ['taskset', '-c', env['CS454_CPU_AFFINITY'], sys.executable,
        str(ROOT/'research/run_allnorm_confirmation.py'), '--plan', str(plan),
        '--output', str(output), '--case-id', case['case_id']]
    error = ''
    try:
        with (folder/'worker.log').open('w') as stream:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=case['grading_seconds_limit']+240)
        error = f'worker exited{child.returncode} without terminal record'
    except (subprocess.TimeoutExpired, OSError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    if not (folder/'result.json').exists():
        record = base.blank_record(case, manifest[case['image_id']], folder)
        record.update(status='worker_error', error=error, failure_stage='worker')
        base.write_json(folder/'result.json', record)
    execution['finished_utc'] = datetime.now(timezone.utc).isoformat()
    base.write_json(folder/'execution.json', execution)
    return json.loads((folder/'result.json').read_text())


def run(plan, output, amendment):
    protocol, manifest = base.verify_plan(plan, checkpoint=True)
    change = json.loads(amendment.read_text())
    for name, record in change['sources'].items():
        if base.digest(name) != record['sha256']:
            raise ValueError(f'parallel controller source changed: {name}')
    groups = cpu_groups(change['cpu_groups'])
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('protocol.json','manifest.csv','source_freeze.json','environment.json'):
            if (output/name).read_bytes() != (plan/name).read_bytes():
                raise ValueError(f'existing experiment changed: {name}')
        target = output/'execution_amendment.json'
        if target.exists() and target.read_bytes() != amendment.read_bytes():
            raise ValueError('different scheduling amendment already applied')
        if not target.exists():
            shutil.copyfile(amendment, target)
        records = base.write_aggregate(output, protocol)
        completed = {row['case_id'] for row in records}
        for case in protocol['cases']:
            if case['case_id'] in completed:
                base.record_exposure(protocol, case, output, complete=True)
        pending = [case for case in protocol['cases'] if case['case_id'] not in completed]
        print(f'Preserved{len(completed)} terminal cases; scheduling{len(pending)} remaining with2workers.', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = {}
            free = [0,1]
            while pending or active:
                while pending and free:
                    slot = free.pop(0)
                    case = pending.pop(0)
                    base.verify_plan(plan)
                    prepare_case(output, case)
                    base.record_exposure(protocol, case, output)
                    print(f'START slot{slot} cpus{sorted(groups[slot])} {case["case_id"]}', flush=True)
                    future = pool.submit(run_case, plan, output, protocol, manifest, case, groups[slot], slot)
                    active[future] = (slot, case)
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    slot, case = active.pop(future)
                    row = future.result()
                    base.record_exposure(protocol, case, output, complete=True)
                    base.write_aggregate(output, protocol)
                    print(json.dumps({key:row[key] for key in
                        ('case_id','status','distance','queries','runtime_seconds','grading_runtime_seconds')}), flush=True)
                    free.append(slot)
        print('All40 cases complete; serial and parallel execution modes are retained explicitly.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.plan, args.output.resolve(), args.amendment)


if __name__ == '__main__':
    main()
