"""Resume a frozen interrupted matrix without repeating completed model trials.

Completed CSV rows and original snapshots remain intact. Partial case directories
are archived; an orphaned completed case is recovered rather than rerun.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import benchmark_attacks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('development','confirmation'),required=True)
    parser.add_argument('--metric',choices=('l2','l0'),required=True)
    args=parser.parse_args()
    protocol=json.loads((args.dataset/'protocol.json').read_text())
    freeze=json.loads((args.dataset/'source_freeze.json').read_text())
    for name,record in freeze.items():
        if args.metric in record.get('metrics',('l2','l0')) and hashlib.sha256(Path(name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'frozen source changed: {name}')
    for name in ('protocol.json','manifest.csv','source_freeze.json'):
        if (args.output/name).read_bytes()!=(args.dataset/name).read_bytes():
            raise ValueError(f'original experiment differs: {name}')
    settings=protocol['primary' if args.metric=='l2' else 'secondary']
    seed=settings[f'{args.split}_seed']
    with (args.dataset/'manifest.csv').open() as stream:
        images=[row for row in csv.DictReader(stream) if row['split']==args.split]
    expected={str(Path(row['path']).resolve()):row for row in images}
    for row in images:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['prepared_sha256']:
            raise ValueError('image checksum changed')
    results_path = args.output / 'results.csv'
    records = []
    if results_path.exists():
        with results_path.open() as stream:
            records = list(csv.DictReader(stream))
    initial_rows=len(records)
    done=set()
    for row in records:
        key=(row['image'],row['method'])
        if key in done or row['image'] not in expected or row['method'] not in ('control','candidate'):
            raise ValueError('duplicate or unexpected existing trial')
        if (row['implementation']!=settings[row['method']] or int(row['seed'])!=seed or row['norm']!=args.metric
            or float(row['seconds_limit'])!=settings['seconds'] or (row['queries_limit'] or None)!=(
                str(settings['queries']) if settings['queries'] is not None else None)
            or int(row['threads'])!=settings['threads']):raise ValueError('existing trial settings mismatch')
        if 'expected_class' in expected[row['image']] and int(row['reference_class'])!=int(expected[row['image']]['expected_class']):
            raise ValueError('labelled eligibility/reference changed')
        done.add(key)
    recovery=args.output/('recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    recovery.mkdir()
    if results_path.exists():
        shutil.copy2(results_path, recovery/'results_before.csv')
    (recovery/'resume_snapshot.py').write_bytes(Path(__file__).read_bytes())
    events=[]
    os.environ.update(CS454_L0_STYLE='parental',CS454_L2_STYLE='energy',CS454_DISCOVERY_REFERENCE='900',
        CS454_DISCOVERY_FRACTION='.55',CS454_BOUNDARY_SUCCESSES='3',CS454_BOUNDARY_FRACTION='.99',CS454_BOUNDARY_ALPHA='.25')
    os.environ.pop('CS454_BOUNDARY_MIN_GENERATIONS',None)
    for name,value in settings.get('environment',{}).items():
        if not name.startswith('CS454_'):raise ValueError('invalid task environment')
        os.environ[name]=str(value)
    for index,image in enumerate(images):
        order=('control','candidate') if index%2==0 else ('candidate','control')
        for kind in order:
            key=(str(Path(image['path']).resolve()),kind)
            if key in done:continue
            output=args.output/f"{image['id']}_{kind}"
            log=args.output/f"{image['id']}_{kind}.log"
            result=None
            if output.exists():
                saved=output/'results.csv'
                rows=list(csv.DictReader(saved.open())) if saved.exists() else []
                if len(rows)==1:
                    result=rows[0]
                    if result['image']!=key[0] or int(result['seed'])!=seed or result['norm']!=args.metric:
                        raise ValueError('orphaned trial does not match expected case')
                    config = json.loads((output/f'config_{args.split}_{kind}.json').read_text())
                    options = config['options']
                    if (options['implementation'] != settings[kind]
                            or options['threads'] != settings['threads']
                            or options['queries'] != settings['queries']
                            or options['seconds'] != settings['seconds']):
                        raise ValueError('orphaned trial settings mismatch')
                    exit_code=0 if result['success']=='True' else 1
                    events.append(dict(case=output.name,action='recovered_completed_orphan'))
                elif rows:raise ValueError('multiple rows in one case directory')
                else:
                    output.rename(recovery/output.name)
                    if log.exists():log.rename(recovery/log.name)
                    events.append(dict(case=output.name,action='archived_partial_case_then_restarted'))
            if result is None:
                options=SimpleNamespace(images=[image['path']],norms=[args.metric],seed=seed,
                    seconds=settings['seconds'],queries=settings['queries'],threads=settings['threads'],
                    implementation=settings[kind],output=output,tag=f'{args.split}_{kind}')
                with log.open('w') as stream:
                    with redirect_stdout(stream):exit_code=benchmark_attacks.run(options)
                result=next(csv.DictReader((output/'results.csv').open()))
            if 'expected_class' in image and int(result['reference_class'])!=int(image['expected_class']):
                raise ValueError('restored runtime no longer satisfies labelled eligibility')
            result.update(split=args.split,method=kind,implementation=settings[kind],experiment_exit=exit_code,
                configuration=str(output/f'config_{args.split}_{kind}.json'))
            records.append(result);done.add(key)
            temporary=args.output/'results.resuming.csv'
            with temporary.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            temporary.replace(results_path)
            (recovery/'events.json').write_text(json.dumps(events,indent=2)+'\n')
            print(json.dumps({key:result[key] for key in ('image','method','distance','queries','status')}),flush=True)
    print(f'Completed matrix: {len(records)} rows; preserved all {initial_rows} existing rows and added {len(records)-initial_rows}.',flush=True)


if __name__=='__main__':main()
