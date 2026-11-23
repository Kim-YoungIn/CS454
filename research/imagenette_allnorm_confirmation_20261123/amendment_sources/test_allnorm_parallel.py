"""Check real scheduler concurrency and orphan recovery without ML imports."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from research import run_allnorm_parallel as scheduler


class ParallelConfirmationTests(unittest.TestCase):
    def test_two_workers_preserve_terminal_case_and_complete_distinct_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, out = root/'plan', root/'run'
            plan.mkdir(); (out/'cases').mkdir(parents=True)
            for name in ('protocol.json','manifest.csv','source_freeze.json','environment.json'):
                (plan/name).write_text('{}'); (out/name).write_text('{}')
            cases = [dict(case_id=f'image{i}_linf', image_id=f'image{i}', norm='linf', seed=457,
                          threads=4, seconds_limit=165, grading_seconds_limit=180) for i in range(5)]
            protocol = dict(cases=cases, environment={})
            manifest = {f'image{i}':dict(path='/tmp/not_loaded.png') for i in range(5)}
            amendment = root/'amendment.json'
            amendment.write_text(json.dumps(dict(sources={},cpu_groups=['0,4,8,12','2,6,10,14'])))
            folder = out/'cases'/cases[0]['case_id']; folder.mkdir()
            first = scheduler.base.blank_record(cases[0],manifest['image0'],folder)
            first.update(status='pass',success=True,queries=100)
            scheduler.base.write_json(folder/'result.json',first)
            original = (folder/'result.json').read_bytes()
            barrier, lock = threading.Barrier(2), threading.Lock()
            state = dict(active=0, maximum=0, seen=[])
            def fake(plan,output,protocol,manifest,case,cpus,slot):
                with lock:
                    state['active']+=1
                    state['maximum']=max(state['maximum'],state['active'])
                    state['seen'].append(case['case_id'])
                barrier.wait(timeout=3)
                folder=output/'cases'/case['case_id']
                row=scheduler.base.blank_record(case,manifest[case['image_id']],folder)
                row.update(status='pass',success=True,distance=1,runtime_seconds=164,
                           grading_runtime_seconds=164.1,queries=500)
                scheduler.base.write_json(folder/'result.json',row)
                with lock:state['active']-=1
                return row
            with (patch.object(scheduler.base,'verify_plan',return_value=(protocol,manifest)),
                  patch.object(scheduler,'run_case',side_effect=fake)):
                scheduler.run(plan,out,amendment)
            self.assertEqual(state['maximum'],2)
            self.assertEqual(set(state['seen']),{c['case_id'] for c in cases[1:]})
            self.assertEqual(len(state['seen']),4)
            self.assertEqual((folder/'result.json').read_bytes(),original)
            self.assertEqual(json.loads((out/'progress.json').read_text())['completed'],5)

    def test_logically_disjoint_groups_cannot_share_physical_core(self):
        with self.assertRaisesRegex(ValueError,'SMT siblings'):
            scheduler.cpu_groups(['0,4,8,12','1,6,10,14'])


if __name__ == '__main__':
    unittest.main()
