"""Exercise durable recovery and environment isolation without model imports."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research import run_allnorm_confirmation as runner


class ConfirmationRunnerTests(unittest.TestCase):
    def fixture(self, root):
        plan, out = root/'plan', root/'run'
        plan.mkdir()
        cases = [dict(case_id='sample_'+norm, image_id='sample', norm=norm, seed=457,
                      seconds_limit=165, grading_seconds_limit=180, threads=4)
                 for norm in ('linf', 'l2')]
        protocol = dict(cases=cases, environment={'CS454_L2_SEARCH':'exploration'})
        manifest = dict(sample={'path':str(root/'image.png')})
        for name in ('protocol.json', 'manifest.csv', 'source_freeze.json', 'environment.json'):
            (plan/name).write_text('{}')
        return plan, out, protocol, manifest

    def fake_completion(self, command, **kwargs):
        out = Path(command[command.index('--output')+1])
        case_id = command[command.index('--case-id')+1]
        folder = out/'cases'/case_id
        case = json.loads((folder/'case.json').read_text())
        row = runner.blank_record(case, {'path':'/tmp/fake.png'}, folder)
        row.update(status='pass', success=True, distance=1, runtime_seconds=164,
                   grading_runtime_seconds=164.1, queries=3)
        runner.write_json(folder/'result.json', row)
        return SimpleNamespace(returncode=0)

    def test_interruption_preserves_completed_trial_and_archives_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, out, protocol, manifest = self.fixture(Path(tmp))
            calls = []
            def interrupted(command, **kwargs):
                case_id = command[command.index('--case-id')+1]
                calls.append(case_id)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return self.fake_completion(command, **kwargs)
            with (patch.object(runner, 'verify_plan', return_value=(protocol, manifest)),
                  patch.object(runner.subprocess, 'run', side_effect=interrupted)):
                with self.assertRaises(KeyboardInterrupt):
                    runner.manager(plan, out)
            completed = out/'cases/sample_linf/result.json'
            original_bytes = completed.read_bytes()
            with (patch.object(runner, 'verify_plan', return_value=(protocol, manifest)),
                  patch.object(runner.subprocess, 'run', side_effect=self.fake_completion) as child):
                runner.manager(plan, out)
            self.assertEqual(child.call_count, 1)
            self.assertEqual(completed.read_bytes(), original_bytes)
            self.assertTrue(list((out/'interrupted').rglob('sample_l2')))
            self.assertEqual(json.loads((out/'progress.json').read_text())['completed'], 2)

    def test_terminal_worker_failure_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, out, protocol, manifest = self.fixture(Path(tmp))
            with (patch.object(runner, 'verify_plan', return_value=(protocol, manifest)),
                  patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=1))):
                runner.manager(plan, out)
            with (patch.object(runner, 'verify_plan', return_value=(protocol, manifest)),
                  patch.object(runner.subprocess, 'run') as child):
                runner.manager(plan, out)
            child.assert_not_called()
            for name in ('sample_linf', 'sample_l2'):
                row = json.loads((out/'cases'/name/'result.json').read_text())
                self.assertEqual(row['status'], 'worker_error')
                self.assertEqual(row['grader_score'], 999999999)

    def test_child_environment_discards_experimental_query_cap(self):
        with patch.dict(runner.os.environ, {'CS454_QUERIES':'2','CS454_L2_SEARCH':'invalid'}):
            env = runner.case_environment({'environment':{'CS454_L2_SEARCH':'exploration'}},
                                          {'threads':4,'seed':457,'seconds_limit':165})
        self.assertNotIn('CS454_QUERIES', env)
        self.assertEqual(env['CS454_L2_SEARCH'], 'exploration')
        self.assertEqual(env['CS454_SEED'], '457')


if __name__ == '__main__':
    unittest.main()
