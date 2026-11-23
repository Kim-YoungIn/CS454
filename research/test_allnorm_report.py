"""Standard-library-only reporting tests; no torch import or classifier calls."""
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.validate_allnorm_confirmation import NORMS, load_run, summarize, write_outputs


def fixture():
    cases, records = {}, {}
    for index in range(2):
        for norm in NORMS:
            case_id = f'confirmation_{index}_{norm}'
            case = dict(case_id=case_id, image_id=f'confirmation_{index}', norm=norm,
                        seed=457, threads=4, seconds_limit=270 if norm == 'perlin' else 165,
                        grading_seconds_limit=300 if norm == 'perlin' else 180)
            cases[case_id] = case
            records[case_id] = dict(case, image=f'/tmp/image{index}.png',
                status='pass', success='True', round_trip='True', distance=str(4**index),
                runtime_seconds='160', grading_runtime_seconds='161', queries='900',
                reference_class=str(index), predicted_class='20', error='')
    return cases, records


class AllnormReportingTests(unittest.TestCase):
    def test_report_imports_no_classifier_or_framework(self):
        # Other suites legitimately load the classifier. Check this module's
        # lazy-import contract in a clean interpreter, independent of order.
        child = subprocess.run([sys.executable, '-c',
            "import sys; import research.validate_allnorm_confirmation; "
            "assert 'util' not in sys.modules; assert 'torch' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr[-2000:])

    def test_pending_and_failed_cases_stay_in_denominator(self):
        cases, records = fixture()
        records.pop('confirmation_1_linf')
        records['confirmation_1_l2'].update(status='error', success='False', distance='', error='attack failed')
        report = summarize(cases, records)
        self.assertEqual(report['linf']['expected_cases'], 2)
        self.assertEqual(report['linf']['success_rate'], .5)
        self.assertEqual(report['linf']['statuses']['pending'], 1)
        self.assertIsNone(report['linf']['failure_inclusive_grader_score_mean'])
        self.assertEqual(report['l2']['success_rate'], .5)
        self.assertEqual(report['l2']['successful_distance_geomean'], 1)
        self.assertEqual(report['l2']['failure_inclusive_grader_score_mean'], 500000000)

    def test_grading_limit_is_authoritative_and_optimizer_overrun_is_diagnostic(self):
        cases, records = fixture()
        records['confirmation_0_l0']['runtime_seconds'] = '166'
        records['confirmation_0_l0']['grading_runtime_seconds'] = '167'
        records['confirmation_1_l0']['grading_runtime_seconds'] = '181'
        report = summarize(cases, records)['l0']
        self.assertEqual(report['success_count'], 1)
        self.assertEqual(report['planned_budget_overruns'], 1)
        self.assertEqual(report['grading_timeouts'], 1)
        self.assertEqual(report['failure_inclusive_grader_score_mean'], 500000000)

    def test_independent_failure_overrides_source_success(self):
        cases, records = fixture()
        verification = [dict(case_id=case_id, verification_status='pass', measured_distance=2)
                        for case_id in cases]
        verification[0].update(verification_status='verification_failure', measured_distance='')
        report = summarize(cases, records, verification)['linf']
        self.assertEqual(report['reported_successes'], 2)
        self.assertEqual(report['independently_verified_successes'], 1)
        self.assertEqual(report['successful_distance_median'], 2)
        self.assertEqual(report['failure_inclusive_grader_score_mean'], 500000000.5)

    def test_nonfinite_or_invalid_distance_is_not_a_reported_success(self):
        cases, records = fixture()
        records['confirmation_0_perlin']['distance'] = 'nan'
        records['confirmation_1_perlin']['distance'] = '-1'
        report = summarize(cases, records)['perlin']
        self.assertEqual(report['success_count'], 0)
        self.assertIsNone(report['successful_distance_geomean'])

    def test_conditional_statistics_and_complete_failure_penalty(self):
        cases, records = fixture()
        report = summarize(cases, records)['l2']
        self.assertEqual(report['successful_distance_min'], 1)
        self.assertEqual(report['successful_distance_median'], 2.5)
        self.assertEqual(report['successful_distance_geomean'], 2)
        self.assertEqual(report['failure_inclusive_grader_score_mean'], 2.5)

    def test_context_mismatch_and_duplicate_results_are_rejected(self):
        cases, records = fixture()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = [dict(id=f'confirmation_{index}', split='confirmation', path=f'/tmp/image{index}.png',
                             expected_class=index, expected_name=f'class{index}', prepared_sha256='unused')
                        for index in range(2)]
            (folder/'protocol.json').write_text(json.dumps({'expected_cases': 8, 'cases': list(cases.values())}))
            with (folder/'manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
            def save(rows):
                with (folder/'results.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            save(list(records.values()))
            self.assertEqual(len(load_run(folder)[3]), 8)
            wrong = copy.deepcopy(list(records.values()))
            wrong[0]['seed'] = 999
            save(wrong)
            with self.assertRaisesRegex(ValueError, 'context'):
                load_run(folder)
            save(list(records.values())+[next(iter(records.values()))])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_run(folder)


if __name__ == '__main__':
    unittest.main()
