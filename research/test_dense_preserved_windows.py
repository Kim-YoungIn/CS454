"""State-continuation tests: AST-isolated production, synthetic forwards only."""
import importlib.util
import math
import os
import sys
import unittest
from unittest.mock import patch

import torch
from research.test_dense_reversible_windows import ROOT, base, candidate as first


spec = importlib.util.spec_from_file_location(
    'preserved_windows_synthetic', ROOT/'research/dense_preserved_windows.py')
candidate = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, epsilon_ball=base,
                **{'research.dense_reversible_windows': first}):
    spec.loader.exec_module(candidate)


class PreservedWindowTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.environment = patch.dict(os.environ, CS454_SEED='454', CS454_QUERIES='1000',
                                     CS454_SECONDS='165', CS454_THREADS='1',
                                     CS454_BOUNDARY_MIN_GENERATIONS='512',
                                     CS454_BOUNDARY_SUCCESSES='3')
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        torch.set_num_threads(self.threads)

    @staticmethod
    def oracle(images):
        own = (5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
        return torch.stack((own, 1-own)).reshape(1, 2)

    @torch.inference_mode()
    def test_pending_coarse_stream_replays_exactly_across_external_valid_gain(self):
        image = torch.zeros(1, 3, 20, 20)
        calls = []
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(images.clone())
            return self.oracle(images)
        with (patch.dict(os.environ, CS454_BOUNDARY_MIN_GENERATIONS='1', CS454_BOUNDARY_SUCCESSES='1'),
              patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
            control = base._Search('control', 'l2')
            control.run()
            reference, calls = calls, []
            trial = candidate.PreservedWindows('unknown', 'l2')
            with patch.object(trial, 'boundary', return_value=None):
                trial.dense()
            start = trial.queries
            state = trial.coarse_state
            self.assertEqual(trial.coarse_capture['pending'], 'contract')
            self.assertEqual(state['pending'], 'warm')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(calls, reference[:start])))
            original_radius = state['radius']
            # A real synthetic-oracle-feasible local improvement leaves the
            # pending24-radius coarse target useful. Do not restart that target.
            trial.evaluate(trial.rescale(trial.best-image, 28.))
            self.assertLess(trial.best_distance, state['seen_best'])
            self.assertGreater(trial.best_distance, state['radius']**2)
            # Unrelated local draws and proposal phases cannot disturb global.
            trial.rng = torch.Generator().manual_seed(123)
            trial.rand(117)
            calls = []
            trial.state_trace = []
            kinds = []
            for _ in range(500):
                kinds.append(trial.coarse_step(state))
            self.assertEqual(state['reanchors'], 0)
            self.assertEqual(len(calls), 500)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(calls, reference[start:start+500])))
            self.assertEqual(kinds[0], 'warm')
            self.assertGreater(kinds.count('warm'), 1)
            self.assertEqual(state['generation'], 1+kinds.count('main'))
            self.assertEqual(state['main_calls'], kinds.count('main'))
            self.assertEqual(state['warm_calls'], kinds.count('warm'))
            self.assertEqual(state['fresh_calls'], kinds.count('fresh'))
            self.assertLess(state['radius'], original_radius)

    @torch.inference_mode()
    def test_warm_success_does_not_advance_clock_or_contract_and_reanchor_requeries(self):
        image = torch.zeros(1, 3, 20, 20)
        with (patch.dict(os.environ, CS454_BOUNDARY_MIN_GENERATIONS='1', CS454_BOUNDARY_SUCCESSES='1'),
              patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=self.oracle)):
            trial = candidate.PreservedWindows('unknown', 'l2')
            with patch.object(trial, 'boundary', return_value=None):
                trial.dense()
            trial.state_trace = []
            state = trial.coarse_state
            generation, radius = state['generation'], state['radius']
            self.assertEqual(trial.coarse_step(state), 'warm')
            self.assertLess(state['pool'][0][0], 0.)
            self.assertEqual(state['radius'], radius)
            self.assertEqual(state['generation'], generation)
            self.assertEqual(trial.coarse_step(state), 'main')
            self.assertEqual(state['generation'], generation+1)
            # A better feasible image now supersedes the pending absolute target.
            local_direction = torch.zeros_like(image)
            local_direction[..., :10, :10] = 1
            trial.evaluate(trial.rescale(local_direction, 6.))
            state['pool'] = [(-999., torch.ones_like(image))]
            self.assertLess(trial.best_distance, state['radius']**2)
            before = trial.queries
            generation = state['generation']
            self.assertEqual(trial.coarse_step(state), 'warm')
            self.assertEqual(trial.queries, before+1)
            self.assertEqual(state['reanchors'], 1)
            self.assertEqual(state['generation'], generation)
            self.assertEqual(len(state['pool']), 1)
            self.assertGreater(state['pool'][0][0], -999.)
            self.assertEqual(trial.state_trace[-1]['event'], 'coarse_reanchor')

    @torch.inference_mode()
    def test_fresh_extra_query_does_not_advance_main_clock_or_stage(self):
        image = torch.zeros(1, 3, 20, 20)
        with (patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=self.oracle)):
            trial = candidate.PreservedWindows('unknown', 'l2')
            trial.evaluate(trial.initial(30.))
            trial.capture(3., [(0., torch.zeros_like(image))], 7, 119, 123)
            trial.state_trace = []
            state = trial.coarse_state
            self.assertEqual(trial.coarse_step(state), 'main')
            self.assertEqual(state['pending'], 'fresh')
            self.assertEqual(state['generation'], 124)
            self.assertEqual(state['stage'], 8)
            self.assertEqual(trial.coarse_step(state), 'fresh')
            self.assertEqual(state['generation'], 124)
            self.assertEqual(state['stage'], 8)
            self.assertEqual(state['fresh_calls'], 1)
            self.assertLessEqual(len(state['pool']), 3)

    @torch.inference_mode()
    def test_full_default_prefix_determinism_and_bounded_windows(self):
        image = torch.zeros(1, 3, 20, 20)
        calls = []
        def oracle(images):
            calls.append(images.clone())
            return self.oracle(images)
        with (patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
            control = base._ExplorationBoundarySearch('control', 'l2')
            control.run()
            reference, calls = calls, []
            trial = candidate.PreservedWindows('unknown', 'l2')
            actual = trial.run()
            actual_calls, calls = calls, []
            second = candidate.PreservedWindows('other-name', 'l2')
            repeated = second.run()
        start = trial.coarse_capture['query']
        self.assertEqual(trial.coarse_capture['generation'], 512)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[:start], actual_calls[:start])))
        self.assertEqual(trial.queries, 1000)
        self.assertEqual(len(actual_calls), 1000)
        self.assertTrue(torch.equal(actual, repeated))
        self.assertEqual(trial.windows, second.windows)
        self.assertEqual(trial.state_trace, second.state_trace)
        self.assertEqual(trial.windows[0]['operator'], 'local')
        self.assertEqual(trial.windows[1]['operator'], 'global')
        self.assertTrue(all(0 < w['calls'] <= 96 for w in trial.windows))
        self.assertTrue(any(torch.equal(actual, proposal) for proposal in actual_calls))
        self.assertEqual(float((actual-image).square().sum()), trial.best_distance)
        for window in trial.windows:
            if window['operator'] == 'local':
                self.assertEqual(window['global_main_proposals'], 0)
        self.assertEqual(sum(w['global_main_proposals'] for w in trial.windows),
                         trial.coarse_state['main_calls'])


if __name__ == '__main__':
    unittest.main()
