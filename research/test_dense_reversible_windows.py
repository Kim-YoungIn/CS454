"""Mechanism tests with synthetic oracles; no VGG or util import is performed."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_isolated():
    # Exercise the real source while excluding the instructor module's eager
    # pretrained-model initialization. This does not modify source files.
    tree = ast.parse((ROOT/'epsilon_ball.py').read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == 'util')
                 and not isinstance(node, ast.If)]
    base = types.ModuleType('epsilon_ball_synthetic')
    exec(compile(tree, str(ROOT/'epsilon_ball.py'), 'exec'), base.__dict__)
    base.get_image_tensor_from = None
    base.get_model_predictions_for = None
    spec = importlib.util.spec_from_file_location(
        'reversible_windows_synthetic', ROOT/'research/dense_reversible_windows.py')
    candidate = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, epsilon_ball=base):
        spec.loader.exec_module(candidate)
    return base, candidate


base, candidate = load_isolated()


class ReversibleWindowTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.environment = patch.dict(os.environ, CS454_SEED='454', CS454_QUERIES='1000',
                                     CS454_SECONDS='165', CS454_THREADS='1',
                                     CS454_BOUNDARY_MIN_GENERATIONS='512')
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        torch.set_num_threads(self.threads)

    def test_scheduler_explores_both_and_limits_consecutive_winner_windows(self):
        select = candidate.ReversibleWindows.choose_operator
        self.assertEqual(select({'local': None, 'global': None}, None, 0), 'local')
        self.assertEqual(select({'local': 10., 'global': None}, 'local', 1), 'global')
        for winner in ('local', 'global'):
            rates = {'local': 0., 'global': 0.}
            rates[winner] = 1.
            self.assertEqual(select(rates, winner, 1), winner)
            self.assertNotEqual(select(rates, winner, 2), winner)

    @torch.inference_mode()
    def test_exact_prefix_budget_archive_and_deterministic_replay(self):
        image = torch.zeros(1, 3, 20, 20)
        proposals = []
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            proposals.append(images.clone())
            own = (5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
            return torch.stack((own, 1-own)).reshape(1, 2)
        with (patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
            control = base._ExplorationBoundarySearch('control', 'l2')
            control.run()
            control_proposals, proposals = proposals, []
            trial = candidate.ReversibleWindows('unseen-path', 'l2')
            result = trial.run()
            trial_proposals, proposals = proposals, []
            second = candidate.ReversibleWindows('different-path', 'l2')
            second_result = second.run()
        event = next(e for e in trial.trace if e['event'] == 'boundary_start')
        self.assertEqual(event['generation'], 512)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(control_proposals[:event['query']], trial_proposals[:event['query']])))
        self.assertEqual(trial.queries, 1000)
        self.assertEqual(len(trial_proposals), trial.queries)
        self.assertTrue(torch.equal(result, second_result))
        self.assertEqual(trial.windows, second.windows)
        self.assertTrue(any(torch.equal(result, p) for p in trial_proposals))
        self.assertEqual(float((result-image).square().sum()), trial.best_distance)
        self.assertTrue(all(a[1] > b[1] for a, b in zip(trial.history, trial.history[1:])))
        self.assertEqual({w['operator'] for w in trial.windows}, {'local', 'global'})
        for index, window in enumerate(trial.windows):
            self.assertGreater(window['calls'], 0)
            self.assertLessEqual(window['calls'], 96)
            self.assertEqual(window['end_query']-window['start_query'], window['calls'])
            self.assertGreaterEqual(window['log_gain'], 0.)
            if index >= 2:
                self.assertGreater(len({w['operator'] for w in trial.windows[index-2:index+1]}), 1)

    @torch.inference_mode()
    def test_rebase_invalidates_fitness_and_failed_probe_keeps_feasible_parent(self):
        image = torch.zeros(1, 3, 2, 2)
        def oracle(images):
            own = (5.9-images.norm()).sigmoid()
            return torch.stack((own, 1-own)).reshape(1, 2)
        with (patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
            search = candidate.ReversibleWindows('unseen', 'l2')
            search.evaluate(torch.ones_like(image)*(6/math.sqrt(image.numel())))
            search.probe_calls = search.probe_successes = 0
            search.probe_trace = []
            search.mate_rescales = search.mate_unused = 0
            state = dict(radius=0., anchor=math.inf, pool=[], feasible=[search.best-image],
                         stage=0, stalled=0, alpha=.25, fraction=.99, queued=None)
            with patch.object(search, 'l2_energy_mutation', side_effect=lambda p, *args: p):
                search.local_step(state, 1, search.queries+96)
            self.assertEqual(search.probe_calls, 1)
            self.assertEqual(search.probe_successes, 0)
            self.assertGreater(search.best.norm(), 5.9)
            self.assertTrue(torch.equal(state['feasible'][0], search.best-image))
            before_query = search.queries
            global_state = dict(radius=99., anchor=math.inf,
                                pool=[(-999., torch.ones_like(image))],
                                stage=44, stalled=12, warm=False)
            search.global_step(global_state, 2)
            self.assertEqual(search.queries, before_query+1)
            self.assertGreater(global_state['pool'][0][0], 0.)
            self.assertEqual(global_state['stage'], 1)
            self.assertAlmostEqual(global_state['radius'], .8*math.sqrt(search.best_distance))

    @torch.inference_mode()
    def test_infeasible_oracle_exhaustion_keeps_failure_explicit(self):
        image = torch.zeros(1, 3, 8, 8)
        with (patch.dict(os.environ, CS454_QUERIES='31'),
              patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for',
                           return_value=torch.tensor([[.75, .25]]))):
            search = candidate.ReversibleWindows('unknown', 'l2')
            with self.assertRaisesRegex(RuntimeError, '31 queries without misclassification'):
                search.run()
        self.assertIsNone(search.best)
        self.assertEqual(search.queries, 31)
        self.assertEqual(search.history, [])

    @torch.inference_mode()
    def test_probe_obeys_one_remaining_window_call(self):
        image = torch.zeros(1, 3, 2, 2)
        def oracle(images):
            own = (1-images.norm()).sigmoid()
            return torch.stack((own, 1-own)).reshape(1, 2)
        with (patch.object(base, 'get_image_tensor_from', return_value=image),
              patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
            search = candidate.ReversibleWindows('unseen', 'l2')
            search.evaluate(torch.ones_like(image))
            search.probe_calls = search.probe_successes = 0
            search.probe_trace = []
            search.mate_rescales = search.mate_unused = 0
            state = dict(radius=0., anchor=math.inf, pool=[], feasible=[search.best-image],
                         stage=0, stalled=0, alpha=.25, fraction=.99, queued=None)
            end_query = search.queries+1
            with patch.object(search, 'l2_energy_mutation', side_effect=lambda p, *args: p):
                search.local_step(state, 1, end_query)
            self.assertEqual(search.queries, end_query)
            self.assertEqual(search.probe_calls, 0)


if __name__ == '__main__':
    unittest.main()
