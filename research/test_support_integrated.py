"""Verify the self-contained support GA preserves the research trajectory."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import epsilon_ball as production
from research import epsilon_support_integrated as integrated
from research.elite_session3_online import OnlineSupportGA


class SupportPortTests(unittest.TestCase):
    def test_original_search_is_unchanged(self):
        def original_search(path):
            tree = ast.parse(Path(path).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_Search'))
        self.assertEqual(original_search(production.__file__), original_search(integrated.__file__))

    @torch.inference_mode()
    def test_complete_discovery_pruning_and_support_trajectory(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        original = torch.zeros(1, 3, 16, 16)
        inputs = []
        def predictions(images):
            inputs.append(images.clone())
            count = images.flatten(1).ne(0).sum(1)
            own = torch.where(count >= 3, .25, .75)
            return torch.stack([own, 1-own], dim=1)
        config = {'CS454_QUERIES': '1200', 'CS454_SECONDS': '165', 'CS454_SEED': '454'}
        try:
            with (patch.dict(os.environ, config),
                  patch.object(production, 'get_image_tensor_from', return_value=original),
                  patch.object(production, 'get_model_predictions_for', side_effect=predictions)):
                research = OnlineSupportGA('unseen-image', 'l0')
                expected = research.run()
            expected_inputs, inputs = inputs, []
            with (patch.dict(os.environ, config),
                  patch.object(integrated, 'get_image_tensor_from', return_value=original),
                  patch.object(integrated, 'get_model_predictions_for', side_effect=predictions)):
                port = integrated._ParentalSparseSearch('unseen-image', 'l0')
                actual = port.run()
            self.assertEqual(len(inputs), len(expected_inputs))
            for before, after in zip(expected_inputs, inputs):
                self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(research.history, port.history)
            self.assertEqual(len(research.chunks), len(port.chunks))
            for before, after in zip(research.chunks, port.chunks):
                self.assertEqual(before, {key: after[key] for key in before})
            for field in ('prune_evaluations', 'cleanup_evaluations', 'cleanup_passes'):
                self.assertEqual(integrated.LAST_RUN[field], sum(c[field] for c in port.chunks))
            self.assertEqual(research.selected_origin, port.selected_origin)
            self.assertTrue(any(c['origin']['kind'] == 'support_ga' for c in port.chunks))
            counts = integrated.LAST_RUN
            self.assertEqual(1 + sum(counts[key] for key in (
                'discovery_evaluations', 'prune_evaluations', 'cleanup_evaluations',
                'support_evaluations')), port.queries)
            self.assertEqual(port.queries, 1200)
            self.assertEqual(port.best_distance, 3)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
