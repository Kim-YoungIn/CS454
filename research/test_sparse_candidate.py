"""Synthetic contract tests for compact rectangle L0 evolution; no VGG forwards."""
import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball
from research.sparse_candidate import RectangleSearch, attack, LAST_RUN


class RectangleTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.image = torch.zeros(1, 19, 23, 3).permute(0, 3, 1, 2)
        self.assertFalse(self.image.is_contiguous())
        self.environment = patch.dict(os.environ, {'CS454_SEED': '454', 'CS454_SECONDS': '165',
                                                  'CS454_QUERIES': '100', 'CS454_THREADS': '1'})
        self.environment.start()
        self.loader = patch.object(epsilon_ball, 'get_image_tensor_from', side_effect=lambda _: self.image.clone())
        self.loader.start()
        self.oracle = patch.object(epsilon_ball, 'get_model_predictions_for', side_effect=self.probabilities)
        self.oracle.start()

    def tearDown(self):
        self.oracle.stop()
        self.loader.stop()
        self.environment.stop()
        torch.set_num_threads(self.threads)

    def probabilities(self, image):
        self.assertFalse(torch.is_grad_enabled())
        different = image.ne(self.image).flatten(1).any(1)
        own = torch.where(different, .1, .9)
        return torch.stack([own, 1 - own], 1)

    @torch.inference_mode()
    def test_render_budget_and_range(self):
        search = RectangleSearch('held-out.jpg')
        for budget in (0, 1, 2, 7, 23, 101, self.image.numel(), self.image.numel() * 3):
            for genome in search.random_genomes(12):
                delta = search.decode(genome, budget)
                self.assertLessEqual(int(delta.count_nonzero()), min(budget, self.image.numel()))
                self.assertLessEqual(float((search.x + delta).abs().max()), 3)
                untouched = delta.eq(0)
                self.assertTrue(torch.equal((search.x + delta)[untouched], search.x[untouched]))
                self.assertTrue(torch.isfinite(delta).all())

    @torch.inference_mode()
    def test_mutation_finite_bounded_and_parents_unchanged(self):
        search = RectangleSearch('held-out.jpg')
        search.reset_population()
        search.fitness = torch.arange(search.POPULATION).float()
        original = search.population.clone()
        for generation in range(48):
            child = search.differential_child(generation % search.POPULATION, generation)
            self.assertEqual(child.shape, (2, 7))
            self.assertTrue(torch.isfinite(child).all())
            self.assertGreaterEqual(float(child.min()), 0)
            self.assertLessEqual(float(child.max()), 1)
        self.assertTrue(torch.equal(search.population, original))

    @torch.inference_mode()
    def test_public_success_and_reproducible_minimal_support(self):
        first = attack('not-in-reference-table.jpg')
        record = dict(LAST_RUN)
        second = attack('different-spelling.jpg')
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(record['history'], LAST_RUN['history'])
        self.assertTrue(LAST_RUN['success'])
        self.assertEqual(LAST_RUN['reference'], 0)
        self.assertEqual(LAST_RUN['distance'], 1)
        self.assertLessEqual(LAST_RUN['queries'], 100)
        self.assertFalse(first.requires_grad)
        search = RectangleSearch('unused.jpg')
        rebuilt = search.x + search.decode(torch.tensor(LAST_RUN['best_genome']), LAST_RUN['best_budget'])
        self.assertTrue(torch.equal(first, rebuilt))

    def test_thread_restore_on_error(self):
        torch.set_num_threads(2)
        with patch.object(epsilon_ball, 'get_model_predictions_for', side_effect=RuntimeError('toy error')):
            with self.assertRaisesRegex(RuntimeError, 'toy error'):
                attack('unused.jpg')
        self.assertEqual(torch.get_num_threads(), 2)


if __name__ == '__main__':
    unittest.main()
