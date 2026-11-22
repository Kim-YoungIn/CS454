"""Exact-image cache contracts using synthetic stationary model probabilities."""
import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball as base
from research import cache_candidate as candidate


class CacheCandidateTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.image = torch.zeros(1, 8, 8, 3).permute(0, 3, 1, 2)
        self.calls = 0
        self.environment = patch.dict(os.environ, {
            'CS454_QUERIES':'48', 'CS454_SECONDS':'165',
            'CS454_THREADS':'1', 'CS454_SEED':'454'})
        self.environment.start()
        self.loader = patch.object(base, 'get_image_tensor_from', side_effect=lambda _:self.image.clone())
        self.loader.start()
        self.oracle = patch.object(base, 'get_model_predictions_for', side_effect=self.predictions)
        self.oracle.start()

    def tearDown(self):
        self.oracle.stop()
        self.loader.stop()
        self.environment.stop()
        torch.set_num_threads(self.threads)

    def predictions(self, image):
        self.calls += 1
        self.assertFalse(torch.is_grad_enabled())
        changed = float(image.abs().max()) > .1
        return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])

    @torch.inference_mode()
    def test_exact_duplicates_avoid_forward_and_preserve_archive(self):
        search = candidate.CachedSearch('unknown.jpg','linf')
        delta = torch.ones_like(search.x)
        margin, actual = search.evaluate(delta)
        archived = search.best.clone()
        first_history = list(search.history)
        duplicate_margin, duplicate_actual = search.evaluate(delta.clone())
        self.assertEqual(self.calls,2)  # One clean and one candidate forward.
        self.assertEqual(search.queries,2)
        self.assertEqual(search.cache_hits,1)
        self.assertEqual(search.proposals,2)
        self.assertEqual(margin,duplicate_margin)
        self.assertTrue(torch.equal(actual,duplicate_actual))
        self.assertTrue(torch.equal(search.best,archived))
        self.assertEqual(search.history,first_history)

    @torch.inference_mode()
    def test_clipped_aliases_hit_but_distinct_images_do_not(self):
        search = candidate.CachedSearch('unknown.jpg','l0')
        first, actual = search.evaluate(torch.full_like(search.x,4))
        same, duplicate_actual = search.evaluate(torch.full_like(search.x,5))
        self.assertEqual(first,same)
        self.assertTrue(torch.equal(actual,duplicate_actual))
        self.assertEqual(search.queries,2)
        self.assertEqual(search.cache_hits,1)
        search.evaluate(torch.full_like(search.x,2))
        self.assertEqual(search.queries,3)
        self.assertEqual(len(search.fitness_cache),2)

    @torch.inference_mode()
    def test_public_success_reproducibility_and_query_accounting(self):
        for norm, function in [('linf',candidate.make_Linf_adversarial_from),
                               ('l2',candidate.make_L2_adversarial_from),
                               ('l0',candidate.make_L0_adversarial_from)]:
            with self.subTest(norm=norm):
                before = self.calls
                one = function('unknown-first.jpg')
                first = dict(candidate.LAST_RUN)
                self.assertEqual(first['queries'],self.calls-before)
                self.assertEqual(first['queries'],first['proposals']-first['cache_hits']+1)
                self.assertLessEqual(first['queries'],48)
                self.assertLessEqual(first['proposals'],first['proposal_limit'])
                self.assertTrue(first['success'])
                two = function('unknown-second.jpg')
                self.assertTrue(torch.equal(one,two))
                self.assertEqual(first['history'],candidate.LAST_RUN['history'])
                self.assertEqual(first['cache_hits'],candidate.LAST_RUN['cache_hits'])

    @torch.inference_mode()
    def test_proposal_guard_terminates_a_collapsed_population(self):
        class CollapsedSearch(candidate.CachedSearch):
            def dense(self):
                delta = torch.ones_like(self.x)
                while self.available():
                    self.evaluate(delta)
        search = CollapsedSearch('unknown.jpg','linf')
        search.proposal_limit = 32
        result = search.run()
        self.assertEqual(search.proposals,32)
        self.assertEqual(search.cache_hits,31)
        self.assertEqual(search.queries,2)
        self.assertEqual(base.LAST_RUN['queries'],2)
        self.assertTrue(torch.equal(result,search.best))

    def test_thread_state_and_error_metadata_restored(self):
        torch.set_num_threads(2)
        base.LAST_RUN.update(stale=True)
        with patch.object(base,'get_model_predictions_for',side_effect=RuntimeError('oracle error')):
            with self.assertRaisesRegex(RuntimeError,'oracle error'):
                candidate.make_Linf_adversarial_from('unknown.jpg')
        self.assertEqual(torch.get_num_threads(),2)
        self.assertNotIn('stale',candidate.LAST_RUN)
        self.assertFalse(base.LAST_RUN)


if __name__=='__main__':
    unittest.main()
