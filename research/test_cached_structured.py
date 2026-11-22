"""Composition checks for cache accounting and hierarchical support pruning."""
import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball as base
from research import cached_structured as candidate


class CachedStructuredTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.image = torch.zeros(1, 8, 8, 3).permute(0,3,1,2)
        self.calls = 0
        self.env = patch.dict(os.environ, {'CS454_QUERIES':'160','CS454_SECONDS':'165',
                                          'CS454_THREADS':'1','CS454_SEED':'454'})
        self.env.start()
        self.loader = patch.object(base,'get_image_tensor_from',side_effect=lambda _:self.image.clone())
        self.loader.start()
        self.oracle = patch.object(base,'get_model_predictions_for',side_effect=self.predictions)
        self.oracle.start()

    def tearDown(self):
        self.oracle.stop()
        self.loader.stop()
        self.env.stop()
        torch.set_num_threads(self.threads)

    def predictions(self,image):
        self.calls += 1
        self.assertFalse(torch.is_grad_enabled())
        changed = image[0,0,0,0].abs() > .5
        return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])

    @torch.inference_mode()
    def test_both_components_retain_essential_element(self):
        search = candidate.CachedStructuredSearch('unknown.jpg')
        self.assertTrue(search.discovery)
        self.assertLess(search.phase_query,search.limit)
        search.discovery = False
        delta = torch.full_like(search.x,3)
        search.evaluate(delta)
        search.evaluate(delta.clone())
        self.assertEqual(search.cache_hits,1)
        search.prune()
        self.assertEqual(search.best_distance,1)
        self.assertEqual(search.queries,self.calls)
        self.assertEqual(search.queries,search.proposals-search.cache_hits+1)
        self.assertGreater(float(search.best[0,0,0,0]),.5)
        self.assertGreater(search.prune_evaluations,0)

    @torch.inference_mode()
    def test_public_repeatability_phase_metadata_and_query_bound(self):
        first = candidate.make_L0_adversarial_from('first-unknown.jpg')
        info = dict(candidate.LAST_RUN)
        second = candidate.make_L0_adversarial_from('second-unknown.jpg')
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(info['history'],candidate.LAST_RUN['history'])
        self.assertEqual(info['cache_hits'],candidate.LAST_RUN['cache_hits'])
        self.assertTrue(info['success'])
        self.assertLessEqual(info['queries'],160)
        self.assertEqual(info['queries'],info['proposals']-info['cache_hits']+1)
        self.assertEqual(info['prune_queries'],info['queries']-info['discovery_queries'])
        self.assertLessEqual(info['discovery_queries'],info['queries'])
        self.assertLessEqual(info['discovery_queries'],88)
        self.assertGreaterEqual(info['prune_evaluations'],info['prune_queries'])

    @torch.inference_mode()
    def test_proposal_guard_limits_duplicate_only_discovery(self):
        class Collapsed(candidate.CachedStructuredSearch):
            def sparse(self):
                while self.available():
                    self.evaluate(torch.full_like(self.x,3))
        search = Collapsed('unknown.jpg')
        search.proposal_limit = 12
        result = search.run()
        self.assertTrue(torch.equal(result,search.best))
        self.assertEqual(search.proposals,12)
        self.assertEqual(search.queries,2)
        self.assertEqual(search.cache_hits,11)
        self.assertFalse(search.available())

    @torch.inference_mode()
    def test_failed_discovery_does_not_stop_at_phase_quota(self):
        with patch.object(base,'get_model_predictions_for',return_value=torch.tensor([[.9,.1]])):
            with self.assertRaisesRegex(RuntimeError,'without misclassification'):
                candidate.make_L0_adversarial_from('unknown.jpg')
        info = candidate.LAST_RUN
        self.assertFalse(info['success'])
        self.assertEqual(info['queries'],160)
        self.assertEqual(info['discovery_queries'],160)
        self.assertEqual(info['prune_queries'],0)
        self.assertEqual(info['queries'],info['proposals']-info['cache_hits']+1)

    @torch.inference_mode()
    def test_wall_deadline_retains_archive_and_transfers_to_pruning(self):
        clock = [1000.]
        def slow_all_required(image):
            clock[0] += 2.5
            changed = int(image.count_nonzero()) == image.numel()
            return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])
        with patch.dict(os.environ,clear=False):
            os.environ.pop('CS454_QUERIES',None)
            with patch.object(base,'get_model_predictions_for',side_effect=slow_all_required), \
                 patch('time.monotonic',side_effect=lambda:clock[0]):
                result = candidate.make_L0_adversarial_from('unknown.jpg')
        info = candidate.LAST_RUN
        self.assertTrue(info['success'])
        self.assertEqual(info['distance'],self.image.numel())
        self.assertEqual(int(result.count_nonzero()),self.image.numel())
        self.assertGreater(info['prune_queries'],0)
        self.assertLess(info['discovery_queries'],info['queries'])
        self.assertLessEqual(info['seconds'],165)
        self.assertEqual(info['seconds'],clock[0]-1000.)
        self.assertEqual(info['queries'],info['proposals']-info['cache_hits']+1)

    def test_error_after_discovery_restores_threads(self):
        torch.set_num_threads(2)
        with patch.object(candidate.CachedStructuredSearch,'prune',side_effect=RuntimeError('prune error')):
            with self.assertRaisesRegex(RuntimeError,'prune error'):
                candidate.make_L0_adversarial_from('unknown.jpg')
        self.assertEqual(torch.get_num_threads(),2)
        self.assertNotIn('success',candidate.LAST_RUN)
        self.assertIsNotNone(candidate.LAST_RUN['discovery_queries'])

    def test_error_restores_thread_state_and_clears_previous_metadata(self):
        torch.set_num_threads(2)
        base.LAST_RUN.update(stale=True)
        candidate.LAST_RUN.update(stale=True)
        with patch.object(base,'get_model_predictions_for',side_effect=RuntimeError('oracle error')):
            with self.assertRaisesRegex(RuntimeError,'oracle error'):
                candidate.make_L0_adversarial_from('unknown.jpg')
        self.assertEqual(torch.get_num_threads(),2)
        self.assertFalse(base.LAST_RUN)
        self.assertFalse(candidate.LAST_RUN)


if __name__=='__main__':
    unittest.main()
