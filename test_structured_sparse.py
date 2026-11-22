"""Memetic sparse-search contracts with a known essential channel element."""
import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball as base
from research import structured_sparse as candidate


class StructuredSparseTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.image = torch.zeros(1, 8, 8, 3).permute(0, 3, 1, 2)
        self.environment = patch.dict(os.environ, {'CS454_QUERIES':'160', 'CS454_SECONDS':'165',
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
        torch.set_num_threads(self.old_threads)

    def predictions(self, image):
        self.assertFalse(torch.is_grad_enabled())
        changed = image[0,0,0,0].abs() > .5
        return torch.tensor([[.1, .9]]) if changed else torch.tensor([[.9,.1]])

    @torch.inference_mode()
    def test_group_deletion_finds_one_essential_channel_element(self):
        search = candidate.StructuredSearch('unknown.jpg')
        search.discovery = False
        search.evaluate(torch.full_like(search.x, 3))
        initial_queries = search.queries
        search.prune()
        self.assertEqual(search.best_distance,1)
        self.assertEqual(int((search.best-search.x).count_nonzero()),1)
        self.assertGreater(abs(float(search.best[0,0,0,0])),.5)
        self.assertLess(search.queries-initial_queries,40)
        self.assertLessEqual(search.queries,160)
        self.assertTrue(search.x.is_contiguous())

    @torch.inference_mode()
    def test_failure_to_prune_retains_feasible_parent(self):
        def all_required(image):
            changed = int(image.count_nonzero()) == image.numel()
            return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])
        with patch.object(base,'get_model_predictions_for',side_effect=all_required):
            search = candidate.StructuredSearch('unknown.jpg')
            search.discovery = False
            search.limit = 12
            search.evaluate(torch.full_like(search.x, 3))
            archived = search.best.clone()
            search.prune()
        self.assertEqual(search.queries,12)
        self.assertTrue(torch.equal(search.best,archived))

    @torch.inference_mode()
    def test_split_preserves_support_exactly(self):
        search = candidate.StructuredSearch('unknown.jpg')
        for count in [2,3,19,64]:
            group = torch.arange(count,dtype=torch.long)
            left,right=search.split_group(group)
            self.assertGreater(len(left),0)
            self.assertGreater(len(right),0)
            self.assertTrue(torch.equal(torch.cat((left,right)).sort().values,group))

    @torch.inference_mode()
    def test_repeated_pass_revisits_failed_deletions_after_context_changes(self):
        from research.repeat_prune import PruneContinuation
        indices = torch.tensor([0,64,128])
        def nonmonotonic(images):
            bits = (images.flatten()[indices] != 0).int()
            mask = int((bits*torch.tensor([1,2,4])).sum())
            return torch.tensor([[.1,.9]]) if mask in (7,5,4) else torch.tensor([[.9,.1]])
        with patch.object(base,'get_model_predictions_for',side_effect=nonmonotonic):
            results=[]
            for kind in (candidate.StructuredSearch,PruneContinuation):
                search=kind('unknown.jpg')
                search.discovery=False
                parent=torch.zeros_like(search.x)
                parent.flatten()[indices]=3
                search.evaluate(parent)
                search.prune()
                results.append(search.best_distance)
            self.assertEqual(results,[2,1])
            self.assertTrue(search.deletion_minimal)
            self.assertGreater(search.cleanup_passes,1)
            self.assertLessEqual(search.queries,160)

    @torch.inference_mode()
    def test_production_uncapped_wall_budget_transfers_to_pruning(self):
        clock=[0.]
        original_oracle=self.predictions
        def slow(images):
            clock[0]+=2.5
            # Every nonempty support is adversarial, allowing a bounded local
            # optimum; this exercises wall-phase transfer without a query cap.
            changed=bool(images.count_nonzero())
            return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])
        with patch.dict(os.environ,{'CS454_SECONDS':'165','CS454_L0_STYLE':'refined'}):
            os.environ.pop('CS454_QUERIES',None)
            with (patch.object(base.time,'monotonic',side_effect=lambda:clock[0]),
                  patch.object(base,'get_model_predictions_for',side_effect=slow)):
                image=base.make_L0_adversarial_from('unknown-wall-budget.jpg')
        self.assertEqual(int(image.count_nonzero()),1)
        self.assertGreater(base.LAST_RUN['prune_evaluations'],0)
        self.assertGreater(base.LAST_RUN['queries'],base.LAST_RUN['discovery_queries'])
        self.assertLessEqual(base.LAST_RUN['seconds'],165)
        self.assertTrue(base.LAST_RUN['deletion_minimal'])

    @torch.inference_mode()
    def test_public_query_seed_is_reproducible(self):
        one=candidate.make_L0_adversarial_from('first-unknown.jpg')
        first=dict(candidate.LAST_RUN)
        two=candidate.make_L0_adversarial_from('second-unknown.jpg')
        self.assertTrue(torch.equal(one,two))
        self.assertEqual(first['history'],candidate.LAST_RUN['history'])
        self.assertEqual(candidate.LAST_RUN['reference'],0)
        self.assertLessEqual(candidate.LAST_RUN['queries'],160)
        self.assertLessEqual(candidate.LAST_RUN['discovery_queries'],88)

    @torch.inference_mode()
    def test_no_success_keeps_entire_discovery_budget(self):
        with patch.object(base,'get_model_predictions_for',return_value=torch.tensor([[.9,.1]])):
            with self.assertRaisesRegex(RuntimeError,'without misclassification'):
                candidate.make_L0_adversarial_from('unknown.jpg')
        self.assertEqual(candidate.LAST_RUN['queries'],160)
        self.assertFalse(candidate.LAST_RUN['success'])


if __name__=='__main__':
    unittest.main()
