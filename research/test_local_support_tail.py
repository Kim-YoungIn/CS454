"""Contracts for the optional neighbor-support tail; no VGG forward passes."""
import os
import unittest
from unittest.mock import patch

import torch
import epsilon_ball as base
from research.local_support_tail import LocalTailSearch, neighbor_alleles


class LocalSupportTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_endpoint_transport_has_no_row_or_channel_wrap(self):
        original = torch.full((1,2,2,3), .5)
        parent = original.clone()
        parent.flatten()[0] = 3
        parent.flatten()[-1] = -3
        weak = neighbor_alleles(original, parent)
        self.assertEqual(weak.flatten()[[1,3,4]].tolist(), [3.,3.,3.])
        self.assertEqual(weak.flatten()[[7,8,10]].tolist(), [-3.,-3.,-3.])
        self.assertEqual(float(weak.flatten()[2]), .5)
        self.assertEqual(float(weak.flatten()[6]), .5)
        self.assertLessEqual(float(weak.abs().max()), 3)

    @torch.inference_mode()
    def test_entire_real_policy_prefix_preserved_and_failed_tail_protects_archive(self):
        original = torch.zeros(1,3,16,16)
        calls = []
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(images.clone())
            own = .25 if images.count_nonzero() >= 3 else .75
            return torch.tensor([[own,1-own]])
        with (patch.dict(os.environ, {'CS454_QUERIES':'500','CS454_SECONDS':'165','CS454_SEED':'454'}),
              patch.object(base,'get_image_tensor_from',return_value=original),
              patch.object(base,'get_model_predictions_for',side_effect=oracle)):
            control = base._ParentalSparseSearch('arbitrary-unseen-image','l0')
            before = control.run()
            prefix, calls = calls, []
            candidate = LocalTailSearch('arbitrary-unseen-image','l0')
            after = candidate.run()
        self.assertLess(control.queries,500)
        self.assertEqual(candidate.queries,500)
        self.assertEqual(len(prefix),control.queries)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(prefix,calls[:len(prefix)])))
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(candidate.selected_origin,control.selected_origin)
        self.assertTrue(candidate.deletion_minimal)
        self.assertEqual(candidate.chunks[-1]['origin']['kind'],'local_support_tail')
        self.assertEqual(candidate.chunks[-1]['promotions'],0)
        stats = base.LAST_RUN
        self.assertEqual(1 + sum(stats[key] for key in ('discovery_evaluations',
                         'prune_evaluations','cleanup_evaluations','support_evaluations')),500)

    @torch.inference_mode()
    def test_new_neighbor_locus_can_replace_and_reduce_a_parent(self):
        original = torch.zeros(1,1,4,4)
        parent = original.clone();parent.flatten()[[0,3,8]] = 3
        def oracle(images):
            active = images.flatten().ne(0)
            feasible = bool(active[[0,3,8]].all() or active[[0,1]].all())
            own = .1 if feasible else .9-.1*int(active[0])-.1*int(active[1])
            return torch.tensor([[own,1-own]])
        def known_complete_policy(search):
            search.discovery = False
            search.evaluate(parent-original)
            search.deletion_minimal = True
            search.selected_origin = dict(kind='discovery',seed=454)
            search.chunks = [dict(origin=search.selected_origin,start_query=1,end_query=2,
                distance=3,deletion_minimal=True,discovery_query=2,discovery_distance=3,
                prune_evaluations=0,cleanup_evaluations=0,cleanup_passes=0)]
        with (patch.dict(os.environ, {'CS454_QUERIES':'150','CS454_SECONDS':'165','CS454_SEED':'454'}),
              patch.object(base,'get_image_tensor_from',return_value=original),
              patch.object(base,'get_model_predictions_for',side_effect=oracle),
              patch.object(base._ParentalSparseSearch,'sparse',new=known_complete_policy)):
            search = LocalTailSearch('unseen','l0');result = search.run()
        self.assertEqual(search.best_distance,2)
        self.assertEqual(int(oracle(result).argmax()),1)
        self.assertTrue(bool(result.flatten()[1]))
        self.assertFalse(search.deletion_minimal)
        self.assertEqual(search.selected_origin['kind'],'local_support_tail')
        self.assertEqual(search.queries,150)


if __name__ == '__main__':
    unittest.main()
