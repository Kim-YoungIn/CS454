import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.repeat_prune import PruneContinuation
from research.quota_prune import QuotaPrune


class QuotaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_larger_budget_preserves_verified_discovery_parent(self):
        image=torch.zeros(1,3,8,8)
        def oracle(images):
            return torch.tensor([[.1,.9]]) if images.abs().max()>.1 else torch.tensor([[.9,.1]])
        class OriginalProbe(PruneContinuation):
            def prune(self):self.parent=self.best.clone()
        class QuotaProbe(QuotaPrune):
            def prune(self):self.parent=self.best.clone()
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',side_effect=lambda _:image.clone()),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle)):
                first=OriginalProbe('held-out-path');first.run()
                os.environ['CS454_QUERIES']='1800'
                second=QuotaProbe('held-out-path');second.run()
            self.assertEqual(first.discovery_queries,second.discovery_queries)
            self.assertTrue(torch.equal(first.parent,second.parent))
        finally:torch.set_num_threads(previous)

    @torch.inference_mode()
    def test_quota_does_not_truncate_unsuccessful_discovery(self):
        image=torch.zeros(1,3,8,8)
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'600','CS454_DISCOVERY_REFERENCE':'100','CS454_SECONDS':'165'}),
                  patch.object(base,'get_image_tensor_from',return_value=image),
                  patch.object(base,'get_model_predictions_for',return_value=torch.tensor([[.9,.1]]))):
                search=QuotaPrune('unknown')
                with self.assertRaises(RuntimeError):search.run()
            self.assertEqual(search.queries,600)
            self.assertIsNone(search.best)
        finally:torch.set_num_threads(previous)

if __name__=='__main__':unittest.main()
