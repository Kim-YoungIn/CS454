import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.repeat_prune import PruneContinuation
from research.prefix_cached_prune import PrefixCachedPrune


class PrefixCacheTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cache_preserves_discovery_proposals_and_parent(self):
        image=torch.zeros(1,3,8,8)
        def oracle(images):
            changed=bool(images.abs().max()>.1)
            return torch.tensor([[.1,.9]]) if changed else torch.tensor([[.9,.1]])
        class UncachedProbe(PruneContinuation):
            def prune(self):
                self.parent=self.best.clone()
        class CachedProbe(PrefixCachedPrune):
            def prune(self):
                self.parent=self.best.clone()
                self.discovery_proposals=self.proposals
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'160','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',side_effect=lambda _:image.clone()),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle)):
                left=UncachedProbe('held-out-path.jpg');left.run()
                right=CachedProbe('held-out-path.jpg');right.run()
            self.assertTrue(torch.equal(left.parent,right.parent))
            self.assertEqual(left.discovery_queries,right.discovery_proposals+1)
            self.assertGreater(right.cache_hits,0)
            self.assertLess(right.discovery_queries,left.discovery_queries)
        finally:
            torch.set_num_threads(old_threads)

if __name__=='__main__':
    unittest.main()
