import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.quota_prune import QuotaPrune
from research.recombine_prune import RecombinePrune


class RecombinationTests(unittest.TestCase):
    def exercise(self,union_feasible):
        image=torch.zeros(1,3,8,8)
        def oracle(images):
            n=int(images.count_nonzero())
            valid=(n>0) if union_feasible else n==3
            return torch.tensor([[.1,.9]]) if valid else torch.tensor([[.9,.1]])
        def discoveries(search):
            child=torch.zeros_like(image)
            if search.discovery_count<=2:
                start=0 if search.discovery_count==1 else 64
                child.flatten()[start:start+3]=3
            search.evaluate(child);search.discovery=False
            search.discovery_queries=search.queries
            search.discovery_distance=search.best_distance if search.best is not None else None
            search.deletion_minimal=True
        old=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',return_value=image),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle),
                  patch.object(QuotaPrune,'sparse',new=discoveries)):
                search=RecombinePrune('held-out-path');result=search.run()
            self.assertLessEqual(search.queries,900)
            self.assertTrue(any(c['origin']['kind']=='union' for c in search.chunks))
            self.assertEqual(int(oracle(result).argmax()),1)
            return search
        finally:torch.set_num_threads(old)

    @torch.inference_mode()
    def test_feasible_union_can_be_pruned_without_losing_parents(self):
        search=self.exercise(True)
        self.assertEqual(search.best_distance,1)
        self.assertEqual(search.selected_origin['kind'],'union')

    @torch.inference_mode()
    def test_failed_union_preserves_global_success(self):
        search=self.exercise(False)
        self.assertEqual(search.best_distance,3)
        self.assertEqual(search.selected_origin,dict(kind='discovery',seed=454))

if __name__=='__main__':unittest.main()
