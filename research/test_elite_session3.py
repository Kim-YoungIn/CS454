import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.elite_session3 import SupportGA,parental_loci
from research.elite_session3_online import OnlineSupportGA
from research.quota_prune import QuotaPrune


class ParentSupportTests(unittest.TestCase):
    def test_parent_values_win_overlap_without_addition(self):
        x=torch.zeros(1,1,1,4)
        strong=torch.tensor([[[[3.,-3.,0.,0.]]]])
        weak=torch.tensor([[[[-3.,0.,3.,0.]]]])
        loci,values,strong_mask,weak_mask=parental_loci(x,strong,weak)
        self.assertEqual(loci.tolist(),[0,1,2])
        self.assertEqual(values.tolist(),[3.,-3.,3.])
        self.assertEqual(strong_mask.tolist(),[True,True,False])
        self.assertEqual(weak_mask.tolist(),[True,False,True])

    @torch.inference_mode()
    def test_genetic_support_can_replace_and_reduce_parent(self):
        x=torch.zeros(1,1,1,8)
        strong=x.clone();strong.flatten()[:3]=3
        weak=x.clone();weak.flatten()[2:6]=3
        def oracle(image):
            active=image.flatten().ne(0)
            feasible=bool(active[:3].all() or (active[0] and active[3]))
            # Infeasible search fitness rewards both useful loci.
            if feasible:return torch.tensor([[.1,.9]])
            quality=int(active[0])+int(active[3])
            return torch.tensor([[.9-.1*quality,.1+.1*quality]])
        with (patch.dict(os.environ,{'CS454_QUERIES':'80','CS454_SECONDS':'165'}),
              patch.object(base,'get_image_tensor_from',return_value=x),
              patch.object(base,'get_model_predictions_for',side_effect=oracle)):
            search=base._Search('arbitrary-heldout-name','l0')
            search.evaluate(strong-x)
            engine=SupportGA(search,strong,weak)
            for size in (1,2,4):
                repaired=engine.repair(engine.strong_mask,size)
                self.assertEqual(int(repaired.sum()),size)
                mutated=engine.mutate(repaired,size,3)
                self.assertEqual(int(mutated.sum()),size)
                crossed=engine.crossover(mutated,repaired,size)
                self.assertEqual(int(crossed.sum()),size)
            engine.run()
        self.assertEqual(search.best_distance,2)
        self.assertEqual(int(oracle(search.best).argmax()),1)
        self.assertGreater(engine.promotions,0)
        self.assertEqual(search.queries,80)

    @torch.inference_mode()
    def test_online_parent_discovery_and_failed_support_stage_protect_archive(self):
        x=torch.zeros(1,3,8,8)
        def oracle(image):
            feasible=int(image.count_nonzero())==3
            return torch.tensor([[.1,.9]]) if feasible else torch.tensor([[.9,.1]])
        def discovery(search):
            child=torch.zeros_like(x)
            start=0 if search.discovery_count==1 else 64
            child.flatten()[start:start+3]=3
            search.evaluate(child);search.discovery=False;search.deletion_minimal=True
        with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165','CS454_SEED':'454'}),
              patch.object(base,'get_image_tensor_from',return_value=x),
              patch.object(base,'get_model_predictions_for',side_effect=oracle),
              patch.object(QuotaPrune,'sparse',new=discovery)):
            search=OnlineSupportGA('unseen-image-path','l0');result=search.run()
        self.assertEqual(search.best_distance,3)
        self.assertEqual(search.discovery_count,2)
        self.assertEqual(search.selected_origin,dict(kind='discovery',seed=454))
        self.assertEqual(search.chunks[-1]['origin']['kind'],'support_ga')
        self.assertEqual(search.chunks[-1]['promotions'],0)
        self.assertEqual(int(oracle(result).argmax()),1)
        self.assertLessEqual(search.queries,900)


if __name__=='__main__':unittest.main()
