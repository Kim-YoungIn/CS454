import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.quota_prune import QuotaPrune
from research.multistart_prune import MultistartPrune


class RestartTests(unittest.TestCase):
    @torch.inference_mode()
    def test_failed_restart_retains_previous_global_success(self):
        image=torch.zeros(1,3,8,8)
        def oracle(images):
            return torch.tensor([[.1,.9]]) if images.count_nonzero() else torch.tensor([[.9,.1]])
        def fake_chunk(search):
            child=torch.zeros_like(image)
            if not search.chunks:child.flatten()[:2]=3
            search.evaluate(child)
            search.discovery=False
            search.discovery_queries=search.queries
            search.discovery_distance=search.best_distance if search.best is not None else None
            search.deletion_minimal=True
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'1800','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',side_effect=lambda _:image.clone()),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle),
                  patch.object(QuotaPrune,'sparse',new=fake_chunk)):
                search=MultistartPrune('held-out-path'); result=search.run()
            self.assertEqual(search.best_distance,2)
            self.assertEqual(int(result.count_nonzero()),2)
            self.assertEqual(len(search.chunks),2)
            self.assertEqual(search.selected_seed,454)
            self.assertEqual(search.queries,3)
            self.assertEqual(search.history,[(2,2)])
        finally:torch.set_num_threads(previous)

    @torch.inference_mode()
    def test_first_uncapped_phase_matches_single_search_clock(self):
        image=torch.zeros(1,3,8,8)
        clock=[0.]
        seen=[]
        def oracle(images):
            clock[0]+=1
            return torch.tensor([[.1,.9]]) if images.count_nonzero() else torch.tensor([[.9,.1]])
        def one_chunk(search):
            seen.append(search.phase_time)
            child=torch.zeros_like(image);child.flatten()[0]=3
            search.evaluate(child)
            search.discovery_queries=search.queries
            search.discovery_distance=search.best_distance
            search.deletion_minimal=True
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',return_value=image),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle),
                  patch.object(base.time,'monotonic',side_effect=lambda:clock[0]),
                  patch.object(QuotaPrune,'sparse',new=one_chunk)):
                os.environ.pop('CS454_QUERIES',None)
                search=MultistartPrune('unknown')
                expected=search.phase_time
                search.run()
            self.assertEqual(seen,[expected])
        finally:torch.set_num_threads(previous)

    @torch.inference_mode()
    def test_global_minimum_stops_restarts(self):
        image=torch.zeros(1,3,8,8)
        def oracle(images):
            return torch.tensor([[.1,.9]]) if images.count_nonzero() else torch.tensor([[.9,.1]])
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',side_effect=lambda _:image.clone()),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle)):
                search=MultistartPrune('unknown');result=search.run()
            self.assertEqual(int(result.count_nonzero()),1)
            self.assertEqual(len(search.chunks),1)
            self.assertLessEqual(search.queries,900)
        finally:torch.set_num_threads(previous)

if __name__=='__main__':unittest.main()
