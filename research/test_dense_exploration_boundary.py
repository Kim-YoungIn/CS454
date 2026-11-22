"""Check exploration gating against the preserved global and local searches."""
import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.dense_exploration_boundary import ExplorationBoundary
from research.dense_radial_probe import RadialProbeDirections


class ExplorationTests(unittest.TestCase):
    @torch.inference_mode()
    def test_zero_warmup_replays_previous_candidate_and_long_warmup_replays_control(self):
        previous_threads = torch.get_num_threads()
        image = torch.zeros(1,3,20,20)
        calls = []
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(images.clone())
            own = (5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
            return torch.stack((own,1-own)).reshape(1,2)
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ, {'CS454_QUERIES':'600','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',return_value=image),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle)):
                for warmup, control_type in ((0,RadialProbeDirections),(1000,base._Search)):
                    with self.subTest(warmup=warmup):
                        calls=[]
                        control = control_type('unknown','l2');expected=control.run()
                        prefix,calls=calls,[]
                        with patch.dict(os.environ,CS454_BOUNDARY_MIN_GENERATIONS=str(warmup)):
                            trial=ExplorationBoundary('unseen','l2');actual=trial.run()
                        self.assertEqual(len(prefix),len(calls))
                        self.assertTrue(all(torch.equal(a,b) for a,b in zip(prefix,calls)))
                        self.assertTrue(torch.equal(expected,actual))
        finally:
            torch.set_num_threads(previous_threads)

    @torch.inference_mode()
    def test_default_gate_preserves_global_prefix_then_activates(self):
        previous_threads = torch.get_num_threads()
        image = torch.zeros(1,3,20,20)
        calls=[]
        def oracle(images):
            calls.append(images.clone())
            own=(5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
            return torch.stack((own,1-own)).reshape(1,2)
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ, {'CS454_QUERIES':'600','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',return_value=image),
                  patch.object(base,'get_model_predictions_for',side_effect=oracle)):
                candidate=ExplorationBoundary('unseen','l2');result=candidate.run()
                prefix,calls=calls,[]
                event=next(e for e in candidate.trace if e['event']=='boundary_start')
                control=base._Search('unknown','l2');control.run()
            self.assertEqual(event['generation'],256)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(prefix[:event['query']],calls[:event['query']])))
            self.assertEqual(candidate.queries,600)
            self.assertEqual(int(oracle(result).argmax()),1)
        finally:
            torch.set_num_threads(previous_threads)


if __name__=='__main__':unittest.main()
