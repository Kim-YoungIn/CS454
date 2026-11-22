"""Verify accepted and failed scalar probes preserve the feasible archive."""
import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research import dense_radial_probe as candidate


class RadialProbeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_spatial_oracle_probes_and_deterministic_archive(self):
        previous_threads = torch.get_num_threads()
        image = torch.zeros(1, 3, 20, 20)
        proposals = []
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            proposals.append(images.clone())
            own = (5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
            return torch.stack((own, 1-own)).reshape(1,2)
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ, {'CS454_SEED': '454', 'CS454_QUERIES': '600',
                                         'CS454_SECONDS': '165', 'CS454_THREADS': '1'}),
                  patch.object(base, 'get_image_tensor_from', return_value=image),
                  patch.object(base, 'get_model_predictions_for', side_effect=oracle)):
                result = candidate.make_L2_adversarial_from('unknown')
                stats = dict(candidate.LAST_RUN)
                self.assertGreater(stats['probe_calls'], 0)
                self.assertGreater(stats['probe_successes'], 0)
                self.assertLess(stats['probe_successes'], stats['probe_calls'])
                self.assertEqual(stats['queries'], 600)
                self.assertTrue(any(torch.equal(result, value) for value in proposals))
                self.assertEqual(int(oracle(result).argmax()), 1)
                second = candidate.make_L2_adversarial_from('another-path')
                self.assertTrue(torch.equal(result, second))
                self.assertEqual(stats['dense_trace'], candidate.LAST_RUN['dense_trace'])
                self.assertEqual(float((result-image).square().sum()), stats['distance'])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
