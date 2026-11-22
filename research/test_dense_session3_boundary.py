"""Contract and discovery-prefix tests using a synthetic probability oracle."""
import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball
from research.dense_session3_boundary import BoundaryDirections, LAST_RUN, make_L2_adversarial_from
from research.dense_session3_fast import FastBoundaryDirections


class BoundaryDirectionTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.image = torch.zeros(1, 3, 20, 20)
        self.proposals = []
        self.region = False
        self.settings = patch.dict(os.environ, CS454_SEED='454', CS454_QUERIES='100',
                                   CS454_SECONDS='165', CS454_THREADS='1',
                                   CS454_L2_STYLE='energy')
        self.settings.start()
        self.loader = patch.object(epsilon_ball, 'get_image_tensor_from',
                                   side_effect=lambda _: self.image.clone())
        self.oracle = patch.object(epsilon_ball, 'get_model_predictions_for',
                                   side_effect=self.probabilities)
        self.loader.start()
        self.oracle.start()

    def tearDown(self):
        self.oracle.stop()
        self.loader.stop()
        self.settings.stop()
        torch.set_num_threads(self.threads)

    def probabilities(self, image):
        self.assertFalse(torch.is_grad_enabled())
        self.proposals.append(image.clone())
        # A deterministic radial decision boundary, with meaningful margins.
        measured = image[..., :10, :10] if self.region else image
        logit = 5 - measured.square().sum().sqrt()
        own = logit.sigmoid()
        return torch.stack((own, 1 - own)).reshape(1, 2)

    @torch.inference_mode()
    def test_discovery_matches_original_through_switch(self):
        candidate = BoundaryDirections('unseen-image.jpg')
        candidate.run()
        prefix = self.proposals[:]
        switch = next(item['query'] for item in candidate.trace if item['event'] == 'boundary_start')
        self.proposals.clear()
        original = epsilon_ball._Search('unseen-image.jpg', 'l2')
        original.run()
        self.assertGreaterEqual(switch, 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(prefix[:switch], self.proposals[:switch])))

    def test_public_contract_archive_and_repeatability(self):
        result = make_L2_adversarial_from('arbitrary.jpg')
        record = dict(LAST_RUN)
        self.assertTrue(record['success'])
        self.assertEqual(record['reference'], 0)
        self.assertLessEqual(record['queries'], 100)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLessEqual(float(result.abs().max()), 3)
        self.assertAlmostEqual(float(result.square().sum()), record['distance'])
        self.assertTrue(any(torch.equal(result, image) for image in self.proposals))
        second = make_L2_adversarial_from('other-path.jpg')
        self.assertTrue(torch.equal(result, second))
        self.assertEqual(record['dense_trace'], LAST_RUN['dense_trace'])

    def test_threads_restored_after_failure(self):
        torch.set_num_threads(2)
        with patch.object(epsilon_ball, 'get_model_predictions_for', side_effect=RuntimeError('oracle error')):
            with self.assertRaisesRegex(RuntimeError, 'oracle error'):
                make_L2_adversarial_from('unseen.jpg')
        self.assertEqual(torch.get_num_threads(), 2)

    @torch.inference_mode()
    def test_fast_variant_bitwise_model_inputs_for_acceptance_and_relaxation(self):
        with patch.dict(os.environ, CS454_QUERIES='600'):
            for region in (False, True):
                with self.subTest(spatial_classifier=region):
                    self.region = region
                    self.proposals.clear()
                    original = BoundaryDirections('arbitrary.jpg')
                    expected = original.run()
                    inputs = self.proposals[:]
                    self.proposals.clear()
                    fast = FastBoundaryDirections('arbitrary.jpg')
                    actual = fast.run()
                    self.assertEqual(len(inputs), len(self.proposals))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, self.proposals)))
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertEqual(original.history, fast.history)
                    self.assertEqual(original.trace, fast.trace)
                    self.assertGreater(fast.mate_unused, 0)
                    events = {event['event'] for event in fast.trace}
                    self.assertIn('boundary_accept' if region else 'boundary_relax', events)


if __name__ == '__main__':
    unittest.main()
