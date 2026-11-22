"""Fast renderer and GA contracts, without running VGG forward passes."""

import unittest
import os
from unittest.mock import patch

import torch

import perlin_fog as fog


class FogGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cached_mixture_matches_reference_renderer(self):
        genes = [(1, 0.07), (3, 0.19), (12, 0.31), (24, 0.09)]
        layers = torch.cat([fog.perlin_image_generator(o) for o, _ in genes])
        weights = torch.tensor([[m for _, m in genes]])
        cached = (weights @ layers.flatten(1)).reshape(1, 3, 224, 224)
        reference = fog.genotype2phenotype(genes)
        torch.testing.assert_close(cached, reference, rtol=1e-6, atol=1e-7)

    def test_batched_blending_is_elementwise(self):
        generator = torch.Generator().manual_seed(0)
        image = torch.randn(1, 3, 224, 224, generator=generator)
        fields = torch.rand(3, 3, 224, 224, generator=generator) * 2
        batched = fog.add_noise_to_image(fields, image)
        individual = torch.cat([fog.add_noise_to_image(f, image) for f in fields.split(1)])
        self.assertTrue(torch.equal(batched, individual))

    def test_ga_returns_valid_reproducible_success_with_toy_classifier(self):
        image = torch.zeros(1, 3, 224, 224)

        def predictions(images):
            brightness = images.flatten(1).mean(1)
            return torch.stack([0.5 - brightness, brightness - 0.5], dim=1).softmax(1)

        threads = torch.get_num_threads()
        with (patch.object(fog, "get_image_tensor_from", return_value=image),
              patch.object(fog, "get_model_predictions_for", side_effect=predictions),
              patch.object(fog, "FOG_GENERATIONS", 2),
              patch.object(fog, "FOG_OCTAVE_GENERATIONS", 2),
              patch.object(fog, "FOG_SECONDS", 30),
              patch.dict(os.environ, {"CS454_THREADS": "1"})):
            first = fog.adv_func("unused")
            second = fog.adv_func("unused")
        self.assertEqual(first, second)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), len(fog.FOG_OCTAVES))
        self.assertTrue(all(type(o) is int and o > 0 and m > 0 for o, m in first))
        rendered = fog.add_noise_to_image(fog.genotype2phenotype(first), image)
        self.assertEqual(int(predictions(rendered).argmax(1)), 1)
        self.assertTrue(torch.equal(image, torch.zeros_like(image)))

    def test_runtime_cap_preserves_a_success_when_budget_expires(self):
        image = torch.zeros(1, 3, 224, 224)
        clock = {"now": 0.0}

        def predictions(images):
            # Simulate a slow CPU without sleeping or using VGG. Any fog is
            # adversarial, while the zero input retains the clean class.
            clock["now"] += 40.0
            changed = images.flatten(1).mean(1) > 0
            own = torch.where(changed, 0.25, 0.75)
            return torch.stack([own, 1 - own], dim=1)

        with (patch.object(fog, "get_image_tensor_from", return_value=image),
              patch.object(fog, "get_model_predictions_for", side_effect=predictions),
              patch.object(fog, "FOG_GENERATIONS", 100000),
              patch.object(fog, "FOG_SECONDS", 100000),
              patch("time.monotonic", side_effect=lambda: clock["now"]),
              patch.dict(os.environ, {"CS454_THREADS": "1"})):
            genes = fog.adv_func("unused")
        self.assertGreater(len(genes), 0)
        self.assertLessEqual(clock["now"], 285.0)
        self.assertLess(fog.LAST_RUN["generations"], 100000)

    def test_timeout_without_success_restores_threads(self):
        image = torch.zeros(1, 3, 224, 224)
        previous = torch.get_num_threads()
        with (patch.object(fog, "get_image_tensor_from", return_value=image),
              patch.object(fog, "get_model_predictions_for", return_value=torch.tensor([[0.9, 0.1]])),
              patch.object(fog, "FOG_SECONDS", 1),
              patch.dict(os.environ, {"CS454_THREADS": "2"})):
            with self.assertRaisesRegex(RuntimeError, "time budget exhausted"):
                fog.adv_func("unused")
        self.assertEqual(torch.get_num_threads(), previous)


if __name__ == "__main__":
    unittest.main()
