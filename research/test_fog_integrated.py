"""Check the self-contained fog port against the isolated research algorithm."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from research.session3_baseline import perlin_fog as baseline
from research import fog_integrated as integrated
from research import fog_octave_ga as candidate


class IntegratedFogTests(unittest.TestCase):
    def test_mixture_search_and_instructor_renderers_are_preserved(self):
        original = ast.parse(Path(baseline.__file__).read_text())
        port = ast.parse(Path(integrated.__file__).read_text())
        functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        copies = {node.name: node for node in port.body if isinstance(node, ast.FunctionDef)}
        for name in ('perlin_image_generator', 'genotype2phenotype', 'add_noise_to_image', 'adv_func'):
            copied = copies['_mixture_search' if name == 'adv_func' else name]
            copied.name = name
            self.assertEqual(ast.dump(functions[name]), ast.dump(copied))

    def test_port_matches_research_genes_and_every_proposal(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        image = torch.zeros(1, 3, 224, 224)
        proposals = []
        def prefix(path):
            baseline.LAST_RUN.clear()
            baseline.LAST_RUN.update(reference=0, queries=1, seconds=.01, generations=36)
            return [(24, 1.)]
        def port_prefix(path):
            genes = prefix(path)
            integrated.LAST_RUN.update(baseline.LAST_RUN)
            return genes
        def layer(octave):
            return torch.full_like(image, float(octave) / 128)
        def predictions(images):
            self.assertFalse(torch.is_grad_enabled())
            proposals.append(images.clone())
            brightness = images.flatten(1).mean(1)
            return torch.stack([.25-brightness, brightness-.25], dim=1).softmax(1)
        try:
            with (patch.dict(os.environ, {'CS454_THREADS': '1', 'CS454_SEED': '454'}),
                  patch.object(baseline, 'adv_func', side_effect=prefix),
                  patch.object(baseline, 'perlin_image_generator', side_effect=layer),
                  patch.object(candidate, 'get_image_tensor_from', return_value=image),
                  patch.object(candidate, 'get_model_predictions_for', side_effect=predictions),
                  patch.object(candidate, 'FOG_GENERATIONS', 3),
                  patch.object(candidate, 'FOG_SECONDS', 30)):
                expected = candidate.adv_func('arbitrary-new-path')
                expected_proposals, proposals = proposals, []
                expected_stats = dict(candidate.LAST_RUN)
            with (patch.dict(os.environ, {'CS454_THREADS': '1', 'CS454_SEED': '454'}),
                  patch.object(integrated, '_mixture_search', side_effect=port_prefix),
                  patch.object(integrated, 'perlin_image_generator', side_effect=layer),
                  patch.object(integrated, 'get_image_tensor_from', return_value=image),
                  patch.object(integrated, 'get_model_predictions_for', side_effect=predictions),
                  patch.object(integrated, 'FOG_OCTAVE_GENERATIONS', 3),
                  patch.object(integrated, 'FOG_SECONDS', 30)):
                actual = integrated.adv_func('arbitrary-new-path')
            self.assertEqual(actual, expected)
            self.assertEqual(len(proposals), len(expected_proposals))
            for actual_input, expected_input in zip(proposals, expected_proposals):
                self.assertTrue(torch.equal(actual_input, expected_input))
            for name in ('queries', 'generations', 'initial_score', 'distance', 'improvements'):
                self.assertEqual(integrated.LAST_RUN[name], expected_stats[name])
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(previous_threads)

    def test_verified_fallback_does_not_need_another_query(self):
        def prefix(path):
            integrated.LAST_RUN.update(reference=0, queries=655, seconds=50., generations=36)
            return [(64, .5)]
        with (patch.object(integrated, '_mixture_search', side_effect=prefix),
              patch.object(integrated, 'FOG_OCTAVE_GENERATIONS', 0),
              patch.object(integrated, 'get_image_tensor_from', return_value=torch.zeros(1,3,224,224)),
              patch.object(integrated, 'get_model_predictions_for', side_effect=AssertionError('redundant query'))):
            self.assertEqual(integrated.adv_func('unseen'), [(64, .5)])
        self.assertEqual(integrated.LAST_RUN['queries'], 655)


    def test_failed_exact_extension_preserves_verified_prefix(self):
        image = torch.zeros(1, 3, 224, 224)
        def prefix(path):
            integrated.LAST_RUN.update(reference=0, queries=1, seconds=.01, generations=36)
            return [(24, 1.)]
        def predictions(images):
            brightness = images.flatten(1).mean(1)
            return torch.stack([.25-brightness, brightness-.25], dim=1).softmax(1)
        with (patch.dict(os.environ, {'CS454_THREADS': '1', 'CS454_SEED': '454'}),
              patch.object(integrated, '_mixture_search', side_effect=prefix),
              patch.object(integrated, 'get_image_tensor_from', return_value=image),
              patch.object(integrated, 'perlin_image_generator', side_effect=lambda o: torch.full_like(image, o/128)),
              patch.object(integrated, 'genotype2phenotype', return_value=torch.zeros_like(image)),
              patch.object(integrated, 'get_model_predictions_for', side_effect=predictions),
              patch.object(integrated, 'FOG_OCTAVE_GENERATIONS', 1),
              patch.object(integrated, 'FOG_SECONDS', 30)):
            genes = integrated.adv_func('unknown')
        self.assertTrue(integrated.LAST_RUN['improvements'])
        self.assertTrue(integrated.LAST_RUN['verification_fallback'])
        self.assertEqual(genes, [(24, 1.)])
        self.assertEqual(integrated.LAST_RUN['distance'], 1.)


if __name__ == '__main__':
    unittest.main()
