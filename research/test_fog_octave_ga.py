import os
import unittest
from unittest.mock import patch
import torch
from research.session3_baseline import perlin_fog as baseline
from research import fog_octave_ga as candidate


class FogOctaveTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.config=(baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES)

    def tearDown(self):torch.set_num_threads(self.threads)

    def prefix(self,path):
        baseline.LAST_RUN.clear()
        baseline.LAST_RUN.update(reference=0,queries=1,seconds=.01)
        return [(24,1.)]

    def layer(self,octave):return torch.full((1,3,224,224),float(octave)/128)

    def predictions(self,images):
        self.assertFalse(torch.is_grad_enabled())
        brightness=images.flatten(1).mean(1)
        return torch.stack([.25-brightness,brightness-.25],dim=1).softmax(1)

    def test_valid_reproducible_genotype_and_config_restore(self):
        image=torch.zeros(1,3,224,224)
        with (patch.dict(os.environ,{'CS454_THREADS':'1','CS454_SEED':'454'}),
              patch.object(baseline,'adv_func',side_effect=self.prefix),
              patch.object(baseline,'perlin_image_generator',side_effect=self.layer),
              patch.object(candidate,'get_image_tensor_from',return_value=image),
              patch.object(candidate,'get_model_predictions_for',side_effect=self.predictions),
              patch.object(candidate,'FOG_GENERATIONS',3),patch.object(candidate,'FOG_SECONDS',30)):
            one=candidate.adv_func('held-out-path');two=candidate.adv_func('held-out-path')
        self.assertEqual(one,two)
        self.assertTrue(all(type(o) is int and 1<=o<=128 and m>0 for o,m in one))
        self.assertLess(sum(m for _,m in one),1)
        self.assertEqual((baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES),self.config)
        self.assertEqual(torch.get_num_threads(),1)
        with torch.inference_mode():
            field=sum(m*self.layer(o) for o,m in one)
            self.assertEqual(int(self.predictions(baseline.add_noise_to_image(field,image)).argmax()),1)

    def test_zero_extension_preserves_verified_full_prefix_without_requery(self):
        observed=[]
        def prefix(path):
            observed.append((baseline.FOG_SECONDS,baseline.FOG_GENERATIONS))
            return self.prefix(path)
        with (patch.dict(os.environ,{'CS454_THREADS':'1'}),
              patch.object(baseline,'adv_func',side_effect=prefix),
              patch.object(candidate,'get_image_tensor_from',return_value=torch.zeros(1,3,224,224)),
              patch.object(candidate,'get_model_predictions_for',side_effect=AssertionError('unnecessary requery')),
              patch.object(candidate,'FOG_GENERATIONS',0),patch.object(candidate,'FOG_SECONDS',30)):
            result=candidate.adv_func('unknown')
        self.assertEqual(result,[(24,1.)])
        self.assertEqual(observed,[(30.,self.config[1])])
        self.assertEqual(candidate.LAST_RUN['queries'],1)

    def test_prefix_error_restores_globals_and_threads(self):
        with (patch.dict(os.environ,{'CS454_THREADS':'2'}),
              patch.object(baseline,'adv_func',side_effect=RuntimeError('prefix failure'))):
            with self.assertRaisesRegex(RuntimeError,'prefix failure'):candidate.adv_func('unknown')
        self.assertEqual((baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES),self.config)
        self.assertEqual(torch.get_num_threads(),1)

if __name__=='__main__':unittest.main()
