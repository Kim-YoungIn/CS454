import os
import unittest
from unittest.mock import patch
import torch
import epsilon_ball as base
from research.value_prune import ValuePrune


class ValueTests(unittest.TestCase):
    @torch.inference_mode()
    def test_value_mutation_escapes_fixed_value_deletion_minimum(self):
        image=torch.full((1,3,8,8),-1.9)
        def predictions(images):
            flat=images.flatten()
            score=torch.maximum(flat[0]+flat[1]-4,3*(-flat[0]-2))
            return torch.stack([torch.zeros_like(score),score]).reshape(1,2).softmax(1)
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'200','CS454_SECONDS':'165','CS454_SEED':'454'}),
                  patch.object(base,'get_image_tensor_from',side_effect=lambda _:image.clone()),
                  patch.object(base,'get_model_predictions_for',side_effect=predictions)):
                search=ValuePrune('unknown.jpg')
                search.discovery=False
                parent=torch.zeros_like(image);parent.flatten()[:2]=3-image.flatten()[:2]
                search.evaluate(parent)
                search.prune()
            self.assertEqual(search.best_distance,1)
            self.assertGreater(search.value_evaluations,0)
            self.assertGreater(search.value_promotions,0)
            self.assertLessEqual(search.queries,200)
            self.assertTrue(torch.equal(search.best.flatten()[2:],image.flatten()[2:]))
            self.assertLessEqual(float(search.best.abs().max()),3)
            self.assertEqual(int(predictions(search.best).argmax()),1)
        finally:
            torch.set_num_threads(previous)

if __name__=='__main__':unittest.main()
