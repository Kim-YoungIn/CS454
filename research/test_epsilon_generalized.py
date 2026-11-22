"""Check a self-contained L2 port against frozen architecture and old routes."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import epsilon_ball as production
from research import epsilon_generalized_candidate as port
from research.dense_exploration_boundary import ExplorationBoundary


class GeneralizedPortTests(unittest.TestCase):
    @torch.inference_mode()
    def test_every_proposal_matches_the_research_candidate(self):
        previous_threads=torch.get_num_threads()
        image=torch.zeros(1,3,20,20)
        proposals=[]
        def oracle(images):
            self.assertFalse(torch.is_grad_enabled())
            proposals.append(images.clone())
            own=(5-images[..., :10, :10].square().sum().sqrt()).sigmoid()
            return torch.stack((own,1-own)).reshape(1,2)
        try:
            torch.set_num_threads(1)
            with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165',
                                        'CS454_BOUNDARY_MIN_GENERATIONS':'512','CS454_SEED':'454'}),
                  patch.object(production,'get_image_tensor_from',return_value=image),
                  patch.object(production,'get_model_predictions_for',side_effect=oracle)):
                reference=ExplorationBoundary('arbitrary','l2');expected=reference.run()
            expected_proposals,proposals=proposals,[]
            with (patch.dict(os.environ,{'CS454_QUERIES':'900','CS454_SECONDS':'165',
                                        'CS454_BOUNDARY_MIN_GENERATIONS':'512','CS454_SEED':'454'}),
                  patch.object(port,'get_image_tensor_from',return_value=image),
                  patch.object(port,'get_model_predictions_for',side_effect=oracle)):
                actual_search=port._ExplorationBoundarySearch('other-path','l2');actual=actual_search.run()
            self.assertEqual(len(expected_proposals),len(proposals))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected_proposals,proposals)))
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(reference.history,actual_search.history)
            self.assertEqual(reference.trace,actual_search.trace)
            self.assertEqual(reference.probe_trace,actual_search.probe_trace)
            self.assertEqual(actual_search.queries,900)
        finally:
            torch.set_num_threads(previous_threads)

    def test_original_core_and_other_norm_routes_are_preserved(self):
        def definitions(path):
            tree=ast.parse(Path(path).read_text())
            return {node.name:ast.dump(node) for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
        old,new=definitions(production.__file__),definitions(port.__file__)
        for name in ('_Search','_SparseRefinement','_SparseRefinementSearch',
                     '_MultistartSparseSearch','_SupportGA','_ParentalSparseSearch',
                     'make_Linf_adversarial_from','make_L0_adversarial_from'):
            self.assertEqual(old[name],new[name])
        for norm,expected in (('linf',port._Search),('l0',port._ParentalSparseSearch),('l2',port._Search)):
            with (patch.dict(os.environ,{'CS454_L2_SEARCH':'original','CS454_L0_STYLE':'parental'}),
                  patch.object(expected,'__init__',return_value=None) as constructor,
                  patch.object(expected,'run',return_value='image')):
                self.assertEqual(port._attack('arbitrary',norm),'image')
                constructor.assert_called_once_with('arbitrary',norm)


if __name__=='__main__':unittest.main()
