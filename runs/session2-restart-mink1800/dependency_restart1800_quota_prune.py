"""Keep the verified 900-query discovery allocation as total budgets increase.

A successful discovery phase uses at most its 900-query-calibrated work quota,
so a larger total budget refines that same phenotype instead of replacing it
with a later, potentially less prunable discovery phenotype. Difficult images
without a successful ancestor retain the original full discovery fallback.
"""
import math
import os
import torch
import epsilon_ball as base
from research.repeat_prune import PruneContinuation

SOURCE_DEPENDENCIES=('research/repeat_prune.py','research/structured_sparse.py')
LAST_RUN={}


class QuotaPrune(PruneContinuation):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.reference_budget=max(2,int(os.environ.get('CS454_DISCOVERY_REFERENCE',900)))
        self.phase_query=max(2,math.ceil(min(self.limit,self.reference_budget)*self.fraction))


def make_L0_adversarial_from(image_path):
    threads=torch.get_num_threads()
    base.LAST_RUN.clear();LAST_RUN.clear()
    search=None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search=QuotaPrune(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='quota_bounded_discovery_and_pruning',
                            discovery_reference=search.reference_budget,discovery_queries=search.discovery_queries,
                            discovery_distance=search.discovery_distance,prune_evaluations=search.prune_evaluations,
                            cleanup_evaluations=search.cleanup_evaluations,cleanup_passes=search.cleanup_passes,
                            deletion_minimal=search.deletion_minimal)
        torch.set_num_threads(threads)
