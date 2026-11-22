"""Compose exact-image fitness caching with hierarchical sparse-support pruning.

Both parents use cooperative super() calls: cache lookup wraps the unchanged
model evaluator, while structured discovery/pruning keeps its phase controls.
Only genuine forward calls consume the query budget in either phase.
"""
import os

import torch

import epsilon_ball as base
from research.cache_candidate import CachedSearch
from research.structured_sparse import StructuredSearch

SOURCE_DEPENDENCIES = ('research/cache_candidate.py', 'research/structured_sparse.py')
LAST_RUN = {}


class CachedStructuredSearch(CachedSearch, StructuredSearch):
    def __init__(self, image_path, norm='l0'):
        super().__init__(image_path, norm)


def make_L0_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    base.LAST_RUN.clear()
    LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, int(os.environ.get('CS454_THREADS', 4))))
        with torch.inference_mode():
            search = CachedStructuredSearch(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='cached_hierarchical_support_pruning',
                            proposals=search.proposals, cache_hits=search.cache_hits,
                            cache_entries=len(search.fitness_cache),
                            proposal_limit=search.proposal_limit,
                            proposal_guard_exhausted=search.proposals >= search.proposal_limit,
                            discovery_fraction=search.fraction,
                            discovery_queries=search.discovery_queries,
                            discovery_distance=search.discovery_distance,
                            prune_evaluations=search.prune_evaluations,
                            prune_queries=(search.queries-search.discovery_queries
                                           if search.discovery_queries is not None else None))
        torch.set_num_threads(previous_threads)
