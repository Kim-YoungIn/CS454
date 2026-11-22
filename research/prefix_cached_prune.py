"""Cache exact fitness while preserving the uncached GA discovery proposal prefix.

At fixed query caps, saved discovery forwards are spent on deletion refinement
rather than additional GA proposals. In wall-clock mode phase transfer still
uses elapsed time, so the identical-prefix guarantee applies only to query caps.
"""
import os
import time
import torch
import epsilon_ball as base
from research.cache_candidate import CachedSearch
from research.repeat_prune import PruneContinuation

SOURCE_DEPENDENCIES = ('research/cache_candidate.py','research/repeat_prune.py','research/structured_sparse.py')
LAST_RUN = {}


class PrefixCachedPrune(CachedSearch, PruneContinuation):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.discovery_proposals=None
        self.discovery_hash=None

    def available(self,reserve=0.,queries=0):
        if self.discovery and self.best is not None:
            if self.proposals+1 >= self.phase_query or time.monotonic() >= self.phase_time:
                return False
        # Preserve CachedSearch's proposal guard and _Search's true-query/time
        # bounds, bypassing only the inherited real-query phase cutoff.
        active=self.discovery
        self.discovery=False
        try:
            return super().available(reserve,queries)
        finally:
            self.discovery=active

    def prune(self):
        if self.best is not None and self.discovery_hash is None:
            self.discovery_hash=self.image_key(self.best).hex()
            self.discovery_proposals=self.proposals
        return super().prune()


def make_L0_adversarial_from(image_path):
    threads=torch.get_num_threads()
    base.LAST_RUN.clear(); LAST_RUN.clear()
    search=None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search=PrefixCachedPrune(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='prefix_preserving_cache_and_pruning',
                            discovery_fraction=search.fraction,discovery_queries=search.discovery_queries,
                            discovery_distance=search.discovery_distance,discovery_proposals=search.discovery_proposals,
                            discovery_hash=search.discovery_hash,proposals=search.proposals,cache_hits=search.cache_hits,
                            prune_evaluations=search.prune_evaluations,cleanup_evaluations=search.cleanup_evaluations,
                            cleanup_passes=search.cleanup_passes,deletion_minimal=search.deletion_minimal,
                            proposal_guard_exhausted=search.proposals>=search.proposal_limit)
        torch.set_num_threads(threads)
