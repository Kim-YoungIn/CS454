"""Hierarchical sparse GA plus repeated single-element deletion passes.

Deleting other elements can change whether an earlier failed deletion succeeds.
A no-change complete pass certifies only single-deletion minimality for the
current fixed values, not a globally minimal adversarial support.
"""
import os
import torch
import epsilon_ball as base
from research.structured_sparse import StructuredSearch

SOURCE_DEPENDENCIES = ('research/structured_sparse.py',)
LAST_RUN = {}


class PruneContinuation(StructuredSearch):
    def __init__(self, image_path, norm='l0'):
        super().__init__(image_path, norm)
        self.cleanup_evaluations = 0
        self.cleanup_passes = 0
        self.deletion_minimal = False

    def prune(self):
        super().prune()
        if self.best is None:
            return
        while self.available():
            active = (self.best-self.x).flatten().nonzero().flatten()
            improved, complete = False, True
            for index in active:
                if not self.available():
                    complete = False
                    break
                child = self.best-self.x
                child.flatten()[index] = 0
                previous = self.best_distance
                self.evaluate(child)
                self.cleanup_evaluations += 1
                improved |= self.best_distance < previous
            self.cleanup_passes += 1
            if complete and not improved:
                self.deletion_minimal = True
                break
            if not complete:
                break


def make_L0_adversarial_from(image_path):
    threads = torch.get_num_threads()
    base.LAST_RUN.clear()
    LAST_RUN.clear()
    search = None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search = PruneContinuation(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='hierarchical_and_repeated_deletion',
                            discovery_fraction=search.fraction,
                            discovery_queries=search.discovery_queries,
                            discovery_distance=search.discovery_distance,
                            prune_evaluations=search.prune_evaluations,
                            cleanup_evaluations=search.cleanup_evaluations,
                            cleanup_passes=search.cleanup_passes,
                            deletion_minimal=search.deletion_minimal)
        torch.set_num_threads(threads)
