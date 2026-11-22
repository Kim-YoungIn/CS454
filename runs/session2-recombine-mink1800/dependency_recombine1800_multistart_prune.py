"""Restart discovery after local support pruning, preserving a global archive."""
import math
import os
import time
import torch
import epsilon_ball as base
from research.quota_prune import QuotaPrune

SOURCE_DEPENDENCIES=('research/quota_prune.py','research/repeat_prune.py','research/structured_sparse.py')
LAST_RUN={}


class MultistartPrune(QuotaPrune):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.initial_seed=int(os.environ.get('CS454_SEED',454))
        self.chunks=[]
        self.selected_seed=None

    def sparse(self):
        best_image,best_distance,best_minimal=None,math.inf,False
        history=[]
        while base._Search.available(self):
            remaining=self.limit-self.queries
            seconds_left=self.seconds-(time.monotonic()-self.started)
            if self.chunks and (remaining<256 or seconds_left<5):
                break
            seed=self.initial_seed+len(self.chunks)
            start=self.queries
            self.rng.manual_seed(seed)
            self.best,self.best_distance=None,math.inf
            self.history=[]
            self.discovery=True
            self.discovery_distance=self.discovery_queries=None
            self.prune_evaluations=self.cleanup_evaluations=self.cleanup_passes=0
            self.deletion_minimal=False
            # Independent controls count one clean forward per run. Reuse the
            # known clean class here, but retain its virtual slot in each quota.
            local_budget=min(remaining+1,self.reference_budget)
            self.phase_query=start+max(1,math.ceil(local_budget*self.fraction)-1)
            self.phase_time=(math.inf if 'CS454_QUERIES' in os.environ else
                             self.started+self.seconds*self.fraction if not self.chunks else
                             time.monotonic()+seconds_left*self.fraction)
            super().sparse()
            self.chunks.append(dict(seed=seed,start_query=start,end_query=self.queries,
                                    distance=self.best_distance if self.best is not None else None,
                                    discovery_query=self.discovery_queries,
                                    discovery_distance=self.discovery_distance,
                                    prune_evaluations=self.prune_evaluations,
                                    cleanup_evaluations=self.cleanup_evaluations,
                                    deletion_minimal=self.deletion_minimal))
            for query,distance in self.history:
                if not history or distance<history[-1][1]:
                    history.append((query,distance))
            if self.best is not None and self.best_distance<best_distance:
                best_image=self.best.clone()
                best_distance=self.best_distance
                best_minimal=self.deletion_minimal
                self.selected_seed=seed
            if self.best is None or best_distance<=1:
                break
        self.best,self.best_distance,self.history=best_image,best_distance,history
        self.deletion_minimal=best_minimal


def make_L0_adversarial_from(image_path):
    threads=torch.get_num_threads()
    base.LAST_RUN.clear();LAST_RUN.clear()
    search=None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search=MultistartPrune(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='multistart_quota_pruning',chunks=search.chunks,
                            selected_seed=search.selected_seed,deletion_minimal=search.deletion_minimal)
        torch.set_num_threads(threads)
