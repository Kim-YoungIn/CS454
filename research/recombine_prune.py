"""Recombine locally pruned GA elites before starting another discovery search."""
import math
import os
import time
from itertools import combinations
import torch
import epsilon_ball as base
from research.multistart_prune import MultistartPrune
from research.quota_prune import QuotaPrune

SOURCE_DEPENDENCIES=('research/multistart_prune.py','research/quota_prune.py',
                     'research/repeat_prune.py','research/structured_sparse.py')
LAST_RUN={}


class RecombinePrune(MultistartPrune):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.selected_origin=None
        self.discovery_count=0

    def sparse(self):
        global_image,global_distance,global_minimal=None,math.inf,False
        history,pool,combined=[],[],set()
        while base._Search.available(self):
            remaining=self.limit-self.queries
            seconds_left=self.seconds-(time.monotonic()-self.started)
            pairs=sorted(combinations(pool,2),key=lambda pair:sum(p['distance'] for p in pair))
            pairs=[pair for pair in pairs if tuple(sorted(p['seed'] for p in pair)) not in combined]
            if pairs and remaining>=1 and seconds_left>=1:
                parents=sorted(pairs[0],key=lambda p:p['seed'])
                key=tuple(p['seed'] for p in parents)
                combined.add(key)
                kind,seed='union',None
            else:
                if self.chunks and (remaining<256 or seconds_left<5):break
                kind,seed='discovery',self.initial_seed+self.discovery_count
                self.discovery_count+=1
            start=self.queries
            self.best,self.best_distance=None,math.inf
            self.history=[]
            self.discovery=(kind=='discovery')
            self.discovery_distance=self.discovery_queries=None
            self.prune_evaluations=self.cleanup_evaluations=self.cleanup_passes=0
            self.deletion_minimal=False
            if kind=='discovery':
                self.rng.manual_seed(seed)
                self.phase_query=start+max(1,math.ceil(min(remaining+1,self.reference_budget)*self.fraction)-1)
                self.phase_time=(math.inf if 'CS454_QUERIES' in os.environ else
                                 self.started+self.seconds*self.fraction if not self.chunks else
                                 time.monotonic()+seconds_left*self.fraction)
                QuotaPrune.sparse(self)
                if self.best is not None and self.deletion_minimal:
                    pool.append(dict(seed=seed,distance=self.best_distance,image=self.best.clone()))
                    pool=sorted(pool,key=lambda p:p['distance'])[:4]
                origin=dict(kind=kind,seed=seed)
            else:
                union=(self.x+(parents[0]['image']-self.x)+(parents[1]['image']-self.x)).clamp(-3,3)
                self.evaluate(union-self.x)
                if self.best is not None:self.prune()
                origin=dict(kind=kind,parents=list(key))
            self.chunks.append(dict(origin=origin,start_query=start,end_query=self.queries,
                                    distance=self.best_distance if self.best is not None else None,
                                    deletion_minimal=self.deletion_minimal))
            for query,distance in self.history:
                if not history or distance<history[-1][1]:history.append((query,distance))
            if self.best is not None and self.best_distance<global_distance:
                global_image=self.best.clone();global_distance=self.best_distance
                global_minimal=self.deletion_minimal;self.selected_origin=origin
            if global_distance<=1 or (kind=='discovery' and self.best is None):break
        self.best,self.best_distance,self.history=global_image,global_distance,history
        self.deletion_minimal=global_minimal


def make_L0_adversarial_from(image_path):
    threads=torch.get_num_threads()
    base.LAST_RUN.clear();LAST_RUN.clear();search=None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search=RecombinePrune(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='pruned_elite_recombination',chunks=search.chunks,
                            selected_origin=search.selected_origin,discovery_count=search.discovery_count,
                            deletion_minimal=search.deletion_minimal)
        torch.set_num_threads(threads)
