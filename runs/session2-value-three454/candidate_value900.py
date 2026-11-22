"""Refine active channel values genetically, then revisit sparse deletions."""
import math
import os
import torch
import epsilon_ball as base
from research.repeat_prune import PruneContinuation

SOURCE_DEPENDENCIES=('research/repeat_prune.py','research/structured_sparse.py')
LAST_RUN={}


class ValuePrune(PruneContinuation):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.value_initial_query=None
        self.value_evaluations=0
        self.value_promotions=0
        self.value_support_reductions=0

    def prune(self):
        super().prune()
        if self.best is None or not self.available():
            return
        self.value_initial_query=self.queries
        while self.available():
            active=(self.best-self.x).flatten().nonzero().flatten()
            own_margin,_=self.evaluate(self.best-self.x)
            genes=self.best.flatten()[active].clone()
            pool=[(own_margin,genes)]
            promoted=False
            support_before=self.best_distance
            for generation in range(40):
                if not self.available():
                    break
                left=pool[0 if self.rand(1).item()<.8 else self.randint(len(pool))][1]
                right=pool[self.randint(len(pool))][1]
                child=torch.where(self.rand(len(active))<.5,left,right).clone()
                count=1 if generation%3!=2 else min(len(active),max(2,math.ceil(len(active)*.15)))
                selected=torch.randperm(len(active),generator=self.rng)[:count]
                if generation%3==0:
                    child[selected]=-child[selected]
                elif generation%3==1:
                    child[selected]=self.rand(count)*6-3
                else:
                    child[selected]=(child[selected]+torch.randn(count,generator=self.rng)*.75).clamp(-3,3)
                delta=torch.zeros_like(self.x)
                delta.flatten()[active]=child-self.x.flatten()[active]
                margin,actual=self.evaluate(delta)
                self.value_evaluations+=1
                if self.best_distance<support_before:
                    self.value_support_reductions+=1
                    promoted=True
                    self.deletion_minimal=False
                    break
                if margin<0:
                    image=self.x+actual
                    pool=sorted(pool+[(margin,image.flatten()[active].clone())],key=lambda item:item[0])[:4]
                    if margin<own_margin-1e-6:
                        # Equal-cardinality feasibility plus stronger margin is
                        # a tie-break only. Never replace a smaller support.
                        self.best=image.clone()
                        own_margin=margin
                        self.value_promotions+=1
                        self.deletion_minimal=False
                        promoted=True
            if not promoted:
                break
            before=self.best_distance
            super().prune()
            self.value_support_reductions+=int(self.best_distance<before)


def make_L0_adversarial_from(image_path):
    threads=torch.get_num_threads()
    base.LAST_RUN.clear();LAST_RUN.clear()
    search=None
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            search=ValuePrune(image_path)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='support_value_ga_and_pruning',
                            discovery_queries=search.discovery_queries,discovery_distance=search.discovery_distance,
                            prune_evaluations=search.prune_evaluations,cleanup_evaluations=search.cleanup_evaluations,
                            deletion_minimal=search.deletion_minimal,value_initial_query=search.value_initial_query,
                            value_evaluations=search.value_evaluations,value_promotions=search.value_promotions,
                            value_support_reductions=search.value_support_reductions)
        torch.set_num_threads(threads)
