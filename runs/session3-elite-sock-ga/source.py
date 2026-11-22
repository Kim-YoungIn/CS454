"""Diagnostic parental-support evolution; saved parents are CLI inputs only.

These experiments do not constitute an end-to-end attack benchmark. The support
operators themselves use only images already evolved in the current search.
"""
import argparse
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import torch
import epsilon_ball as base
from verify_artifacts import read_tensor


def parental_loci(original, strong, weak):
    """Strong-parent values win overlaps, so crossover never adds amplitudes."""
    strong_delta, weak_delta = strong-original, weak-original
    strong_mask, weak_mask = strong_delta.ne(0), weak_delta.ne(0)
    delta = torch.where(strong_mask, strong_delta, weak_delta)
    loci = delta.flatten().nonzero().flatten()
    return loci, delta.flatten()[loci], strong_mask.flatten()[loci], weak_mask.flatten()[loci]


class SupportGA:
    """Fixed-cardinality parental-support GA with a protected feasible archive."""
    def __init__(self, search, strong, weak):
        self.search = search
        self.loci, self.values, strong_mask, self.weak_mask = parental_loci(search.x,strong,weak)
        self.strong_mask = strong_mask
        self.attempts = self.promotions = 0
        self.seen,self.duplicates=set(),0
        self.population = []

    def choose(self, values, count):
        if count <= 0:return values[:0]
        order=torch.randperm(len(values),generator=self.search.rng)[:count]
        return values[order]

    def repair(self, mask, count):
        mask=mask.clone()
        active=mask.nonzero().flatten()
        if len(active)>count:mask[self.choose(active,len(active)-count)]=False
        if int(mask.sum())<count:
            inactive=(~mask).nonzero().flatten()
            mask[self.choose(inactive,count-int(mask.sum()))]=True
        return mask

    def mutate(self, mask, count, exchanges=1):
        mask=self.repair(mask,count)
        active=mask.nonzero().flatten()
        inactive=(~mask).nonzero().flatten()
        exchanges=min(exchanges,len(active),len(inactive))
        removed=self.choose(active,exchanges)
        added=self.choose(inactive,exchanges)
        mask[removed]=False;mask[added]=True
        return mask

    def crossover(self, left, right, count):
        choices=torch.rand(len(left),generator=self.search.rng)<.5
        return self.repair(torch.where(choices,left,right),count)

    def run(self):
        search=self.search
        count=max(1,int(search.best_distance)-1)
        while search.available() and count>0:
            if not self.population or self.attempts%8==0:
                child=self.repair(self.strong_mask,count)
            elif len(self.population)>1 and self.attempts%4==0:
                left=self.population[search.randint(len(self.population))][1]
                right=self.population[search.randint(len(self.population))][1]
                child=self.crossover(left,right,count)
                child=self.mutate(child,count,1)
            else:
                parent=self.population[search.randint(len(self.population))][1]
                child=self.mutate(parent,count,1 if search.rand(1).item()<.8 else 3)
            delta=torch.zeros_like(search.x)
            delta.flatten()[self.loci[child]]=self.values[child]
            key=bytes(child.tolist())
            if key in self.seen:self.duplicates+=1
            self.seen.add(key)
            previous=search.best_distance
            margin,_=search.evaluate(delta)
            self.attempts+=1
            self.population.append((margin,child.clone()))
            self.population=sorted(self.population,key=lambda p:p[0])[:4]
            if search.best_distance<previous:
                self.promotions+=1
                count=int(search.best_distance)-1
                self.strong_mask=(search.best-search.x).flatten()[self.loci].ne(0)
                self.population=[]


def priority_prune(search,strong,weak,weak_first=True):
    """Hierarchical deletion with weak-only or strong-only loci visited first."""
    loci,values,strong_mask,weak_mask=parental_loci(search.x,strong,weak)
    delta=torch.zeros_like(search.x);delta.flatten()[loci]=values
    # Allow a larger union as a local trajectory; caller protects parent globally.
    search.best=None;search.best_distance=math.inf
    search.evaluate(delta)
    if search.best is None:return
    width=search.x.shape[-1];height=search.x.shape[-2]
    partitions=(~strong_mask, strong_mask & ~weak_mask, strong_mask & weak_mask)
    if not weak_first:partitions=(partitions[1],partitions[0],partitions[2])
    for selected in partitions:
        queue=[];sequence=0
        for channel in range(search.x.shape[1]):
            group=loci[selected & (loci//(height*width)==channel)]
            if len(group):heapq.heappush(queue,(-len(group),sequence,group));sequence+=1
        while queue and search.available():
            _,_,group=heapq.heappop(queue)
            group=group[(search.best-search.x).flatten()[group]!=0]
            if not len(group):continue
            child=search.best-search.x;child.flatten()[group]=0
            margin,_=search.evaluate(child)
            if margin>=0 and len(group)>1:
                for part in search.split_group(group):
                    heapq.heappush(queue,(-len(part),sequence,part));sequence+=1
    # Reconsider all failures after successful deletions changed the context.
    while search.available():
        previous=search.best_distance
        active=(search.best-search.x).flatten().nonzero().flatten()
        for index in active:
            if not search.available():return
            child=search.best-search.x;child.flatten()[index]=0;search.evaluate(child)
        if search.best_distance==previous:return


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--parents',nargs=2,type=Path,required=True)
    parser.add_argument('--prior-queries',type=int,required=True)
    parser.add_argument('--queries',type=int,required=True)
    parser.add_argument('--seed',type=int,default=454)
    parser.add_argument('--mode',choices=('ga','weakfirst','strongfirst'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.queries<4 or args.prior_queries<0:parser.error('queries must be >=4 and prior queries nonnegative')
    if args.output.exists():parser.error('output must not exist')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    search=base._SparseRefinementSearch(args.image,'l0')
    search.discovery=False;search.limit=args.queries;search.rng.manual_seed(args.seed)
    parents=[read_tensor(path,search.x) for path in args.parents]
    for parent in parents:
        if int(base.get_model_predictions_for(parent).argmax())==search.reference:
            raise ValueError('parent retains clean prediction')
        search.queries+=1
    original_costs=[search.distance(p-search.x) for p in parents]
    parents.sort(key=lambda image:search.distance(image-search.x))
    strong,weak=parents
    global_image,global_distance=strong.clone(),search.distance(strong-search.x)
    initial=global_distance
    if args.mode=='ga':
        search.best=strong.clone();search.best_distance=global_distance
        engine=SupportGA(search,strong,weak);engine.run()
    else:
        engine=None;priority_prune(search,strong,weak,args.mode=='weakfirst')
    if search.best is not None and search.best_distance<global_distance:
        global_image,global_distance=search.best.clone(),search.best_distance
    elapsed=time.monotonic()-search.started
    predicted=int(base.get_model_predictions_for(global_image).argmax())
    if predicted==search.reference:raise AssertionError('protected archive lost feasibility')
    result=dict(experiment='posthoc_parental_support',mode=args.mode,seed=args.seed,
                image=args.image,parent_paths=[str(p) for p in args.parents],
                parent_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.parents],
                parent_distances=original_costs,
                initial_distance=initial,distance=global_distance,local_distance=search.best_distance,
                diagnostic_queries=search.queries,prior_queries=args.prior_queries,
                queries_limit=args.queries,
                total_charged_queries=search.queries+args.prior_queries,
                seconds=elapsed,reference=search.reference,predicted=predicted,
                post_validation_queries=1,threads=4,torch_version=str(torch.__version__),
                attempts=engine.attempts if engine else None,promotions=engine.promotions if engine else None,
                repeated_supports=engine.duplicates if engine else None,
                min_margin=search.min_margin,history=search.history,
                caveat='Saved-parent mechanism diagnostic, not an end-to-end benchmark.')
    torch.save(global_image,args.output/'best.pt')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'epsilon_ball_source.py').write_bytes(Path('epsilon_ball.py').read_bytes())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
