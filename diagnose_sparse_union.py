"""Post-hoc sparse-elite recombination diagnostic; never an attack entry point.

Parents must already be evolved and saved. Charge their generation budget via
--prior-queries; this diagnostic does not claim an end-to-end online benchmark.
"""
import argparse
import hashlib
import json
import math
import time
from contextlib import chdir
from pathlib import Path
import torch
import epsilon_ball as base
from util import get_viz_for
from verify_artifacts import read_tensor


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--parents',nargs=2,type=Path,required=True)
    parser.add_argument('--prior-queries',type=int,required=True)
    parser.add_argument('--queries',type=int,default=187)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.queries<4 or args.prior_queries<0:parser.error('queries must be at least4; prior queries nonnegative')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):parser.error('output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    started=time.monotonic()
    search=base._SparseRefinementSearch(args.image,'l0')
    search.discovery=False
    search.limit=args.queries
    parents=[read_tensor(path,search.x) for path in args.parents]
    costs=[]
    for parent in parents:
        predicted=int(base.get_model_predictions_for(parent).argmax())
        search.queries+=1
        if predicted==search.reference:raise ValueError('parent retains clean class')
        costs.append(search.distance(parent-search.x))
    selected=min(range(2),key=lambda i:costs[i])
    global_image,global_distance=parents[selected].clone(),costs[selected]
    union=(search.x+(parents[0]-search.x)+(parents[1]-search.x)).clamp(-3,3)
    margin,_=search.evaluate(union-search.x)
    feasible=search.best is not None
    initial=search.best_distance if feasible else None
    if feasible:search.prune()
    if search.best is not None and search.best_distance<global_distance:
        global_image,global_distance=search.best.clone(),search.best_distance
        source='union'
    else:source=f'parent{selected+1}'
    data=dict(experiment='posthoc_elite_union',image=args.image,
              parent_paths=[str(p) for p in args.parents],parent_distances=costs,
              parent_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.parents],
              union_initial_distance=initial,union_feasible=feasible,union_margin=margin,
              distance=global_distance,selected=source,diagnostic_queries=search.queries,
              charged_prior_queries=args.prior_queries,total_charged_queries=args.prior_queries+search.queries,
              seconds=time.monotonic()-started,deletion_minimal=search.deletion_minimal,
              caveat='Posthoc mechanism diagnostic, not an online end-to-end implementation or runtime measurement.')
    (output/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    (output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'epsilon_ball_source.py').write_bytes(Path('epsilon_ball.py').read_bytes())
    with chdir(output):
        get_viz_for(global_image.clone(),save_to=f'{Path(args.image).stem}_l0.png')
        import matplotlib.pyplot as plt
        plt.close('all')
    print(json.dumps(data),flush=True)

if __name__=='__main__':main()
