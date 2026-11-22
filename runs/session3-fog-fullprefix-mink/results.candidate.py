"""Extend the verified fog screen with a genetic search over integer octaves."""
import os
import time
import numpy as np
import torch
import perlin_fog as baseline
from util import get_image_tensor_from, get_model_predictions_for

SOURCE_DEPENDENCIES=('perlin_fog.py',)
FOG_SECONDS=270.
FOG_GENERATIONS=24
FOG_OCTAVES=baseline.FOG_OCTAVES
LAST_RUN={}


def adv_func(image_path):
    started=time.monotonic()
    deadline=started+min(285.,max(1.,float(FOG_SECONDS)))
    old_threads=torch.get_num_threads()
    old_config=(baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES)
    LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1,int(os.environ.get('CS454_THREADS',4))))
        with torch.inference_mode():
            # Preserve the complete existing search as a successful fallback.
            # If it converges early, use the remaining time for new octaves.
            baseline.FOG_SECONDS=min(285.,max(1.,float(FOG_SECONDS)))
            baseline.FOG_GENERATIONS=max(0,int(os.environ.get('CS454_FOG_PREFIX_GENERATIONS',old_config[1])))
            baseline.FOG_OCTAVES=FOG_OCTAVES
            genes=baseline.adv_func(image_path)
            stats=dict(baseline.LAST_RUN)
            baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES=old_config
            image=get_image_tensor_from(image_path)
            reference=stats['reference']
            queries=stats['queries']
            best=[(int(o),float(m)) for o,m in genes]
            best_cost=sum(m for _,m in best)
            initial=best_cost
            forward_time=4*stats.get('seconds',0.)/max(1,queries)
            layer_time=0.
            layers={}
            rng=np.random.default_rng(int(os.environ.get('CS454_SEED',454)))
            population_size=16
            max_octave=128

            def available():
                return time.monotonic()<deadline-max(3.,2*forward_time+2*layer_time)

            def evaluate(octaves,amplitudes):
                nonlocal queries,forward_time,layer_time
                scores=np.full(len(octaves),np.inf)
                for offset in range(0,len(octaves),4):
                    fields=[]
                    for octave in octaves[offset:offset+4]:
                        octave=int(octave)
                        if not available():break
                        if octave not in layers:
                            t=time.monotonic()
                            layers[octave]=baseline.perlin_image_generator(octave)
                            layer_time=max(layer_time,time.monotonic()-t)
                        fields.append(layers[octave])
                    if not fields or not available():break
                    amplitudes_chunk=torch.tensor(amplitudes[offset:offset+len(fields)],dtype=image.dtype).view(-1,1,1,1)
                    fog=torch.cat(fields)*amplitudes_chunk
                    t=time.monotonic()
                    p=get_model_predictions_for(baseline.add_noise_to_image(fog,image))
                    forward_time=max(forward_time,time.monotonic()-t)
                    queries+=len(fields)
                    own=p[:,reference].clamp_min(1e-35).log()
                    p[:,reference]=-1
                    rival=p.max(1).values.clamp_min(1e-35).log()
                    scores[offset:offset+len(fields)]=(own-rival).tolist()
                return scores

            population=np.array(list(FOG_OCTAVES)+list(rng.integers(1,max_octave+1,size=population_size-len(FOG_OCTAVES))))
            completed=0
            improvements=[]
            for generation in range(FOG_GENERATIONS):
                if not available():break
                target=best_cost*.985
                scores=evaluate(population,np.full(population_size,target))
                completed+=1
                order=np.argsort(scores)
                parents=population[order[:4]].copy()
                if scores[order[0]]<0:
                    octave=int(parents[0])
                    low,high=0.,target
                    for _ in range(9):
                        if not available():break
                        middle=(low+high)/2
                        if evaluate([octave],[middle])[0]<0:high=middle
                        else:low=middle
                    if high<best_cost:
                        best,best_cost=[(octave,float(high))],float(high)
                        improvements.append((queries,octave,best_cost))
                left=parents[rng.integers(4,size=population_size-4)]
                right=parents[rng.integers(4,size=population_size-4)]
                alpha=rng.random(population_size-4)
                children=alpha*left+(1-alpha)*right
                children+=rng.normal(0,(2.,8.,24.)[generation%3],len(children))
                children=np.rint(children).clip(1,max_octave).astype(int)
                children[-2:]=rng.integers(1,max_octave+1,size=2)
                population=np.concatenate([parents,children])
            # The baseline fallback was already reference-verified. Newly
            # improved genotypes need their own exact reference verification.
            if improvements:
                for _ in range(2):
                    rendered=baseline.add_noise_to_image(baseline.genotype2phenotype(best),image)
                    queries+=1
                    if int(get_model_predictions_for(rendered).argmax())!=reference:break
                    best=[(o,m*1.001) for o,m in best]
                    best_cost=sum(m for _,m in best)
                else:raise RuntimeError('integer-octave result failed exact renderer verification')
            LAST_RUN.update(queries=queries,generations=completed,reference=reference,
                            initial_score=initial,distance=best_cost,seconds=time.monotonic()-started,
                            prefix_queries=stats['queries'],prefix_seconds=stats.get('seconds'),
                            prefix_generations=stats.get('generations'),
                            unique_octaves=len(layers),improvements=improvements)
            return best
    finally:
        baseline.FOG_SECONDS,baseline.FOG_GENERATIONS,baseline.FOG_OCTAVES=old_config
        torch.set_num_threads(old_threads)
