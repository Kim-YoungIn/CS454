import argparse
import os
import random
from datetime import datetime

import torch
from perlin_noise import PerlinNoise

from util import *

LAST_RUN = {}

def perlin_image_generator(octaves):
    size = min(octaves*5, 224)
    noise = PerlinNoise(octaves=octaves, seed=1)
    pic = [[noise([i/size, j/size]) for j in range(size)] for i in range(size)]
    pic_tensor = TRANSFORM(torch.Tensor(pic).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0)
    pic_tensor = (pic_tensor - pic_tensor.min())/(pic_tensor.max()-pic_tensor.min())
    return pic_tensor

def genotype2phenotype(genes):
    noise = get_random_noise()*0
    for octave, magnitude in genes:
        noise += magnitude * perlin_image_generator(octave)
    return noise

def add_noise_to_image(noise, image):
    noise = noise.clip(0, 1)
    actual_white = torch.Tensor([[[2.2489]], [[2.428]], [[2.64]]]).repeat(1, 224, 224).unsqueeze(0)
    return (1-noise)*image + noise*actual_white

def adv_func(image_path):
    """Evolve a nonnegative mixture of the instructor's fixed Perlin fields.

    The chromosome is a distribution over octaves; a separate scalar is the
    exact grading cost. Selection uses the untargeted log-probability margin,
    and successful chromosomes undergo a scalar boundary search. No model
    derivatives, reference labels, or image-specific recipes are used.
    """
    import time

    import numpy as np

    started = time.monotonic()
    deadline = started + min(285.0, max(1.0, float(FOG_SECONDS)))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(os.environ.get('CS454_THREADS', 4))))
    try:
        with torch.inference_mode():
            image = get_image_tensor_from(image_path)
            reference = int(get_model_predictions_for(image).argmax(1).item())
            queries = 1
            forward_seconds = 0.0

            def available():
                return time.monotonic() < deadline - max(3.0, 2 * forward_seconds)
            # Preserve the renderer above verbatim. These exact layers are
            # reused only inside this search, never installed into util.py.
            octaves = (1, 2, 3, 4, 6, 8, 12, 24)
            layers = torch.cat([perlin_image_generator(o) for o in octaves])
            rng = np.random.default_rng(int(os.environ.get('CS454_SEED', 454)))
            population_size = 16

            def normalize(values):
                values = np.maximum(values, 0)
                totals = values.sum(axis=1, keepdims=True)
                values[totals[:, 0] == 0, 0] = 1
                return values / values.sum(axis=1, keepdims=True)

            def margins(directions, magnitudes):
                nonlocal queries, forward_seconds
                weights = torch.as_tensor(
                    directions * np.asarray(magnitudes).reshape(-1, 1),
                    dtype=image.dtype,
                )
                fields = (weights @ layers.flatten(1)).reshape(-1, 3, 224, 224)
                results = np.full(len(directions), np.inf)
                # Small batches bound activation memory on ordinary CPUs.
                offset = 0
                for field in fields.split(4):
                    # Reserve time for exact regeneration and a final forward
                    # pass, accounting for the CPU speed actually observed.
                    if not available():
                        break
                    before = time.monotonic()
                    predictions = get_model_predictions_for(add_noise_to_image(field, image))
                    forward_seconds = max(forward_seconds, time.monotonic() - before)
                    queries += len(field)
                    own = predictions[:, reference].clamp_min(1e-35).log()
                    predictions[:, reference] = -1
                    rival = predictions.max(1).values.clamp_min(1e-35).log()
                    results[offset:offset + len(field)] = (own - rival).tolist()
                    offset += len(field)
                return results

            directions = np.concatenate([
                np.eye(len(octaves)),
                rng.dirichlet(np.ones(len(octaves)), population_size - len(octaves)),
            ])
            magnitude = 0.5
            scores = margins(directions, np.full(population_size, magnitude))
            while not np.any(scores < 0):
                magnitude *= 2
                if not available():
                    raise RuntimeError("Fog time budget exhausted before finding an adversary")
                scores = margins(directions, np.full(population_size, magnitude))

            # Bracket every successful initial direction at once, avoiding a
            # bias toward the first octave that happens to change the class.
            candidates = directions[scores < 0]
            lo = np.zeros(len(candidates))
            hi = np.full(len(candidates), magnitude)
            for _ in range(9):
                if not available():
                    break
                mid = (lo + hi) / 2
                successful = margins(candidates, mid) < 0
                hi = np.where(successful, mid, hi)
                lo = np.where(successful, lo, mid)
            winner = int(np.argmin(hi))
            best_direction, best_magnitude = candidates[winner].copy(), float(hi[winner])
            initial_magnitude = best_magnitude

            target = best_magnitude * 0.98
            directions[0] = best_direction
            stale = 0
            completed = 0
            for generation in range(FOG_GENERATIONS):
                if not available():
                    break
                scores = margins(directions, np.full(population_size, target))
                completed += 1
                ranking = np.argsort(scores)
                parents = directions[ranking[:4]].copy()
                if scores[ranking[0]] < 0:
                    candidate = parents[0]
                    lower, upper = 0.0, target
                    for _ in range(9):
                        if not available():
                            break
                        middle = (lower + upper) / 2
                        if margins(candidate[None], [middle])[0] < 0:
                            upper = middle
                        else:
                            lower = middle
                    best_direction, best_magnitude = candidate.copy(), upper
                    target = best_magnitude * 0.98
                    stale = 0
                else:
                    stale += 1
                left = parents[rng.integers(4, size=population_size - 4)]
                right = parents[rng.integers(4, size=population_size - 4)]
                children = np.where(rng.random(left.shape) < 0.5, left, right)
                # Alternate broad mutations and fine local refinement.
                step = (0.12, 0.04, 0.015)[generation % 3]
                children += rng.normal(0, step, children.shape)
                directions = np.concatenate([parents, normalize(children)])
                directions[0] = best_direction
                if stale and stale % 12 == 0:
                    directions[-4:] = rng.dirichlet(np.ones(len(octaves)) * 0.3, 4)
                if stale >= 36:
                    break

            # Keep the last verified successful endpoint. Zero coefficients
            # are omitted so every submitted magnitude is strictly positive.
            for _ in range(2):
                genes = [(octave, float(weight * best_magnitude))
                         for octave, weight in zip(octaves, best_direction) if weight > 0]
                regenerated = add_noise_to_image(genotype2phenotype(genes), image)
                queries += 1
                if int(get_model_predictions_for(regenerated).argmax(1).item()) != reference:
                    break
                # The reference renderer accumulates layers in a different
                # floating-point order from the batched search matrix product.
                best_magnitude *= 1.001
            else:
                raise RuntimeError("Fog candidate failed reference-renderer verification")
            LAST_RUN.clear()
            LAST_RUN.update(queries=queries, generations=completed, reference=reference,
                            seconds=time.monotonic() - started, initial_score=initial_magnitude,
                            distance=best_magnitude)
            return genes
    finally:
        torch.set_num_threads(previous_threads)


# Fixed generation limits keep completed runs reproducible. A wall-clock guard
# returns the best verified genotype early on slower grading hardware.
FOG_SECONDS = min(285.0, max(1.0, float(os.environ.get('CS454_FOG_SECONDS', 270.0))))
FOG_GENERATIONS = max(0, int(os.environ.get('CS454_FOG_GENERATIONS', 80)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Perlin Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    args = parser.parse_args()
    
    noise_genotype = adv_func(args.filename)
    noise_img = genotype2phenotype(noise_genotype)
    org_img = get_image_tensor_from(args.filename)
    adv_image = add_noise_to_image(noise_img, org_img)
    get_viz_for(adv_image)
