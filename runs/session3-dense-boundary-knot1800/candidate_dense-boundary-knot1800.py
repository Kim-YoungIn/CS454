"""Experimental feasible-direction evolution after dense GA discovery.

Keep the original energy-GA discovery prefix through three successful radii.
Then evolve directions near the verified feasible boundary, using one model
query per proposal and small radial improvements. All selection is genetic;
no gradients, finite differences, surrogate, or image-specific data are used.
This is research code, independent of the production implementation.
"""
import math

import torch

import epsilon_ball
from epsilon_ball import _Search, _setting

LAST_RUN = {}


class BoundaryDirections(_Search):
    def __init__(self, image_path, norm='l2'):
        super().__init__(image_path, norm)
        if norm != 'l2':
            raise ValueError('Boundary direction experiment supports L2 only')
        self.trace = []
        self.switch_successes = max(1, _setting('BOUNDARY_SUCCESSES', 3, int))

    def rescale(self, delta, radius):
        return self.project(delta * (radius / max(1e-12, float(delta.norm()))), radius)

    def dense(self):
        # Deliberately reproduce the original search prefix, including random
        # draws, restart rules and scaled warm-start queries.
        radius, pool = 30., []
        stage = stalled = generation = successes = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds) if self.best is not None else 0.,
                             queries=12 if self.best is not None else 0):
            if not pool:
                delta = self.initial(radius)
            else:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
                mate = pool[self.randint(len(pool))][1]
                fraction = max(.0004, .25 * (.5 ** (stage // 35)))
                delta = self.mutate(parent, mate, radius, fraction, generation)
            margin, delta = self.evaluate(delta)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            stalled = 0 if margin < previous - 1e-5 else stalled + 1
            stage += 1
            generation += 1
            if margin < 0:
                successes += 1
                if successes >= self.switch_successes:
                    self.trace.append(dict(event='boundary_start', query=self.queries,
                                           distance=self.best_distance))
                    self.boundary(generation)
                    return
                radius *= .8
                pool = []
                if self.available():
                    score, scaled = self.evaluate(self.project(delta * .8, radius))
                    pool = [(score, scaled)]
                stage = stalled = 0
            elif self.best is None and (stage >= 220 or stalled >= 100):
                radius *= 1.6
                pool = []
                stage = stalled = 0
            elif stalled >= 120:
                if self.available():
                    score, fresh = self.evaluate(self.initial(radius))
                    pool = sorted(pool[:2] + [(score, fresh)], key=lambda item: item[0])
                stalled = 0
        self.polish()

    def boundary(self, generation):
        self.polish()
        target_fraction = _setting('BOUNDARY_FRACTION', .99, float)
        if not .9 <= target_fraction < 1:
            raise ValueError('BOUNDARY_FRACTION must be in [.9, 1)')
        alpha = _setting('BOUNDARY_ALPHA', .25, float)
        if not 0 < alpha <= 1:
            raise ValueError('BOUNDARY_ALPHA must be in (0, 1]')
        target = math.sqrt(self.best_distance) * target_fraction
        pool = []
        queued = None
        feasible = [self.best - self.x]
        stage = stalled = accepted = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds), queries=12):
            if pool:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
            else:
                parent = self.rescale(feasible[0], target)
            mate = (pool[self.randint(len(pool))][1] if pool and generation % 7 == 0
                    else self.rescale(feasible[self.randint(len(feasible))], target))
            fraction = max(.0004, .04 * .5 ** (stage // 50))
            if queued is not None:
                child, queued = queued, None
            else:
                mutant = self.l2_energy_mutation(parent, mate, target, fraction, generation)
                child = self.rescale((1 - alpha) * parent + alpha * mutant, target)
            margin, delta = self.evaluate(child)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            stalled = 0 if margin < previous - 1e-5 else stalled + 1
            stage += 1
            generation += 1
            if margin < 0:
                accepted += 1
                # Retain actual scored directions; the global archive itself
                # remains the exact clipped image saved by _Search.evaluate.
                feasible.insert(0, delta.clone())
                feasible = feasible[:4]
                self.trace.append(dict(event='boundary_accept', query=self.queries,
                                       distance=self.best_distance, stage_queries=stage,
                                       alpha=alpha, fraction=target_fraction))
                target = math.sqrt(self.best_distance) * target_fraction
                pool = []
                stage = stalled = 0
            elif stalled >= 120:
                # A smaller angular move is useful near a narrow boundary;
                # keep the best failed direction to preserve model evidence.
                alpha = max(.04, alpha * .7)
                stalled = 0
            if stage and stage % 256 == 0:
                target_fraction = (1 + target_fraction) / 2
                target = math.sqrt(self.best_distance) * target_fraction
                queued = self.rescale(pool[0][1], target)
                pool = []
                self.trace.append(dict(event='boundary_relax', query=self.queries,
                                       fraction=target_fraction, alpha=alpha))
        self.trace.append(dict(event='boundary_end', query=self.queries, accepted=accepted))
        self.polish()


def make_L2_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    LAST_RUN.clear()
    epsilon_ball.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, _setting('THREADS', 4, int)))
        with torch.inference_mode():
            search = BoundaryDirections(image_path)
            return search.run()
    finally:
        if search is not None:
            LAST_RUN.update(epsilon_ball.LAST_RUN, dense_schedule='boundary_directions',
                            dense_trace=search.trace)
        torch.set_num_threads(previous_threads)
