"""Retain global energy evolution before switching to local boundary directions.

The minimum exploration stage addresses premature switching on easy initial
images. It uses search work, never image identity, labels or fixture scores.
"""
import math

import torch
import epsilon_ball
from epsilon_ball import _setting
from research.dense_radial_probe import RadialProbeDirections

LAST_RUN = {}
SOURCE_DEPENDENCIES = ('research/dense_radial_probe.py', 'research/dense_session3_boundary.py')


class ExplorationBoundary(RadialProbeDirections):
    def __init__(self, image_path, norm='l2'):
        super().__init__(image_path, norm)
        self.minimum_generations = max(0, _setting('BOUNDARY_MIN_GENERATIONS', 256, int))

    def dense(self):
        # Deliberately reproduce the original search prefix, including random
        # draws, restart rules and scaled warm-start queries.
        radius, pool = 30., []
        stage = stalled = generation = successes = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds) if self.best is not None else 0.,
                             queries=12 if self.best is not None else 0):
            if successes >= self.switch_successes and generation >= self.minimum_generations:
                self.trace.append(dict(event='boundary_start', query=self.queries,
                                       distance=self.best_distance, generation=generation))
                self.boundary(generation)
                return
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
                if successes >= self.switch_successes and generation >= self.minimum_generations:
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


def make_L2_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    LAST_RUN.clear()
    epsilon_ball.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, _setting('THREADS', 4, int)))
        with torch.inference_mode():
            search = ExplorationBoundary(image_path)
            return search.run()
    finally:
        if search is not None:
            LAST_RUN.update(epsilon_ball.LAST_RUN, dense_schedule='exploration_then_boundary',
                            minimum_generations=search.minimum_generations,
                            dense_trace=search.trace,
                            mate_rescales=getattr(search, 'mate_rescales', 0),
                            mate_unused=getattr(search, 'mate_unused', 0),
                            probe_calls=getattr(search, 'probe_calls', 0),
                            probe_successes=getattr(search, 'probe_successes', 0),
                            probe_trace=getattr(search, 'probe_trace', []))
        torch.set_num_threads(previous_threads)
