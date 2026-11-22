"""Trajectory-preserving mate preparation optimization of boundary evolution.

Unused mate draws are still consumed. Energy mutation reads mate only every
seventh generation, so other generations need no mate normalization.
"""
import math

import torch

import epsilon_ball
from epsilon_ball import _setting
from research.dense_session3_boundary import BoundaryDirections

LAST_RUN = {}
SOURCE_DEPENDENCIES = ('research/dense_session3_boundary.py',)


class FastBoundaryDirections(BoundaryDirections):
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
        self.mate_rescales = self.mate_unused = 0
        stage = stalled = accepted = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds), queries=12):
            if pool:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
            else:
                parent = self.rescale(feasible[0], target)
            if pool and generation % 7 == 0:
                mate = pool[self.randint(len(pool))][1]
            else:
                # Preserve the original random draw even when mutation will
                # never read this mate. This is essential for exact replay.
                index = self.randint(len(feasible))
                if generation % 7:
                    mate = parent
                    self.mate_unused += 1
                else:
                    mate = self.rescale(feasible[index], target)
                    self.mate_rescales += 1
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
                feasible.insert(0, delta.clone())
                feasible = feasible[:4]
                self.trace.append(dict(event='boundary_accept', query=self.queries,
                                       distance=self.best_distance, stage_queries=stage,
                                       alpha=alpha, fraction=target_fraction))
                target = math.sqrt(self.best_distance) * target_fraction
                pool = []
                stage = stalled = 0
            elif stalled >= 120:
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
            search = FastBoundaryDirections(image_path)
            return search.run()
    finally:
        if search is not None:
            LAST_RUN.update(epsilon_ball.LAST_RUN, dense_schedule='boundary_directions_fast',
                            dense_trace=search.trace,
                            mate_rescales=getattr(search, 'mate_rescales', 0),
                            mate_unused=getattr(search, 'mate_unused', 0))
        torch.set_num_threads(previous_threads)
