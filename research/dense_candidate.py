"""Experimental continuation scheduler for the verified spatial genetic search.

Only dense-budget continuation changes. Fitness, projection, model access,
mutation and crossover are inherited from epsilon_ball._Search. Production is
unchanged until fixed-budget experiments justify promotion.

After an unsuccessful stage, move the target radius toward the last successful
radius instead of searching indefinitely at the same 20% contraction. Reuse
both the strongest unsuccessful direction and the successful archive. A fast
success makes the next contraction a little more ambitious; difficult stages
make it gentler. All selected genomes are evaluated at their current radius.
"""
import math

import torch

import epsilon_ball
from epsilon_ball import _Search, _setting

LAST_RUN = {}


class DenseContinuation(_Search):
    def __init__(self, image_path, norm):
        super().__init__(image_path, norm)
        self.patience = max(10, _setting('DENSE_PATIENCE', 256, int))
        self.stage_trace = []

    def archive_radius(self):
        if self.norm == 'l2':
            return math.sqrt(self.best_distance)
        return self.best_distance

    def delta_radius(self, delta):
        if self.norm == 'linf':
            return float(delta.abs().max())
        return float(delta.norm())

    def at_radius(self, delta, radius):
        return self.project(delta * (radius / max(1e-12, self.delta_radius(delta))), radius)

    def dense(self):
        radius = .4 if self.norm == 'linf' else 30.
        contraction = .8
        pool, queued = [], []
        stage = shape_stage = stalled = generation = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds) if self.best is not None else 0.,
                             queries=12 if self.best is not None else 0):
            warm_start = bool(queued)
            if warm_start:
                delta = queued.pop(0)
            elif not pool:
                delta = self.initial(radius)
            else:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
                mate = pool[self.randint(len(pool))][1]
                fraction = max(.0004, .25 * (.5 ** (shape_stage // 35)))
                delta = self.mutate(parent, mate, radius, fraction, generation)
            margin, delta = self.evaluate(delta)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            stalled = 0 if margin < previous - 1e-5 else stalled + 1
            # Warm-start evaluations consume model queries but do not advance
            # the genetic crossover cadence or coarse-to-fine mutation clock.
            if not warm_start:
                stage += 1
                shape_stage += 1
                generation += 1
            if pool[0][0] < 0:
                # Reuse the strongest feasible elite, including a cached scaled
                # offspring; do not wait for a weaker child to become feasible.
                winner = pool[0][1]
                feasible = self.archive_radius()
                if stage <= 20:
                    contraction = max(.8, contraction - .025)
                self.stage_trace.append(dict(query=self.queries, event='success',
                                             stage_queries=stage, radius=radius,
                                             best_distance=self.best_distance,
                                             contraction=contraction))
                radius = feasible * contraction
                queued = [self.at_radius(winner, radius)]
                pool = []
                stage = shape_stage = stalled = 0
            elif self.best is None and (stage >= 220 or stalled >= 100):
                self.stage_trace.append(dict(query=self.queries, event='expand', radius=radius,
                                             stage_queries=stage))
                radius *= 1.6
                pool, queued = [], []
                stage = shape_stage = stalled = 0
            elif self.best is not None and stage >= self.patience:
                previous_radius = radius
                elite = pool[0][1]
                contraction = min(.99, (1. + contraction) / 2.)
                radius = self.archive_radius() * contraction
                self.stage_trace.append(dict(query=self.queries, event='relax',
                                             stage_queries=stage, from_radius=previous_radius,
                                             radius=radius, contraction=contraction,
                                             best_margin=pool[0][0],
                                             best_distance=self.best_distance))
                queued = [self.at_radius(elite, radius),
                          self.at_radius(self.best - self.x, radius)]
                pool = []
                stage = stalled = 0
                # Keep the finer spatial mutation scale learned during an
                # unsuccessful stage, rather than returning to broad blocks.
            elif stalled >= 120:
                if self.available():
                    score, fresh = self.evaluate(self.initial(radius))
                    pool = sorted(pool[:2] + [(score, fresh)], key=lambda item: item[0])
                stalled = 0
        self.polish()


def _attack(image_path, norm):
    previous_threads = torch.get_num_threads()
    search = None
    LAST_RUN.clear()
    epsilon_ball.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, _setting('THREADS', 4, int)))
        with torch.inference_mode():
            search = DenseContinuation(image_path, norm)
            return search.run()
    finally:
        if search is not None:
            metadata = dict(epsilon_ball.LAST_RUN)
            metadata.update(dense_schedule='adaptive_continuation',
                            dense_patience=search.patience, dense_trace=search.stage_trace)
            LAST_RUN.clear()
            LAST_RUN.update(metadata)
        torch.set_num_threads(previous_threads)


def make_Linf_adversarial_from(image_path):
    return _attack(image_path, 'linf')


def make_L2_adversarial_from(image_path):
    return _attack(image_path, 'l2')
