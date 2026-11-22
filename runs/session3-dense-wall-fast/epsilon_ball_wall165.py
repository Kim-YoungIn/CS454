"""Gradient-free, elitist genetic searches for the course's three distances.

The public entry points return images, not perturbations. Search uses only model
probabilities; no backward pass, surrogate model, or seed-image lookup is used.
CS454_SECONDS/QUERIES/SEED/THREADS provide reproducible experimental controls.
"""
import argparse
import heapq
import math
import os
import time

import torch
import torch.nn.functional as F

from util import get_image_tensor_from, get_model_predictions_for, get_viz_for

LAST_RUN = {}


def _setting(name, default, cast):
    return cast(os.environ.get('CS454_' + name, default))


class _Search:
    def __init__(self, image_path, norm):
        self.started = time.monotonic()
        self.seconds = max(1., min(170., _setting('SECONDS', 165., float)))
        self.limit = max(2, _setting('QUERIES', 100000, int))
        self.rng = torch.Generator(device='cpu').manual_seed(_setting('SEED', 454, int))
        self.x = get_image_tensor_from(image_path).contiguous()
        self.norm = norm
        self.l2_style = os.environ.get('CS454_L2_STYLE', 'energy')
        self.queries = 0
        self.best = None
        self.best_distance = math.inf
        self.min_margin = math.inf
        self.history = []
        self.forward_seconds = 0.
        probabilities = get_model_predictions_for(self.x)[0]
        self.queries += 1
        self.reference = int(probabilities.argmax())

    def rand(self, *shape):
        return torch.rand(shape, generator=self.rng)

    def randint(self, high):
        return int(torch.randint(max(1, high), (), generator=self.rng))

    def available(self, reserve=0., queries=0):
        # Leave at least two recent forward times for cleanup/slow CPU variation.
        slack = max(1., 2 * self.forward_seconds)
        return self.queries < self.limit - queries and time.monotonic() - self.started < self.seconds - slack - reserve

    def distance(self, delta):
        if self.norm == 'linf':
            return float(delta.abs().max())
        if self.norm == 'l2':
            return float(delta.square().sum())
        return int(torch.count_nonzero(delta))

    def evaluate(self, delta):
        image = (self.x + delta).clamp(-3, 3)
        before = time.monotonic()
        probabilities = get_model_predictions_for(image)[0]
        self.forward_seconds = max(self.forward_seconds, time.monotonic() - before)
        self.queries += 1
        logp = probabilities.clamp_min(1e-35).log()
        other = logp.clone()
        other[self.reference] = -torch.inf
        margin = float(logp[self.reference] - other.max())
        self.min_margin = min(self.min_margin, margin)
        actual = image - self.x
        if int(probabilities.argmax()) != self.reference:
            distance = self.distance(actual)
            if distance < self.best_distance:
                self.best, self.best_distance = image.clone(), distance
                self.history.append((self.queries, distance))
        return margin, actual

    def square(self, fraction):
        height, width = self.x.shape[-2:]
        side = max(1, min(height, width, round(math.sqrt(fraction * height * width))))
        row, col = self.randint(height - side + 1), self.randint(width - side + 1)
        return row, col, side

    def signs(self, *shape):
        return (self.rand(*shape) < .5).float().mul_(2).sub_(1)

    def project(self, delta, radius):
        if self.norm == 'linf':
            delta = delta.clamp(-radius, radius)
        else:
            delta = delta * min(1., radius / max(1e-12, float(delta.norm())))
        return (self.x + delta).clamp(-3, 3) - self.x

    def initial(self, radius):
        if self.norm == 'linf':
            # Vertical stripes supply high-contrast starting diversity cheaply.
            return self.project(radius * self.signs(1, 3, 1, self.x.shape[-1]).expand_as(self.x), radius)
        style = self.l2_style
        if style == 'portfolio':
            style = ('additive', 'energy', 'sign')[self.randint(3)]
        if style == 'sign':
            return self.project((radius / math.sqrt(self.x.numel())) *
                                self.signs(1, 3, 1, self.x.shape[-1]).expand_as(self.x), radius)
        if style == 'energy':
            delta = torch.zeros_like(self.x)
            side = max(2, self.x.shape[-1] // 5)
            for row in range(0, self.x.shape[-2] - side + 1, side):
                for col in range(0, self.x.shape[-1] - side + 1, side):
                    delta[:, :, row:row+side, col:col+side] = self.l2_pattern(side)
            return self.project(delta * (radius / float(delta.norm())), radius)
        small = self.signs(1, 3, 5, 5)
        delta = F.interpolate(small, size=self.x.shape[-2:], mode='bilinear', align_corners=False)
        return self.project(delta * (radius / float(delta.norm())), radius)

    def l2_pattern(self, side):
        # Smooth positive/negative lobes retain energy on a small spatial support.
        axis = torch.linspace(-1, 1, side)
        vertical, horizontal = torch.meshgrid(axis, axis, indexing='ij')
        bump = torch.exp(-4 * (vertical.square() + horizontal.square()))
        bump *= torch.where(vertical < 0, -1., 1.)
        if self.rand(1).item() < .5:
            bump = bump.T
        return bump.reshape(1, 1, side, side) * self.signs(1, 3, 1, 1)

    def l2_energy_mutation(self, parent, mate, radius, fraction, generation):
        child = parent.clone()
        row, col, side = self.square(fraction)
        rr, cc, _ = self.square(fraction)
        first = (slice(None), slice(None), slice(row, row+side), slice(col, col+side))
        second = (slice(None), slice(None), slice(rr, rr+side), slice(cc, cc+side))
        old = child[first].clone()
        # Recover the energy in two windows, handling overlap without counting twice.
        child[first] = 0
        child[second] = 0
        spare = max(0., radius * radius - float(child.square().sum()))
        if generation % 7 == 0:
            old = mate[first].clone()
        old /= old.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        pattern = self.l2_pattern(side)
        pattern /= pattern.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        direction = old + pattern
        direction /= direction.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        child[first] = direction * math.sqrt(spare / 3)
        return self.project(child, radius)

    def mutate(self, parent, mate, radius, fraction, generation):
        style = ('energy', 'additive', 'sign')[generation % 3] if self.l2_style == 'portfolio' else self.l2_style
        if self.norm == 'l2' and style == 'energy':
            return self.l2_energy_mutation(parent, mate, radius, fraction, generation)
        child = parent.clone()
        row, col, side = self.square(fraction)
        region = (slice(None), slice(None), slice(row, row + side), slice(col, col + side))
        if generation % 7 == 0:
            # Cross a separate block so mutation does not erase inheritance.
            cr, cc, cs = self.square(fraction)
            cross = (slice(None), slice(None), slice(cr, cr+cs), slice(cc, cc+cs))
            child[cross] = mate[cross]
        if self.norm == 'linf':
            child[region] = radius * self.signs(1, 3, 1, 1)
        elif style == 'sign':
            child[region] = (radius / math.sqrt(self.x.numel())) * self.signs(1, 3, 1, 1)
        else:
            # Structured local mutations retain useful signal outside the patch.
            patch = self.signs(1, 3, 1, 1).expand(1, 3, side, side)
            step = radius * (.2 if generation % 3 else .5)
            child[region] += patch * (step / float(patch.norm()))
        return self.project(child, radius)

    def dense(self):
        radius = .4 if self.norm == 'linf' else 30.
        pool = []
        stage = 0
        stalled = 0
        generation = 0
        while self.available(reserve=max(2., 14 * self.forward_seconds) if self.best is not None else 0.,
                             queries=12 if self.best is not None else 0):
            if not pool or (self.norm == 'l2' and self.l2_style == 'portfolio' and generation < 4):
                delta = self.initial(radius)
            else:
                # A small truncation-selected population; elites never reevaluated.
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
                # Archive is independent of this harder budget's population.
                radius *= .8
                pool = []
                if self.available():
                    scaled = self.project(delta * .8, radius)
                    score, scaled = self.evaluate(scaled)
                    pool = [(score, scaled)]
                stage, stalled = 0, 0
            elif self.best is None and (stage >= 220 or stalled >= 100):
                radius *= 1.6
                pool = []
                stage, stalled = 0, 0
            elif stalled >= 120:
                # Escaping local minima while preserving the current elite.
                if self.available():
                    score, fresh = self.evaluate(self.initial(radius))
                    pool = sorted(pool[:2] + [(score, fresh)], key=lambda item: item[0])
                stalled = 0
        self.polish()

    def sparse_initial(self, count):
        delta = torch.zeros_like(self.x)
        # Contiguous supports are more effective than unrelated isolated pixels.
        side = min(self.x.shape[-1], max(1, math.ceil(math.sqrt(count / 3))))
        row = self.randint(self.x.shape[-2] - side + 1)
        col = self.randint(self.x.shape[-1] - side + 1)
        values = 3 * self.signs(1, 3, 1, 1)
        delta[:, :, row:row+side, col:col+side] = values - self.x[:, :, row:row+side, col:col+side]
        active = delta.flatten().nonzero().flatten()
        if len(active) > count:
            delta.flatten()[active[count:]] = 0
        return delta

    def sparse_mutate(self, parent, mate, count, fraction, generation):
        child = parent.clone()
        row, col, side = self.square(fraction)
        region = (slice(None), slice(None), slice(row, row+side), slice(col, col+side))
        # Occasionally relocate supports rather than just changing their values.
        if generation % 4 == 0:
            child[region] = mate[region]
        else:
            child[region] = 3 * self.signs(1, 3, 1, 1) - self.x[region]
        active = child.flatten().nonzero().flatten()
        if len(active) > count:
            # Mutated patch takes priority; drop random old support elements.
            mask = torch.zeros_like(child, dtype=torch.bool)
            mask[region] = True
            old = (child.ne(0) & ~mask).flatten().nonzero().flatten()
            remove = len(active) - count
            candidates = old if len(old) >= remove else active
            chosen = torch.randperm(len(candidates), generator=self.rng)[:remove]
            child.flatten()[candidates[chosen]] = 0
        return child

    def sparse(self):
        count = 3072
        pool = []
        stage, generation, stalled = 0, 0, 0
        while self.available():
            if not pool or generation % 23 == 0:
                delta = self.sparse_initial(count)
            else:
                parent = pool[0 if self.rand(1).item() < .85 else self.randint(len(pool))][1]
                mate = pool[self.randint(len(pool))][1]
                fraction = max(1 / (224*224), min(count / (3*224*224), .08) * .5 ** (stage // 40))
                delta = self.sparse_mutate(parent, mate, count, fraction, generation)
            margin, delta = self.evaluate(delta)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            stalled = 0 if margin < previous - 1e-5 else stalled + 1
            stage += 1
            generation += 1
            if margin < 0:
                # Random group deletion supplies offspring at a smaller L0 budget.
                count = max(1, int(self.best_distance * .85))
                active = delta.flatten().nonzero().flatten()
                delta = delta.clone()
                chosen = torch.randperm(len(active), generator=self.rng)[count:]
                delta.flatten()[active[chosen]] = 0
                pool = []
                if self.available():
                    score, delta = self.evaluate(delta)
                    pool = [(score, delta)]
                stage, stalled = 0, 0
            elif self.best is None and (stage >= 180 or stalled >= 90):
                count = min(self.x.numel(), int(count * 1.7))
                pool = []
                stage, stalled = 0, 0
            elif self.best is not None and stage >= 100:
                # Restart pruning from a known feasible support instead of drift.
                delta = self.best - self.x
                active = delta.flatten().nonzero().flatten()
                count = max(1, int(self.best_distance * .97))
                chosen = torch.randperm(len(active), generator=self.rng)[count:]
                delta.flatten()[active[chosen]] = 0
                pool = []
                if self.available():
                    score, delta = self.evaluate(delta)
                    pool = [(score, delta)]
                stage, stalled = 0, 0

    def polish(self):
        # Dense search reserves both queries and wall time for boundary refinement.
        if self.best is None:
            return
        delta = self.best - self.x
        low, high = 0., 1.
        for _ in range(12):
            if not self.available():
                break
            middle = (low + high) / 2
            margin, _ = self.evaluate(delta * middle)
            if margin < 0:
                high = middle
            else:
                low = middle

    def run(self):
        if self.norm == 'l0':
            self.sparse()
        else:
            self.dense()
        LAST_RUN.clear()
        LAST_RUN.update(norm=self.norm, l2_style=self.l2_style, queries=self.queries, seconds=time.monotonic()-self.started,
                        reference=self.reference, success=self.best is not None, min_margin=self.min_margin,
                        distance=self.best_distance if self.best is not None else None, history=self.history)
        if self.best is None:
            raise RuntimeError(f'Genetic search exhausted {self.queries} queries without misclassification')
        return self.best.detach().clone()


class _SparseRefinement(_Search):
    def __init__(self, image_path, norm='l0'):
        super().__init__(image_path, norm)
        self.discovery = True
        self.fraction = float(os.environ.get('CS454_DISCOVERY_FRACTION', .55))
        if not 0 < self.fraction <= 1:
            raise ValueError('CS454_DISCOVERY_FRACTION must be in (0, 1]')
        self.reference_budget = max(2, _setting('DISCOVERY_REFERENCE', 900, int))
        self.phase_query = max(2, math.ceil(min(self.limit, self.reference_budget) * self.fraction))
        self.phase_time = (math.inf if 'CS454_QUERIES' in os.environ else
                           self.started + self.seconds * self.fraction)
        self.prune_evaluations = 0
        self.discovery_distance = None
        self.discovery_queries = None

    def available(self, reserve=0., queries=0):
        if self.discovery and self.best is not None:
            if self.queries >= self.phase_query or time.monotonic() >= self.phase_time:
                return False
        return super().available(reserve, queries)

    def split_group(self, group):
        """Median split the widest coordinate, with channels already separated."""
        height, width = self.x.shape[-2:]
        rows = (group // width) % height
        cols = group % width
        coordinates = rows if int(rows.max()-rows.min()) >= int(cols.max()-cols.min()) else cols
        order = coordinates.argsort(stable=True)
        midpoint = len(group)//2
        return group[order[:midpoint]], group[order[midpoint:]]

    def prune(self):
        if self.best is None:
            return
        height, width = self.x.shape[-2:]
        active = (self.best - self.x).flatten().nonzero().flatten()
        queue, sequence = [], 0
        for channel in range(self.x.shape[1]):
            group = active[(active // (height*width)) == channel]
            if len(group):
                heapq.heappush(queue, (-len(group), sequence, group))
                sequence += 1
        while queue and self.available():
            _, _, group = heapq.heappop(queue)
            delta = self.best - self.x
            group = group[delta.flatten()[group] != 0]
            if not len(group):
                continue
            child = delta.clone()
            child.flatten()[group] = 0
            margin, _ = self.evaluate(child)
            self.prune_evaluations += 1
            if margin >= 0 and len(group) > 1:
                for part in self.split_group(group):
                    heapq.heappush(queue, (-len(part), sequence, part))
                    sequence += 1
        # If every individual deletion was exhausted, spending more identical
        # queries would add no information. The public archive remains feasible.

    def sparse(self):
        super().sparse()
        self.discovery = False
        self.discovery_distance = self.best_distance if self.best is not None else None
        self.discovery_queries = self.queries
        self.prune()

class _SparseRefinementSearch(_SparseRefinement):
    def __init__(self, image_path, norm='l0'):
        super().__init__(image_path, norm)
        self.cleanup_evaluations = 0
        self.cleanup_passes = 0
        self.deletion_minimal = False

    def prune(self):
        super().prune()
        if self.best is None:
            return
        while self.available():
            active = (self.best-self.x).flatten().nonzero().flatten()
            improved, complete = False, True
            for index in active:
                if not self.available():
                    complete = False
                    break
                child = self.best-self.x
                child.flatten()[index] = 0
                previous = self.best_distance
                self.evaluate(child)
                self.cleanup_evaluations += 1
                improved |= self.best_distance < previous
            self.cleanup_passes += 1
            if complete and not improved:
                self.deletion_minimal = True
                break
            if not complete:
                break

    def run(self):
        try:
            return super().run()
        finally:
            LAST_RUN.update(l0_style='quota_hierarchical_and_repeated_deletion',
                            discovery_reference=self.reference_budget,
                            discovery_fraction=self.fraction,
                            discovery_queries=self.discovery_queries,
                            discovery_distance=self.discovery_distance,
                            prune_evaluations=self.prune_evaluations,
                            cleanup_evaluations=self.cleanup_evaluations,
                            cleanup_passes=self.cleanup_passes,
                            deletion_minimal=self.deletion_minimal)


class _MultistartSparseSearch(_SparseRefinementSearch):
    def __init__(self,image_path,norm='l0'):
        super().__init__(image_path,norm)
        self.initial_seed=int(os.environ.get('CS454_SEED',454))
        self.chunks=[]
        self.selected_seed=None

    def sparse(self):
        best_image,best_distance,best_minimal=None,math.inf,False
        history=[]
        while _Search.available(self):
            remaining=self.limit-self.queries
            seconds_left=self.seconds-(time.monotonic()-self.started)
            if self.chunks and (remaining<256 or seconds_left<5):
                break
            seed=self.initial_seed+len(self.chunks)
            start=self.queries
            self.rng.manual_seed(seed)
            self.best,self.best_distance=None,math.inf
            self.history=[]
            self.discovery=True
            self.discovery_distance=self.discovery_queries=None
            self.prune_evaluations=self.cleanup_evaluations=self.cleanup_passes=0
            self.deletion_minimal=False
            # Independent controls count one clean forward per run. Reuse the
            # known clean class here, but retain its virtual slot in each quota.
            local_budget=min(remaining+1,self.reference_budget)
            self.phase_query=start+max(1,math.ceil(local_budget*self.fraction)-1)
            self.phase_time=(math.inf if 'CS454_QUERIES' in os.environ else
                             self.started+self.seconds*self.fraction if not self.chunks else
                             time.monotonic()+seconds_left*self.fraction)
            super().sparse()
            self.chunks.append(dict(seed=seed,start_query=start,end_query=self.queries,
                                    distance=self.best_distance if self.best is not None else None,
                                    discovery_query=self.discovery_queries,
                                    discovery_distance=self.discovery_distance,
                                    prune_evaluations=self.prune_evaluations,
                                    cleanup_evaluations=self.cleanup_evaluations,
                                    deletion_minimal=self.deletion_minimal))
            for query,distance in self.history:
                if not history or distance<history[-1][1]:
                    history.append((query,distance))
            if self.best is not None and self.best_distance<best_distance:
                best_image=self.best.clone()
                best_distance=self.best_distance
                best_minimal=self.deletion_minimal
                self.selected_seed=seed
            if self.best is None or best_distance<=1:
                break
        self.best,self.best_distance,self.history=best_image,best_distance,history
        self.deletion_minimal=best_minimal

    def run(self):
        try:
            return super().run()
        finally:
            LAST_RUN.update(l0_style='multistart_quota_refinement',
                            chunks=self.chunks, selected_seed=self.selected_seed,
                            discovery_evaluations=sum(max(0, c['discovery_query']-c['start_query'])
                                                      for c in self.chunks if c['discovery_query'] is not None),
                            prune_evaluations=sum(c['prune_evaluations'] for c in self.chunks),
                            cleanup_evaluations=sum(c['cleanup_evaluations'] for c in self.chunks))


def _attack(image_path, norm):
    threads = torch.get_num_threads()
    LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, _setting('THREADS', 4, int)))
        with torch.inference_mode():
            style = os.environ.get('CS454_L0_STYLE', 'multistart')
            if norm == 'l0' and style not in ('original', 'refined', 'multistart'):
                raise ValueError('CS454_L0_STYLE must be original, refined, or multistart')
            search_type = _Search
            if norm == 'l0':
                search_type = {'original': _Search, 'refined': _SparseRefinementSearch,
                               'multistart': _MultistartSparseSearch}[style]
            return search_type(image_path, norm).run()
    finally:
        torch.set_num_threads(threads)


def make_Linf_adversarial_from(image_path):
    return _attack(image_path, 'linf')


def make_L2_adversarial_from(image_path):
    return _attack(image_path, 'l2')


def make_L0_adversarial_from(image_path):
    return _attack(image_path, 'l0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='L-norm Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    parser.add_argument('-n', '--norm', type=str.lower, required=True, choices=('inf', '2', '0'))
    args = parser.parse_args()
    attack = {'inf': make_Linf_adversarial_from, '2': make_L2_adversarial_from, '0': make_L0_adversarial_from}[args.norm]
    with torch.inference_mode():
        get_viz_for(attack(args.filename))
