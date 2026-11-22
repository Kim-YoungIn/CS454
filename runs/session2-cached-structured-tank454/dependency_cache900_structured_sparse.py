"""Experimental memetic GA: spatial discovery followed by hierarchical support pruning.

The discovery GA is the verified implementation. Pruning deletes channel-element
support groups from a successful phenotype; each accepted deletion must still
change the clean argmax. This module never reads saved attacks or seed labels.
"""
import heapq
import math
import os
import time

import torch

import epsilon_ball as base

LAST_RUN = {}


class StructuredSearch(base._Search):
    def __init__(self, image_path, norm='l0'):
        super().__init__(image_path, norm)
        self.discovery = True
        self.fraction = float(os.environ.get('CS454_DISCOVERY_FRACTION', .55))
        if not 0 < self.fraction <= 1:
            raise ValueError('CS454_DISCOVERY_FRACTION must be in (0, 1]')
        self.phase_query = max(2, math.ceil(self.limit * self.fraction))
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


def make_L0_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    base.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, int(os.environ.get('CS454_THREADS', 4))))
        with torch.inference_mode():
            search = StructuredSearch(image_path)
            return search.run()
    finally:
        LAST_RUN.clear()
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='hierarchical_support_pruning',
                            discovery_fraction=search.fraction,
                            discovery_queries=search.discovery_queries,
                            discovery_distance=search.discovery_distance,
                            prune_evaluations=search.prune_evaluations)
        torch.set_num_threads(previous_threads)
