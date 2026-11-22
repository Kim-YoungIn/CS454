"""Spend only production's unused L0 tail on one-parent spatial support evolution.

The complete current discovery/restart/parental policy runs first. No established
search work is displaced. A synthetic neighbor image provides possible alleles;
it is never treated as an independently feasible parent or submitted directly.
"""
import os
import time

import torch
import epsilon_ball as base

LAST_RUN = {}


def neighbor_alleles(original, parent):
    """Transport parent's actual channel values to valid eight-neighbor loci.

    When sources disagree at one target, the smallest source index wins. The
    inherited parent overrides overlaps later in _SupportGA. No channel or row
    wrapping is allowed, and the normalized endpoint remains within [-3, 3].
    """
    height, width = original.shape[-2:]
    active = parent.ne(original).flatten().nonzero().flatten()
    weak = original.clone()
    if not len(active):
        return weak
    offsets = torch.tensor([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                            if r or c], dtype=torch.long)
    source = active.repeat_interleave(len(offsets))
    shift = offsets.repeat(len(active), 1)
    channels = source // (height * width)
    rows = (source // width) % height + shift[:, 0]
    cols = source % width + shift[:, 1]
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    target = channels[valid] * (height * width) + rows[valid] * width + cols[valid]
    source = source[valid]
    order = target.argsort(stable=True)
    target, source = target[order], source[order]
    first = torch.ones(len(target), dtype=torch.bool)
    first[1:] = target[1:] != target[:-1]
    weak.flatten()[target[first]] = parent.flatten()[source[first]]
    return weak


class LocalTailSearch(base._ParentalSparseSearch):
    def sparse(self):
        super().sparse()
        if self.best is None or self.best_distance <= 1 or not base._Search.available(self):
            return
        self.discovery = False
        parent = self.best.clone()
        weak = neighbor_alleles(self.x, parent)
        novel = weak.ne(self.x) & parent.eq(self.x)
        if not bool(novel.any()):
            return
        before, start = self.best_distance, self.queries
        operator_seed = self.initial_seed + 1000003
        self.rng.manual_seed(operator_seed)
        engine = base._SupportGA(self, parent, weak)
        engine.run()
        origin = dict(kind='local_support_tail', seed=operator_seed)
        if self.best_distance < before:
            self.selected_origin = origin
            self.deletion_minimal = False
        self.chunks.append(dict(origin=origin, start_query=start, end_query=self.queries,
                                distance=self.best_distance, deletion_minimal=self.deletion_minimal,
                                discovery_query=None, discovery_distance=None,
                                prune_evaluations=0, cleanup_evaluations=0, cleanup_passes=0,
                                attempts=engine.attempts, promotions=engine.promotions,
                                repeated_supports=engine.duplicates, initial_distance=before,
                                novel_loci=int(novel.sum())))


def make_L0_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    base.LAST_RUN.clear()
    LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, int(os.environ.get('CS454_THREADS', 4))))
        with torch.inference_mode():
            search = LocalTailSearch(image_path, 'l0')
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN, implementation='parental_then_local_support_tail')
        torch.set_num_threads(previous_threads)
