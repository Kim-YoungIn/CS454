"""Experimental exact-image fitness cache for the verified spatial GA.

Cache keys hash the complete clipped CPU image, so distinct genotypes that
produce exactly the same evaluator input share a result. No approximation,
model modification, or persisted attack lookup is used. Only actual model
forwards count toward CS454_QUERIES; proposals are separately bounded so a
collapsed population cannot spin forever without consuming model queries.
"""
import hashlib
import os

import torch

import epsilon_ball as base

LAST_RUN = {}


class CachedSearch(base._Search):
    def __init__(self, image_path, norm):
        super().__init__(image_path, norm)
        self.fitness_cache = {}
        self.proposals = 0
        self.cache_hits = 0
        self.proposal_limit = max(256, 64 * self.limit)

    def available(self, reserve=0., queries=0):
        return self.proposals < self.proposal_limit and super().available(reserve, queries)

    @staticmethod
    def image_key(image):
        return hashlib.sha256(image.detach().contiguous().numpy().tobytes()).digest()

    def evaluate(self, delta):
        image = (self.x + delta).clamp(-3, 3)
        self.proposals += 1
        key = self.image_key(image)
        if key in self.fitness_cache:
            self.cache_hits += 1
            # This exact phenotype was already processed by the archive. Its
            # fixed norm and deterministic model fitness cannot have changed.
            return self.fitness_cache[key], image - self.x
        margin, actual = super().evaluate(delta)
        self.fitness_cache[key] = margin
        return margin, actual


def _attack(image_path, norm):
    previous_threads = torch.get_num_threads()
    search = None
    base.LAST_RUN.clear()
    LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, int(os.environ.get('CS454_THREADS', 4))))
        with torch.inference_mode():
            search = CachedSearch(image_path, norm)
            return search.run()
    finally:
        LAST_RUN.update(base.LAST_RUN)
        if search is not None:
            LAST_RUN.update(implementation='exact_image_fitness_cache',
                            proposals=search.proposals, cache_hits=search.cache_hits,
                            cache_entries=len(search.fitness_cache),
                            proposal_limit=search.proposal_limit,
                            proposal_guard_exhausted=search.proposals >= search.proposal_limit)
        torch.set_num_threads(previous_threads)


def make_Linf_adversarial_from(image_path):
    return _attack(image_path, 'linf')


def make_L2_adversarial_from(image_path):
    return _attack(image_path, 'l2')


def make_L0_adversarial_from(image_path):
    return _attack(image_path, 'l0')
