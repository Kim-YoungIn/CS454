"""Forward-free component profile of discarded boundary mate construction.

This isolates vector preparation, not classifier runtime. Run from repo root:
  TORCH_HOME=/tmp/cs454-torch python -m research.dense_session3_profile
The search is allocated without invoking its model-evaluating constructor.
"""
import json
import statistics
import time

import torch

from research.dense_session3_boundary import BoundaryDirections


def measure(skip_unused, proposals=512):
    search = object.__new__(BoundaryDirections)
    search.x = torch.zeros(1, 3, 224, 224)
    search.norm = 'l2'
    search.rng = torch.Generator().manual_seed(454)
    delta = torch.randn(search.x.shape, generator=search.rng)
    delta *= 20 / float(delta.norm())
    feasible = [delta.clone() for _ in range(4)]
    started = time.perf_counter()
    for generation in range(proposals):
        index = search.randint(4)
        # In the real nonempty pool, every seventh generation already selects
        # an appropriately scaled pool tensor. The other six rescalings are
        # discarded by energy mutation and can be skipped without arithmetic.
        if generation % 7 and not skip_unused:
            mate = search.rescale(feasible[index], 20.)
        else:
            mate = delta
    return time.perf_counter() - started


def main():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with torch.inference_mode():
            measure(False, 32)
            measure(True, 32)
            # Alternate ordering to reduce startup and CPU-frequency bias.
            original, optimized = [], []
            for repetition in range(7):
                if repetition % 2:
                    optimized.append(measure(True))
                    original.append(measure(False))
                else:
                    original.append(measure(False))
                    optimized.append(measure(True))
        print(json.dumps(dict(threads=4, proposals=512, repetitions=7,
                              original_seconds=original, optimized_seconds=optimized,
                              median_original_seconds=statistics.median(original),
                              median_optimized_seconds=statistics.median(optimized),
                              note='Forward-free mate preparation only; no model calls. '
                                   'May run alongside another experiment; no full-attack speed claim.'),
                         indent=2))
    finally:
        torch.set_num_threads(previous)


if __name__ == '__main__':
    main()
