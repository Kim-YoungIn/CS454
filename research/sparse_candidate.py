"""Research alternative: evolve a compact two-rectangle L0 chromosome.

This module is not the submitted/default implementation. The genome describes
positions, aspect ratios, colors and channel subsets rather than a dense tensor.
Differential mutation and binomial crossover search a 14-dimensional genotype;
selection is by the same model-probability margin as the verified baseline.
No gradients, external image data, saved adversarial tensors or class lookup.
"""
import math
import os
import time

import torch

from epsilon_ball import _Search, _setting

LAST_RUN = {}


class RectangleSearch(_Search):
    POPULATION = 16
    PATCHES = 2
    PARAMETERS = 7

    def __init__(self, image_path):
        super().__init__(image_path, 'l0')
        self.best_genome = None
        self.best_budget = None
        self.stage_history = []
        self.population = None
        self.fitness = None
        self.genome_queries = 0

    def random_genomes(self, count):
        genomes = self.rand(count, self.PATCHES, self.PARAMETERS)
        # Extreme colors provide contrast; subsequent mutation permits interiors.
        genomes[:, :, 3:6] = (genomes[:, :, 3:6] >= .5).float()
        # Mix RGB and one/two-channel supports from the first population.
        genomes[:count // 2, :, 6] = 6.5 / 7
        return genomes

    def decode(self, genome, budget):
        """Render bounded, in-frame rectangles with at most budget changed elements."""
        height, width = self.x.shape[-2:]
        delta = torch.zeros_like(self.x)
        budget = max(0, min(self.x.numel(), int(budget)))
        for index, gene in enumerate(genome.clamp(0, 1)):
            allocation = budget // self.PATCHES + (index < budget % self.PATCHES)
            if not allocation:
                continue
            mask = min(7, 1 + int(float(gene[6]) * 7))
            channels = [channel for channel in range(3) if mask & (1 << channel)]
            channels = channels[:allocation]
            area = min(height * width, allocation // len(channels))
            if not area:
                continue
            aspect = math.exp(4 * (float(gene[2]) - .5))
            patch_height = max(1, min(height, area, round(math.sqrt(area * aspect))))
            patch_width = max(1, min(width, area // patch_height))
            row = round(float(gene[0]) * (height - patch_height))
            col = round(float(gene[1]) * (width - patch_width))
            for channel in channels:
                value = 6 * float(gene[3 + channel]) - 3
                region = (0, channel, slice(row, row + patch_height), slice(col, col + patch_width))
                delta[region] = value - self.x[region]
        return delta

    def differential_child(self, target, generation):
        """Differential mutation plus forced binomial crossover, bounded to [0,1]."""
        order = torch.randperm(self.POPULATION, generator=self.rng)
        order = order[order != target]
        first, second, third = [int(i) for i in order[:3]]
        elite = int(self.fitness.argmin())
        if generation % 2:
            donor = self.population[elite] + .7 * (self.population[first] - self.population[second])
        else:
            donor = self.population[first] + .7 * (self.population[second] - self.population[third])
        inherited = self.rand(self.PATCHES, self.PARAMETERS) < .7
        inherited.flatten()[self.randint(inherited.numel())] = True
        child = torch.where(inherited, donor, self.population[target])
        if generation % 11 == 0:
            # A small fresh mutation prevents irreversible discrete mask collapse.
            child += .08 * torch.randn(child.shape, generator=self.rng)
        return child.clamp(0, 1)

    def assess(self, genome, budget):
        previous = self.best_distance
        margin, _ = self.evaluate(self.decode(genome, budget))
        self.genome_queries += 1
        if self.best_distance < previous:
            self.best_genome = genome.clone()
            self.best_budget = budget
        return margin

    def reset_population(self, inherited=None):
        fresh = self.random_genomes(self.POPULATION)
        if inherited is not None:
            keep = min(len(inherited), self.POPULATION // 2)
            fresh[:keep] = inherited[:keep]
        if self.best_genome is not None:
            fresh[0] = self.best_genome
        self.population = fresh
        self.fitness = torch.full((self.POPULATION,), torch.inf)

    def search(self):
        budget = min(1536, self.x.numel())
        stage = 0
        generation = 0
        self.reset_population()
        while self.available():
            index = stage % self.POPULATION
            if stage < self.POPULATION:
                child = self.population[index].clone()
            elif stage % 41 == 0:
                child = self.random_genomes(1)[0]
            else:
                child = self.differential_child(index, generation)
            margin = self.assess(child, budget)
            if margin < float(self.fitness[index]):
                self.population[index] = child
                self.fitness[index] = margin
            stage += 1
            generation += 1
            if self.best_distance == 1:
                break  # Zero changed elements reproduces the clean reference.
            if margin < 0 or stage >= 160:
                self.stage_history.append(dict(queries=self.queries, budget=budget,
                                               margin=float(self.fitness.min()), successful=margin < 0))
                elites = self.population[self.fitness.argsort()[:self.POPULATION // 2]].clone()
                if margin < 0:
                    budget = max(1, int(self.best_distance * .8))
                elif self.best is not None:
                    budget = max(1, int(self.best_distance * .95))
                else:
                    budget = min(self.x.numel(), math.ceil(budget * 1.6))
                self.reset_population(elites)
                stage = 0
        LAST_RUN.clear()
        LAST_RUN.update(algorithm='rectangle-de', norm='l0', queries=self.queries,
                        seconds=time.monotonic() - self.started,
                        reference=self.reference, success=self.best is not None,
                        distance=None if self.best is None else self.best_distance,
                        min_margin=self.min_margin, history=self.history,
                        stage_history=self.stage_history,
                        best_budget=self.best_budget,
                        best_genome=None if self.best_genome is None else self.best_genome.tolist())
        if self.best is None:
            raise RuntimeError(f'Rectangle genetic search exhausted {self.queries} queries without misclassification')
        return self.best.detach().clone()


def attack(image_path):
    """Return the best successful image, honoring standard CS454 research controls."""
    LAST_RUN.clear()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(max(1, _setting('THREADS', 4, int)))
        with torch.inference_mode():
            return RectangleSearch(image_path).search()
    finally:
        torch.set_num_threads(threads)


def make_L0_adversarial_from(image_path):
    return attack(image_path)
