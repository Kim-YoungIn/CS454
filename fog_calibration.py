"""Measure coefficient recovery for a fixed fog field using the course renderer."""

import csv
import json
import sys
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch

from perlin_fog import add_noise_to_image, genotype2phenotype, perlin_image_generator
from util import (
    IMAGES_ORG_PREDS,
    get_image_tensor_from,
    get_L0_norm_for,
    get_L2_norm_for,
    get_Linf_norm_for,
    get_model_most_likely_for,
    get_viz_for,
)


# The small reference expression is taken from the assignment's JSON example.
REFERENCE = [(3, 0.4), (12, 0.1)]
OCTAVES = (3, 6, 12, 24)
POPULATION = 32
GENERATIONS = 40


def fit_coefficients(basis, observed, seed, decaying):
    rng = np.random.default_rng(seed)
    population = rng.uniform(0, 0.5, (POPULATION, basis.shape[1]))
    for generation in range(GENERATIONS):
        errors = np.mean((population @ basis.T - observed) ** 2, axis=1)
        parents = population[np.argsort(errors)[:8]]
        if generation == GENERATIONS - 1:
            return parents[0]
        # Keep the best row. Each child inherits one parental weight per octave.
        left, right = parents[rng.integers(8, size=(2, POPULATION - 1))]
        children = np.where(rng.random(left.shape) < 0.5, left, right)
        step = 0.08
        if decaying:
            step *= 0.95 ** generation
        children += rng.normal(0, step, children.shape)
        population = np.vstack([parents[0], children.clip(0, 0.5)])


def check_artifact_round_trip(field):
    with TemporaryDirectory() as directory:
        with chdir(directory):
            Path("fog.json").write_text(json.dumps(REFERENCE))
            torch.manual_seed(73)
            regenerated = genotype2phenotype(json.loads(Path("fog.json").read_text()))
        if not torch.equal(regenerated, field):
            raise ValueError("Fog changed after JSON reload and RNG reseeding")
        for path in IMAGES_ORG_PREDS:
            original = get_image_tensor_from(path)
            rendered = add_noise_to_image(regenerated, original)
            with chdir(directory):
                try:
                    get_viz_for(rendered, save_to="control.png")
                finally:
                    plt.close()
                saved = torch.load("raw_control.png", weights_only=True)
            if not torch.equal(saved, rendered):
                raise ValueError(f"{path}: the writer changed the rendered tensor")
            direct_prediction = get_model_most_likely_for(rendered)
            if get_model_most_likely_for(saved) != direct_prediction:
                raise ValueError(f"{path}: prediction changed after reload")
            for metric in (get_Linf_norm_for, get_L2_norm_for, get_L0_norm_for):
                if metric(saved - original) != metric(rendered - original):
                    raise ValueError(f"{path}: {metric.__name__} changed after reload")


def evaluate_recipe(output, recipe_name, recipe, field, layers):
    observed = field[0].numpy().astype(np.float64)
    # All channels, one pixel per 16x16 cell. Assess on a disjoint offset grid.
    fit_basis = layers[:, ::16, ::16].reshape(-1, len(OCTAVES))
    fit_field = observed[:, ::16, ::16].ravel()
    check_basis = layers[:, 8::16, 8::16].reshape(-1, len(OCTAVES))
    check_field = observed[:, 8::16, 8::16].ravel()
    reference_weights = dict(recipe)
    weights = np.array([reference_weights.get(o, 0) for o in OCTAVES])
    def report(method, seed, count, fitted):
        rmse = np.sqrt(np.mean((check_basis @ fitted - check_field) ** 2))
        max_error = np.max(np.abs(fitted - weights))
        output.writerow([recipe_name, method, seed, count, f"{rmse:.10g}", f"{max_error:.10g}"])

    exact_fit = np.linalg.lstsq(fit_basis, fit_field, rcond=None)[0]
    report("least_squares", "", "", exact_fit)
    for seed in range(10):
        for decaying in (False, True):
            fitted = fit_coefficients(fit_basis, fit_field, seed, decaying)
            name = "decaying" if decaying else "constant"
            report(name, seed, POPULATION * GENERATIONS, fitted)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    field = genotype2phenotype(REFERENCE)
    check_artifact_round_trip(field)
    layers = np.stack([perlin_image_generator(o)[0].numpy() for o in OCTAVES], axis=-1)
    layers = layers.astype(np.float64)
    output = csv.writer(sys.stdout, lineterminator="\n")
    output.writerow(["recipe", "method", "seed", "evaluated_rows", "check_grid_rmse", "max_weight_error"])
    evaluate_recipe(output, "two_octaves", REFERENCE, field, layers)
    dense_recipe = [(3, 0.20), (6, 0.15), (12, 0.10), (24, 0.05)]
    evaluate_recipe(output, "four_octaves", dense_recipe, genotype2phenotype(dense_recipe), layers)


if __name__ == "__main__":
    main()
