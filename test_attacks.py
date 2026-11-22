"""Fast search-contract tests with synthetic probabilities, without VGG forwards.

Run from the repository root: python -m unittest -v test_attacks
These verify algorithm invariants; they do not estimate real-image performance.
"""

import os
import unittest
from unittest.mock import patch

import torch

import epsilon_ball as attacks


class SearchContractTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.original = torch.zeros(1, 3, 16, 16)
        self.calls = 0
        self.env = patch.dict(os.environ, {
            "CS454_SECONDS": "170", "CS454_QUERIES": "48",
            "CS454_SEED": "13", "CS454_THREADS": "1",
            "CS454_L2_STYLE": "additive",
        })
        self.env.start()
        self.loader = patch.object(attacks, "get_image_tensor_from", side_effect=lambda _: self.original.clone())
        self.loader.start()
        self.oracle = patch.object(attacks, "get_model_predictions_for", side_effect=self.probabilities)
        self.oracle.start()

    def tearDown(self):
        self.oracle.stop()
        self.loader.stop()
        self.env.stop()
        torch.set_num_threads(self.previous_threads)

    def probabilities(self, image):
        self.calls += 1
        self.assertFalse(torch.is_grad_enabled(), "model evaluation should disable autograd")
        self.assertTrue(bool(torch.isfinite(image).all()))
        self.assertLessEqual(float(image.abs().max()), 3.)
        changed = float((image - self.original).abs().max()) > .01
        result = torch.full((1, 12), .001)
        # Reference index deliberately differs from every supplied seed label.
        result[0, 7] = .09 if changed else .9
        result[0, 2] = .9 if changed else .09
        return result / result.sum(dim=1, keepdim=True)

    @torch.inference_mode()
    def test_public_entrypoints_return_valid_adversarial_images(self):
        functions = (
            ("linf", attacks.make_Linf_adversarial_from),
            ("l2", attacks.make_L2_adversarial_from),
            ("l0", attacks.make_L0_adversarial_from),
        )
        for norm, function in functions:
            with self.subTest(norm=norm):
                before = self.calls
                image = function("held-out-not-in-reference-table.jpg")
                self.assertEqual(image.shape, self.original.shape)
                self.assertEqual(image.dtype, self.original.dtype)
                self.assertTrue(bool(torch.isfinite(image).all()))
                self.assertFalse(image.requires_grad)
                self.assertLessEqual(float(image.abs().max()), 3.)
                self.assertEqual(attacks.LAST_RUN["reference"], 7)
                self.assertTrue(attacks.LAST_RUN["success"])
                self.assertLessEqual(attacks.LAST_RUN["queries"], 48)
                self.assertEqual(attacks.LAST_RUN["queries"], self.calls - before)
                self.assertNotEqual(int(self.probabilities(image).argmax()), 7)
                distance = attacks._Search.distance
                proxy = type("Norm", (), {"norm": norm})()
                self.assertEqual(distance(proxy, image - self.original), attacks.LAST_RUN["distance"])
                history = [entry[1] for entry in attacks.LAST_RUN["history"]]
                self.assertTrue(all(a > b for a, b in zip(history, history[1:])))

    @torch.inference_mode()
    def test_fixed_query_seed_repeats_without_using_global_rng(self):
        for function in (attacks.make_Linf_adversarial_from, attacks.make_L2_adversarial_from, attacks.make_L0_adversarial_from):
            with self.subTest(function=function.__name__):
                torch.manual_seed(107)
                before = torch.random.get_rng_state().clone()
                first = function("first-spelling.jpg")
                history = list(attacks.LAST_RUN["history"])
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                # A different path or unrelated global RNG use must not affect the local seed.
                torch.rand(17)
                second = function("second-spelling.jpg")
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(history, attacks.LAST_RUN["history"])

    @torch.inference_mode()
    def test_projection_enforces_actual_norm_after_clipping(self):
        self.original = torch.linspace(-2.99, 2.99, self.original.numel()).reshape_as(self.original)
        proposed = torch.linspace(-20, 20, self.original.numel()).reshape_as(self.original)
        for norm, radius in (("linf", .1), ("l2", 2.)):
            with self.subTest(norm=norm):
                search = attacks._Search("unseen.jpg", norm)
                projected = search.project(proposed, radius)
                self.assertTrue(bool(torch.isfinite(projected).all()))
                self.assertLessEqual(float((search.x + projected).abs().max()), 3.000001)
                if norm == "linf":
                    self.assertLessEqual(float(projected.abs().max()), radius + 1e-6)
                else:
                    self.assertLessEqual(float(projected.square().sum()), radius * radius + 1e-5)

    @torch.inference_mode()
    def test_l2_styles_are_reproducible_under_fixed_query_cap(self):
        for style in ("additive", "energy", "sign", "portfolio"):
            with self.subTest(style=style), patch.dict(os.environ, {"CS454_L2_STYLE": style}):
                first = attacks.make_L2_adversarial_from("unseen-style-first.jpg")
                first_history = list(attacks.LAST_RUN["history"])
                queries = attacks.LAST_RUN["queries"]
                second = attacks.make_L2_adversarial_from("unseen-style-second.jpg")
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first_history, attacks.LAST_RUN["history"])
                self.assertEqual(queries, attacks.LAST_RUN["queries"])
                self.assertLessEqual(queries, 48)
                self.assertNotEqual(int(self.probabilities(second).argmax()), 7)

    @torch.inference_mode()
    def test_l2_mutations_stay_finite_and_bounded_with_overlapping_windows(self):
        windows = (
            ((0, 0, 4), (10, 10, 4)),  # disjoint windows
            ((3, 3, 6), (5, 4, 6)),    # partially overlapping windows
            ((4, 4, 5), (4, 4, 5)),    # exactly overlapping windows
            ((0, 0, 1), (0, 0, 1)),    # single-element spatial windows
        )
        for style in ("additive", "energy", "sign", "portfolio"):
            with patch.dict(os.environ, {"CS454_L2_STYLE": style}):
                for clipping in (False, True):
                    self.original = torch.zeros(1, 3, 16, 16)
                    if clipping:
                        self.original = torch.linspace(-2.99, 2.99, self.original.numel()).reshape_as(self.original)
                    search = attacks._Search("unseen-mutation.jpg", "l2")
                    radius = 3.
                    proposed = torch.linspace(-1, 1, self.original.numel()).reshape_as(self.original)
                    parent = search.project(proposed / proposed.norm() * radius, radius)
                    mate = search.project(-parent, radius)
                    for locations in windows:
                        for generation in (0, 1, 2, 7):
                            with self.subTest(style=style, clipping=clipping, windows=locations, generation=generation):
                                before_parent, before_mate = parent.clone(), mate.clone()
                                with patch.object(search, "square", side_effect=locations):
                                    child = search.mutate(parent, mate, radius, .1, generation)
                                self.assertEqual(child.shape, self.original.shape)
                                self.assertTrue(bool(torch.isfinite(child).all()))
                                self.assertLessEqual(float(child.square().sum()), radius * radius + 1e-5)
                                self.assertLessEqual(float((search.x + child).abs().max()), 3.000001)
                                self.assertTrue(torch.equal(parent, before_parent))
                                self.assertTrue(torch.equal(mate, before_mate))

    @torch.inference_mode()
    def test_l2_mutation_can_leave_zero_parent_and_handles_cancellation(self):
        for style in ("additive", "energy", "sign", "portfolio"):
            with self.subTest(style=style), patch.dict(os.environ, {"CS454_L2_STYLE": style}):
                search = attacks._Search("unseen-zero-parent.jpg", "l2")
                parent = torch.zeros_like(search.x)
                for generation in (0, 1, 2, 7):
                    child = search.mutate(parent, parent, 2., .1, generation)
                    self.assertTrue(bool(torch.isfinite(child).all()))
                    self.assertGreater(float(child.square().sum()), 0.)
                    self.assertLessEqual(float(child.square().sum()), 4.00001)
        search = attacks._Search("unseen-cancellation.jpg", "l2")
        parent = torch.zeros_like(search.x)
        parent[:, :, 2:4, 2:4] = 1.
        # Force normalized old and injected directions to be exact opposites.
        with patch.object(search, "square", side_effect=((2, 2, 2), (2, 2, 2))), \
             patch.object(search, "l2_pattern", return_value=-torch.ones(1, 3, 2, 2)):
            child = search.l2_energy_mutation(parent, parent, 4., .1, 1)
        self.assertTrue(bool(torch.isfinite(child).all()))
        self.assertLessEqual(float(child.square().sum()), 16.00001)

    @torch.inference_mode()
    def test_sparse_mutations_respect_channel_element_budget(self):
        # Match the noncontiguous channel-first layout produced by the real loader.
        self.original = torch.zeros(1, 16, 16, 3).permute(0, 3, 1, 2)
        self.assertFalse(self.original.is_contiguous())
        search = attacks._Search("unseen.jpg", "l0")
        # Nonmultiples of three distinguish channel-element budgets from RGB pixels.
        for count in (1, 7, 23, 101):
            with self.subTest(count=count):
                parent = search.sparse_initial(count)
                mate = search.sparse_initial(count)
                self.assertGreater(int(torch.count_nonzero(parent)), 0)
                self.assertLessEqual(int(torch.count_nonzero(parent)), count)
                for generation in range(12):
                    child = search.sparse_mutate(parent, mate, count, .1, generation)
                    self.assertLessEqual(int(torch.count_nonzero(child)), count)
                    self.assertLessEqual(float((search.x + child).abs().max()), 3.)
                    untouched = child == 0
                    self.assertTrue(torch.equal((search.x + child)[untouched], search.x[untouched]))
                    parent = child

    @torch.inference_mode()
    def test_archive_keeps_lowest_distance_success_when_offspring_fail(self):
        for norm in ("linf", "l2", "l0"):
            with self.subTest(norm=norm):
                search = attacks._Search("unseen.jpg", norm)
                large = torch.zeros_like(search.x)
                large.flatten()[:20] = .2
                small = torch.zeros_like(search.x)
                small.flatten()[:7] = .1
                self.assertLess(search.evaluate(large)[0], 0)
                self.assertLess(search.evaluate(small)[0], 0)
                archived = search.best.clone()
                saved_distance = search.best_distance
                self.assertLess(search.evaluate(large)[0], 0)
                self.assertGreater(search.evaluate(torch.zeros_like(search.x))[0], 0)
                self.assertTrue(torch.equal(search.best, archived))
                self.assertEqual(search.best_distance, saved_distance)
                self.assertNotEqual(int(self.probabilities(archived).argmax()), search.reference)

    def test_thread_setting_restored_when_oracle_raises(self):
        torch.set_num_threads(2)
        with patch.object(attacks, "get_model_predictions_for", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                attacks.make_Linf_adversarial_from("unseen.jpg")
        self.assertEqual(torch.get_num_threads(), 2)

    @torch.inference_mode()
    def test_deadline_preserves_success_with_default_and_clamped_budget(self):
        functions = (
            ("linf", attacks.make_Linf_adversarial_from),
            ("l2", attacks.make_L2_adversarial_from),
            ("l0", attacks.make_L0_adversarial_from),
        )
        for requested, expected in ((None, 165.), ("999999", 170.)):
            for norm, function in functions:
                with self.subTest(seconds=requested, norm=norm), patch.dict(os.environ, {
                    "CS454_QUERIES": "10000", "CS454_L2_STYLE": "portfolio",
                }):
                    if requested is None:
                        os.environ.pop("CS454_SECONDS", None)
                    else:
                        os.environ["CS454_SECONDS"] = requested
                    clock = [1000.]
                    successes = []
                    outcomes = []

                    def slow_oracle(image):
                        clock[0] += 2.5
                        probabilities = self.probabilities(image)
                        successful = int(probabilities.argmax()) != 7
                        outcomes.append(successful)
                        if successful:
                            successes.append(image.clone())
                        return probabilities

                    with patch.object(attacks.time, "monotonic", side_effect=lambda: clock[0]), \
                         patch.object(attacks, "get_model_predictions_for", side_effect=slow_oracle):
                        # Inspect configured deadline as well as actual public-entrypoint stopping.
                        search = attacks._Search("deadline-held-out.jpg", norm)
                        self.assertEqual(search.seconds, expected)
                        clock[0] = 1000.
                        successes.clear()
                        outcomes.clear()
                        image = function("deadline-held-out.jpg")
                    self.assertTrue(attacks.LAST_RUN["success"])
                    self.assertLess(attacks.LAST_RUN["queries"], 10000)
                    self.assertEqual(attacks.LAST_RUN["queries"], len(outcomes))
                    self.assertEqual(attacks.LAST_RUN["seconds"], clock[0] - 1000.)
                    self.assertLessEqual(attacks.LAST_RUN["seconds"], expected)
                    self.assertLess(attacks.LAST_RUN["seconds"], 180.)
                    self.assertGreater(attacks.LAST_RUN["seconds"], 100.)
                    self.assertTrue(any(torch.equal(image, success) for success in successes))
                    self.assertNotEqual(int(self.probabilities(image).argmax()), 7)
                    if norm != "l0":
                        # Refinement eventually crosses back to a failed perturbation;
                        # the returned archive must still contain a prior success.
                        self.assertIn(False, outcomes[outcomes.index(True) + 1:])

    @torch.inference_mode()
    def test_no_success_raises_instead_of_returning_clean_image(self):
        def constant_probabilities(image):
            result = torch.zeros(1, 12)
            result[0, 7] = 1.
            return result
        with patch.object(attacks, "get_model_predictions_for", side_effect=constant_probabilities):
            with self.assertRaisesRegex(RuntimeError, "without misclassification"):
                attacks.make_Linf_adversarial_from("unseen.jpg")
        self.assertFalse(attacks.LAST_RUN["success"])
        self.assertLessEqual(attacks.LAST_RUN["queries"], 48)


if __name__ == "__main__":
    unittest.main()
