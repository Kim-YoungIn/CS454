# Imagenette all-norm confirmation results

Success basis: independent one-thread validation.

| Attack | Success / frozen cases | Median distance* | Geometric mean* | Failure-inclusive mean score | Max grading seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| linf | 10 / 10 | 0.0126621 | 0.0146885 | 0.0332215 | 162.869 |
| l2 | 10 / 10 | 15.2024 | 17.5801 | 451.527 | 163.256 |
| l0 | 10 / 10 | 5.5 | 7.90162 | 36.3 | 164.172 |
| perlin | 10 / 10 | 0.168201 | 0.155102 | 0.220662 | 228.11 |

*Distance statistics condition on successful cases. L2 is the sum of squared differences; L0 counts changed channel elements; fog uses the sum of positive magnitudes.

One fixed seed and one confirmation photograph per each of ten classes; the clean-correct eligibility rule is not an estimate of classifier accuracy. Imagenette is ImageNet-derived and may overlap VGG pretraining. These are absolute performance measurements of one frozen implementation; there is no paired comparator and no claim of an algorithmic improvement. All failures and incomplete cases remain in the denominator. Distances across different norms are not comparable. A consumed confirmation bank must not be reused as fresh independent validation after tuning on these outcomes.

The failure-inclusive mean assigns the grader penalty 999999999 to failed cases; it is omitted until all cases in a norm have finished. Frozen optimizer ceilings are 165/270 seconds; grading ceilings are 180/300 seconds and include mandatory final scoring/rendering. Optimizer overruns are recorded separately; only the authoritative grading ceiling determines the time-related grade penalty.

Full per-image outcomes, including failures: `per_image.csv`. Independent verification details: `verification.csv` (validation mode only).

Execution conditions:11 serial cases and29 cases with two concurrent workers, each pinned to four disjoint physical cores. Shared memory bandwidth and package power remain uncontrolled. See [execution-aware full table](../execution-report-v2/execution_results.md) and [10×4 wide CSV](../execution-report-v2/image_norm_wide.csv). No standalone-throughput or cross-mode algorithm-improvement claim is made.
