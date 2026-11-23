# All-norm confirmation with explicit execution strata

Independent one-thread successes: **40/40**. All forty cases remain in the denominator; each norm has ten cases.

| Norm | Verified successes / 10 | Failed cases | Failure-inclusive mean grade score |
| --- | ---: | ---: | ---: |
| linf | 10/10 | 0 | 0.0332215 |
| l2 | 10/10 | 0 | 451.527 |
| l0 | 10/10 | 0 | 36.3 |
| perlin | 10/10 | 0 | 0.220662 |

## Ten-image × four-attack outcomes

Cells show independently measured distance and execution mode: **S** = serial, unrestricted affinity/four Torch threads; **P0/P1** = parallel worker pinned to the corresponding CPU group. Failed cases retain their failure label; they are not omitted.

| Image (class) | Linf | Squared L2 | L0 elements | Fog magnitude |
| --- | ---: | ---: | ---: | ---: |
| confirmation_n01440764_00 (tench) | 0.0670925 [S] | 316.85 [S] | 66 [S] | 0.468539 [S] |
| confirmation_n02102040_01 (English springer) | 0.00235724 [S] | 0.216176 [S] | 1 [S] | 0.0634788 [S] |
| confirmation_n02979186_03 (cassette player) | 0.0112536 [S] | 23.3055 [P0] | 2 [S] | 0.225533 [S] |
| confirmation_n03000684_00 (chain saw) | 0.00188899 [P0] | 0.770055 [P1] | 1 [P0] | 0.0279209 [P1] |
| confirmation_n03028079_00 (church) | 0.0140705 [P0] | 7.09933 [P1] | 2 [P0] | 0.147926 [P1] |
| confirmation_n03394916_01 (French horn) | 0.16276 [P1] | 4039.34 [P0] | 177 [P1] | 0.628906 [P0] |
| confirmation_n03417042_00 (garbage truck) | 0.0273671 [P1] | 80.174 [P0] | 9 [P0] | 0.257237 [P1] |
| confirmation_n03425413_00 (gas pump) | 0.00898963 [P0] | 4.36172 [P1] | 8 [P0] | 0.0813156 [P1] |
| confirmation_n03445777_00 (golf ball) | 0.0274409 [P1] | 36.9618 [P0] | 94 [P1] | 0.188477 [P0] |
| confirmation_n03888257_00 (parachute) | 0.0089941 [P0] | 6.18852 [P1] | 3 [P0] | 0.117289 [P1] |

## Execution metadata for every case

CPU groups: P0 = `0,4,8,12`; P1 = `2,6,10,14`. Entry and grading seconds are the originally recorded times, not remeasured validation times.

| Image | Attack | Mode | CPU affinity | Entry seconds | Grading seconds | Actual queries | Verification |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| confirmation_n01440764_00 | linf | S | unrestricted; four Torch threads | 160.8 | 160.861 | 2515 | pass |
| confirmation_n01440764_00 | l2 | S | unrestricted; four Torch threads | 160.964 | 161.051 | 2375 | pass |
| confirmation_n01440764_00 | l0 | S | unrestricted; four Torch threads | 164.027 | 164.085 | 2434 | pass |
| confirmation_n01440764_00 | perlin | S | unrestricted; four Torch threads | 105.672 | 106.336 | 997 | pass |
| confirmation_n02102040_01 | l2 | S | unrestricted; four Torch threads | 162.663 | 162.735 | 2443 | pass |
| confirmation_n02102040_01 | l0 | S | unrestricted; four Torch threads | 34.0424 | 34.1239 | 498 | pass |
| confirmation_n02102040_01 | perlin | S | unrestricted; four Torch threads | 140.108 | 140.735 | 1496 | pass |
| confirmation_n02102040_01 | linf | S | unrestricted; four Torch threads | 162.801 | 162.869 | 2501 | pass |
| confirmation_n02979186_03 | l0 | S | unrestricted; four Torch threads | 164.055 | 164.141 | 2435 | pass |
| confirmation_n02979186_03 | perlin | S | unrestricted; four Torch threads | 121.548 | 121.95 | 1033 | pass |
| confirmation_n02979186_03 | linf | S | unrestricted; four Torch threads | 162.464 | 162.52 | 2328 | pass |
| confirmation_n02979186_03 | l2 | P0 | 0,4,8,12 | 161.398 | 161.515 | 1741 | pass |
| confirmation_n03000684_00 | perlin | P1 | 2,6,10,14 | 226.825 | 228.11 | 1756 | pass |
| confirmation_n03000684_00 | linf | P0 | 0,4,8,12 | 162.27 | 162.372 | 1762 | pass |
| confirmation_n03000684_00 | l2 | P1 | 2,6,10,14 | 163.033 | 163.14 | 1615 | pass |
| confirmation_n03000684_00 | l0 | P0 | 0,4,8,12 | 52.6908 | 52.8134 | 498 | pass |
| confirmation_n03028079_00 | linf | P0 | 0,4,8,12 | 162.023 | 162.103 | 1555 | pass |
| confirmation_n03028079_00 | l2 | P1 | 2,6,10,14 | 163.149 | 163.256 | 1567 | pass |
| confirmation_n03028079_00 | l0 | P0 | 0,4,8,12 | 164.057 | 164.14 | 1854 | pass |
| confirmation_n03028079_00 | perlin | P1 | 2,6,10,14 | 160.651 | 161.853 | 1024 | pass |
| confirmation_n03394916_01 | l2 | P0 | 0,4,8,12 | 162.08 | 162.22 | 1712 | pass |
| confirmation_n03394916_01 | l0 | P1 | 2,6,10,14 | 162.38 | 162.472 | 1724 | pass |
| confirmation_n03394916_01 | perlin | P0 | 0,4,8,12 | 142.08 | 142.856 | 1030 | pass |
| confirmation_n03394916_01 | linf | P1 | 2,6,10,14 | 162.002 | 162.1 | 1939 | pass |
| confirmation_n03417042_00 | l0 | P0 | 0,4,8,12 | 164.053 | 164.139 | 1908 | pass |
| confirmation_n03417042_00 | perlin | P1 | 2,6,10,14 | 127.898 | 128.33 | 1024 | pass |
| confirmation_n03417042_00 | linf | P1 | 2,6,10,14 | 159.284 | 159.377 | 1577 | pass |
| confirmation_n03417042_00 | l2 | P0 | 0,4,8,12 | 159.407 | 159.497 | 1564 | pass |
| confirmation_n03425413_00 | perlin | P1 | 2,6,10,14 | 137.083 | 137.575 | 1042 | pass |
| confirmation_n03425413_00 | linf | P0 | 0,4,8,12 | 161.634 | 161.732 | 1795 | pass |
| confirmation_n03425413_00 | l2 | P1 | 2,6,10,14 | 162.323 | 162.471 | 1658 | pass |
| confirmation_n03425413_00 | l0 | P0 | 0,4,8,12 | 164.047 | 164.148 | 1691 | pass |
| confirmation_n03445777_00 | linf | P1 | 2,6,10,14 | 162.204 | 162.31 | 1632 | pass |
| confirmation_n03445777_00 | l2 | P0 | 0,4,8,12 | 161.666 | 161.798 | 1601 | pass |
| confirmation_n03445777_00 | l0 | P1 | 2,6,10,14 | 164.053 | 164.128 | 1804 | pass |
| confirmation_n03445777_00 | perlin | P0 | 0,4,8,12 | 148.136 | 149.201 | 1005 | pass |
| confirmation_n03888257_00 | l2 | P1 | 2,6,10,14 | 162.442 | 162.534 | 1646 | pass |
| confirmation_n03888257_00 | l0 | P0 | 0,4,8,12 | 164.045 | 164.172 | 1670 | pass |
| confirmation_n03888257_00 | perlin | P1 | 2,6,10,14 | 156.001 | 156.865 | 1078 | pass |
| confirmation_n03888257_00 | linf | P0 | 0,4,8,12 | 159.475 | 159.526 | 1963 | pass |

## Descriptive strata

| Execution | Norm | Verified successes / cases in stratum | Median entry seconds | Median actual queries |
| --- | --- | ---: | ---: | ---: |
| S | linf | 3/3 | 162.464 | 2501 |
| S | l2 | 2/2 | 161.814 | 2409 |
| S | l0 | 3/3 | 164.027 | 2434 |
| S | perlin | 3/3 | 121.548 | 1033 |
| P0/P1 | linf | 7/7 | 162.002 | 1762 |
| P0/P1 | l2 | 8/8 | 162.202 | 1630.5 |
| P0/P1 | l0 | 7/7 | 164.047 | 1724 |
| P0/P1 | perlin | 7/7 | 148.136 | 1030 |

The first11 terminal trials used unrestricted scheduling and four Torch threads; the remaining29 used two concurrent workers, each pinned to four distinct physical cores with one SMT thread per core. The groups do not overlap physical cores. Memory bandwidth, CPU package power, frequency and host activity remain shared. These strata contain different cases and were not randomized between execution modes. Their timing, call counts, distances and success rates are descriptive; no same-case counterfactual establishes speedup, slowdown or an algorithmic gain. This is one seed and one confirmation photograph per each of ten classes. Imagenette may overlap VGG pretraining. All40 cases, including failures, remain in the total success denominator; every norm has ten cases. Distance units differ across norms and must not be compared across attacks. The confirmation bank is now consumed and must not be relabeled fresh after tuning.

Mandatory attack+distance/render+classification uses180/300 seconds. Frozen optimizer165/270 targets are separate diagnostics. Failed cases retain999999999.

Runtime totals are sums of recorded per-case wall times, not elapsed makespan or CPU seconds. Parallel case intervals overlap. Query totals are actual model-evaluation examples as recorded by each attack.

Full forty-case details: `execution_cases.csv`. Wide ten-image table: `image_norm_wide.csv`. Stratum counts and input/source hashes: `execution_summary.json`.
