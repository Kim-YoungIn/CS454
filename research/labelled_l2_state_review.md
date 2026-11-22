# Development-only review: preserving coarse L2 search state

This static review uses only the church, French-horn and garbage-truck DEVELOPMENT `results.csv` metadata in `runs/labelled-v2-l2-baseline-development/`, plus current/frozen algorithm source. It runs system Python only: no torch/model imports, forwards, tensor loads or confirmation data. The frozen reversible candidate and production sources remain unchanged; their recorded hashes still match. The parent is evaluating that candidate before considering another architecture.

## What the measured regressions establish

All six named baseline/current trials reach exactly 900 calls at seed456 and four threads. Their entry-point times are57.6–61.5seconds, well below165seconds, so the regressions are measured at a matched call ceiling rather than attributable to time truncation.

| Development image | Global-only final squared L2 | Current512 final | Ratio | Transition call | Calls since last global success | Next global-only success | Calls after transition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| church | 96.353851 | 106.717606 | 1.108 | 517 | 141 | 546 | 29 |
| French horn | 150.773834 | 198.767532 | 1.318 | 516 | 87 | 639 | 123 |
| garbage truck | 61.606171 | 100.911346 | 1.638 | 517 | 174 | 523 | 6 |

The “since last success” count includes the explicitly queried scaled warm start, rather than pretending every call was a mutation proposal. At transition the in-progress global targets have squared radii150.994944,235.9296 and150.994944, respectively. None has just become infeasible or been shown exhausted; the prefix ends at the fixed generation gate. The original continuation succeeds at those targets after29,123 and6 additional calls. Church's next later global success is at743, horn's at873 and truck's at664/886. This is direct evidence that the replaced operator's original continuation still has useful progress available.

Local evolution does make progress:30,24 and37 accepted directions, respectively, plus3,3 and2 successful extra radial probes. The33,27 and39 total probes consume8.9%,7.3% and10.5% of the post-transition calls through888. No `boundary_relax` event occurs, alpha stays.25, and the maximum completed local-stage lengths are37,29 and33 calls. A policy that returns to global evolution only when local progress stalls would therefore not detect these losses: local is productive but has lower realized improvement over the available interval.

At the common call888, global-only/current distances are96.636757/107.661781,150.994949/198.767532 and61.847527/102.104370. From each shared transition archive, log-distance gain per subsequent actual call is0.002406/0.002115,0.002399/0.001660 and0.003609/0.002258. These are retrospective comparisons, not an online oracle for which operator would work next.

## What cannot yet be concluded

The metadata records archive improvements, not the pre-transition pool, best failed margin or mutation RNG state. It cannot prove that losing a particular elite caused a final regression, that the imminent global success was predictable, or that another interleaving would reproduce the original continuation. Switching operator, discarding state and changing RNG consumption occur together. The evidence supports preserving the option to continue; it does not establish a new attack's performance.

These are selected regression cases supplied by the parent. They diagnose a mechanism and are not an unbiased estimate of the method's overall development performance. Read the full development matrix before selecting an architecture, and preserve the independent confirmation bank.

## Static weakness of the frozen reversible candidate

`ReversibleWindows.boundary()` creates an empty global state after a first96-call local window and shared12-call polish. Thus it retains neither the existing failed pool nor its work toward the current target. In addition, `global_step()` resets radius, stage and pool whenever the shared best distance improves between global visits. Frequent small local gains can therefore cause repeated restarts at progressively harder targets. The same `.8*best_radius` target may look aggressive while the global operator repeatedly pays discovery costs and never completes a long stage. This is a plausible failure mechanism, not a measured claim about the as-yet-unseen candidate results.

There is an important distinction: **a new archive best does not itself invalidate previously evaluated fitness.** Margin depends on the image, reference and fixed model, not on archive distance. The frozen candidate chooses to move its radius, which changes candidate coordinates and requires fresh fitness. That choice is optional. If the old global target remains strictly below the new best radius, its existing elite images and margins remain valid and still optimize an improving distance. Resetting them after every1% local contraction is unnecessary for correctness. Keep the current candidate frozen so this concern is tested independently rather than silently changing the intervention.

## Next architecture, if the frozen evaluation warrants it

Prefer a **state-preserving reversible continuation** over another warmup/window sweep:

1. Capture the actual pre-transition global radius, four elites, stage/stall counters, proposal generation and RNG state. Also preserve a pending contraction/warm-start action if transition occurred immediately after a success. Do not reconstruct this population from the feasible archive.
2. Keep a global target fixed across local visits while its squared radius remains below the best feasible squared distance. Local improvements need no global resets in that interval. Retire/reanchor the target only once it is no longer guaranteed to improve the archive, or on a separately justified stagnation/restart rule. If any image is rescaled, query it again before assigning fitness.
3. Give each operator its own deterministic RNG state and proposal counter. The current candidate's shared counter/RNG means local calls also alter global crossover/mutation phases. Separate state permits a strong synthetic property: inserting local windows preserves the original global proposal subsequence until a target is deliberately retired.
4. Initially retain the already-frozen96-call allocation rule to isolate state preservation from new scheduler tuning. Continue using one shared protected feasible archive, actual-call accounting and shared final polish. Log target retirements, cross-operator resets, inherited pool size, target age and operator-local proposal counts so a failed experiment remains interpretable.

The global target may become only slightly smaller than the best archive, reducing the intended aggression; old elites can still stagnate; and preserving an unproductive stage consumes budget. Retaining state is therefore a testable hypothesis rather than a dominance argument. A meaningful ablation should compare this frozen architecture with the existing reversible candidate at the same calls and then matched165seconds. Do not claim recovery from the three traces alone, or promote based on selected cases. No implementation or model experiment for this next architecture was performed in this review.
