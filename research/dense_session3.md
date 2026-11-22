# Dense search investigation — session 3 (2026-11-21)

## Question and preserved control

Production L2 energy mutation is unchanged. Its knot trajectory reaches squared
L2 943.718 at query 826, 603.980 at query 1203, and 386.547 at query 2482 in the
earlier default-budget run. A 900-query test cannot evaluate either late plateau.
The previous patience-80 continuation was worse; patience 256 matched the
900-query control but never relaxed within that budget.

The first experiment here compares production with the existing continuation
at 1800 queries, 165 seconds, seed 454, four CPU threads, knot only, in sequential
isolated CPU windows. Query caps include the clean model evaluation. If either
run hits the wall-clock limit, its actual query count is reported separately.

## Separate representation prototype

`dense_session3_boundary.py` preserves the original energy GA through three
successful radii, then evolves verified feasible directions near their scalar
decision boundary. The trigger counts search events and is independent of the
image identity and query number. A 12-query bisection initializes the boundary.
Each genetic proposal blends a structured energy mutation with its parent,
normalizes to a radius 1% below the current feasible archive, and uses one model
query. Four failed directions are selected by classifier margin; up to four
successful directions supply crossover mates. Failure reduces angular strength
and, after 256 proposals, halves the radial decrement. The exact clipped image
scored by the shared evaluator remains the only successful archive.

This tests whether frequent feasible improvements outperform spending hundreds
of queries on a fixed 20% contraction. It does not infer gradients or train a
surrogate. It is intentionally separate from production and is not a verified
performance improvement.

Synthetic tests: `python -m unittest -v research.test_dense_session3_boundary`
initially passed three cases: exact discovery proposal prefix through the switch,
deterministic successful archive and query/bounds contract on an arbitrary path,
and CPU-thread restoration after an exception. The oracle is a radial binary
classifier, so these tests establish mechanics rather than VGG effectiveness.

## Measurements

| Implementation | Knot squared L2 | Queries | Entrypoint seconds | Exact reload |
| --- | ---: | ---: | ---: | --- |
| Production energy GA | 601.622803 | 1800 | 126.669 | Pass |
| Continuation, patience 256 | 586.773376 | 1800 | 117.715 | Pass |
| Feasible-direction evolution | 571.531006 | 1800 | 153.693 | Pass |

Directories: `runs/session3-dense-control-knot1800/` and
`runs/session3-dense-cont256-knot1800/`. Each contains the exact source snapshot,
input/configuration hashes, raw image and independent reload check. The candidate
used `CS454_DENSE_PATIENCE=256`; implementations were `epsilon_ball` and
`research.dense_candidate`, respectively. Other settings are listed above and
fully recorded in each configuration JSON. Runs were sequential with no other
VGG experiment; use the equal query budgets for comparison rather than treating
the 9-second timing difference as an algorithmic speedup.

Continuation improved the matched score by 2.47%. Its first three successes
matched the production query positions 364, 586 and 826. Relaxation finally
activated at queries 1084, 1342 and 1601. The strongest failed direction became
feasible immediately after each relaxation, at squared distances 764.416,
671.853 and 614.355. Final scalar polishing improved the last to 586.773. The
production control reproduced its next success at query 1203 (603.980), and
polished to 601.623.

This comparison actually exercises late relaxation, unlike the earlier
900-query no-op. It remains one image and one seed, with a modest improvement;
production is preserved. It also does not improve the canonical knot artifact
386.170 obtained with a larger default-budget run. Next: test the distinct
boundary representation at the same budget, then evaluate any promising dense
candidate on other images and seeds before considering production changes.

Attribution caveat: the continuation's archive-radius rescaling produces tiny
FP32 differences before the first relaxation (query-826 squared L2 943.718323
versus 943.718384). The success positions match, but the prefix is not bitwise
identical. A very-large-patience continuation control would isolate scheduler
causality; broader paired images and seeds are still needed for generalization.

## Boundary representation result

`runs/session3-dense-boundary-knot1800/` records the distinct representation
experiment. Explicit settings were `CS454_L2_STYLE=energy`,
`CS454_BOUNDARY_SUCCESSES=3`, `CS454_BOUNDARY_FRACTION=.99`, and
`CS454_BOUNDARY_ALPHA=.25`; the other budgets and seed match the control.

The original discovery prefix reproduced the exact recorded squared distances
at queries 364, 586 and 826. Initial scalar bisection reached 890.127258 by query
837. Direction evolution then accepted 22 feasible improvements, from 872.420
at query 859 to 572.090 at query 1753. Final polishing reached 571.531006 at query
1799. No radial relaxation or angular-strength reduction activated; the longest
successful stage took 149 proposals. The result is 5.00% below the matched-query
production control and 2.60% below continuation on this case.

The run consumed 153.693 seconds, substantially more than the control's 126.669
seconds. Runs were sequential without another VGG experiment, but no claim is
made that this timing gap is solely algorithmic overhead. Actual default-time
paired runs and proposal-cost profiling are needed before ranking wall-clock
efficiency. The 1800-query candidate remained under the 165-second budget.
Neither candidate is promoted: these measurements cover only knot, seed 454,
and neither beats the canonical 386.170 artifact from the earlier larger-query
default-budget run. Priorities are another image/seed pair, an actual-time
comparison, and measurement of the extra normalization/mixing cost. Preserve
both architectures: the boundary result justifies further research beyond
changing the fixed-radius scheduler.

## Trajectory-preserving cleanup and timing check

`dense_session3_fast.py` preserves the original boundary module and skips
constructing a rescaled feasible mate on generations where energy mutation
never reads it. It still consumes the original random-index draw. A cache was
considered and rejected: on crossover generations a nonempty pool already
supplies the mate, so rescaled feasible mates have no useful reuse before the
target/archive changes.

The expanded four-test suite in `runs/session3-dense-tests.txt` passed. It
compares every model input bitwise, final image, search history and event trace
between original and fast boundary searches over 600 synthetic queries. A
radial oracle covers relaxation; a spatial oracle covers accepted improvements.

The forward-free four-thread microprofile (`dense_session3_profile.py`, output
`runs/session3-dense-mate-profile.json`) alternated original/optimized order for
seven repetitions of 512 proposals. Median discarded-mate preparation cost
fell from 0.05104 to 0.00106 seconds. This is only about 0.1 ms/proposal saved,
so it cannot explain the earlier 27-second whole-run timing gap. It is a small
safe cleanup, not evidence of a material attack speedup. The profile ran beside
another model experiment and makes no claim about end-to-end throughput.

Next scheduled experiment: an isolated production-versus-fast-boundary pair
with the actual default 165-second budget and no restrictive query cap. The
original boundary module and every earlier run remain unchanged.

## Actual-time pair

Sequential isolated knot calls, seed 454, four threads, no query cap, 165-second ceiling:

| Implementation | Squared L2 | Model calls | Seconds |
| --- | ---: | ---: | ---: |
| Production | 601.622803 | 2011 | 163.077 |
| Fast feasible-direction evolution | 559.887268 | 1896 | 163.074 |

Both raw round trips pass. The candidate improves the objective by 6.94% in this pair while using fewer calls; it accepts 23 boundary contractions. The control remains in the plateau after query 1203 and never reaches its next generic contraction at query 2482. Historical default runs reached that contraction on a faster CPU window and produced 386, so this does not establish universal wall-clock superiority or improve the canonical artifact. A second image with early boundary activation (sock, seed 454, 900 queries) is a high-value next comparison. Sources: `runs/session3-dense-wall-{control,fast}/`. No production promotion.

## Second image: sock

At exactly 900 calls/165-second ceiling, seed 454/four threads: production147.622498 (75.328s), fast boundary134.045715 (71.221s), 9.20% lower squared distance. Both round trips passed. The first three successes match at91/186/297. Production next accepts235.930 at446 and150.995 at719; the candidate accepts50 smaller contractions through875 and polishes to134.046. Sources: `runs/session3-dense-sock-{control,fast}/`. This strengthens the representation evidence, but three other images and additional seeds remain untested.

A bounded radial-probe candidate now tests whether accepted directions support larger scalar contractions. It tries at most three extra2% steps per accepted direction and records every outcome, retains only verified feasible images, and preserves polish reserves. Compare `research/dense_radial_probe.py` against the untouched fast-boundary baseline on sock900/knot1800. Consecutive successes may benefit; frequent failed probes may instead waste calls.
