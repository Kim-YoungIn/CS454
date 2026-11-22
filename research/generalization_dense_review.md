# Development-only L2 generalization review

This review reads only CSV/JSON traces in `runs/generalization-v1-l2-development/` and the two candidate source files. It uses system Python, performs no model imports or new benchmarks, and does not inspect confirmation predictions/results. Extracted evidence is in `research/generalization_dense_review_data.json`.

## What actually regresses

All eight calls complete exactly900 model queries and succeed; entry-point times54–62 seconds do not explain the distance differences. The frozen radial candidate wins one of four development images:

| Image | Production squared L2 | Radial squared L2 | Radial/control | Switch query | Local accepts | Successful/total probes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| astronaut | 2.713446 | 7.184147 | 2.648 | 9 | 104 | 33/133 |
| coffee | 383.721069 | 305.327454 | 0.796 | 370 | 46 | 2/48 |
| brick | 150.847519 | 154.180222 | 1.022 | 279 | 41 | 0/41 |
| rocket | 6.631104 | 11.816244 | 1.782 | 23 | 94 | 28/120 |

At query256, production already has squared L2 **10.376293/10.376294** on astronaut/rocket; radial has **40.629181/47.384041**. Production beats the radial candidate's final900-query result by queries **293/234**, respectively. These are substantial opportunity costs, not merely different final-polish roundoff.

The raw distance mean falls135.978→119.627 because coffee's absolute scale dominates, despite three regressions. The geometric mean paired ratio is1.400 and the median paired ratio1.402; worst ratio2.648. Report the actual per-image grader distances alongside these diagnostics. Do not select a broadly claimed generalization improvement from raw-distance mean alone; the assignment grader itself reports separate distances rather than specifying this diagnostic aggregation.

## Mechanism supported by the traces

Three successful global proposals do not establish that global progress has slowed. On astronaut the third success is already at query9; on rocket it is at23. The candidate permanently replaces a still-productive aggressive contraction search with much smaller direction changes. Production repeatedly contracts radius by0.8, corresponding to squared-distance factor0.64, while local accepts contract radius0.99 (squared factor0.9801), with occasional extra0.98 probes (factor0.9604). Therefore104 or94 local successes can still be less useful than a handful of aggressive global successes. Acceptance count is a poor measure of optimization progress.

Coffee has much slower initial discovery and switches at370; local evolution improves its final distance by20.43%. Brick switches at279; its41 accepted1% contractions plus final polish cannot quite match production's later235.930 at442 and150.995 at867. None of its41 extra2% probes succeeds. The local/global tradeoff is image-dependent; the supplied knot/sock evidence did not expose the early-switch failure mode.

## What the minimum256 gate should and should not fix

`research/dense_exploration_boundary.py` retains the same global proposal prefix until both three successes and256 global proposals are reached. It should directly protect astronaut and rocket from switching at9/23. The threshold counts proposals, not total queries, so warm-start and polish calls mean the actual switch query will exceed256. It may switch during an unfinished global stage once the count crosses256; it does not establish that the stage has become unproductive.

Coffee's and brick's third successes already occur after256 global proposals, accounting for the few warm-start calls. Their switch logic, RNG state and ensuing boundary trajectory should remain unchanged by this gate. In particular, the mild brick regression should remain; the new gate does not fix every observed tradeoff. Verify that prediction against the new development trace rather than treating four new calls as four independent algorithmic tests.

The gate is a reasonable development hypothesis. The value256 is a fixed work threshold chosen after observing the development failures, not a theorem about image difficulty or a validated population-wide optimum. Preserve the frozen original candidate/control and do not inspect confirmation results while adjusting it.

## Next generic architecture if the gated result remains mixed

Prefer a **reversible allocation between aggressive global contraction and local boundary evolution**, rather than another permanent switch determined by an absolute success count. Retain both operators' state and the exact global feasible archive. Give each operator bounded windows; rank recent progress by log(distance_before/distance_after) per actual model call or wall second, not by how many accepted proposals it makes. Retain a minimum exploration share so initial local success cannot permanently eliminate the aggressive operator. A local window that yields little progress should return work to global contraction; a slow global stage can lend work to local evolution.

A simpler first experiment is temporary boundary refinement only after an observed global-stage slowdown relative to recent successful stages, then resuming the global20% target from the improved feasible archive. This also avoids interpreting early easy contractions as a reason to stop exploring. Avoid repeatedly paying12-query polish initialization costs when alternating; keep operator state and RNG streams explicit. No such architecture is implemented or claimed to improve scores here.

Use the same development query ceilings and then actual165-second budgets to evaluate allocation overhead. Freeze the resulting candidate and selection decision before a single confirmation pass. Four development images can diagnose this failure mechanism but cannot establish broad statistical generalization; successful transfer still needs an independent confirmation evaluation and additional seeds.

## Is one additional512-proposal scale justified?

Yes, as a coarse diagnostic of the same premature-switch mechanism, not as a fine threshold sweep. Parent-reported development result: the256 gate reduces astronaut's regression to3.029705 versus2.713446, but does not eliminate it. The original control trace shows additional large global improvements after the256 scale: astronaut reaches4.250130 at488, rocket reaches6.640828 at513, brick reaches235.929596 at442, and coffee reaches603.979797 at515. A512-global-proposal gate includes the necessary clean/warm-start calls, so its actual query switch occurs later than512 and should preserve these global improvements before local refinement. This gives a trace-based reason to test one doubling rather than interpolating many nearby thresholds to fit astronaut.

The same evidence also identifies the cost: coffee benefits from the original local switch at370, so forcing roughly another145 global calls can remove productive local work and may reduce its gain. Brick's original256 gate is inert;512 would finally change its trajectory by retaining the fourth coarse contraction. These are generic allocation tradeoffs, not evidence that512 is optimal across images.

Predeclare this as the second and final coarse work scale for the current development investigation. Compare its four-image paired outcomes with production and256 using success, geometric-mean ratio, worst regression and actual time. If no frozen candidate meets the chosen promotion criterion, retain production and move to reversible operator allocation rather than continuing a threshold sweep. Freeze the selection before the one confirmation pass; do not use confirmation outcomes to select256 versus512.


Parent completion note: the512 scale subsequently won4/4 development pairs (ratio0.776),4/4 reserved confirmation pairs (0.788) and both predeclared165-second pairs (0.599). Its exact self-contained port was promoted after72 tests and independent artifact validation. The analysis above was performed on development records before those confirmation results, without inspecting them.
