# Reversible L2 allocation candidate — 2026-11-21

Status: **research-only; no real-image attack evidence at source freeze**. Production and canonical artifacts are unchanged. Review used production source, `PROGRESS.md` and the consumed fixture bank's historical mechanism review. No Imagenette confirmation image, row or output was inspected. Fresh labelled development measurements were being run independently by the parent agent; none informed this candidate's constants or implementation.

## Hypothesis and intervention

The production method permanently replaces aggressive radius contraction with local boundary evolution after 512 proposals and three successes. Previous traces show that many small accepted contractions can yield less squared-distance improvement than a few global contractions. The minimum-work gate fixed a demonstrated early-switch failure, but does not establish that aggressive contraction remains unproductive for the rest of a 165-second attack.

`research/dense_reversible_windows.py` inherits the production global prefix exactly and overrides only its later boundary stage. It maintains two search states:

- **Local:** production-style `.99` radius direction evolution, four failed-direction elites and four feasible directions, angular mixing, bounded three `.98` scalar probes and failure-driven relaxation.
- **Global:** four-elite energy mutation at `.8` times the verified feasible radius, with scaled warm starts and stalled-stage fresh candidates. Failed elites persist across windows when their radius remains valid.

After one shared initialization polish, the first 96-model-call window goes to local evolution, and the second to global evolution. Later windows prefer the larger exponentially smoothed `log(distance_before/distance_after) / actual_calls` (weight .5 on the newest window). Exact ties prefer local. At most two consecutive windows go to either operator, so neither can disappear. Calls include warm starts and radial probes; each window stops after at most 96 calls, and all steps honor the existing 12-call/final-polish time reserve. End polish remains shared and is not charged to either operator's reward.

The constants 96, .5 and maximum-two are **architectural starting values, not tuned optima**. Ninety-six calls makes a bounded window small relative to normal 165-second throughput while leaving the aggressive GA's failed elites useful on later visits. The minimum share is deliberately simple; a success count would reward many tiny gains, and a permanent switch cannot recover from a wrong allocation decision. Query-based reward permits a clean fixed-call comparison; actual 165-second trials must determine whether it also improves grading performance.

## Correctness and accounting

Only `evaluate()` can change the shared feasible archive. Failed probes never become feasible parents. An improvement by the other operator invalidates all margins at the old target radius: the affected operator resets its failed pool and explicitly queries a new warm candidate. It does not rescale a population and reuse stale scores. Unchanged targets retain operator state across windows; local feasible directions are reusable geometrically and need no stored margin.

The candidate is still population/elite selection, crossover and structured energy mutation with probability-based fitness, plus verified scalar contractions; it uses no derivatives or surrogate model. It reads neither saved attacks nor image identities/classes. Required public interface is `make_L2_adversarial_from(image_path)`. `LAST_RUN['windows']` records operator, start/end model calls, distances, log gain and smoothed rate, including partial final windows. `epsilon_ball.py` is an explicit source dependency.

## Synthetic verification, not performance evidence

Five tests pass in `research/test_dense_reversible_windows.py`. They exercise the real production AST with the eager `util` import removed, so no VGG is loaded; all classifier forwards use synthetic oracles. Tests verify exact production-prefix replay through the 512-proposal switch, deterministic 1000-call replay and matching archives, operator exploration, bounded call windows, stale-score invalidation, archive preservation after a failed radial probe, probe budget enforcement, and explicit failure after an infeasible 31-call run. These establish mechanism/correctness properties only. The spatial oracle favors structured mutations and cannot support an adversarial performance claim.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CS454_THREADS=1 \
  /tmp/cs454-baseline-env/bin/python -m unittest research.test_dense_reversible_windows -v
```

Final pre-freeze execution: five tests passed in 0.600 seconds; process exited at `2026-11-20T21:55:49Z`. Test/import overlap with the parent's fixed-900-call development audit was explicitly permitted and reported; those trials must reach their call ceiling. No actual-time claims should use concurrent execution. Freeze manifest: `research/labelled_l2_candidate_freeze.json`.

## Predeclared interpretation and limitations

Evaluate the frozen candidate against the current default on the same **development** images, seeds, threads and 900-call ceiling before any confirmation. Keep failures and inactive candidates; report per-image squared distances, success counts, geometric-mean paired ratio, worst ratio and activation. Candidate selection is the parent's independent responsibility; this document makes no promotion claim. An apparent query advantage needs separate matched-165-second evaluation before claiming improvement under the assignment objective. Preserve the production freeze throughout. Confirmation must remain untouched until the final candidate/budget/decision rule is frozen.

Main risks are minimum global allocation displacing productive local work, 96-call windows underestimating global improvements requiring long uninterrupted stages, cross-operator target improvements repeatedly invalidating global elites, delayed/noisy rates and a minimum-share choice that is wrong for an individual image. Rates measure realized improvement, not counterfactual utility. Global warm starts use actual archived distance (rather than the original nominal radius), so this is an allocation/target-reanchoring candidate, not an isolated scheduler ablation. Global-stage fresh queries occur on a later bounded step rather than as an unbounded extra call. The next iteration should use traces to distinguish those mechanisms; do not search many windows/weights on the small labelled bank or mistake synthetic wins for generalization.
