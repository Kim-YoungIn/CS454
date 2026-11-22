# Parental sparse-support investigation — November 21

Question: can genetic recombination of two locally pruned L0 parents use the
remaining 1800-query budget better than blind union/pruning? Existing online
controls: mink63+41→38 with 1613 parent-generation calls; sock23+49→42, rejected
in favor of23, after1501 parent-generation calls.

`elite_session3.py` is an explicitly posthoc mechanism diagnostic, not an online
attack. Saved images are CLI inputs only. The `SupportGA` operator works from
two already evolved images, inherits parental channel-element values, evolves
fixed-cardinality support masks using mutation/crossover, and ranks feasible and
infeasible candidates by the untargeted classifier margin. Successful offspring
reduce the target cardinality; the better feasible parent remains protected.
`weakfirst` and `strongfirst` change the hierarchical deletion order on a union.
Parent values from the stronger parent win overlap conflicts.

Both measured parent pairs have **zero overlapping changed channel elements**.
Thus additive saturation cannot explain sock's regression for these parents;
deletion trajectory and selected support are the real distinctions here.

Three synthetic tests pass:

```sh
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python -m unittest -v research.test_elite_session3
```

They verify overlap-value inheritance, fixed support counts through mutation and
crossover, and successful gene replacement reducing a three-element feasible
parent to two elements while protecting feasibility. The online-policy test
also evolves its parents in the same call and proves that a failed support GA
cannot overwrite the stronger feasible discovery archive.

Diagnostic protocol: four threads, seed454,187 total diagnostic calls for mink
or299 for sock, including one clean classification and two parent checks. Add
the known parent-generation charge1613 or1501 respectively; final independent
classification validation is recorded separately. Compare mink against38 as
well as its41-element parent; compare sock against its23-element parent. The
separate best22-element sock value-refinement artifact remains stronger still.
Wall times exclude module/model import and final validation and do not claim
end-to-end online timing.

Example (change mode and fresh output directory for matched controls):

```sh
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python -m research.elite_session3 \
  --image images/mink.jpg \
  --parents runs/session2-production-five454/raw_mink_l0.png runs/session2-repeat-five455/raw_mink_l0.png \
  --prior-queries 1613 --queries 187 --seed 454 --mode ga \
  --output runs/session3-elite-mink-ga
```

## Saved-parent results

| Image | Operator | Best parent | Selected result | Local result | Diagnostic calls | Charged total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mink | support GA |41|32|32|187|1800|
| mink | weak-first deletion |41|41|41|127|1740|
| sock | support GA |23|22|22|299|1800|
| sock | weak-first deletion |23|23|47|153|1654|

Raw records and source snapshots: `runs/session3-elite-{mink,sock}-{ga,weakfirst}/`.
GA search made nine promotions on mink (first40 at diagnostic call6, final32 at
159) and one on sock (22 at135). Repeated masks consumed0/184 mink GA calls and
14/296 sock GA calls, so duplicate-only caching is not the main opportunity on
these cases. Weak-first deletion failed to improve either parent; sock's local
trajectory again entered a worse basin. Protected parent selection prevented
regression. Timings are descriptive, with a synthetic-test/model-weight import
overlapping the final weak-first run; comparisons use equal query ceilings.

`elite_session3_online.py` now implements the successful operator after two
independent discoveries are locally pruned. It uses a deterministic operator
RNG stream (`initial_seed + pair_index`) and reads no saved parents. Online benchmarking completed: mink 32 in 1800 calls/125.812s and sock 22 in 1800 calls/115.683s, seed 454, four threads. Both reproduced the diagnostic winners bitwise and passed independent one-thread classification and parent-value hash checks. Validation is in `runs/session3-elite-online1800/independent_validation.json`. Broader online evidence is still needed before production promotion.

Completed online check (one process, sequential cases; no saved-parent inputs):

```sh
TORCH_HOME=/tmp/cs454-torch MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_attacks.py \
  --implementation research.elite_session3_online --images mink sock --norms l0 \
  --queries 1800 --seconds 165 --threads 4 --seed 454 \
  --output runs/session3-elite-online1800 --tag support1800
```

Online operators count three fewer repeated clean/parent-check calls than the
diagnostic, so an1800-query run can execute187 rather than184 support proposals
on mink and299 rather than296 on sock, provided the same parents are generated
before the deadline. Account for this small distinction when interpreting
exact final scores. Source snapshots and chunks must establish activation and
the actual parent-generation prefix. Independently validate any promoted raw
artifact at one thread as well as the four search threads.

Remaining investigations after the online check: validate activated crossover
on additional independently generated parent pairs/seeds; measure actual165s
default calls; test combination with free-amplitude value evolution only if
support GA plateaus. Existing supplied-parent evidence concerns two cases and
does not establish held-out image generalization. Cache instrumentation shows
few repeats here, so exact fitness caching has lower expected impact than
better search moves or allocating time between discovery and recombination.

## Seed-455 online replication

At the same 1800-query/165-second ceilings and four threads, production mink control41 (1595 calls/135.148s) versus support31 (1800/150.894s); sock control46 (1691/142.511s) versus support43 (1800/145.997s). Parents are mink41/53 and sock49/46; the operator made ten and three promotions. All raw round trips passed. These use spare work the control could not allocate to another discovery, rather than establishing equal-actual-query efficiency. The four activated image/seed comparisons now all improve. No held-out image claim is made. Exact sources and configurations: `runs/session3-l0-control455/`, `runs/session3-l0-support455/`.

A self-contained candidate, `research/epsilon_support_integrated.py`, retains the preserved `_Search` class and adds the operator/policy. It awaits bitwise synthetic proposal checks and an actual default-duration mink run before production promotion.
