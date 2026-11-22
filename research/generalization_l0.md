> Current result: four external development pairs give one activated win (coffee39→33) and three inactive ties at900 calls. Full-time benefit remains unproven; production stays unchanged. See `docs/history/2026-11-21-generalization.md`. The frozen dependency records predate the unrelated L2 addition; historical full-file reproduction should use the captured baseline, whose L0 classes remain AST-identical to production.

# One-parent L0 tail investigation

## Question and frozen candidate

The production parental-support GA requires two completed locally pruned parents. On an unseen image, expensive pruning may leave only one parent, or the policy may have fewer than 256 calls left and decline a fresh discovery. The current extra representation then never activates. Discovery itself can also struggle on unseen images: its initial contiguous RGB block and 900-call-calibrated phase allocation were measured on the five supplied images. A neighbor tail addresses the unused-allowance case only; it does not establish a solution to difficult discovery or slow pruning.

`research/local_support_tail.py` defines `LocalTailSearch`, which first executes **the entire current production policy**, then runs only if a feasible image has L0 > 1 and `_Search.available()` still permits work. It builds a candidate allele universe from that image and valid eight-neighbor positions in the same channel. A neighbor inherits the source's actual normalized endpoint value, not its delta. The smallest source index wins conflicting neighbor values; the original parent wins overlap conflicts. The weak synthetic image supplies alleles and is never assumed feasible. The existing fixed-cardinality support GA attempts k−1 elements with margin-based selection, mutation, crossover and a protected feasible archive.

The independent tail RNG seed is `initial_seed + 1000003`. No gradients, labels, saved tensors, image names, dataset statistics or image-specific configurations are used. Production files were not changed. The candidate and tests were frozen **before any real-image attack evaluations or confirmation results**; no confirmation images/predictions were inspected. Exact candidate, test and production dependency SHA256 hashes are in `research/generalization_l0_frozen.json`.

## Safety and accounting

- Existing discovery, restart and parental-support work is not deliberately diverted. Fixed-query synthetic traces show every production oracle input as the candidate's exact prefix. Wall-clock stopping can still differ slightly near a deadline; compare the actual entry-point times and query counts.
- Only `_Search.evaluate` can improve the inherited image. Failed replacement candidates cannot overwrite its support, feasible tensor, selected origin or deletion-minimality flag. A successful replacement clears deletion-minimality, because the new support has not undergone a complete deletion pass.
- A tail chunk has origin `local_support_tail`, initial/final distance, query boundaries, novel-locus count, attempts, promotions and repeated-support count. No tail chunk means no activation. Discovery counts are unchanged; all tail attempts contribute to `support_evaluations`. The identity remains clean1 + discovery + pruning + cleanup + support = total calls.
- The same entry-point wall deadline and query limit remain authoritative. Neighbor construction has CPU/memory overhead and needs actual-time measurement, especially for very large inherited supports. Exceptions or overhead can still cause a practical regression even though the successful-image archive is monotonic.
- No improvement is expected if production already consumes its allowance, never finds an adversary, reaches L0=1, or has no novel neighbor alleles. This deliberately narrow candidate is not a broad generalization claim.

## Tests and exact next experiment

Three tests pass without VGG forwards (0.069 seconds): endpoint transport and no row/channel wrapping; complete real production-policy prefix preservation plus a failed tail protecting its feasible archive and exact query accounting; a synthetic locally minimal three-element parent improved to two elements by introducing a new neighbor locus.

```sh
TORCH_HOME=/tmp/cs454-torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/cs454-baseline-env/bin/python -m unittest -v research.test_local_support_tail
```

The parent agent is constructing an independent public-image development/confirmation split in `research/generalization_v1/protocol.json`. Respect its frozen split. Compare production and the frozen candidate on development inputs first at **900-call / 165-second ceilings**, seed454, four threads, in sequential CPU windows. Report success, actual L0, actual calls/time, and whether a tail activated. Do not report inactive ties as evidence of effective replacement search. Use a fresh output directory per implementation; examples:

```sh
TORCH_HOME=/tmp/cs454-torch MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_attacks.py \
  --images DEVELOPMENT_IMAGE_PATHS --norms l0 --queries 900 --seconds 165 \
  --seed 454 --threads 4 --output FRESH_CONTROL --tag generalization900

TORCH_HOME=/tmp/cs454-torch MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_attacks.py \
  --implementation research.local_support_tail \
  --images DEVELOPMENT_IMAGE_PATHS --norms l0 --queries 900 --seconds 165 \
  --seed 454 --threads 4 --output FRESH_CANDIDATE --tag generalization900
```

A supplied-image mechanism check on mink at900 is also interpretable: known production first-parent completion850 leaves50 spare calls. It is not held-out evidence. The four independent development images need not have the same activation pattern. Run the confirmation split only once after freezing the selection decision; retain the existing production if evidence is weak. A second seed and actual165-second evaluations are needed before promotion. Larger1800-call trials can diagnose spare-tail activation but may be exact ties because the existing two-parent GA consumes all remaining allowance.

Use success rate first, then paired distance ratios on mutually successful cases, worst paired regression, and actual work/time. Clean model argmax is the attack reference, not a verified semantic ground-truth label; an eight-image bank gives useful transfer evidence but cannot establish broad population generalization.
