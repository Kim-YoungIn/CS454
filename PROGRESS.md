# CS454 project status — 2026-11-23

## Current verified solution

All required interfaces are implemented and self-contained. Searches infer the clean class and use model probabilities without gradients, saved attacks or image/class-specific recipes. Instructor utilities/grader and the three fog renderer functions are preserved. Production sources still byte-match `research/generalization_session_promoted/`.

- **Linf:** four-elite signed spatial-block GA with radius refinement.
- **L2:** global energy GA for at least512 main proposals and three successes, then feasible-direction GA and bounded scalar probes. `CS454_L2_SEARCH=original` retains the global-only route. **Grading L2 is sum of squares.**
- **L0:** bounded discovery, hierarchical/repeated channel deletion, protected restarts and parental-support evolution (`parental`). **Grading L0 counts changed channel elements.** The neighbor-tail candidate remains research-only.
- **Fog:** full mixture GA plus up to24 integer-octave generations, exact renderer verification and protected fallback.

Defaults: four Torch threads,165s per norm (hard configurable max170), fog270s (hard max285). **93 distinct tests pass; all40 new confirmation artifacts pass independent one-thread validation and retain the same predicted classes.** Canonical20 artifacts and source hashes are preserved.

## Completed all-norm generalization confirmation

The user explicitly authorized consuming all ten previously reserved Imagenette confirmation images. Frozen before outcomes: one clean-correct photo/class across ten classes, seed457, no query cap, all four attacks. **40/40 succeed; no180/300-second grading timeout or165/270-second optimizer-target overrun.** Fog grading time includes exact regeneration and classification.

| Attack | Verified success | Median grading distance | Maximum grading seconds |
| --- | ---: | ---: | ---: |
| Linf | 10/10 | 0.0126621 | 162.869 |
| Squared L2 | 10/10 | 15.2024 | 163.256 |
| L0 elements | 10/10 | 5.5 | 164.172 |
| Fog magnitude | 10/10 | 0.168201 | 228.110 |

At the user's request, after11 serial cases the remaining29 ran as two concurrent workers, each pinned to four disjoint physical cores according to the exposed CPU topology. Existing results were preserved without reruns. Shared memory bandwidth, package power/frequency and host scheduling remain uncontrolled. **These mixed execution conditions are recorded; this is absolute transfer evidence, not a comparative algorithm/throughput improvement claim.** One seed and ten classes do not establish population-wide generalization; Imagenette may overlap VGG pretraining. The bank is now consumed and cannot serve as fresh holdout after tuning.

Evidence: [norm summary](runs/imagenette-allnorm-confirmation-validation-20261123/report/results.md), [full10×4 table with execution conditions](runs/imagenette-allnorm-confirmation-validation-20261123/execution-report-v2/execution_results.md), [wide CSV](runs/imagenette-allnorm-confirmation-validation-20261123/execution-report-v2/image_norm_wide.csv), [independent checks](runs/imagenette-allnorm-confirmation-validation-20261123/report/verification.csv), [tests](runs/imagenette-allnorm-confirmation-validation-20261123/all-tests.txt), [source audit](runs/imagenette-allnorm-confirmation-validation-20261123/source-audit.json). Protocol/source/runtime freeze: `research/imagenette_allnorm_confirmation_20261123/`; live/terminal cases: `runs/imagenette-allnorm-confirmation-20261123/`. **All requested jobs and validation are complete.**

## Research decisions and next priorities

The earlier labelled development study rejected both L2 window variants: first reversible165s ratio1.122; preserved-state1.110, worst1.244. Production was retained before this confirmation. Current-vs-global900-call development was mixed (5wins/5losses, ratio1.040), and the two-class full-time audit depended on timing controls. Do not repeat those candidates or reinterpret them as confirmed improvements. Details: [labelled L2 history](docs/history/2026-11-21-labelled-research.md), [closed study](runs/labelled-v2-completion-study/study.json). The earlier eight-image fixture bank is also already consumed.

1. Optimize only on development data; keep the frozen production control, use matched actual-time runs and fresh timing controls, examine worst cases/another seed, and obtain a new independent bank before claiming further generalization improvements. French horn remains costly here (squaredL2 4039.34,L0 177); this is a limitation, not permission to tune a class-specific rule.
2. Investigate L0 fixed-support exhaustion plus novel neighboring loci. A consumed development case spent136/156 support calls on repeated masks. Earlier generic discovery caches had mixed results; preserve discovery/pruning prefixes and prove actual165-second gains. The existing neighbor tail has one activated900-call win; full-time benefit remains unmeasured.
3. A compact spatial-coefficient L2 genotype is a different proposal hypothesis if allocation designs stall. Avoid another window/gate sweep. Preserve failure-inclusive reporting and distinguish more actual work from a better operator.

## Canonical assignment artifacts

`submission/` retains20 raw tensors,20 plots,five fog JSON files, scores, provenance and checksums. The best artifacts mix methods/seeds/budgets; **they are not a uniform benchmark of the current default**. They were previously independently verified; their checksum manifest still matches.

| Image | Linf | Squared L2 | L0 | Fog magnitude |
| --- | ---: | ---: | ---: | ---: |
| knot | 0.163680 | 386.170 | 109 | 0.912109 |
| mink | 0.067076 | 235.239 | 31 | 0.413872 |
| orca | 0.162440 | 940.956 | 82 | 0.501953 |
| sock | 0.042834 | 131.672 | 22 | 0.123452 |
| tank | 0.103578 | 595.750 | 84 | 0.903415 |

Exact origins: [verification](submission/verification.csv), [selection](submission/selection.csv). Earlier bundles remain under `runs/submission-session*/`. The PDF report remains the user's responsibility.

## Reproduction and retained history

Run from the repository root. `.runtime/` contains ignored persistent dependencies and VGG weights; `python3 research/restore_runtime_links.py` restores `/tmp/cs454-baseline-env` and `/tmp/cs454-torch` aliases. Python/package/checkpoint hashes are in each frozen plan. Use fresh report/run destinations; completed terminal trials must not be repeated.

```sh
TORCH_HOME="$PWD/.runtime/torch-cache" .runtime/python-env/bin/python \
  research/validate_allnorm_confirmation.py \
  --run runs/imagenette-allnorm-confirmation-20261123 --output FRESH_REPORT
TORCH_HOME="$PWD/.runtime/torch-cache" .runtime/python-env/bin/python \
  verify_artifacts.py submission --perlin
```

Historical evidence and exact resume commands: [all-norm confirmation](docs/history/2026-11-23-allnorm-confirmation.md), [labelled L2](docs/history/2026-11-21-labelled-research.md), [earlier generalization](docs/history/2026-11-21-generalization.md), [November21](docs/history/2026-11-21-research.md), [original277-line log](docs/history/2026-11-20-research-log.md). Prior production snapshots remain under `research/session3_promoted/` and `research/session3_baseline/`; strict historical guards require captured dependencies. `experiments.csv` is a clean-classification baseline, not an adversarial optimization result.
