# CS454 project status — 2026-11-23

## Current verified implementation

`epsilon_ball.py` implements all required norm entry points; `perlin_fog.py` implements fog. Searches infer the clean prediction, use model probabilities without gradients and accept arbitrary supported image paths. Instructor README, utilities, grader and supplied fog-rendering functions remain unchanged. Submission implementations are self-contained.

- **Linf:** unchanged four-elite spatial GA, signed blocks and radius refinement.
- **L2 default:** global energy GA for at least **512 proposal generations** and three successful stages, then feasible-direction GA with bounded scalar probes. This prevents premature local refinement on initially easy images. **The grader uses sum of squared differences.** `CS454_L2_SEARCH=original` preserves the previous global-only route; `CS454_BOUNDARY_MIN_GENERATIONS` is an experimental override, not a proven optimum.
- **L0 default (`parental`):** bounded discovery, hierarchical/repeated channel deletion, protected restarts and fixed-cardinality parental-support evolution. **L0 counts changed channel elements.** `CS454_L0_STYLE=multistart`, `refined`, `original` retain controls. The new one-parent neighbor tail remains research-only.
- **Fog:** unchanged full mixture GA followed by up to 24 integer-octave generations within remaining time, exact renderer verification and protected mixture fallback. `CS454_FOG_GENERATIONS` controls mixtures; `CS454_FOG_OCTAVE_GENERATIONS` controls the extension.

Norm defaults remain four threads / 165 seconds (hard configurable maximum 170); fog remains 270 seconds (hard maximum 285). **81 tests pass. All 50 distinct labelled-study tensors pass fresh one-thread verification.** Earlier one-thread verification passed all 20 canonical and 28 first-bank generalization artifacts; canonical and labelled-bank checksum manifests still match.

## Generalization evidence

The first external bank confirmed the current L2 method on four fixture images at 900 calls (geometric-mean squared-distance ratio **0.788**) and two at 165 seconds (**0.599**). This was a small bank using the clean argmax as its reference. Its confirmation set has already been used. [Earlier evidence](docs/history/2026-11-21-generalization.md).

The labelled Imagenette bank has ten clean-correct development and ten reserved confirmation images across ten classes. The 900-call development screens use seed 456 and four threads. Every method succeeded on every input.

| Candidate vs. control | Wins / pairs | Geometric-mean squared L2 ratio | Worst ratio |
| --- | ---: | ---: | ---: |
| Current search vs. original global GA | 5 / 10 | 1.040 | 1.638 |
| First reversible candidate vs. current search | 0 / 10 | 1.176 | 1.333 |
| State-preserving candidate vs. current search | 2 / 10 | 1.050 | 1.234 |

Both window candidates fail the frozen **165-second** selection: first reversible loses2/2 (ratio **1.122**); preserved-state wins1/2 but has ratio **1.110**, worst **1.244**. **Reject both; production and all ten labelled confirmation images remain untouched.** State retention produced some global gains but did not offset displaced local refinement. [Completed study and decisions](runs/labelled-v2-completion-study/study.json).

The separate original-global165-second audit is mixed: cassette4.248, gas pump631.773. Its geometric-mean ratio is0.990 against earlier current controls,1.123 against later controls. All control measurements are retained; changing CPU throughput makes the denominator consequential. These two development cases do not settle broader generalization. [Detailed evidence and limitations](docs/history/2026-11-21-labelled-research.md).

## Next priorities

1. Broaden matched165-second L2 evaluation beyond these two classes and check another seed. Keep timings sequential, record actual calls, use fresh controls, and retain all failures/regressions. Do not repeat the rejected window variants or infer wall-time quality from900-call screens alone.
2. Test L0 fixed-support exhaustion plus novel neighboring loci. One consumed development case spent136/156 support calls on repeated masks; cache/exhaustion handling may free useful work, but earlier generic discovery caches already had mixed results. Preserve the discovery/pruning prefix and measure actual165-second gains before promotion. The existing one-parent tail has only one activated900-call win; its full-time benefit remains unmeasured.
3. Keep the ten labelled confirmation images reserved until a frozen method qualifies. If revisiting L2, a compact spatial-coefficient genotype tests a different proposal representation; avoid another window/gate sweep. Generalization of Linf/fog is also still limited. Imagenette covers ten classes and may overlap classifier pretraining. [Data protocol](research/generalization_imagenette_v2/README.md).

## Canonical assignment artifacts

`submission/` still contains 20 raw tensors, 20 plots, five fog JSON files, provenance, verified scores and checksums. These best artifacts mix methods, seeds and budgets; **they are not a uniform benchmark of the new default**. This session preserves them intact.

| Image | Linf | Squared L2 | L0 | Fog magnitude |
| --- | ---: | ---: | ---: | ---: |
| knot | 0.163680 | 386.170 | 109 | 0.912109 |
| mink | 0.067076 | 235.239 | 31 | 0.413872 |
| orca | 0.162440 | 940.956 | 82 | 0.501953 |
| sock | 0.042834 | 131.672 | 22 | 0.123452 |
| tank | 0.103578 | 595.750 | 84 | 0.903415 |

Exact values/origins: [verification](submission/verification.csv), [selection](submission/selection.csv). Old bundles remain in `runs/submission-session3-before-improvements/`, `runs/submission-session2-before-recombination/`, `runs/submission-session1/`.

## Reproduction and retained records

Run from the repository root. Persistent caches are ignored under `.runtime/`; run `python3 research/restore_runtime_links.py` after temporary-directory resets. Environment: `/tmp/cs454-baseline-env/bin/python`, `TORCH_HOME=/tmp/cs454-torch`, `MPLCONFIGDIR=/tmp/cs454-plot-cache`. Recreate missing temporary dependencies from `torch_cpu.txt` / `requirements.txt`; versions and checkpoint hashes are recorded in run metadata. Use fresh output paths.

```sh
TORCH_HOME=/tmp/cs454-torch MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_attacks.py \
  --images mink sock --norms l2 --queries 900 --seconds 165 \
  --threads 4 --seed 455 --output NEW --tag trial
# Set CS454_L2_SEARCH=original for the previous L2 route; remove --queries for wall-time evaluation.
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python \
  verify_artifacts.py submission --perlin
```

Frozen production: `research/generalization_session_promoted/`; previous production: `research/session3_promoted/`; earlier baseline: `research/session3_baseline/`. Generalization protocols, manifests, algorithm/source freezes and exact commands are beside the run configurations. Strict historical source guards may require their captured dependencies in an isolated reproduction checkout; create new plans for current sources instead of changing old protocols. The labelled archive is pinned by official MD5 and recorded SHA256; selection retains all exclusions and does not filter on attack difficulty.

Completed labelled outputs: [50 unique trial records](runs/labelled-v2-completion-study/results.csv), [independent verification](runs/labelled-v2-completion-validation/artifacts.csv), [81-test log](runs/labelled-v2-completion-validation/tests.txt), [source audit](runs/labelled-v2-completion-source-audit.json). All planned jobs have finished; no promotion is pending.

Detailed history: [current labelled session](docs/history/2026-11-21-labelled-research.md), [prior generalization session](docs/history/2026-11-21-generalization.md), [earlier November 21 session](docs/history/2026-11-21-research.md), [original November 20 record](docs/history/2026-11-20-research-log.md). The original 277-line log is intact. `experiments.csv` is a clean-classification control and `grader_results.csv` is historical starter failure output. Saved-parent diagnostics are not online benchmarks. Attack entry points never read saved attacks or known seed labels. The PDF report remains the user's responsibility.
