# CS454 project status — 2026-11-22

## Current verified implementation

`epsilon_ball.py` implements all required norm entry points; `perlin_fog.py` implements fog. Searches infer the clean prediction, use model probabilities without gradients and accept arbitrary supported image paths. Instructor README, utilities, grader and supplied fog-rendering functions remain unchanged. Submission implementations are self-contained.

- **Linf:** unchanged four-elite spatial GA, signed blocks and radius refinement.
- **L2 default:** global energy GA for at least **512 proposal generations** and three successful stages, then feasible-direction GA with bounded scalar probes. This prevents premature local refinement on initially easy images. **The grader uses sum of squared differences.** `CS454_L2_SEARCH=original` preserves the previous global-only route; `CS454_BOUNDARY_MIN_GENERATIONS` is an experimental override, not a proven optimum.
- **L0 default (`parental`):** bounded discovery, hierarchical/repeated channel deletion, protected restarts and fixed-cardinality parental-support evolution. **L0 counts changed channel elements.** `CS454_L0_STYLE=multistart`, `refined`, `original` retain controls. The new one-parent neighbor tail remains research-only.
- **Fog:** unchanged full mixture GA followed by up to 24 integer-octave generations within remaining time, exact renderer verification and protected mixture fallback. `CS454_FOG_GENERATIONS` controls mixtures; `CS454_FOG_OCTAVE_GENERATIONS` controls the extension.

Norm defaults remain four threads / 165 seconds (hard configurable maximum 170); fog remains 270 seconds (hard maximum 285). **72 tests pass. All 20 canonical artifacts pass fresh one-thread verification; 28 selected generalization artifacts also independently pass.** Existing artifact checksums remain unchanged.

## Generalization evidence and limits

An eight-image external bank was split before attacks: development astronaut/coffee/brick/rocket at seed 454; reserved confirmation camera/chelsea/grass/motorcycle at seed 455. All paired L2 screens use exactly 900 model calls. Confirmation settings remained frozen through the predeclared full-time block.

| L2 comparison | Pairs | Candidate wins | Successes (control/candidate) | Geometric-mean squared-distance ratio |
| --- | ---: | ---: | --- | ---: |
| Old early-switch radial candidate, development | 4 | 1 | 4/4 / 4/4 | 1.400 |
| Minimum exploration 256, development | 4 | 2 | 4/4 / 4/4 | 0.872 |
| Selected exploration 512, development | 4 | 4 | 4/4 / 4/4 | **0.776** |
| Frozen selected port, reserved confirmation | 4 | 4 | 4/4 / 4/4 | **0.788** |
| Same frozen port, actual 165-second confirmation | 2 | 2 | 2/2 / 2/2 | **0.599** |

Actual-time cases: camera **158.252→99.669**, motorcycle **16.205→9.243** squared L2; all calls finish in 162.8–163.0 seconds. Actual model counts differ, so these establish matched-time performance rather than universal per-query efficiency. The earlier candidate's supplied-image gains did not generalize; its premature switch at query 9 on astronaut motivated the revised architecture.

This small fixture bank uses clean argmax references and includes low-confidence/semantically misaligned classifications. It is not a representative ImageNet validation set or evidence of classifier-training-unseen inputs. Its confirmation is now consumed; later tuning must use it as historical evidence. [Current research log](docs/history/2026-11-21-generalization.md), [36 unique trials](runs/generalization-experiment-index.csv), [tests/audit/verification](runs/generalization-final-validation/).

## Next priorities

1. **Active labelled audit:** all ten Imagenette development images, seed456, current512 default versus prior global-only GA at900 queries/165 seconds. Sources and budgets are frozen in `research/labelled_v2_baseline_plan/`; results stream to `runs/labelled-v2-l2-baseline-development/`. The environment interruption preserved14 rows; pinned runtime was restored, one exact tensor/history replay passed, and only unfinished trials are resuming. Ten confirmation images remain untouched. Freeze new experiment methods/budgets in a separate plan; preserve its data-only protocol. [Bank and scope](research/generalization_imagenette_v2/README.md).
2. Evaluate broader actual 165-second performance, more seeds and the remaining supplied-image matrix with the new default. A complete default-duration matrix is still unmeasured. Consider reversible global/local windows driven by improvement per call/second if fixed-stage exploration shows new weaknesses; do not tune against consumed confirmation.
3. Preserve `research/local_support_tail.py` as a mechanism candidate: external development coffee **39→33** at a 900-call ceiling; three inactive ties. Full-time production may already consume all allowance, so default-time benefit is unproven. Test active stage allocation before promotion. Notes: `research/generalization_l0.md`.
4. Later: eight-thread attack scores; external Linf/fog generalization; fog fitness-repeat caching and a larger integer-generation allowance under the same 270-second ceiling. Keep real calls, proposals, partial failures and phase activation separate.

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

Frozen production: `research/generalization_session_promoted/`; previous production: `research/session3_promoted/`; earlier baseline: `research/session3_baseline/`. Generalization protocols, manifests, algorithm/source freezes and exact commands are beside the run configurations. Strict historical source guards may require their captured dependencies in an isolated reproduction checkout; create new plans for current sources instead of changing old protocols. The next labelled archive is pinned by official MD5 and recorded SHA256; selection retains all exclusions and does not filter on attack difficulty.

Detailed history: [current labelled session](docs/history/2026-11-21-labelled-research.md), [prior generalization session](docs/history/2026-11-21-generalization.md), [earlier November 21 session](docs/history/2026-11-21-research.md), [original November 20 record](docs/history/2026-11-20-research-log.md). The original 277-line log is intact. `experiments.csv` is a clean-classification control and `grader_results.csv` is historical starter failure output. Saved-parent diagnostics are not online benchmarks. Attack entry points never read saved attacks or known seed labels. The PDF report remains the user's responsibility.
