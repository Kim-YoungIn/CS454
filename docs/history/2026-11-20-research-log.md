# Current state — 2026-11-20

The L0 default now combines genetic discovery, bounded discovery work, hierarchical/repeated support deletion, and protected independent restarts. All ten matched image/seed comparisons improved over the original GA at a 900-query ceiling. The integrated default completed an uncapped mink call in 164.1 seconds and retained its strongest 41-element result across three searches. Original and faster single-search modes remain available through `CS454_L0_STYLE`.

`submission/` contains the best verified artifacts across recorded runs. Current best L0 counts are knot 109, mink 38, orca 82, sock 22, tank 84 channel elements; these include multiple seeds and experimental refinements, so use the controlled tables below for algorithm comparisons. All 20 saved cases pass independent validation, all 48 contract tests pass, and the unchanged instructor grader accepts the revised interface. The prior canonical bundles are preserved under `runs/submission-session1/` and `runs/submission-session2-before-recombination/`. Dense and fog search are unchanged in this continuation. Genuine held-out image performance remains unmeasured. See the latest continuation priorities at the end and exact scores/provenance in `submission/verification.csv` and `submission/selection.csv`.

# Clean VGG-11 baseline

On 2026-11-20, two independent CPU processes produced byte-identical CSV results for all five seed images. Each top prediction matched the class ID in `util.IMAGES_ORG_PREDS`. The lowest clean top-class probability was 0.695805 for sock. The other four exceeded 0.993. These measurements establish clean predictions only. Held-out performance and adversarial distances are unmeasured.

## Experiment

- `baseline.py` uses the instructor's image loader and prediction helper, with one PyTorch CPU thread and inference mode. The input files, SHA-256 digests, class IDs, and probabilities appear in `experiments.csv`.
- A synthetic difference tensor changes two channels of one pixel by −0.5 and +0.25. The grader's helpers returned 0.5 for Linf, 0.3125 for squared L2, and 2 for element-wise L0. The script checks all three before processing images.
- The environment used Python 3.14.4, torch 2.15.0+cpu, torchvision 0.29.2+cpu, Pillow 13.0.0, and NumPy 2.5.4. The VGG-11 checkpoint SHA-256 was `8a7190461cb581dbcbd2c3b58f43cf12a0bca3ccf92898f867e4a9e9b898c464`.

## Reproduction

Run from the repository root. The environment and checkpoint used in this session are temporary. To recreate the baseline environment on a fresh machine, use the following commands.

```sh
python3 -m venv /tmp/cs454-baseline-env
/tmp/cs454-baseline-env/bin/python -m pip install -r torch_cpu.txt Pillow==13.0.0 numpy==2.5.4
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python baseline.py > /tmp/cs454-clean.csv
diff -u experiments.csv /tmp/cs454-clean.csv
```

The first import of `util.py` downloads the checkpoint if the cache is empty. The exact CSV comparison above reproduced on this CPU. When comparing across platforms, check the image hashes and class IDs first, then compare probabilities allowing for floating-point differences.

## Assignment status

On 2026-11-20 the instructor's simulator ran all five seed images through all four entry points. All 15 required cases and all five bonus cases raised `NotImplementedError`. The per-case score of 999999999 is the simulator's failure sentinel. No case produced a measured adversarial distance or a submission tensor. The figures under `images/examples/` and `images/init_viz/` are instructor-provided.

`grader_results.csv` preserves the simulator's 20 rows. Reproduce the run with `TORCH_HOME=/tmp/cs454-torch OMP_NUM_THREADS=1 /tmp/cs454-baseline-env/bin/python grade.py -n audit -i all -l all` from the repository root after installing the dependencies listed below. Wall times can vary.

## Fog coefficient recovery, 2026-11-20

`fog_calibration.py` tests how accurately a GA can recover a known opacity map from the four instructor-rendered octave layers at 3, 6, 12, and 24. The fitness is mean squared reconstruction error. Both targets have total magnitude 0.5. The first uses the README's example weights (0.4, 0, 0.1, 0). After observing that a decreasing mutation scale won on 8 of 10 seeds, we repeated the comparison on (0.20, 0.15, 0.10, 0.05) to check the result away from zero-weight boundaries.

Each method starts from the same uniform population for a given seed. We evaluate 32 rows for each of 40 generations, retain one elite, and sample parents uniformly from the best eight. Each child inherits one parent's weight independently at each octave, then receives Gaussian mutations clipped to keep coefficients within [0, 0.5]. The constant mutation standard deviation is 0.08. The decaying schedule multiplies it by 0.95 each generation. Both methods use the same random draws for paired seeds 0–9.

Training uses one spatial sample every 16 pixels in both dimensions across all channels. Evaluation uses the same spacing offset by eight pixels on both axes. Lower root mean squared error (RMSE) on the second grid is better. Least squares on the training grid provides a numerical reference because the supplied fog field is linear in its magnitudes before blending with an image.

| Target | Constant GA median RMSE | Decaying GA median RMSE | Seeds favoring decay | Least-squares RMSE |
| --- | ---: | ---: | ---: | ---: |
| Two active octaves | 0.002806 | 0.000597 | 8/10 | 8.18e−9 |
| Four active octaves | 0.006796 | 0.001727 | 10/10 | 1.12e−8 |

The decaying schedule gave lower median error on both fixed fields. Two seeds on the first field favored constant mutation. These trials measure weight recovery with known candidate octaves. They supply no estimate of misclassification rate or held-out image scores. A follow-up calibration could fit weights from clipped, blended observations and measure how well the original weights remain identifiable.

For the first recipe, JSON reload and a different torch seed reproduced the exact opacity tensor. On all five seed images the instructor's visualization writer produced raw tensors identical to the tensors passed to it. Top-class ID, probability, and all three distance helper outputs matched on reload. These temporary files serve as serialization controls.

### Reproduction

Using the baseline environment above, install the two additional packages and run from the repository root. Two independent processes produced identical `fog_calibration.csv` files on the recorded environment.

```sh
/tmp/cs454-baseline-env/bin/python -m pip install matplotlib==3.11.2 perlin-noise==1.13
OPENBLAS_NUM_THREADS=1 TORCH_HOME=/tmp/cs454-torch MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cs454-plot-cache \
    /tmp/cs454-baseline-env/bin/python fog_calibration.py > /tmp/cs454-fog.csv
diff -u fog_calibration.csv /tmp/cs454-fog.csv
```

## Saved submission verification

`verify_artifacts.py` reads the submission files specified by the README and writes one CSV row per case. It checks tensor shape, dtype, and finiteness, then uses the supplied distance helpers and compares VGG-11's top class against the reference class. For the optional fog cases it checks the JSON pairs and compares the rebuilt, clipped image against the saved tensor with absolute tolerance 1e−6 and relative tolerance 1e−5. Fog distance and classification use the rebuilt image and its gene magnitudes, as in the instructor's simulator.

Run from the repository root in the baseline environment. Omit `--perlin` to check only the 15 required files. Exit code 0 indicates that every requested case passed. Exit code 1 indicates a missing or invalid artifact or a prediction that retained the reference class.

```sh
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python verify_artifacts.py /path/to/submission --perlin
```

The empty-directory test returned 15 missing cases. A mixed 20-case control directory returned four unchanged classifications, five invalid cases, and 11 missing cases. Controls covered an exported clean image, a fixed single-element difference, the README's fixed fog recipe, a wrong tensor shape, a corrupt file, a NaN, a boolean value in place of an octave, and inconsistent JSON and tensor files. The measured distances matched the instructor's helpers. Both invocations exited with code 1 as expected.

## Genetic attack implementation, 2026-11-20 (ongoing)

Implemented the three public norm entry points as gradient-free, elitist spatial genetic searches. Fitness is the clean-class log probability minus the strongest rival's log probability. The clean class is inferred from the supplied model, including for paths outside the five-image table. Four truncation-selected elites, block crossover, structured mutations, and an independently maintained successful archive allow harder perturbation budgets without losing a valid output. Returned tensors are clipped to the evaluator's range; the instructor's model, metrics, grading code, and fog renderer remain unchanged.

Linf uses signed spatial blocks; L2 initially uses local color-block additions with radius projection; L0 evolves extreme-valued channel-element supports and prunes successful supports. L2 is being compared with a two-window energy-redistribution representation using smooth positive/negative lobes. This is still derivative-free. `CS454_SECONDS`, `CS454_QUERIES`, `CS454_SEED`, and `CS454_THREADS` control norm experiments. Default norm searches use 165 seconds and four CPU threads, with a hard internal maximum of 170 seconds and a forward-time reserve. Fixed query caps support reproducible comparisons; a wall deadline can shorten a run on slower hardware.

The first 350-query sock trials (`runs/preliminary/sock_v1.txt`) found Linf 0.06710887 and squared L2 575.9985. Those trials predate fixes and are preliminary, not a complete adversarial baseline. Review caught an overwritten Linf crossover and an unreachable boundary-polishing phase. Actual model tests exposed noncontiguous loader output: flattened indexed writes modified copies, defeating L0 support pruning. The engine now makes its input contiguous, and synthetic tests cover this regression. The v1 source is retained under `research/v1/` for interpretation of these early measurements, not as the recommended implementation.

`benchmark_attacks.py` uses the instructor's metrics, saves raw tensors and class plots through `get_viz_for`, reloads tensors independently, and checks exact distances and predictions. It records source/input hashes, configuration, query counts, runtimes, and the history of successful distances. New runs should use a fresh output directory. `select_artifacts.py` independently classifies candidates and selects the lowest successful grader distance into an empty output directory; it is an offline artifact tool, never an input to the attack functions. `test_attacks.py` checks contracts against synthetic probability oracles, rather than making performance claims from mocks.

### First complete adversarial baseline and L2 comparison

`runs/ga-v2/results.csv` records all 15 norm cases with seed 454, four threads, an 80-second safety ceiling and a 450-query cap. Linf and L0 succeeded on all five images. The additive-block L2 method succeeded only on mink (1980.988) and sock (574.877); knot, orca and tank failed. Successful cases consumed 450 queries, with entry-point runtimes 25.2–31.6 seconds on this CPU. Failed dense searches reserved 12 polishing queries but had no successful archive to polish, so stopped at 438; this accounting defect was corrected for subsequent comparisons. These are bounded short experiments, not measurements of the full default time budget.

The structured-energy L2 variant (`runs/l2-energy-450/`) uses the same seed/thread/query ceilings and succeeded on all five images. It initializes tiled smooth two-lobe patterns, then recovers energy from two spatial windows (counting overlaps only once) and reallocates it into a mutated first window. Selection, crossover, projection and budget contraction remain genetic and derivative-free. Measured squared distances were knot 2277.079, mink 574.595, orca 2144.848, sock 367.022 and tank 2240.321. It improved both common successful cases and rescued all three earlier failures, so it replaces additive-block mutation as the default L2 representation. The 12-query failure-accounting change is a minor experimental confound; do not attribute the entire success-rate difference solely to representation.

A further `portfolio` experiment mixes energy redistribution, additive color patches and bounded sign-block replacement within the same elite population. It is retained as an explicit `CS454_L2_STYLE` research option alongside `energy`, `additive`, and `sign`; default selection will follow measured results. `runs/l2-portfolio-450/` and the first fog benchmark run concurrently on separate processes. Their fixed-query outcomes can be compared when query caps are reached, but their runtimes should not be treated as isolated CPU timing comparisons. Configuration metadata marks this context.

### Optional fog attack: verified rendering, broader octave search

`adv_func` now evolves nonnegative octave mixtures with truncation selection, uniform crossover, Gaussian mutation, occasional fresh Dirichlet directions, and scalar boundary searches on successful mixtures. It caches the exact instructor-rendered basis only inside a call. Final genes are checked through the unchanged reference renderer; zero coefficients are omitted, and every submitted pair has a positive integer octave and positive finite magnitude. The default pool is `(1,2,3,4,6,8,12,24,32,48,64,96)`. The default cap is 80 generations with a 36-generation stale stop and a 270-second guard; the hard configurable maximum is 285 seconds. The successful archive is retained on early stopping. Full 80-generation runs have not yet been measured against the model.

`runs/fog-v1/` and `runs/fog-v2/` compare eight versus twelve candidate octaves at population 16, seed 454, four threads, ten generations and a 90-second cap. All ten trials completed all ten generations, passed classification, and passed independent JSON/tensor regeneration checks. These runs overlapped the fixed-query norm portfolio experiment; their measured 14.6–29.4 second call times are descriptive, not isolated timing comparisons.

| Image | Eight-octave cost | Twelve-octave cost | Selected octave (twelve) |
| --- | ---: | ---: | ---: |
| knot | 0.912109375 | 0.912109375 | 24 |
| mink | 0.962890625 | 0.556640625 | 64 |
| orca | 0.501953125 | 0.501953125 | 24 |
| sock | 0.3349609375 | 0.169921875 | 48 |
| tank | 1.078125 | 0.9765625 | 64 |

Every winner was a pure-octave candidate from initial screening. The GA's subsequent mixture search did not improve these five ten-generation results. The measured improvement is a broader octave pool, not evidence that mixture evolution or 80 generations improves performance. Retaining the longer default extends the verified search prefix without discarding its successful archive and may help on other images; that generalization remains unmeasured.

Reproduce the twelve-octave experiment from the repository root in the baseline environment, choosing a fresh `NEW` directory:

```sh
TORCH_HOME=/tmp/cs454-torch MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_fog.py \
  --generations 10 --seconds 90 --threads 4 --seed 454 \
  --octaves 1 2 3 4 6 8 12 24 32 48 64 96 \
  --output NEW/results.csv --artifacts NEW
```

The norm portfolio comparison (`runs/l2-portfolio-450/`) completed all 450 queries for every case but succeeded on only mink, sock and tank. Squared distances were 2170.978, 347.708 and 2249.178. It failed on knot and orca; the pure energy operator succeeded on all five and is the chosen default. The slightly better sock portfolio artifact is retained for offline selection without introducing any image-specific behavior into the attack.

`runs/baseline-submission/verification.csv` independently verifies a complete set of 15 required norm artifacts from the first successful baselines, using the validator's one-thread evaluation rather than search's four. All 15 passed. Floating-point L2 sums differ in the last few decimal places with reduction thread counts; classification and support counts agreed.

### Isolated 900-query refinement

`runs/ga-final-900/` uses the chosen defaults, seed 454, four threads, a 900-query cap and a 165-second wall ceiling. All 15 calls reached exactly 900 queries, succeeded, and passed exact saved-tensor round-trip checks. All 15 distances improved over the best successful 450-query candidate from earlier variants. Entry-point runtimes were 50.23–54.65 seconds (total 781.5 seconds), excluding model import, plotting and independent reload verification.

| Image | Linf | Squared L2 | L0 channel elements |
| --- | ---: | ---: | ---: |
| knot | 0.163680 | 890.127 | 627 |
| mink | 0.067076 | 235.239 | 113 |
| orca | 0.162440 | 940.956 | 176 |
| sock | 0.042834 | 147.622 | 39 |
| tank | 0.103578 | 595.750 | 2240 |

Reproduce in a fresh output directory (use the baseline Python environment and checkpoint cache described above):

```sh
TORCH_HOME=/tmp/cs454-torch MPLCONFIGDIR=/tmp/cs454-plot-cache \
  /tmp/cs454-baseline-env/bin/python benchmark_attacks.py \
  --images knot mink orca sock tank --norms linf l2 l0 \
  --queries 900 --seconds 165 --threads 4 --seed 454 \
  --output NEW --tag final900
```

The initial inference microbenchmark is preserved in `runs/preliminary/inference_profile.csv`: knot input repeated to batches 1/4/8, threads 1/2/4, inference mode, one warm-up followed by the mean of three forwards per setting. Four threads was fastest among those tested (0.0675 seconds for one image versus 0.1425 at one thread). Small batches improve throughput but delay sequential genetic selection; norm search uses single offspring, while fog uses batches of four. Eight and sixteen CPU threads were not tested in that initial microbenchmark; see the continuation profiler below.

### Full-duration check, reproducibility and final artifacts

`runs/default-budget/` measures the public default L2 configuration on knot without a query cap. It completed 2,766 model evaluations (including the clean input) in 162.772 seconds, returned a successful image, and passed export/reload verification. Squared L2 improved from the 900-query value 890.127 to 386.170. This is a measured default-duration call within the three-minute limit; the full default-duration matrix has not yet been run. Reproduce with `benchmark_attacks.py --images knot --norms l2 --seconds 165 --threads 4 --seed 454 --output NEW --tag default165` in the same environment.

`runs/validation/grader-smoke.txt` is from the unchanged instructor grader on sock, all four public entry points, with `CS454_QUERIES=100`, `CS454_SECONDS=165`, `CS454_THREADS=4`, `OMP_NUM_THREADS=1`, `CS454_FOG_GENERATIONS=10`, and `CS454_FOG_SECONDS=90`. All four cases passed (5.9–16.9 seconds each). This also checks normal caller autograd state and restoration of the caller's thread count; it is an interface smoke test, not the final score benchmark.

`runs/repeatability/` repeats the 450-query energy-L2 sock experiment in an independent process. The tensor, distance, predicted class, query count, and entire successful-distance history matched exactly (`runs/validation/repeatability.json`). `runs/validation/unit-tests.txt` records 16 passing tests covering norm bounds, channel-element support, noncontiguous inputs, untouched elements, held-out path/class handling, archived success, fixed-query randomness, crossover/mutation geometry, clipping, serialization-related rendering, successful and failed deadlines, and thread restoration. These synthetic tests establish contracts, not real held-out attack performance.

`submission/selection.csv` records the source of every selected artifact. Fourteen norm artifacts come from the 900-query run, knot L2 comes from the full-duration run, and the five bonus cases come from the wider octave pool. The folder contains exactly 20 raw tensor files, 20 classification plots and five fog JSON files. `submission/verification.csv` records all 20 passing independent checks at one CPU thread; `submission/SHA256SUMS` protects the saved files against accidental changes. The instructor's utility and grader files remain byte-identical to the supplied versions, and the three instructor fog-renderer functions remain unchanged.

Recheck the final bundle from the repository root:

```sh
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python verify_artifacts.py submission --perlin
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python -m unittest -v test_attacks test_fog_geometry
```

### Continuation priorities

- Run the full default-duration matrix and multiple fixed seeds. The 900-query matrix is complete, but only knot L2 has a measured full-duration call. Keep experiment directories immutable and compare success rate first, then distances under equal budgets.
- Improve tank L0 first: 450→900 queries reduced its support only from 2382 to 2240, whereas the other L0 cases improved much more. Test channel-wise and spatial-group deletion, selective value mutation, and support relocation against the current baseline.
- Investigate gentler dense-budget contraction or restarting near the successful boundary. Knot L2 spent 1279 queries between its 603.98 and 386.55 successful distances in the full-duration run; the current 20% radius reduction can create long difficult stages. Nonuniform color-energy allocation is another untested L2 representation.
- Measure genuine held-out images when available. The search derives clean classes and never looks up artifacts or seed filenames, but all actual classifier experiments so far use the five supplied images. Explore fog octave neighborhoods and mixture representations; separate gains from initial octave screening from any gains due to GA evolution. Full 80-generation fog behavior remains unmeasured. Profile eight/sixteen inference threads before changing the verified four-thread default.

The implementation and submission artifacts are a working milestone. The PDF report remains the user's responsibility; no report has been generated.

## Continuation research — 2026-11-20

The verified production implementation and its current submission are preserved while alternatives run through `benchmark_attacks.py --implementation MODULE`. The harness now snapshots both the frozen base and candidate module, plus declared Python-source dependencies, and records the actual entry-point module. `research/session2_baseline/` preserves the production source, progress record, and verification table at the start of this session. Candidate code does not read saved attacks or reference-label tables.

The first investigation targets L0's weakest result: tank changed 2240 channel elements after 900 queries, only a small improvement over its 2382 at 450. Hypothesis: random subset deletion destroys useful elements alongside irrelevant ones; hierarchical channel/spatial ablation can remove much more redundant support from an already successful genetic phenotype. `research/structured_sparse.py` retains the spatial GA for approximately 55% of the budget after a success exists, then greedily tests whole-channel deletion and splits failed spatial groups by their widest coordinate. Every accepted deletion preserves a model-confirmed misclassification. The discovery stage retains its full budget if no adversary has yet been found. Five synthetic tests cover essential-element recovery, failed-deletion archive retention, exact support partitions, reproducibility and no-success budget behavior.

At seed 454, four threads and exactly 900 queries, the prototype reduced tank from 2240 to **99** changed channel elements (`runs/session2-structured-tank454/`). Discovery ended at query 496 with distance 2382; 404 pruning evaluations produced the improvement. Raw export/reload checks passed. This establishes a promising single-image result, not yet a production improvement or a generalization claim. The 87.7-second runtime overlapped an independent fixed-query candidate and is not an isolated speed comparison.

The independent compact-genome alternative (`research/sparse_candidate.py`) evolves two rectangles with positions, aspect ratios, continuous RGB colors and channel-subset genes using differential mutation and binomial crossover. On the same tank/seed/query settings it scored **6228**, worse than 2240; its first success occurred only at query 642 after several failed support-budget stages. It is rejected for promotion; source, four passing synthetic tests, configuration and artifacts remain in `runs/session2-rect-tank454/`. Both initial L0 trials reached all 900 queries, making the objective comparison independent of their concurrent runtimes.

A fresh isolated profiler (`runs/session2-thread-profile.csv`, source `profile_inference.py`) measured threads 1/2/4/8/16, batches 1/4, two warm-ups and five timed forwards per setting. Eight threads averaged 0.0718 seconds/image at batch one versus 0.0830 for four, but variance at four threads was substantial. The four-thread attack default remains unchanged for controlled algorithm comparisons; this short profiler alone does not establish attack quality or stable timing gains.

### Cross-image and second-seed validation

The 55% discovery / hierarchical-pruning variant improved all five provided images at seed 454 under the same 900-query ceiling. Some trials stopped early after exhausting the hierarchy. A correctness-oriented extension revisits single-element deletions after later successful deletions, because classifier behavior is nonmonotonic: an earlier failed deletion can become valid in a changed image. A synthetic counterexample establishes this difference. Repeated passes stop on an unchanged complete pass (single-deletion minimality at fixed values only) or the normal query/time limit.

A fresh, paired seed-455 control evaluates the original GA and the repeated-pruning hybrid at four threads and a 900-query ceiling. All ten cases succeeded and passed raw export/reload checks. Every paired objective improved. These are two seeds on the supplied five images, not genuine held-out validation. Overlapping fixed-query trials are marked in metadata; wall times are not used to attribute algorithm improvements.

| Image | Original 454 | Hierarchical 454 | Original 455 | Repeated-pruning 455 |
| --- | ---: | ---: | ---: | ---: |
| knot | 627 | 185 | 761 | 109 |
| mink | 113 | 64 | 137 | 41 |
| orca | 176 | 84 | 2113 | 84 |
| sock | 39 | 26 | 178 | 49 |
| tank | 2240 | 99 | 1417 | 97 |

The verified hybrid is being integrated as the public L0 default in `epsilon_ball.py`. The original `_Search` class remains AST-identical to the preserved baseline; `CS454_L0_STYLE=original` selects the old public behavior for further controls. The hybrid is self-contained in the required file and does not depend on research modules. Dense and fog algorithms are unchanged. Integration tests and a fresh production seed-454 matrix check the public implementation; the original submission remains untouched until replacement artifacts pass independent validation.

### Rejected or unresolved follow-ups

- Adaptive dense continuation with an 80-query relaxation patience scored knot squared L2 1462.463 at 900 queries, worse than 890.127. It abandoned promising aggressive stages too early. A 256-query follow-up scored 890.127136, an effective floating-point tie, with no actual relaxation event by query 900. It does not establish a late-plateau improvement. Both traces and source versions remain in `runs/dense-cont-knot900/` and `runs/dense-cont256-knot900/`; production dense search is unchanged.
- Exact-image fitness caching correctly avoided 140 of 1039 proposals in the cached hierarchical tank trial, but the 900-real-query discovery cutoff admitted extra GA proposals and changed the parent phenotype. The discovery support was slightly smaller (2310 vs 2382), yet final support was worse (130 vs 99). This is a search-trajectory/phase-allocation effect, not evidence that cached fitness is wrong. The result and dependencies are retained in `runs/session2-cached-structured-tank454/`.
- A follow-up cache preserves the uncached discovery proposal prefix and spends saved model calls on pruning. Its synthetic test confirms identical discovery parents and fewer real queries. Actual model evaluation is underway; it has not been adopted.

### Production refinement, cache isolation and value-stage follow-up

The self-contained public integration passed its seed-454 matrix: knot185, mink63, orca82, sock23, tank99 at no more than900 queries. Repeated deletion improved three hierarchy-only outputs (64→63,84→82,26→23). The original `_Search` class and public function signatures remain AST-identical to the preserved baseline; the private refined class adds the new phases, and the original mode stays available. The same method improved all five matched seed-455 controls (109,41,84,49,97 versus761,137,2113,178,1417). The default RNG seed remains454; the second seed is validation, not an image-specific code path.

`research/prefix_cached_prune.py` fixes the cache experiment's allocation confound: with a fixed query cap, discovery stops after the same number of GA proposals (including cached duplicates) as the uncached method, then saved real forwards become extra pruning budget. A synthetic exact-parent test passed. On tank454 it scored85 at900 real model queries, versus99uncached, using111 cache hits and the same2382-element discovery parent. This supports a query-efficiency improvement under that protocol. Its uncapped wall time phase still uses elapsed time and does not guarantee the same discovery prefix, so it has not replaced production caching behavior. Full source/dependencies and diagnostics are in `runs/session2-prefix-cache-tank454/`.

`research/value_prune.py` evolves current active channel values after fixed-value deletion stalls, retaining four feasible elites by untargeted margin, using crossover and sign/uniform/Gaussian mutations, then retrying deletion. It only replaces an equal-size archive with a stronger feasible phenotype and always retains lower-cardinality results. The first experiment's mink export failed: reconstructing a scored image as `x+(image-x)` changed a floating-point endpoint by an ulp. The export validator caught this before promotion. The prototype now archives the exact clipped candidate that was evaluated, with a regression test with a nonzero original input. Corrected reruns passed: mink stayed63 after40 value mutations; sock improved23→22 after120 value evaluations and two equal-support promotions. Orca exhausted its900-query budget before this phase activated. The corrected variant remains experimental, with results in `runs/session2-value-corrected454/`; the failed first snapshot remains for diagnosis.

These deletion-minimality flags refer to complete unchanged passes under the implementation's FP32 removal operator, at the current values; they do not certify a globally minimal support. The current full contract suite records41 passing tests in `runs/session2-validation/tests.txt`. Genuine held-out data remains unavailable and unmeasured.

An actual uncapped default L0 test on tank is in progress. A larger resource budget changes the current55% discovery allocation and can select a different parent, so fixed900-query results alone do not establish default-duration scores. `research/quota_prune.py` is a separate hypothesis: cap successful discovery work at the verified900-query allocation as total budget increases, while preserving the full fallback if no adversary is found. It is not adopted without a model comparison.

### Bounded discovery and protected restarts

The first full-duration single hybrid on tank used 2325 model evaluations and returned 95 elements in 163.2 seconds (`runs/session2-default-tank454/`). Keeping successful discovery at the work allocation calibrated by the 900-query trial instead used 1119 evaluations and returned 84 elements in 83.4 seconds (`runs/session2-quota-default-tank454/`). Both had the same 165-second ceiling and seed 454, and ran without another classifier experiment. The later 1980-element discovery parent was less prunable than the earlier 2382-element parent. More discovery work and a smaller intermediate support therefore did not imply a better final result.

Production now limits successful discovery using `min(total_query_limit,900)` as the reference budget, together with the existing wall-phase guard; images without any successful ancestor retain the full original discovery fallback. For caps ≤900 this is identical to the validated earlier allocation. Two additional synthetic tests verify equal discovery parents as total budgets increase and full discovery budget retention on failure.

A further multistart policy reuses the remaining allowance after local pruning converges, changes the local RNG seed, and preserves a separate global best image across successes, incomplete restarts and failed restarts. The original image and clean class are reused; clean classification is counted once. Restarts have one virtual clean slot in their work quota to reproduce independent-run proposal prefixes. At an 1800-query/165-second ceiling the comparison was:

| Image | Single-search distance | Single actual queries | Multistart distance | Multistart actual queries |
| --- | ---: | ---: | ---: | ---: |
| knot | 136 | 1470 | 136 | 1642 |
| mink | 63 | 850 | 41 | 1613 |
| orca | 82 | 945 | 82 | 1636 |
| sock | 23 | 640 | 23 | 1800 |
| tank | 84 | 1119 | 84 | 1800 |

All ten calls passed raw round-trip checks. Four image pairs tied and mink improved 63→41. This uses additional otherwise-idle actual evaluations after convergence; it is not a per-query-efficiency gain. Several late restarts remained incomplete at the time/query ceiling; the returned global archive retained the earlier stronger image. Trials on the other four images overlapped their controls, so reported wall times are not isolated speed comparisons. Source, traces and configuration are under `runs/session2-restart-*` and `runs/session2-quota-*`. The second mink restart reproduced the independent seed 455 tensor exactly (`runs/session2-validation/restart-equivalence.json`).

The validated multistart policy is the L0 default. `CS454_L0_STYLE=refined` selects the faster single quota/pruning search, and `original` selects the preserved original GA. `research/session2_quota_baseline/` preserves the single-search production source. The default policy retains the first search's verified prefix and global best; the 46-test suite additionally covers failed-restart retention, the singleton lower bound and matching the first uncapped phase's clock. The final integrated default is undergoing an actual 165-second mink call before handoff.

### Final default-duration check and sparse-elite recombination

The integrated default completed its real uncapped mink call in **164.068 seconds**, within the three-minute limit, using 2488 model evaluations. Its three local searches produced supports of 63, 41 and 71; the returned global archive correctly retained the second search's 41-element image. The final source and full trace are preserved in `runs/session2-production-restart-default/`. Global query/evaluation totals cover all searches; phase-specific discovery fields refer to the last local search, while `chunks` records each search separately.

The unproductive third restart suggested reusing the already optimized sparse parents. `diagnose_sparse_union.py` combined the 63- and 41-element parents and pruned their union. The union initially changed 104 channel elements and had a strong adversarial margin; it pruned to 38. That initial experiment was explicitly post hoc, charged the known 1613 parent-generation evaluations plus its 134 diagnostic calls, and made no end-to-end implementation or runtime claim.

`research/recombine_prune.py` then implemented the actual online policy: retain up to four locally pruned discovery elites, recombine an untried pair by adding their perturbations and clipping, prune feasible offspring, and preserve the global best if the combination fails or is worse. Independent random discovery continues when suitable pairs are unavailable. No saved tensors are read by this attack entry point. Two synthetic tests cover feasible recombination and failed-union archive protection.

The end-to-end mink call reproduced **38 elements in 1744 evaluations and 121.658 seconds**, below the same 1800-query/165-second ceilings. The trace contains discovery at seeds 454 and 455, followed by a union/pruning stage; thus the gain is not merely a post-hoc artifact. A second online test on sock retained the stronger 23-element parent: its union offspring pruned only to 42, and the global archive correctly rejected that regression. The best saved sock result remains the separate 22-element value-refinement artifact. Both online recombination cases passed export/reload verification.

Recombination is retained as a promising research candidate rather than replacing the broadly verified production default after two activated cases. Its source, declared dependencies, traces and artifacts are in `runs/session2-recombine-mink1800/` and `runs/session2-recombine-sock1800/`.

### Immediate continuation priorities

1. Validate pruned-elite recombination on the other images, more seeds, and actual default time budgets. The strongest lesson is to compare candidate unions with the global archive: a union can improve one image and be much worse than its stronger parent on another. Test deletion order biased toward weaker-parent-exclusive genes, or crossover of selected support groups, rather than assuming that blind union pruning is optimal.
2. Extend the late dense-stagnation experiment beyond query 1203, where the long knot L2 plateau actually starts. The 256-query patience trial did not activate at 900 queries, so it did not resolve this hypothesis. Consider feasible-direction evolution ranked by verified boundary distance as a substantially different representation.
3. Refactor cache accounting into explicit proposal/work versus real-forward units before composing it with restart budgets. The negative cache trial demonstrates that changing the discovery trajectory can erase a theoretical query saving. Preserve and test a controlled prefix before attributing improvements to caching.
4. Collect genuine held-out evidence when available. All classifier trials still use the five supplied images. More seeds test search variance, not image-distribution generalization. Profile eight-thread attack throughput and score jointly before changing the four-thread default.

Reproduce a research candidate with `benchmark_attacks.py --implementation research.recombine_prune --images mink --norms l0 --queries 1800 --seconds 165 --threads 4 --seed 454 --output NEW --tag recombine1800`. Historical original-GA controls can use `CS454_L0_STYLE=original` or the frozen `research.session2_baseline.epsilon_ball` module. Keep result directories immutable, and keep best-artifact selection separate from fixed-budget algorithm comparisons.

### Verified handoff state

The final suite has 48 passing tests (`runs/session2-validation/tests.txt`). The unchanged instructor grader passed the integrated L0 interface (`runs/session2-validation/grader-smoke.txt`), and `source-audit.json` confirms the original search, public function definitions, utility/grader/README files and three instructor fog-renderer functions remain preserved. The code is self-contained in the required files; research modules are optional alternatives, not runtime dependencies of the submission implementation.

The final canonical bundle passes all 20 independent one-thread checks and includes the online recombination result of 38 for mink. Its selection table records the mixed seeds/variants and budgets, and its checksums cover all saved artifacts. This best-artifact collection should not be confused with the fixed-budget algorithm tables. Original canonical bundles are retained, and no instructor code or seed images were changed.

The next strongest investigation is broader validation of recombination of locally optimized sparse elites. The online mink result improved the stronger parent (41→38), while sock's union was worse than its stronger parent (42 versus23) and was correctly rejected. Parent-aware deletion order or partial support crossover may improve that asymmetry. Validate more images, seeds and actual time budgets before replacing the current protected-restart default.

Run all current contracts from the repository root:

```sh
TORCH_HOME=/tmp/cs454-torch /tmp/cs454-baseline-env/bin/python -m unittest -v \
  test_attacks test_fog_geometry test_structured_sparse \
  research.test_sparse_candidate research.test_cache_candidate research.test_cached_structured \
  research.test_prefix_cached_prune research.test_value_prune research.test_quota_prune \
  research.test_multistart_prune research.test_recombine_prune
```
