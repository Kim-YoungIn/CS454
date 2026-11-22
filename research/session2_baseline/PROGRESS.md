# Current state — 2026-11-20

All three norm attacks and the optional fog GA are implemented. `submission/` contains the strongest verified tensors, classification plots, five fog genotypes, selection provenance, and checksums. All 20 cases passed independent saved-artifact validation; all 16 contract tests and all four public-interface grader smoke cases passed. Fixed 900-query searches improved all 15 norm scores over the earlier 450-query runs. One full default-duration L2 search improved knot further, to 386.170, in 162.8 seconds. Held-out performance remains unmeasured. Historical records and continuation priorities follow; the latest exact distances are in `submission/verification.csv`.

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

The initial inference microbenchmark is preserved in `runs/preliminary/inference_profile.csv`: knot input repeated to batches 1/4/8, threads 1/2/4, inference mode, one warm-up followed by the mean of three forwards per setting. Four threads was fastest among those tested (0.0675 seconds for one image versus 0.1425 at one thread). Small batches improve throughput but delay sequential genetic selection; norm search uses single offspring, while fog uses batches of four. Eight and sixteen CPU threads remain unprofiled.

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
