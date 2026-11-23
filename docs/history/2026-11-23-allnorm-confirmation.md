# Imagenette all-norm confirmation — 2026-11-23

## Authorized scope and frozen decision

The user explicitly requested using all ten reserved Imagenette confirmation images and completing generalization across Linf, L2, L0 and optional fog. This releases the prior reservation for assessing the already selected production implementation. It does not revive or promote the two rejected window variants. Both production sources byte-match `research/generalization_session_promoted/`; all five instructor seed artifacts remain unchanged.

The evaluation is a single, fixed-seed absolute-performance assessment: ten clean-correct photographs across ten classes, four attacks per photograph, seed457, four CPU threads, no query ceiling,165-second norm and270-second fog optimizer targets. The actual grading limits180/300 seconds remain authoritative. Source/protocol/runtime/weights and image hashes are frozen before any confirmation attack in `research/imagenette_allnorm_confirmation_20261123/`. Norm order rotates by fixed image index, all calls are sequential and fresh child processes reset model/search state. All cases have the same declared clean-reference warmup before timing. There is no other project model/test workload during actual-time evaluation.

## Correctness and timing

The runner measures both attack-entry time and grader-equivalent attack plus required scoring/prediction. Fog's latter clock includes the original full `genotype2phenotype` and `add_noise_to_image` render. Plotting, tensor export and independent reload are excluded from this grading clock. Optimizer-target overshoots are reported separately and do not invent a grade failure below180/300. Returned tensors, exact raw reloads, valid positive-integer/positive-finite fog genotypes and exact regeneration are checked. After all40 calls, an independent one-thread validator rechecks expected clean labels, flips, authoritative distances and fog rendering. L2 uses sum of squared normalized channel differences; L0 counts changed channel elements.

`research/run_allnorm_confirmation.py` uses an exclusive file lock, atomic per-case terminal JSON and aggregate CSV/progress files. Completed successes and failures are never repeated. Incomplete interrupted case folders are archived with explicit recovery events, then resumed. Runtime errors/timeouts stay in the40-case denominator with the grader's999999999 penalty. Ten standard-library-only tests pass, including interrupted-matrix recovery, failure retention, environment/query-cap isolation and failure-inclusive summary logic. No confirmation model query was used to build the harness. `research/validate_allnorm_confirmation.py` produces the final per-image table, per-norm success and conditional distance statistics, failure-inclusive scores and limitations.

## Live output and reproduction

Live output: `runs/imagenette-allnorm-confirmation-20261123/`; terminal result/logs in `cases/`, aggregate `results.csv`, live `progress.json`. Exposure updates conservatively when a case starts, in `research/labelled_v2_usage.json`. The bank's immutable creation-time README/protocol retain their original state; the usage ledger is authoritative for current exposure.

```sh
python3 research/restore_runtime_links.py
.runtime/python-env/bin/python research/run_allnorm_confirmation.py \
  --plan research/imagenette_allnorm_confirmation_20261123 \
  --output runs/imagenette-allnorm-confirmation-20261123
# Same command resumes missing/nonterminal cases; it does not repeat completed cases.
TORCH_HOME="$PWD/.runtime/torch-cache" .runtime/python-env/bin/python \
  research/validate_allnorm_confirmation.py \
  --run runs/imagenette-allnorm-confirmation-20261123 --output FRESH_REPORT
```

## Interpretation constraints

These images are now consumed confirmation data as evaluated and must not be reused as fresh holdout after tuning. One seed, one photo/class and ten classes give limited transfer evidence. Imagenette is ImageNet-derived and may overlap VGG's training data; no classifier-training-unseen claim is made. Initially correct classification is an eligibility condition, not a measured accuracy estimate. There is no paired alternative implementation, so scores do not establish an algorithmic improvement. Keep failures and timeouts rather than reporting only attractive successful perturbations. Any later optimization uses development data and needs a new independent bank for further confirmation.

## User-requested parallel scheduling amendment

After11 terminal cases, the user asked to proceed in parallel. Host inspection shows8physical/16logical CPUs, full0–15affinity, noCPU quota and about8GiBavailable memory. Two workers now receive disjoint physical cores: logical CPUs0,4,8,12 and2,6,10,14, four Torch/OMP/MKL threads each. Each chooses one SMT thread per physical core; the groups alternate physical indices rather than separating potential fast/slow core clusters. Memory bandwidth, CPU package power/frequency and host workloads remain shared, so resource isolation is partial. Results from the serial and parallel phases must be labeled; no equivalent standalone-throughput or comparative algorithm-improvement claim is justified.

The original protocol/source freeze is byte-preserved. A separate `execution_amendment.json` changes scheduling only, pinned with the new controller's source and tests. Neither attacks/settings nor image eligibility/seed/budgets/metrics change. Two standard-library scheduler tests pass in0.009s; they verify concurrent distinct tasks, preserved terminal results and rejection of SMT-sibling overlap. These tiny tests overlapped the last ongoing serial call; no second model/framework workload ran then.

For a safe transition, only the coordinator was paused viaSIGSTOP. Its active attack finished and saved its terminal result. After verifying that child had exited, the paused coordinator was terminated and released the file lock. The parallel controller recovered the orphaned completed record without rerunning it:11serial results preserved,29remaining. Transition evidence is in the run's `serial-coordinator-transition.json`. The old shell's143 exit is this intentional administrative stop, not an attack failure. Per-case `execution.json` and metadata environment record CPU affinity and concurrency mode for every new call. Exposure and aggregate updates remain single-writer; each worker writes only its own atomic result.

Resume the amended evaluation with:
```sh
.runtime/python-env/bin/python research/run_allnorm_parallel.py \
  --plan research/imagenette_allnorm_confirmation_20261123 \
  --output runs/imagenette-allnorm-confirmation-20261123 \
  --amendment research/imagenette_allnorm_confirmation_20261123/execution_amendment.json
```

## Completed result and independent validation

All40 terminal trials and all40 independent one-thread checks pass; the predicted class matches the original four-thread result in every case. Each attack succeeds10/10. All optimizer targets and180/300-second grading limits are met. The original instructor renderer independently regenerates each of ten saved fog genotypes. There are40 raw tensors,40 plots and10geneJSON files. No terminal attack was repeated, no failure or image was removed, and no source/setting was tuned on confirmation.

| Attack | Successes | Min distance | Median distance | Max distance | Max grading seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| linf | 10/10 | 0.00188899 | 0.0126621 | 0.16276 | 162.869 |
| l2 | 10/10 | 0.216176 | 15.2024 | 4039.34 | 163.256 |
| l0 | 10/10 | 1 | 5.5 | 177 | 164.172 |
| perlin | 10/10 | 0.0279209 | 0.168201 | 0.628906 | 228.110 |

The large French-horn squared-L2 value4039.34 and L0 count177 remain visible in the individual table and maxima; success does not imply minimal or uniformly small perturbations. Two L0cases achieve the minimum positive count1. These are absolute transfer measurements for one frozen source on one seed/photo perclass. They are not an improvement comparison or a universal generalization guarantee. The confirmation bank is now fully consumed; future tuning on these results requires a new independent bank for further validation.

The final combined suite passes93unique tests. The successful rerun executes96tests because one three-test module was accidentally listed twice; no attack was repeated. An initial test assumed an otherwise empty process import registry and failed when the earlier suite legitimately loadedTorch/util. It now checks lazy imports in a clean interpreter. The evaluation worker, validator and production sources remain byte-identical to their freezes; only the test was corrected. The initial failed test log is retained. The supplemental report initially included an uninterpretable mixed-unit cross-norm score mean; v2 removes it (null for mixed groups) and retains meaningful per-norm means. The initial report is explicitly marked superseded. This correction changes no measured result, verification or source.

Authoritative outputs:
- `runs/imagenette-allnorm-confirmation-validation-20261123/report/summary.json` and `report/verification.csv`: original independent validator.
- `runs/imagenette-allnorm-confirmation-validation-20261123/execution-report-v2/execution_results.md` and `image_norm_wide.csv`: all10×4outcomes withS/Pmode,CPUaffinity,times,queries.
- `runs/imagenette-allnorm-confirmation-validation-20261123/all-tests.txt`: successful final combined tests.
- `runs/imagenette-allnorm-confirmation-validation-20261123/source-audit.json`: frozen source/renderer/instructor invariants, counts and analysis-tool hashes.
- Per-run `SHA256SUMS`: raw tensors, genes, logs, metadata and reports; canonical47-file and original-bank manifests also pass unchanged.

All requested evaluation, validation and record updates are complete; no jobs, promotion or approval remain pending. Preserve the canonical submission and production. Continue any optimization on development data with fresh timing controls, and distinguish expensive inputs, more actual work and resource-mode differences from an algorithmic improvement.
