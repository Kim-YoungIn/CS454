# Generalization research — 2026-11-21, continuation

Current state belongs in `PROGRESS.md`; this log retains experimental evidence.

## Scope and preservation

Production and the 20 verified submission artifacts remain preserved at session start. Primary task: improve transfer to new image inputs under the same local VGG-11 and authoritative metrics, rather than optimize only the supplied five. The instructor sources/renderers are unchanged. No gradients or image-specific label/routing tables are used.

## Frozen external-image bank

`research/generalization_v1/` contains eight official scikit-image v0.25.2 fixtures, source URLs, official registry checksums, original bytes and uniformly prepared RGB PNGs. The four/four split, methods and budgets were fixed before any clean predictions or attacks. Development: astronaut/coffee/brick/rocket, seed 454. Confirmation: camera/chelsea/grass/motorcycle_left, seed 455. No confidence-based filtering or replacement of difficult cases. Source freeze and manifest hashes preserve the exact inputs and methods. Details/attribution/scope: the dataset README.

The attack reference is clean classifier argmax; semantic correctness is not independently annotated. This small bank is held out from our attack development, not asserted excluded from VGG's pretraining or representative of ImageNet. Confirmation is a one-time check after model selection. Later tuning must treat it as historical evidence and acquire a new reserved bank.

The first paired screen uses 900-query / 165-second ceilings, four threads, alternating method order by image, preserved production versus the frozen radial-boundary candidate. Budget failures remain in the matrix. `benchmark_generalization.py` wraps the existing independent raw-tensor export/reload checks and records all configurations. `compare_generalization.py` reports success first, common-success distance ratios, worst regression and phase activation. Equal ceilings do not imply equal actual work; single-seed pairs are descriptive rather than statistical evidence of population performance.

## Development failure and architectural revision

The frozen radial-boundary candidate regresses on astronaut: production squared L2 **2.713446** versus candidate **7.184147**, both exactly 900 calls. Its fine phase begins after only three successes at query **9**; production continues making large radius contractions and reaches **2.720083 at query 833**. Thus unconditional switching after three successes is not generally reliable. Coffee improves **383.721069→305.327454**; brick has a smaller regression **150.847519→154.180222**. No confirmation outcomes have been inspected.

`research/dense_exploration_boundary.py` keeps the global energy GA for at least **256 proposal generations**, while retaining the three-success prerequisite and protected feasible archive. Slow discoveries whose third success occurs later retain the earlier candidate's switch. Easy early discoveries continue exploring global directions before local angular refinement. This uses observed search work, not filename, clean class or a per-image parameter. The 256 setting is a single development hypothesis, not a proven optimum. Zero warmup replays the old local candidate; an unreachable warmup replays production.

The revised source/test and selected development protocol are frozen in `research/generalization_v1_adaptive/`; image paths reference the same bank. This is a development revision using the same still-unexamined confirmation images, not a second independent image set. Existing candidate and original protocol remain intact. Run the new proposal-prefix contracts and matched development comparisons before selecting a confirmation method.

## Independent L0 hypothesis

An independent review produced `research/local_support_tail.py`, frozen before real-image attacks. It runs the complete production policy first, then uses otherwise idle allowance to evolve a single parent's support plus valid eight-neighbor alleles. It protects the known feasible image and transports actual channel endpoints without row/channel wrapping. Three synthetic tests pass, including the complete production input prefix and exact work accounting. It cannot help hard discovery that already exhausts the budget. Design, source hashes and limitations: `research/generalization_l0.md`, `research/generalization_l0_frozen.json`. The predefined secondary comparison uses the same development/confirmation split and 900-query ceilings. No production promotion is justified yet.

## Completed initial development screen and exploration256

All four methods/cases retain success. Original radial candidate: **1 win /3 losses**, geometric-mean ratio **1.399578**, worst **2.647610**. Rocket also regresses **6.631104→11.816244**. Sources: `runs/generalization-v1-l2-development/` and its summary directory. The original positive supplied-image evidence did not generalize to this development bank, so no default promotion occurs.

Exploration256 changes the paired ratios to astronaut **1.116553**, coffee **0.795702**, brick **1.022093**, rocket **0.637070**: **2 wins /2 losses**, geometric mean **0.872121**, worst **1.116553**. It reduces the large early-switching failures without removing all regressions. Sources: `runs/generalization-v1-l2-adaptive-development/`. Controls are reused from the identical fixed-query source/seed/budget; current-window speed is not inferred from that reuse.

The second and final coarse work scale512 is frozen in `research/generalization_v1_warmup512/`. Independent development-only review found trace support: major control contractions remain at astronautq488, rocketq513, brickq442 and coffeeq515. Waiting512 proposal generations should preserve those steps, while potentially displacing productive local work on coffee. This is a planned coarse-scale comparison, not a fine threshold sweep. Review: `research/generalization_dense_review.md`. No confirmation results have been inspected.

`research/epsilon_generalized_candidate.py` prepares a self-contained port for possible confirmation, with default exploration512 and `CS454_L2_SEARCH=original` for the old route. Production remains unchanged. Source/core and bitwise proposal tests must pass before it can replace the research-only candidate in confirmation. A preliminary port missing a standard-library import was caught by synthetic tests before any real-image run; it was corrected and the exploration contracts passed.

## Development selection frozen

Exploration512 improves **all4/4** development pairs, preserving success4/4: astronaut2.713446→2.393057, coffee383.721069→292.018616, brick150.847519→140.716187, rocket6.631104→3.845240. Geometric-mean squared-L2 ratio **0.776234**, worst **0.932837**. No finer threshold search is performed. Exact results/summary: `runs/generalization-v1-l2-warmup512-development/` and the matching `-summary/` directory.

The self-contained port passes bitwise input, final tensor, history and probe-trace comparison at900 synthetic calls. Its original `_Search`, L0/Linf classes/wrappers and old-L2 route are preserved. All7 focused tests pass (`runs/generalization-selected-contracts.txt`). Independent static review also found equivalent dense/boundary/rescale ASTs and preserved other-norm routing.

Selected protocol/source/settings are now frozen in `research/generalization_v1_selected/`. Confirmation is four reserved images at seed455/900 calls; `research/generalization_v1_selected_wall/` adds the predeclared first/last confirmation images camera/motorcycle_left at165 seconds with no query cap. Both blocks use the same frozen method, with no tuning between outcomes. Clean references for all8 images are recorded after selection, without filtering (`research/generalization_v1/clean.csv`). Source guards in the selected protocol are scoped by metric to actual dependencies; original snapshots remain recorded.

The explicit promotion criteria are in `research/generalization_v1_selected/promotion_rule.json`, fixed before inspecting confirmation attack results. They require success preservation, a beneficial confirmation aggregate with bounded worst-case loss, both predeclared full-time pairs within1% of control or better, and contracts/source audits. Failure leaves production unchanged; confirmation is not reused for parameter tuning.

## L0 development result and budget limitation

The neighbor tail succeeds on all four development cases with **1 win /3 ties**: coffee **39→33**, using900 calls versus control672. Astronaut and rocket both reach the global L0 minimum1 at498 calls, so the tail does not run; brick3→3 exhausts900 with no tail. Only coffee activates the new representation. Geometric-mean ratio0.959097 includes those inactive ties; they are not evidence of useful replacement search. Source: `runs/generalization-v1-l0-development/` and summary.

This is an equal-ceiling improvement using228 otherwise idle calls, not equal actual work. At a full165-second budget, production may instead finish a second parent and spend all remaining time in its existing support GA, leaving no tail allowance. Therefore this small-budget gain does not establish grading-time gains and does not justify default promotion. A future active stage allocation would need matched full-time evidence; the frozen conservative suffix remains available as a research candidate.

## Confirmation execution and independent verification plan

The selected self-contained port and its512 setting remain unchanged throughout the reserved-image block. First completed pairs: camera **573.698242→367.497925**, chelsea **39.563095→36.323112**, both900 calls, seed455, exact four-thread raw round trips. These are partial results, not the final confirmation aggregate. No settings are changed after inspection.

`validate_generalization.py` will separately load successful saved tensors and the original input at one CPU thread, recompute authoritative metrics, verify the original reference and untargeted prediction change, and retain original unsuccessful trials explicitly. L0 must match exactly; squared-L2 reductions permit1e-6 relative numeric tolerance across thread counts. It does not treat rival-class changes as failure if the reference class remains displaced. Run only after the real-model benchmark queue completes.

## Complete reserved900-query confirmation

The frozen self-contained exploration512 method improves **all4/4** confirmation pairs at seed455, with success4/4 and no success regressions. Geometric-mean squared-L2 ratio **0.788251**, worst **0.918106**. Distances: camera573.698242→367.497925, chelsea39.563095→36.323112, grass4.250130→3.084477, motorcycle_left25.308018→22.891394. Both methods use exactly900 calls for every case; all four-thread raw round trips pass. Sources: `runs/generalization-v1-l2-confirmation/`, corresponding summary/pairs.

This confirmation uses different images and seed455 after development selection at seed454. It supports transfer on this small bank, not population-wide or pretrained-training-unseen claims. The bank is now consumed as confirmation. No algorithm/settings changes occur after inspecting it. The predeclared full165-second camera/motorcycle block is running with the same frozen method; promotion still depends on those results and independent one-thread validation.

## Predeclared actual165-second confirmation

| Image | Control squared L2 | Selected port squared L2 | Control calls/time | Candidate calls/time |
| --- | ---: | ---: | --- | --- |
| camera | 158.252380 | **99.668648** | 2196 /162.832s | 2299 /162.977s |
| motorcycle_left | 16.205042 | **9.242699** | 2401 /162.778s | 2243 /162.877s |

Both calls per pair share165-second ceilings and no query cap; all succeed with exact four-thread raw round trips. Camera uses more actual candidate work, motorcycle less; this is grading-time performance evidence, not a claim of uniformly better per-query efficiency. Together with four confirmation900-query wins, both full-time pairs satisfy the pre-inspection promotion criteria. The algorithms/settings remained fixed through both blocks. Independent one-thread artifact validation and final-source contracts remain the last gates before promotion.

## Next labelled bank provenance

The original eight-fixture bank includes low-confidence/semantically misaligned clean predictions (astronaut bobsled0.147, brick oscilloscope0.199, grass croquet ball0.040). Its original scope explicitly uses clean argmax and does not claim semantic accuracy or ImageNet representativeness. This reinforces the need for labelled, clean-correct next-session inputs rather than a broader claim from eight fixtures.

A second independent review verified the official Imagenette2-160 archive and ImageNet1000 label mapping. Only the99,003,388-byte archive was downloaded/hash-checked during the CPU benchmark; no listing/decompression/model work overlapped the actual-time windows. MD5 matches the official torchvision loader; SHA256 and primary provenance are in `research/generalization_next_bank.md`. After benchmarks, `research/prepare_labelled_generalization.py` fixes deterministic class/role pools before any clean forwards, chooses the first clean-correct example per class/role with a finite10-candidate limit, and retains all exclusions/duplicates/missing classes. This eligibility condition mirrors the assignment's initially correct seeds; selected100% clean accuracy is not a classifier accuracy estimate. Ten classes and160px source resolution still limit coverage, and no VGG-pretraining-unseen claim is made. No attacks will be run on the new confirmation bank this session.

## Final integration, integrity and continuation

The exact self-contained port tested in confirmation is now the L2 default in `epsilon_ball.py`; `CS454_L2_SEARCH=original` retains the old global-only GA. L0/Linf classes and public wrappers remain AST-identical. The original `_Search` remains intact, as do instructor bytes/renderers/interfaces. Production source is frozen in `research/generalization_session_promoted/`; the preceding production remains in `research/session3_promoted/`. Final source hash/audit: `runs/generalization-final-validation/source-audit.json`.

All **72 tests pass in 19.385 seconds** after integration. **28/28** selected generalization tensors pass independent one-thread authoritative-metric/reference/prediction checks; no original unsuccessful rows were discarded. The existing **20/20** assignment artifacts also pass fresh one-thread verification and their prior47-entry checksums remain unchanged. No canonical artifact was replaced by external-bank output. Validation files: `runs/generalization-final-validation/`. The deduplicated trial index contains **36 fresh real-model trials**; reused fixed-query controls retain their original configuration pointers and comparison membership instead of being counted as fresh measurements.

The labelled Imagenette bank is now prepared under `research/generalization_imagenette_v2/`: **20 clean-correct images, ten development and ten confirmation, ten semantic classes**. Fixed pools had200 candidates;28 were scanned before filling all roles/classes, with8 clean-label mismatches recorded. All20 selected original-byte hashes are distinct. Eligibility selection is a condition rather than a measured classifier accuracy; original ImageNet/VGG-training overlap is not ruled out. No adversarial attacks have been run on this bank. The current eight-fixture bank is historical confirmation, while the new bank's confirmation attacks remain untouched. See the bank README, data protocol, environment/checkpoint hash, candidate/eligibility tables, selector snapshot and checksums.

An initial selector import-path error occurred after fixed-pool extraction and before classifier import/forward. Its partial output remains at `research/generalization_imagenette_v2_initial_import_failure/`. Adding the project import path fixed it; the retried `pools_before_clean.csv` is byte-identical, so no image-selection decision changed. Source/data preparation and clean eligibility were executed after all timed attack benchmarks ended. No dataset installation or classifier/utility modification was needed.

No model jobs remain pending. Next session should use the labelled development bank to assess the promoted method under actual grading-time budgets and investigate reversible global/local allocation only if evidence supports it. Freeze selection before using its ten confirmation images. L0 neighbor evolution remains research-only; stronger reduced-cap results do not establish full-time gains. Existing assignment artifacts remain available for the report and future offline best selection.
