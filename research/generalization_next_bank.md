> Current status: preparation is now complete in `research/generalization_imagenette_v2/` (10 development + 10 confirmation images, no attacks). The download-stage statements and proposed protocol below are historical; the final data protocol, exclusions and selector snapshot are in that bank.

# Next labelled generalization bank — Imagenette160

## Verified primary sources and download

Use **Imagenette2, 160px**, an ImageNet subset with ten semantic labels. Fastai states that the December2019 revision uses a70/30 train/validation split and resizes the shorter image side to160 while preserving aspect ratio. Its splits are Imagenette splits; they do **not** establish that a pretrained VGG has never seen an image. [Official fastai repository](https://github.com/fastai/imagenette#the-datasets).

Pinned archive URL: `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz`. The [official torchvision v0.24.1 loader](https://raw.githubusercontent.com/pytorch/vision/v0.24.1/torchvision/datasets/imagenette.py) specifies this URL, its MD5 and the WNID/name mapping below. The versioned source is used for provenance only; do not change the assignment's installed libraries or preprocessing.

Downloaded on2026-11-21 to **`/tmp/cs454-imagenette2-160.tgz`**:

- Exact bytes: **99,003,388** (94.42MiB; below the authorized200MB limit).
- MD5: **`e793b78cc4c9e9a4ccc0c1155377a412`**, matches the official loader.
- SHA256: **`64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`**.
- HTTP HEAD: `Content-Length:99003388`, `Last-Modified: Sat, 06 Feb 2021 19:53:36 GMT`; the multipart ETag is not treated as a content MD5.

**Only download and byte hashing were performed. No archive listing, decompression, image reads or model imports/runs occurred.** No current dataset, confirmation outputs or production files were changed. Extraction is deferred until the parent agent ends the active CPU benchmark queue and authorizes it. The temporary archive can be reacquired from the pinned URL and verified against both hashes.

## Labels and the1000-class VGG indices

Imagenette's own indices0–9 are **not** the VGG's1000-class indices. The WNID/name pairs come from the official torchvision loader above. The1000-class indices below were cross-checked against the [TensorFlow class-index JSON](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json), which [official Keras prediction decoding references](https://raw.githubusercontent.com/keras-team/keras/master/keras/src/applications/imagenet_utils.py), and the corresponding lines in the assignment's `category_names.txt`.

| WNID | Semantic label | VGG/ImageNet1000 index |
| --- | --- | ---: |
| n01440764 | tench | 0 |
| n02102040 | English springer | 217 |
| n02979186 | cassette player | 482 |
| n03000684 | chain saw | 491 |
| n03028079 | church | 497 |
| n03394916 | French horn | 566 |
| n03417042 | garbage truck | 569 |
| n03425413 | gas pump | 571 |
| n03445777 | golf ball | 574 |
| n03888257 | parachute | 701 |

The class-index JSON was downloaded separately to `/tmp/cs454-imagenet-class-index.json`:35,363bytes, MD5 `c2c37ea517e94d9795004a39431a14cb` (matches Keras's published file hash), SHA256 `a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc`. It is metadata, not an archive listing. No framework was imported to verify these mappings.

## Proposed next-session protocol, before inspecting attack results

1. After the CPU queue finishes, extract safely to a new data directory and record the archive checksum. Use original WNID folder labels, never the optional synthetic noisy-label columns. Do not modify the current scikit-image bank or its results.
2. Preregister a deterministic class-stratified ordering of Imagenette `val` members using SHA256 of a fixed protocol seed plus relative path. Partition ordered candidates into disjoint development/confirmation pools **before** clean eligibility checks. This is a new attack-algorithm split, not a claim about the VGG training split.
3. Choose the first clean-correct image from each pool/class using the unchanged instructor loader and classifier. Clean-correct means argmax equals the official1000-class label. Prespecify a finite scan limit; retain all exclusions and failure-to-fill classes. Do not select by attack difficulty or tune a confidence cutoff after seeing results. Report clean confidence and correct/incorrect counts so the bank's selection is transparent.
4. An initial manageable bank is **ten development plus ten confirmation images**, one per class in each role. Freeze its manifest with archive/image hashes, source split, relative path, WNID, official label index, deterministic rank, clean result and algorithm role. Sampling more images per class later is a new named protocol, not retroactive replacement of hard cases.
5. Optimize on development only at matched query/time ceilings. Rank success first and show per-image distances, paired ratios, worst regression, actual queries and seconds; a raw-distance mean alone can hide regressions on easier images. Freeze code/configuration and selection before one confirmation pass. Keep confirmation attack results out of tuning.
6. Interpret this as **transfer to labelled ImageNet-style examples not used in attack development**. Imagenette derives from ImageNet and the fastai split may overlap VGG pretraining material; do not call it pretrained-unseen or a large-population generalization result. The160px version is upsampled by the authoritative224px pipeline, so preserve that preprocessing and report the source resolution. The ten relatively easy classes give better semantic alignment than the current OOD fixtures but still limited class coverage.

This is a prepared next-session plan and verified download, not a new measured attack result or an already frozen image selection. No current confirmation data was inspected.
