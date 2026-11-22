# Fixed external-image bank for attack generalization

Eight public scikit-image v0.25.2 fixtures were selected and split **before clean predictions or attacks**. `protocol.json` fixes the split, compared algorithms, query/time ceilings and random seeds. `source_freeze.json` pins the algorithms and dependencies. `manifest.csv` pins upstream and prepared image SHA256 hashes. `originals/` retains upstream bytes; `images/` converts each decoded image to RGB PNG without cropping, augmentation or resizing. The unchanged instructor preprocessing performs its usual resize/normalization.

- Development: astronaut, coffee, brick, rocket; seed 454.
- Confirmation: camera, chelsea, grass, motorcycle_left; seed 455.

The source is the [official scikit-image data collection](https://scikit-image.org/docs/stable/api/skimage.data.html). Download URLs use the fixed v0.25.2 tag; every original is checked against its official registry SHA256 (`upstream_registry.py`, parsed as literals rather than executed). Photographer/source attribution is available in those official per-image entries. The bank intentionally includes people, objects and textures. It is small, not a representative ImageNet validation set.

Attack references are the local VGG-11 clean argmax. These experiments measure prediction changes relative to that reference, not independently verified semantic classification accuracy. Images are held out from this project's attack development; no assertion is made that the pretrained classifier never encountered them. Repeated model queries during each attack are part of the authorized black-box optimization, not training on the confirmation bank. Once confirmation results are inspected, those images are historical confirmation and cannot be reused as an untouched test set for later tuning.

`research/prepare_generalization.py` validates the bank offline; `--download` retrieves missing originals if networking is available. `benchmark_generalization.py` runs each pair sequentially, alternating method order by image. Expected budget-limited attack failures remain in results. `compare_generalization.py` reports all successes/failures first, paired ratios on common successes, the worst ratio and activation; it rejects incomplete pairs. Equal ceilings do not imply equal actual work. All outputs must use fresh directories.
