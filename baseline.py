"""Reproduce the clean-image controls and check how the grader measures distance."""

import csv
import hashlib
import sys
from pathlib import Path

import torch

from util import (
    IMAGES_ORG_PREDS,
    get_image_tensor_from,
    get_L0_norm_for,
    get_L2_norm_for,
    get_Linf_norm_for,
    get_model_most_likely_for,
)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    # Two channels at one spatial location distinguish element count from pixel count.
    delta = torch.zeros(1, 3, 1, 1)
    delta[0, :2, 0, 0] = torch.tensor([-0.5, 0.25])
    metrics = get_Linf_norm_for, get_L2_norm_for, get_L0_norm_for
    measured = tuple(metric(delta) for metric in metrics)
    if measured != (0.5, 0.3125, 2):
        raise ValueError(f"Unexpected distance convention: {measured}")

    rows = []
    for path, reference in IMAGES_ORG_PREDS.items():
        image = get_image_tensor_from(path)
        probability, predicted = get_model_most_likely_for(image)
        if predicted != reference:
            raise ValueError(f"{path}: expected class {reference}, got {predicted}")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rows.append((path, digest, reference, predicted, probability))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("image", "sha256", "reference_class", "predicted_class", "probability"))
    writer.writerows(rows)


if __name__ == "__main__":
    main()
