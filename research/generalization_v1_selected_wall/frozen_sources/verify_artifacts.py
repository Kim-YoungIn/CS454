"""Check fixed CS454 submission files against the supplied model and metrics."""

import argparse
import csv
import json
import math
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


METRICS = {"linf": get_Linf_norm_for, "l2": get_L2_norm_for, "l0": get_L0_norm_for}


def read_tensor(path, original):
    with path.open("rb") as stream:
        try:
            tensor = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"{path.name}: unable to load a PyTorch tensor") from error
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.dtype != original.dtype
        or tensor.shape != original.shape
    ):
        raise ValueError(f"{path.name}: expected a dense {original.dtype} tensor of shape {tuple(original.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path.name}: tensor contains nonfinite values")
    return tensor


def check_fog(path, original, saved):
    from perlin_fog import add_noise_to_image, genotype2phenotype

    genes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(genes, list) or len(genes) > 50:
        raise ValueError(f"{path.name}: expected a JSON list of at most 50 pairs")
    for gene in genes:
        if not isinstance(gene, list) or len(gene) != 2:
            raise ValueError(f"{path.name}: each pair must contain an octave and a magnitude")
        octave, magnitude = gene
        if type(octave) is not int or octave <= 0:
            raise ValueError(f"{path.name}: octaves must be positive integers")
        if type(magnitude) not in (int, float) or not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(f"{path.name}: magnitudes must be positive finite numbers")
    distance = sum(magnitude for _, magnitude in genes)
    if not math.isfinite(distance):
        raise ValueError(f"{path.name}: total magnitude must be finite")
    rendered = add_noise_to_image(genotype2phenotype(genes), original)
    if not torch.allclose(saved, rendered.clip(-3, 3), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{path.name}: regenerated image differs from the saved tensor")
    return rendered, distance


def check_case(directory, stem, metric, original, reference):
    raw_path = directory / f"raw_{stem}_{metric}.png"
    try:
        image = read_tensor(raw_path, original)
        if metric == "perlin":
            image, distance = check_fog(directory / f"{stem}_perlin.json", original, image)
        else:
            distance = METRICS[metric](image - original)
            if not math.isfinite(distance):
                raise ValueError(f"{raw_path.name}: distance is nonfinite")
    except FileNotFoundError as error:
        return "", "", "missing", Path(error.filename).name
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        return "", "", "invalid", str(error)

    _, predicted = get_model_most_likely_for(image)
    status = "pass" if predicted != reference else "class_unchanged"
    return distance, predicted, status, ""


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the submission files")
    parser.add_argument("--perlin", action="store_true", help="Include all five optional Perlin cases")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    torch.set_num_threads(1)
    metrics = (*METRICS, "perlin") if args.perlin else METRICS
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("image", "metric", "distance", "reference_class", "predicted_class", "status", "detail"))
    all_passed = True
    for path, reference in IMAGES_ORG_PREDS.items():
        original = get_image_tensor_from(path)
        stem = Path(path).stem
        for metric in metrics:
            distance, predicted, status, detail = check_case(args.directory, stem, metric, original, reference)
            writer.writerow((stem, metric, distance, reference, predicted, status, detail))
            all_passed &= status == "pass"
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
