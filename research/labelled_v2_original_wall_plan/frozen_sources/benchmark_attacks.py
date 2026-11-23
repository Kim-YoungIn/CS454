"""Run and export norm attacks using the instructor's metrics and visualization writer.

Run from the repository root, for example:
  python benchmark_attacks.py --images sock knot --norms linf --queries 300 \
      --seconds 165 --output runs/linf-smoke --tag baseline
Each completed case is immediately appended to results.csv so interrupted runs
retain their measurements. Use a separate output directory for each experiment
to preserve earlier artifacts.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import time
from contextlib import chdir
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

import epsilon_ball
from util import (
    IMAGES_ORG_PREDS,
    get_image_tensor_from,
    get_L0_norm_for,
    get_L2_norm_for,
    get_Linf_norm_for,
    get_model_most_likely_for,
    get_viz_for,
)
from verify_artifacts import read_tensor


METRICS = {"linf": get_Linf_norm_for, "l2": get_L2_norm_for, "l0": get_L0_norm_for}
ATTACKS = {
    "linf": epsilon_ball.make_Linf_adversarial_from,
    "l2": epsilon_ball.make_L2_adversarial_from,
    "l0": epsilon_ball.make_L0_adversarial_from,
}
COLUMNS = (
    "tag", "image", "norm", "seed", "seconds_limit", "queries_limit", "threads",
    "runtime_seconds", "distance", "reference_class", "predicted_class",
    "predicted_probability", "success", "status", "queries", "engine_elapsed",
    "round_trip", "error", "engine_metadata",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_images(names):
    known = {Path(path).stem: Path(path).resolve() for path in IMAGES_ORG_PREDS}
    paths = []
    for name in names:
        path = known.get(name, Path(name).resolve())
        if not path.is_file():
            raise ValueError(f"image does not exist: {name}")
        if path not in paths:
            paths.append(path)
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("images must have distinct stems to avoid artifact filename collisions")
    return paths


def reference_for(path, original):
    _, clean_class = get_model_most_likely_for(original)
    references = {Path(name).resolve(): index for name, index in IMAGES_ORG_PREDS.items()}
    reference = references.get(path, clean_class)
    if clean_class != reference:
        raise ValueError(f"{path}: clean prediction {clean_class} differs from reference {reference}")
    return reference


def export_and_verify(image, original, reference, metric, output, filename):
    """Independently reload the serialized tensor and compare authoritative results."""
    distance = metric(image - original)
    probability, predicted = get_model_most_likely_for(image)
    if not math.isfinite(distance):
        raise ValueError("attack produced a nonfinite distance")
    # The supplied writer clips and then unnormalizes its argument in place.
    with chdir(output):
        try:
            get_viz_for(image.clone(), save_to=filename)
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")
    saved = read_tensor(output / f"raw_{filename}", original)
    if not torch.equal(saved, image):
        raise ValueError("export changed the attack tensor (the writer clips to [-3, 3])")
    saved_distance = metric(saved - original)
    saved_probability, saved_class = get_model_most_likely_for(saved)
    if saved_distance != distance or (saved_probability, saved_class) != (probability, predicted):
        raise ValueError("distance or prediction changed after independent tensor reload")
    return distance, probability, predicted, predicted != reference


def make_metadata(args, images, output, module):
    source_names = ("benchmark_attacks.py", "epsilon_ball.py", "util.py", "grade.py", "verify_artifacts.py")
    source = Path("epsilon_ball.py").read_bytes()
    snapshot = output / f"epsilon_ball_{args.tag}.py"
    snapshot.write_bytes(source)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "options": {**vars(args), "output": str(output)},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "matplotlib")},
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": {name: file_hash(name) for name in source_names},
        "source_snapshot": {"epsilon_ball.py": snapshot.name},
        "images_sha256": {str(path): file_hash(path) for path in images},
        "environment": {name: value for name, value in os.environ.items() if name.startswith("CS454_") or name in ("TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "metric_semantics": {"linf": "max absolute normalized channel difference", "l2": "sum of squared normalized channel differences", "l0": "number of changed channel elements"},
        "timing": "entry point call only; excludes model import, clean reference check, visualization and independent reload verification",
    }
    metadata["source_sha256"]["epsilon_ball.py"] = hashlib.sha256(source).hexdigest()
    metadata["implementation"] = module.__name__
    if module is not epsilon_ball:
        candidate_source = Path(module.__file__).read_bytes()
        candidate_snapshot = output / f"candidate_{args.tag}.py"
        candidate_snapshot.write_bytes(candidate_source)
        metadata["source_snapshot"][module.__name__] = candidate_snapshot.name
        metadata["source_sha256"][module.__name__] = hashlib.sha256(candidate_source).hexdigest()
    for dependency in getattr(module, "SOURCE_DEPENDENCIES", ()):
        dependency_path = Path(dependency)
        dependency_source = dependency_path.read_bytes()
        dependency_snapshot = output / f"dependency_{args.tag}_{dependency_path.name}"
        dependency_snapshot.write_bytes(dependency_source)
        metadata["source_snapshot"][str(dependency_path)] = dependency_snapshot.name
        metadata["source_sha256"][str(dependency_path)] = hashlib.sha256(dependency_source).hexdigest()
    (output / f"config_{args.tag}.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def run(args):
    images = resolve_images(args.images)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.norms = list(dict.fromkeys(args.norms))
    if (output / f"config_{args.tag}.json").exists() or (output / f"epsilon_ball_{args.tag}.py").exists():
        raise ValueError(f"tag {args.tag!r} already exists in {output}; use a fresh output directory or tag")
    for path in images:
        for norm in args.norms:
            for filename in (f"{path.stem}_{norm}.png", f"raw_{path.stem}_{norm}.png"):
                if (output / filename).exists():
                    raise ValueError(f"artifact already exists: {output / filename}; use a fresh output directory")
    torch.set_num_threads(args.threads)
    settings = {"CS454_SEED": args.seed, "CS454_SECONDS": args.seconds, "CS454_THREADS": args.threads}
    if args.queries is not None:
        settings["CS454_QUERIES"] = args.queries
    for key, value in settings.items():
        os.environ[key] = str(value)
    if args.queries is None:
        os.environ.pop("CS454_QUERIES", None)
    os.environ.setdefault("MPLBACKEND", "Agg")
    module = importlib.import_module(args.implementation)
    names = {"linf": "make_Linf_adversarial_from", "l2": "make_L2_adversarial_from", "l0": "make_L0_adversarial_from"}
    attacks = {}
    for norm in args.norms:
        function = getattr(module, names[norm], None)
        if not callable(function):
            raise ValueError(f"{args.implementation} does not implement {names[norm]}")
        attacks[norm] = function
    make_metadata(args, images, output, module)
    results_path = output / "results.csv"
    if results_path.exists() and results_path.stat().st_size:
        with results_path.open(newline="") as stream:
            if next(csv.reader(stream)) != list(COLUMNS):
                raise ValueError(f"incompatible CSV columns: {results_path}")
    write_header = not results_path.exists() or not results_path.stat().st_size
    all_passed = True
    with results_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
            stream.flush()
        for path in images:
            original = get_image_tensor_from(str(path))
            reference = reference_for(path, original)
            for norm in args.norms:
                row = dict.fromkeys(COLUMNS, "")
                row.update(tag=args.tag, image=str(path), norm=norm, seed=args.seed,
                           seconds_limit=args.seconds, queries_limit=args.queries or "",
                           threads=args.threads, reference_class=reference,
                           success=False, round_trip=False)
                # Reinitialize for each case; case order must not change randomness.
                torch.manual_seed(args.seed)
                module.LAST_RUN = {}
                started = time.perf_counter()
                try:
                    image = attacks[norm](str(path))
                    row["runtime_seconds"] = time.perf_counter() - started
                    if not isinstance(image, torch.Tensor) or image.shape != original.shape or image.dtype != original.dtype:
                        raise ValueError("attack returned an incompatible image tensor")
                    if not torch.isfinite(image).all():
                        raise ValueError("attack returned a nonfinite image tensor")
                    distance, probability, predicted, success = export_and_verify(
                        image, original, reference, METRICS[norm], output, f"{path.stem}_{norm}.png")
                    row.update(distance=distance, predicted_class=predicted,
                               predicted_probability=probability, success=success,
                               round_trip=True, status="pass" if success else "class_unchanged")
                except Exception as error:
                    if row["runtime_seconds"] == "":
                        row["runtime_seconds"] = time.perf_counter() - started
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                metadata = dict(getattr(module, "LAST_RUN", {}))
                row.update(queries=metadata.get("queries", ""),
                           engine_elapsed=metadata.get("elapsed", metadata.get("seconds", "")),
                           engine_metadata=json.dumps(metadata, sort_keys=True))
                writer.writerow(row)
                stream.flush()
                all_passed &= row["success"]
                print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--implementation", default="epsilon_ball", help="Python module containing required attack entry points")
    parser.add_argument("--images", nargs="+", default=list(Path(path).stem for path in IMAGES_ORG_PREDS), help="seed stems or image paths")
    parser.add_argument("--norms", nargs="+", choices=tuple(METRICS), default=list(METRICS))
    parser.add_argument("--seed", type=int, default=454)
    parser.add_argument("--seconds", type=float, default=165, help="entry point wall-clock budget")
    parser.add_argument("--queries", type=int, help="optional model-query cap for paired experiments")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="run", help="experiment name used in CSV and config filename")
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.seconds) or args.threads <= 0 or (args.queries is not None and args.queries <= 0):
        parser.error("seconds, threads, and supplied query cap must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscore, dot, or hyphen")
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
