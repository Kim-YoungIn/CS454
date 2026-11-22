"""Reproduce fog GA measurements and optionally write bonus submission files."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from contextlib import chdir
from pathlib import Path

import torch

import perlin_fog
from util import IMAGES_ORG_PREDS, get_image_tensor_from, get_model_most_likely_for, get_viz_for
from verify_artifacts import check_fog, read_tensor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="perlin_fog", help="Module exposing adv_func and fog controls")
    parser.add_argument("--images", nargs="+", default=list(IMAGES_ORG_PREDS))
    parser.add_argument("--generations", type=int, default=perlin_fog.FOG_GENERATIONS)
    parser.add_argument("--seconds", type=float, default=perlin_fog.FOG_SECONDS)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CS454_THREADS", 4)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CS454_SEED", 454)))
    parser.add_argument("--octaves", type=int, nargs="+", default=list(perlin_fog.FOG_OCTAVES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.generations < 0 or args.seconds <= 0 or not math.isfinite(args.seconds) or args.threads <= 0:
        parser.error("generations must be nonnegative; seconds and threads must be positive and finite")
    if not 1 <= len(args.octaves) <= 16 or len(set(args.octaves)) != len(args.octaves) or min(args.octaves) <= 0:
        parser.error("octaves must contain between 1 and 16 distinct positive integers")
    metadata_path = args.output.with_suffix(".metadata.json")
    source_path = args.output.with_suffix(".source.py")
    if any(path.exists() for path in (args.output, metadata_path, source_path)):
        parser.error("output files already exist; choose a new output path to preserve previous runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CS454_THREADS"] = str(args.threads)
    os.environ["CS454_SEED"] = str(args.seed)
    torch.set_num_threads(args.threads)
    module=importlib.import_module(args.implementation)
    module.FOG_GENERATIONS = args.generations
    module.FOG_SECONDS = min(285.0, max(1.0, args.seconds))
    module.FOG_OCTAVES = tuple(args.octaves)
    metadata = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "seed": args.seed,
        "seconds": module.FOG_SECONDS,
        "generations": module.FOG_GENERATIONS,
        "octaves": module.FOG_OCTAVES,
        "source_snapshot": str(source_path),
        "context": os.environ.get("CS454_CONTEXT", "isolated"),
        "timing": "entry-point call only; excludes model import, artifact rendering and verification",
        "environment": {name: value for name, value in os.environ.items()
                        if name.startswith("CS454_") or name in ("TORCH_HOME", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torchvision", "numpy", "Pillow", "perlin-noise")},
        "sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in ["perlin_fog.py", "util.py", "grade.py", "benchmark_fog.py", *args.images]},
    }
    metadata["implementation"]=module.__name__
    if module is not perlin_fog:
        candidate=args.output.with_suffix(".candidate.py")
        candidate.write_bytes(Path(module.__file__).read_bytes())
        metadata["candidate_snapshot"]=str(candidate)
        metadata["sha256"][module.__name__]=hashlib.sha256(candidate.read_bytes()).hexdigest()
    dependencies = {}
    for index, dependency in enumerate(getattr(module, "SOURCE_DEPENDENCIES", ())):
        dependency_path = Path(dependency)
        snapshot = args.output.with_suffix(f".dependency_{index}_{dependency_path.stem}.py")
        snapshot.write_bytes(dependency_path.read_bytes())
        dependencies[str(dependency_path)] = str(snapshot)
        metadata["sha256"][str(dependency_path)] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    metadata["dependency_snapshots"] = dependencies
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    source_path.write_bytes(Path("perlin_fog.py").read_bytes())
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        all_passed = True
        writer.writerow(["image", "generations", "budget_seconds", "elapsed_seconds", "score",
                         "reference_class", "predicted_class", "status", "genes",
                         "initial_score", "completed_generations", "queries", "round_trip", "engine_metadata"])
        for path in args.images:
            started = time.monotonic()
            genes = module.adv_func(path)
            elapsed = time.monotonic() - started
            image = get_image_tensor_from(path)
            rendered = perlin_fog.add_noise_to_image(perlin_fog.genotype2phenotype(genes), image)
            _, reference = get_model_most_likely_for(image)
            _, predicted = get_model_most_likely_for(rendered)
            status = "pass" if predicted != reference else "class_unchanged"
            all_passed &= status == "pass"
            round_trip = ""
            if args.artifacts:
                stem = Path(path).stem
                with chdir(args.artifacts):
                    Path(f"{stem}_perlin.json").write_text(json.dumps(genes) + "\n")
                    get_viz_for(rendered.clone(), save_to=f"{stem}_perlin.png")
                    import matplotlib.pyplot as plt
                    plt.close("all")
                saved = read_tensor(args.artifacts / f"raw_{stem}_perlin.png", image)
                rebuilt, rebuilt_cost = check_fog(args.artifacts / f"{stem}_perlin.json", image, saved)
                _, rebuilt_class = get_model_most_likely_for(rebuilt)
                if not torch.equal(saved, rendered) or rebuilt_cost != sum(m for _, m in genes) or rebuilt_class != predicted:
                    raise RuntimeError(f"{path}: artifact round-trip changed tensor, score, or prediction")
                round_trip = True
            writer.writerow([path, args.generations, module.FOG_SECONDS, elapsed,
                             sum(m for _, m in genes), reference, predicted, status, json.dumps(genes),
                             module.LAST_RUN["initial_score"], module.LAST_RUN["generations"],
                             module.LAST_RUN["queries"], round_trip, json.dumps(module.LAST_RUN,sort_keys=True)])
            stream.flush()
            print(path, "score", sum(m for _, m in genes), "seconds", elapsed, status, flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
