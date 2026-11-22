"""Reproduce CPU inference throughput across thread counts and small batches.

Run from the repository root. This profiles the authoritative prediction helper,
using contiguous normalized inputs as in the attack engine. Example:
  TORCH_HOME=/tmp/cs454-torch python profile_inference.py --output runs/cpu-profile.csv
Model import and metadata creation are excluded from forward timings.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from util import get_image_tensor_from, get_model_predictions_for


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='images/knot.jpg')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if min(args.threads + args.batches + [args.repeats]) < 1 or args.warmup < 0:
        parser.error('threads, batches and repeats must be positive; warmup must be nonnegative')
    metadata_path = args.output.with_suffix('.metadata.json')
    if args.output.exists() or metadata_path.exists():
        parser.error('output already exists; use a fresh output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = get_image_tensor_from(args.image).contiguous()
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
        'options': {**vars(args), 'output': str(args.output)},
        'python': platform.python_version(), 'platform': platform.platform(),
        'cpu_model': cpu_model, 'logical_cpus': os.cpu_count(),
        'packages': {name: version(name) for name in ('torch', 'torchvision', 'Pillow', 'numpy')},
        'input_shape': list(image.shape), 'input_strides': list(image.stride()),
        'input_dtype': str(image.dtype), 'mkldnn_enabled': torch.backends.mkldnn.enabled,
        'sha256': {str(path): digest(path) for path in ('profile_inference.py', 'util.py', args.image)},
        'environment': {name: value for name, value in os.environ.items()
                        if name in ('TORCH_HOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'timing': 'perf_counter around prediction helper only, CPU synchronous; warmup excluded',
        'context': os.environ.get('CS454_CONTEXT', 'isolated'),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    previous_threads = torch.get_num_threads()
    columns = ['threads', 'batch', 'repeats', 'mean_seconds', 'median_seconds', 'min_seconds',
               'mean_seconds_per_image', 'images_per_second', 'predicted_class', 'sample_seconds']
    try:
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            for threads in args.threads:
                torch.set_num_threads(threads)
                for batch in args.batches:
                    inputs = image.repeat(batch, 1, 1, 1)
                    for _ in range(args.warmup):
                        get_model_predictions_for(inputs)
                    samples = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        probabilities = get_model_predictions_for(inputs)
                        samples.append(time.perf_counter() - started)
                    predicted = probabilities.argmax(1)
                    if not bool(torch.all(predicted == predicted[0])):
                        raise RuntimeError('Identical batch elements received different predictions')
                    mean = statistics.mean(samples)
                    row = dict(threads=threads, batch=batch, repeats=args.repeats,
                               mean_seconds=mean, median_seconds=statistics.median(samples),
                               min_seconds=min(samples), mean_seconds_per_image=mean / batch,
                               images_per_second=batch / mean, predicted_class=int(predicted[0]),
                               sample_seconds=json.dumps(samples))
                    writer.writerow(row)
                    stream.flush()
                    print(json.dumps(row), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
