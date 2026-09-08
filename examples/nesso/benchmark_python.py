"""Compare Python state-check overhead without running Nesso model inference.

The original pretrained checkpoint is loaded on CPU. Only signature creation
and disposal are timed; no model forward, prediction or GPU operation runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def original_state_signature(model, tensor_signature):
    """Original three-traversal reference, including repeated aliased paths."""
    modules = tuple((name, id(module), type(module), module.training)
                    for name, module in model.named_modules(remove_duplicate=False))
    tensors = tuple((name, tensor_signature(value)) for name, value in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))))
    return modules, tensors


def canonical(signature):
    """Normalize the private one-pass representation to the original layout."""
    modules, parameters, buffers = signature
    tensors = tuple((f'{path}.{name}' if path else name, value)
                    for path, name, value in parameters + buffers)
    return modules, tensors


def summary(values):
    return {'median': statistics.median(values), 'mean': statistics.mean(values),
            'std': statistics.pstdev(values), 'min': min(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Local directory containing hparams.json and model.safetensors')
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON report; existing paths are never overwritten')
    parser.add_argument('--rounds', type=int, default=40,
                        help='Alternating reference/candidate pairs (default: 40)')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 1:
        parser.error('rounds and threads must be positive')
    if os.path.lexists(args.output):
        parser.error(f'Output already exists: {args.output}')
    checkpoint_files = [args.checkpoint / name for name in ('hparams.json', 'model.safetensors')]
    if not args.checkpoint.is_dir() or any(not path.is_file() for path in checkpoint_files):
        parser.error('checkpoint must be a local directory with hparams.json and model.safetensors')

    # Optional model dependencies are deliberately absent from the help path.
    import torch
    from nesso.model.models.nesso1 import Nesso1
    from compressme import nesso_runtime

    torch.set_num_threads(args.threads)
    script = Path(__file__).resolve()
    runtime_source = Path(inspect.getsourcefile(nesso_runtime)).resolve()
    model_source = Path(inspect.getsourcefile(Nesso1)).resolve()
    provenance = {
        'benchmark_sha256': sha256(script),
        'runtime_sha256': {runtime_source.name: sha256(runtime_source)},
        'nesso_model_source_sha256': sha256(model_source),
        'checkpoint_sha256': sha256(checkpoint_files[1]),
        'hparams_sha256': sha256(checkpoint_files[0]),
    }
    load_start = time.perf_counter()
    model = Nesso1.from_pretrained(args.checkpoint, map_location='cpu').eval().requires_grad_(False)
    load_seconds = time.perf_counter() - load_start
    signature = nesso_runtime._tensor_signature
    functions = {
        'original_three_walks': lambda: original_state_signature(model, signature),
        'current_one_walk': lambda: nesso_runtime._state_signature(model),
    }
    original = functions['original_three_walks']()
    current = functions['current_one_walk']()
    if canonical(current) != original:
        raise RuntimeError('Current state signature does not preserve the complete reference contents')
    module_paths, tensor_paths = len(original[0]), len(original[1])
    del original, current

    for _ in range(5):
        for function in functions.values():
            function()
    samples = {name: [] for name in functions}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for pair in range(args.rounds):
            order = tuple(functions)
            if pair % 2:
                order = tuple(reversed(order))
            for name in order:
                start = time.perf_counter_ns()
                functions[name]()
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
    finally:
        if gc_enabled:
            gc.enable()

    if canonical(functions['current_one_walk']()) != functions['original_three_walks']():
        raise RuntimeError('State signature agreement changed during the benchmark')
    if (sha256(script) != provenance['benchmark_sha256']
            or sha256(runtime_source) != provenance['runtime_sha256'][runtime_source.name]
            or sha256(model_source) != provenance['nesso_model_source_sha256']):
        raise RuntimeError('Benchmark or model/runtime source changed during the benchmark')

    old_median = statistics.median(samples['original_three_walks'])
    new_median = statistics.median(samples['current_one_walk'])
    ratios = [old / new for old, new in zip(samples['original_three_walks'], samples['current_one_walk'])]
    report = {
        'model': 'Nesso-1', 'nesso_revision': nesso_runtime.NESSO_REVISION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'device': 'cpu', 'threads': args.threads,
        'checkpoint_directory_name': args.checkpoint.name,
        'timing_scope': 'CPU signature creation and disposal only; no model forward, prediction or GPU operations; checkpoint load, hashing and validation excluded',
        'gc_during_timing': 'disabled for both implementations; restored afterward',
        'checkpoint_load_seconds': load_seconds,
        'module_paths': module_paths, 'tensor_paths': tensor_paths,
        'signature_contents_exact': True, 'pairs': args.rounds,
        'warmup_calls_per_implementation': 5,
        'samples_ms': samples,
        'summary_ms': {name: summary(values) for name, values in samples.items()},
        'metadata_speedup_from_medians': old_median / new_median,
        'paired_metadata_speedup': summary(ratios),
        'estimated_three_checks_saved_ms': 3 * (old_median - new_median),
        'whole_model_speedup': None,
        **provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also closes the race after the early existence check.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({
        'signature_contents_exact': True,
        'median_ms': {name: values['median'] for name, values in report['summary_ms'].items()},
        'metadata_speedup_from_medians': report['metadata_speedup_from_medians'],
        'estimated_three_checks_saved_ms': report['estimated_three_checks_saved_ms'],
    }, indent=2))


if __name__ == '__main__':
    main()
