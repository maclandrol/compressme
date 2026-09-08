"""Inspect safe finite-profile artifacts and test/measure exact XOR reconstruction."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from profile_deltas import XorProfileLookup, bytes_equal, registered_bytes


def summarize(values):
    return {'median_us': statistics.median(values), 'mean_us': statistics.mean(values),
            'sample_sd_us': statistics.stdev(values) if len(values) > 1 else 0,
            'samples_us': values}


def pair_stats(base, target):
    xor = base.numpy().view(np.uint32) ^ target.numpy().view(np.uint32)
    n, width = xor.shape
    column_bytes = 2 if width <= 65536 else 4
    candidates = {}
    for bits in (8, 16):
        flags = xor >= (1 << bits)
        row_counts = flags.sum(axis=1)
        high_bytes = 4 if bits == 8 else 2
        candidates[str(bits)] = {
            'dense_low_bytes': xor.size * bits // 8,
            'exception_values': int(flags.sum()), 'max_exceptions_per_row': int(row_counts.max()),
            'padded_exception_bytes': int(n * row_counts.max() * (column_bytes + high_bytes)),
            'total_bytes': int(xor.size * bits // 8 + n * row_counts.max() * (column_bytes + high_bytes)),
            'csr_bytes_lower_layout': int(xor.size * bits // 8 + flags.sum() * (column_bytes + high_bytes) + (n+1)*4),
        }
    nonzero = xor != 0
    return {'values': xor.size, 'equal_values': int((xor == 0).sum()),
            'nonzero_byte_counts_low_to_high': [int((((xor >> (8*i)) & 255) != 0).sum()) for i in range(4)],
            'highest_xor': int(xor.max()), 'raw_bytes': int(xor.nbytes),
            'candidates': candidates,
            'sparse_full_xor_csr_bytes': int(nonzero.sum() * (column_bytes + 4) + (n+1)*4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=21)
    parser.add_argument('--inner', type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Native MPS access is required for this measurement')
    manifest_path = args.artifact / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    weight_path = args.artifact / manifest['weights_file']
    sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if sha != manifest['weights_sha256']:
        raise ValueError('Artifact weights differ from its manifest')
    state = load_file(str(weight_path))
    names = list(state)
    tables = [state[name] for name in names]
    if not all(t.dtype == torch.float32 and t.shape == tables[0].shape for t in tables):
        raise ValueError('This bounded study requires matching dense float32 profile tables')
    original_bytes = sum(t.numel() * t.element_size() for t in tables)
    base_options = []
    for base in range(len(tables)):
        candidate = XorProfileLookup(tables, base_route=base)
        base_options.append({'base_route': base, 'resident_bytes': registered_bytes(candidate),
                             'modes': [route.mode for route in candidate.routes]})
    candidate = XorProfileLookup(tables, base_route=0).to(args.device)
    original = [table.to(args.device) for table in tables]
    report = {'torch': str(torch.__version__), 'platform': platform.platform(), 'machine': platform.machine(),
              'execution_device': args.device, 'artifact': str(args.artifact), 'weights_sha256': sha,
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'prototype_sha256': hashlib.sha256(Path(__file__).with_name('profile_deltas.py').read_bytes()).hexdigest(),
              'shape': list(tables[0].shape), 'table_names': names,
              'original_registered_bytes': original_bytes, 'base_options': base_options,
              'bulk_base_registered_bytes': registered_bytes(candidate),
              'bulk_base_reduction_fraction': 1 - registered_bytes(candidate) / original_bytes,
              'pair_statistics': {f'{i}->{j}': pair_stats(tables[i], tables[j])
                                  for i in range(len(tables)) for j in range(i+1, len(tables))},
              'state': {name: {'shape': list(t.shape), 'dtype': str(t.dtype), 'bytes': t.numel()*t.element_size()}
                        for name, t in candidate.state_dict().items()},
              'scope': 'Exact selected-row value-byte reconstruction of existing profile tables, not a new whole-model gate'}
    cases = 0
    with torch.inference_mode():
        ids_cases = [torch.tensor(i, device=args.device) for i in range(tables[0].shape[0])]
        ids_cases += [torch.arange(tables[0].shape[0], device=args.device),
                      torch.tensor([[0,1,2], [5,7,3]], device=args.device).t(),
                      torch.empty((2,0,3), dtype=torch.long, device=args.device),
                      torch.zeros((2,3,4), dtype=torch.int32, device=args.device)]
        for route, table in enumerate(original):
            for ids in ids_cases:
                assert bytes_equal(candidate(ids, route=route), F.embedding(ids, table))
                cases += 1
    report['value_byte_validation'] = {'accepted': True, 'comparisons': cases, 'different_bytes': 0}
    def sync():
        if args.device == 'mps': torch.mps.synchronize()
    timings = {}
    cases = [('bulk_32_ids', 0, 32), ('alternate_1_id', 1, 1)]
    if len(tables) > 2:
        cases.append(('small_2_ids', 2, 2))
    rng = random.Random(554)
    with torch.inference_mode():
        for name, route, rows in cases:
            ids = torch.arange(rows, device=args.device).remainder(tables[0].shape[0])
            functions = {'full_table': lambda: F.embedding(ids, original[route]),
                         'xor_lookup': lambda: candidate(ids, route=route)}
            values = {key: [] for key in functions}
            for _ in range(10):
                for function in functions.values(): function()
            sync()
            for _ in range(args.rounds):
                order = list(functions)
                rng.shuffle(order)
                for key in order:
                    sync()
                    start = time.perf_counter_ns()
                    for _ in range(args.inner): functions[key]()
                    sync()
                    values[key].append((time.perf_counter_ns() - start) / 1000 / args.inner)
            timings[name] = {key: summarize(samples) for key, samples in values.items()}
            timings[name]['median_overhead_us'] = statistics.median(values['xor_lookup']) - statistics.median(values['full_table'])
    report['timings'] = timings
    report['timing_scope'] = 'Tiny lookup-only calls, preloaded IDs and state, 4 CPU threads; interleaved synchronized batches, no whole-model speed claim'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('execution_device', 'original_registered_bytes', 'bulk_base_registered_bytes', 'bulk_base_reduction_fraction', 'value_byte_validation')}, indent=2))
    print(json.dumps({name: {'overhead_us': value['median_overhead_us'],
                            'full_us': value['full_table']['median_us'],
                            'xor_us': value['xor_lookup']['median_us']} for name, value in timings.items()}, indent=2))


if __name__ == '__main__':
    main()
