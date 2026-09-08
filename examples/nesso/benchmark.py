"""Paired, seeded native Nesso-1 output and latency checks on prepared inputs.

Preparation, checkpoint loading and validation are outside the timed interval.
Every timed call still computes all native metadata. This is an execution
preservation benchmark, not an assay-accuracy benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['tiny20', 'fragment130'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--variants', nargs='+', default=['copy', 'both'],
                        choices=['baseline', 'guards', 'copy', 'conditioning', 'both', 'packed', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 1:
        parser.error('rounds and threads must be positive')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve earlier reports')
    if not args.checkpoint.is_dir() or any(not (args.checkpoint / name).is_file() for name in ('hparams.json', 'model.safetensors')):
        parser.error('--checkpoint must be an existing local directory containing hparams.json and model.safetensors; this benchmark does not download weights')

    import torch
    from prepare import load_batch
    from nesso.model.models.nesso1 import Nesso1
    from compressme.validation import compare_outputs, seeded
    import compressme.validation as validation_module

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    model = Nesso1.from_pretrained(args.checkpoint).eval().requires_grad_(False)
    model.use_kernels = False
    model.predict_args = dict(model.predict_args or {}, recycling_steps=5,
                              refine_protein_inference=True,
                              refine_protein_cutoff=22.0,
                              refine_protein_tokens_budget=256,
                              affinity_protein_cutoff=15.0, save_metadata=True)
    model.to(device)
    sync()
    load_seconds = time.perf_counter() - start
    stats_by_variant = {}

    def context(variant):
        if variant == 'baseline':
            return nullcontext()
        from compressme.nesso_runtime import nesso_inference_optimizations
        return nesso_inference_optimizations(model, immutable_request=True,
                                           single_chunk=variant in {'copy', 'both', 'all'},
                                           cache_esm=variant in {'conditioning', 'both', 'all'},
                                           pack_triangles=variant in {'packed', 'all'})

    def run(batch, variant, native='predict'):
        # Restore the RNG before each native reference/candidate call, including
        # evaluation-mode random masks used by the pinned upstream Pairformer.
        with seeded(args.seed), torch.no_grad():
            sync()
            start = time.perf_counter()
            with context(variant) as runtime_stats:
                if native == 'forward':
                    out = model(batch, recycling_steps=5, refine_protein_inference=True)
                else:
                    out = model.predict_step(batch, 0)
            sync()
            elapsed = time.perf_counter() - start
            if runtime_stats is not None:
                stats_by_variant[variant] = dict(runtime_stats)
            if out.get('exception', False):
                raise RuntimeError('Native prediction failed')
            # Snapshot after timing. No output is retained on the accelerator.
            out = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                   for k, v in out.items()}
        return elapsed, out

    def summary(metrics):
        return {'tensor_leaves': len(metrics),
                'bitwise_identical': all(x['bitwise'] for x in metrics.values()),
                'max_abs': max((x['max_abs'] for x in metrics.values()), default=0.0),
                'metrics': metrics}

    storages = {}
    for tensor in model.state_dict().values():
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    import importlib.metadata
    report = {'model': 'Nesso-1', 'device': str(device), 'platform': platform.platform(),
              'torch': torch.__version__, 'nesso': importlib.metadata.version('nesso'),
              'threads': args.threads, 'precision': 'float32; native metadata casts retained',
              'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'use_kernels': False, 'seed': args.seed, 'predict_args': model.predict_args,
              'checkpoint_sha256': sha256(args.checkpoint / 'model.safetensors'),
              'hparams_sha256': sha256(args.checkpoint / 'hparams.json'),
              'benchmark_sha256': sha256(Path(__file__)),
              'preparation_script_sha256': sha256(Path(__file__).with_name('prepare.py')),
              'validation_script_sha256': sha256(Path(validation_module.__file__)),
              'parameters': sum(p.numel() for p in model.parameters()),
              'unique_registered_storage_bytes': sum(storages.values()),
              'checkpoint_load_and_device_seconds': load_seconds,
              'timing_scope': 'warmed native predict_step, all metadata; includes per-call adapter construction, state audit, request setup/cleanup and native pocket-selection transfers; excludes checkpoint load, ESM, YAML/RDKit, file I/O, external input/output transfers, RNG reset and validation',
              'quality_scope': 'numerical preservation of native outputs; no labelled biological accuracy evaluation',
              'cases': []}
    import nesso
    root = Path(nesso.__file__).parent
    report['nesso_source_sha256'] = {str(p.relative_to(root)): sha256(p)
                                    for p in sorted(root.rglob('*.py'))}
    import compressme
    cm_root = Path(compressme.__file__).parent
    report['runtime_sha256'] = {n: sha256(cm_root / n)
                                for n in ('nesso_runtime.py', 'chunking.py', 'contractions.py')
                                if (cm_root / n).exists()}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    name = variant = None
    phase = 'validation setup'
    try:
        with torch.no_grad():
            for name in args.cases:
                phase = 'fixture loading'
                variant = None
                case_dir = args.fixtures / name
                files = list(case_dir.glob('*batch*.safetensors'))
                if len(files) != 1:
                    raise ValueError(f'Expected one prepared batch in {case_dir}: {files}')
                batch = load_batch(case_dir, device=str(device))
                item = {'case': name, 'batch_sha256': sha256(files[0]),
                        'batch_metadata_sha256': sha256(case_dir / 'batch.json'),
                        'tokens': int(batch['token_pad_mask'].sum()),
                        'padded_tokens': batch['token_pad_mask'].shape[1], 'variants': {}}
                report['cases'].append(item)
                phase = 'reference forward and predict self-repeat'
                print(f'{name}: {item["tokens"]} tokens, {device}, validating native outputs', flush=True)
                _, original_forward = run(batch, 'baseline', 'forward')
                _, repeated_forward = run(batch, 'baseline', 'forward')
                item['reference_forward_self_repeat'] = summary(compare_outputs(original_forward, repeated_forward))
                del repeated_forward
                _, original = run(batch, 'baseline')
                _, repeated = run(batch, 'baseline')
                item['reference_predict_self_repeat'] = summary(compare_outputs(original, repeated))
                del repeated
                if not all(item[k]['bitwise_identical'] for k in ('reference_forward_self_repeat', 'reference_predict_self_repeat')):
                    report['status'] = 'reference_self_repeat_failed'
                    save()
                    raise RuntimeError('Reference self-repeat is not byte-identical; inspect the saved report')
                for variant in args.variants:
                    phase = 'candidate forward and predict validation'
                    item['variants'][variant] = {'status': 'validating', 'pairs': []}
                    _, forward = run(batch, variant, 'forward')
                    _, prediction = run(batch, variant)
                    result = {'forward': summary(compare_outputs(original_forward, forward)),
                              'predict_step': summary(compare_outputs(original, prediction)),
                              'runtime_counters': stats_by_variant.get(variant), 'pairs': []}
                    item["variants"][variant] = result
                    del forward, prediction
                    if not (result['forward']['bitwise_identical'] and result['predict_step']['bitwise_identical']):
                        result['status'] = 'rejected_output_mismatch'
                        item['variants'][variant] = result
                        print(f'  {variant}: output mismatch; no accepted timing claim', flush=True)
                        continue
                    phase = 'warmup and paired timed output validation'
                    # Warm both paths, then alternate order to limit thermal/order bias.
                    run(batch, 'baseline'); run(batch, variant)
                    for index in range(args.rounds):
                        order = ['baseline', variant] if index % 2 == 0 else [variant, 'baseline']
                        measured = []
                        for path in order:
                            elapsed, prediction = run(batch, path)
                            gate = summary(compare_outputs(original, prediction))
                            if not gate['bitwise_identical']:
                                result['status'] = 'rejected_timed_output_mismatch'
                                result['failed_gate'] = gate
                                item['variants'][variant] = result
                                save()
                                raise RuntimeError('Timed output mismatch; inspect the saved report')
                            measured.append((path, elapsed, gate))
                        # A baseline-only run intentionally measures baseline twice.
                        base = measured[0 if index % 2 == 0 else 1]
                        candidate = measured[1 if index % 2 == 0 else 0]
                        result['pairs'].append({'reference_seconds': base[1], 'candidate_seconds': candidate[1],
                                                'speedup': base[1] / candidate[1],
                                                'reference_gate': base[2], 'candidate_gate': candidate[2]})
                        print(f'  {variant} round {index+1}: {base[1]:.4f} -> {candidate[1]:.4f} s; {base[1]/candidate[1]:.3f}x', flush=True)
                    ratios = [p['speedup'] for p in result['pairs']]
                    result.update(median_paired_speedup=statistics.median(ratios), paired_speedup_range=[min(ratios), max(ratios)],
                                  reference_median_seconds=statistics.median(p['reference_seconds'] for p in result['pairs']),
                                  candidate_median_seconds=statistics.median(p['candidate_seconds'] for p in result['pairs']))
                    result['status'] = 'accepted_byte_identical'
                    item['variants'][variant] = result
                if args.profile:
                    with seeded(args.seed), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        model.predict_step(batch, 0)
                        sync()
                    table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25)
                    item['cpu_dispatch_profile'] = table
                    print(table, flush=True)
                del batch, original, original_forward
                report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
                report['peak_rss_scope'] = 'whole benchmark process including reference/candidate output snapshots; not a per-variant comparison'
                report['status'] = ('accepted_byte_identical' if all(
                    v.get('status') == 'accepted_byte_identical'
                    for c in report['cases'] for v in c['variants'].values()) else 'candidate_rejected')
                save()
    except Exception as exc:
        if report.get('status') != 'reference_self_repeat_failed':
            report['status'] = 'validation_exception'
        report['failure'] = {'case': name, 'variant': variant, 'phase': phase,
                             'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise
    print(f'Saved {args.output}', flush=True)
    if report['status'] != 'accepted_byte_identical':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
