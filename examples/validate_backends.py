"""Complete-output, same-backend validation; a missing GPU is never a pass.

Custom providers are trusted local Python, specified as /path/provider.py:build.
A provider receives (device, options) and returns BackendSuite. Its cases must
return the complete model output, not an output selector. Factories are loaded
sequentially; reference outputs are detached and copied to CPU before reuse.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from collections.abc import Mapping
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class BackendCase:
    name: str
    call: Callable
    seed: int = 1729


@dataclasses.dataclass
class BackendSuite:
    reference_factory: Callable
    candidate_factory: Callable
    cases: Callable
    metadata: dict = dataclasses.field(default_factory=dict)
    source_files: tuple = ()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_inventory():
    cuda_available = torch.cuda.is_available()
    return {
        'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
        'torch': str(torch.__version__), 'cuda_build_version': torch.version.cuda,
        'cuda_available': cuda_available,
        'cuda_devices': [dict(index=i, name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i)),
                              total_memory=torch.cuda.get_device_properties(i).total_memory)
                         for i in range(torch.cuda.device_count())] if cuda_available else [],
        'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(),
        'nvidia_smi_found': shutil.which('nvidia-smi') is not None,
    }


def device_available(device):
    device = torch.device(device)
    if device.type == 'cpu':
        return device.index in (None, 0)
    if device.type == 'mps':
        return device.index in (None, 0) and torch.backends.mps.is_available()
    if device.type == 'cuda':
        return torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count()
    return False


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def settings():
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'mps_fallback_environment': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'mps_fast_math_environment': os.environ.get('PYTORCH_MPS_FAST_MATH'),
        'cpu_threads': torch.get_num_threads(),
        'execution': 'torch.inference_mode; autocast explicitly disabled',
    }


@contextlib.contextmanager
def execution_settings(determinism='inherit', tf32=False, threads=4):
    before = settings()
    try:
        if determinism != 'inherit':
            torch.use_deterministic_algorithms(determinism == 'strict', warn_only=False)
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = False
        if determinism == 'strict':
            torch.backends.cudnn.deterministic = True
        yield settings()
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'],
                                          warn_only=before['deterministic_warn_only'])
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = before['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = before['cudnn_allow_tf32']
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
        torch.set_num_threads(before['cpu_threads'])


def tree_structure(value):
    if isinstance(value, torch.Tensor):
        return ('tensor',)
    if isinstance(value, Mapping):
        return (type(value), tuple((k, tree_structure(v)) for k, v in value.items()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, tree_structure(getattr(value, f.name)))
                                   for f in dataclasses.fields(value)))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(tree_structure(v) for v in value))
    return (type(value),)


def snapshot_output(value):
    from compressme.validation import tensor_leaves
    leaves = tensor_leaves(value)
    copied = {name: item.detach().cpu().resolve_neg().clone() if isinstance(item, torch.Tensor) else item
              for name, item in leaves.items()}
    return tree_structure(value), copied


def compare_snapshots(reference, candidate, atol, rtol):
    from compressme.validation import compare_outputs
    if reference[0] != candidate[0]:
        raise ValueError('Complete output container structure changed')
    metrics = compare_outputs(reference[1], candidate[1])
    if not metrics:
        raise ValueError('No tensor outputs were inspected')
    tensor_values = [v for v in reference[1].values() if isinstance(v, torch.Tensor)]
    return {
        'tensor_leaves': len(tensor_values),
        'non_tensor_leaves': len(reference[1]) - len(tensor_values),
        'logical_tensor_bytes': sum(v.numel() * v.element_size() for v in tensor_values),
        'accepted_tolerance': all(m['max_abs'] <= atol and m['relative_l2'] <= rtol for m in metrics.values()),
        'bitwise_identical': all(m['bitwise'] for m in metrics.values()),
        'numeric_equal': all(m['exact'] for m in metrics.values()),
        'max_abs': max(m['max_abs'] for m in metrics.values()),
        'max_relative_l2': max(m['relative_l2'] for m in metrics.values()),
        'metrics': metrics,
    }


def model_info(model, device):
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Factories must return a torch.nn.Module')
    from compressme.packed_embedding import PackedFrozenEmbedding
    from compressme.finite_lookup import _resident_storage_bytes
    host_buffers = set()
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module) is PackedFrozenEmbedding:
            module._check()
            prefix = path + '.' if path else ''
            host_buffers.update(prefix + name for name in ('payload', 'offsets'))
    wrong = [name for name, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
             if name not in host_buffers and (t.device.type != device.type
                or device.type == 'cuda' and device.index is not None and t.device.index != device.index)]
    if wrong:
        raise ValueError(f'Model state is not on requested device: {wrong[:4]}')
    return {'class': type(model).__module__ + '.' + type(model).__qualname__,
            'parameters': sum(v.numel() for v in model.parameters()),
            'dtypes': sorted({str(v.dtype) for v in model.parameters()}),
            'registered_storage_bytes_all_devices': _resident_storage_bytes(model),
            'permitted_cpu_codec_buffers': sorted(host_buffers)}


def run_backend(provider, *, device='cpu', options=None, atol=1e-5, rtol=1e-5,
                require_bitwise=False, determinism='inherit', tf32=False, threads=4):
    report = {'device': str(device), 'hardware': hardware_inventory(), 'status': 'running',
              'accepted': False, 'accepted_tolerance': False, 'bitwise_identical': False,
              'atol': atol, 'relative_l2_tolerance': rtol, 'require_bitwise': require_bitwise,
              'rule': 'Every floating leaf: max_abs <= atol AND relative_l2 <= rtol; discrete/non-tensor outputs exact. Source self-repeat must also pass.',
              'scope': 'Complete outputs returned by the trusted cases, on this backend; not biological accuracy, cross-device equivalence, or a speed benchmark.',
              'source_hashes': {str(Path(__file__).resolve()): sha256(__file__)}, 'cases': []}
    reference = candidate = None
    try:
        if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
            report.update(atol=repr(atol), relative_l2_tolerance=repr(rtol))
            raise ValueError('Tolerances must be finite and nonnegative')
        if determinism not in ('inherit', 'strict', 'off') or threads < 1:
            raise ValueError('Invalid deterministic setting or thread count')
        destination = torch.device(device)
        if not device_available(destination):
            report.update(status='unavailable', reason=f'{destination} is not available; no model validation ran')
            return report
        from compressme.validation import seeded, compare_outputs
        report['source_hashes'][inspect.getsourcefile(compare_outputs)] = sha256(inspect.getsourcefile(compare_outputs))
        with execution_settings(determinism, tf32, threads) as active:
            report['settings'] = active
            suite = provider(destination, options or {})
            if not isinstance(suite, BackendSuite):
                raise TypeError('Provider must return BackendSuite')
            report['model'] = suite.metadata
            for path in suite.source_files:
                report['source_hashes'][str(Path(path).resolve())] = sha256(path)
            with seeded(0):
                reference = suite.reference_factory(destination).eval().requires_grad_(False)
            report['reference'] = model_info(reference, destination)
            with seeded(1729):
                cases = list(suite.cases(reference))
            if not cases or len({case.name for case in cases}) != len(cases):
                raise ValueError('Cases must be nonempty and have unique names')
            expected = []
            with torch.inference_mode(), torch.autocast(destination.type, enabled=False):
                for case in cases:
                    with seeded(case.seed):
                        first = snapshot_output(case.call(reference, destination))
                    synchronize(destination)
                    with seeded(case.seed):
                        repeat = snapshot_output(case.call(reference, destination))
                    synchronize(destination)
                    check = compare_snapshots(first, repeat, atol, rtol)
                    report['cases'].append({'name': case.name, 'seed': case.seed, 'source_self_repeat': check})
                    expected.append(first)
                    print('SOURCE', case.name, check['accepted_tolerance'], flush=True)
            del reference
            reference = None
            gc.collect()
            if destination.type == 'mps': torch.mps.empty_cache()
            elif destination.type == 'cuda': torch.cuda.empty_cache()
            with seeded(0):
                candidate = suite.candidate_factory(destination).eval().requires_grad_(False)
            report['candidate'] = model_info(candidate, destination)
            with torch.inference_mode(), torch.autocast(destination.type, enabled=False):
                for case, first, row in zip(cases, expected, report['cases']):
                    with seeded(case.seed):
                        actual = snapshot_output(case.call(candidate, destination))
                    synchronize(destination)
                    row['candidate'] = compare_snapshots(first, actual, atol, rtol)
                    print('CANDIDATE', case.name, row['candidate']['accepted_tolerance'],
                          row['candidate']['max_abs'], flush=True)
            all_checks = [c[k] for c in report['cases'] for k in ('source_self_repeat', 'candidate')]
            report['accepted_tolerance'] = all(c['accepted_tolerance'] for c in all_checks)
            report['bitwise_identical'] = all(c['bitwise_identical'] for c in all_checks)
            report['accepted'] = report['accepted_tolerance'] and (not require_bitwise or report['bitwise_identical'])
            report['status'] = 'passed' if report['accepted'] else 'failed'
            report['candidate_tensor_comparisons'] = sum(c['candidate']['tensor_leaves'] for c in report['cases'])
    except Exception as error:
        report.update(status='failed', accepted=False, error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        del reference, candidate
    return report


def import_local(path, name):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def builtin_provider(device, options):
    recipe = options['recipe']
    artifact = Path(options['artifact']).resolve(strict=True)
    checkpoint = Path(options['reference_weights']).resolve(strict=True)
    source_files = list((artifact / 'architecture').rglob('*.py'))
    source_files.append(artifact / 'manifest.json')
    metadata = {'recipe': recipe, 'artifact': str(artifact), 'reference_weights': str(checkpoint)}
    if recipe.startswith('moljepa'):
        from compressme.moljepa_io import moljepa_factory, load_moljepa
        from safetensors.torch import load_file
        metadata['reference_sha256'] = sha256(checkpoint)
        def reference(d):
            model = moljepa_factory(artifact / 'architecture')
            model.load_state_dict(load_file(str(checkpoint)), strict=True)
            return model.to(d)
        def cases(model):
            smiles = json.loads((ROOT / 'benchmarks/verification_smiles.json').read_text())['smiles']
            for start in range(0, len(smiles), 4):
                values = smiles[start:start + 4]
                for attention in (False, True):
                    yield BackendCase(f'smiles-{start}-{start + len(values)}-attention-{attention}',
                                      lambda m, d, values=values, attention=attention: m(values, return_attn=attention))
            yield BackendCase('isolated-helium-attention', lambda m, d: m(['[He]'], return_attn=True))
            if recipe == 'moljepa-full':
                generator = torch.Generator().manual_seed(9922)
                supplied = []
                for spec in model.model.modalities_spec:
                    if spec['name'] == 'graph': continue
                    shape = (3, spec['node_dim']) if spec['output'] == 'atoms' else (spec['dim'],)
                    supplied.append({spec['name']: torch.randn(shape, generator=generator)})
                yield BackendCase('all-optional-modalities', lambda m, d: m(smiles[:len(supplied)], embeddings_data=supplied, return_attn=True))
        source_files.extend(ROOT.joinpath('src/compressme').glob('*.py'))
        return BackendSuite(reference, lambda d: load_moljepa(artifact, device=d), cases, metadata, tuple(source_files))
    if recipe == 'state-st':
        from compressme.state_io import state_factory, load_state_st
        source = import_local(ROOT / 'examples/build_state_st.py', '_backend_state_probes')
        expected = '121ff54db0e6f9f53b2777bbf45139757367ed827a111824213e6bf3777b2618'
        metadata['reference_sha256'] = sha256(checkpoint)
        if metadata['reference_sha256'] != expected: raise ValueError('Unexpected State ST checkpoint')
        def reference(d):
            model = state_factory(artifact / 'architecture')
            original = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
            model.load_state_dict(original['state_dict'], strict=True)
            return model.to(d)
        def cases(model):
            for batch, spec in source.probes(model):
                def call(m, d, batch=batch, spec=spec):
                    moved = {k: v.to(d) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    return m.predict_step(moved, 0, padded=spec['padded'])
                yield BackendCase(f"cells-{spec['cells']}-padded-{spec['padded']}", call)
            yield BackendCase('all-32000-token-ids', lambda m, d: m.transformer_backbone.embed_tokens(torch.arange(32000, device=d)))
        source_files.extend([ROOT / 'examples/build_state_st.py', ROOT / 'src/compressme/constant_embeddings.py', ROOT / 'src/compressme/state_io.py'])
        return BackendSuite(reference, lambda d: load_state_st(artifact, device=d), cases, metadata, tuple(source_files))
    if recipe == 'boltz2':
        import copy
        from compressme.boltz2_io import boltz2_factory, load_boltz2
        safe = import_local(ROOT / 'experiments/boltz2-weight-audit/safe_state.py', '_backend_safe_boltz_state')
        native = Path(options.get('native_dir') or ROOT / 'experiments/boltz2-runtime')
        sys.path.insert(0, str(native))
        from sharing_probe import batch_from_existing
        def reference(d):
            bundle = boltz2_factory(artifact / 'architecture')
            for member, filename in [('confidence', 'boltz2_conf.ckpt'), ('affinity', 'boltz2_aff.ckpt')]:
                state, _ = safe.load_tensor_state(checkpoint, filename)
                bundle[member].load_state_dict(state, strict=True)
            return bundle.to(d)
        def cases(model):
            conf_dir = native / 'cpu-fp32-default-sampling'
            for member in ('confidence', 'affinity'):
                base = conf_dir if member == 'confidence' else native / 'cpu-aff-fp32-default-sampling'
                dm, batch = batch_from_existing(base, native / 'mols', conf_dir / 'predictions' if member == 'affinity' else None)
                def call(m, d, member=member, dm=dm, batch=batch):
                    moved = dm.transfer_batch_to_device(copy.deepcopy(batch), d, 0)
                    output = m[member].predict_step(moved, 0)
                    if output.get('exception'): raise ValueError('Native Boltz returned an exception')
                    return output
                yield BackendCase(member, call)
        metadata.update(scope='20-residue protein plus ethanol; native complete FP32 confidence and affinity default schedules',
                        reference_manifest_sha256=sha256(checkpoint / 'manifest.json'))
        source_files.extend([native / 'sharing_probe.py', ROOT / 'experiments/boltz2-weight-audit/safe_state.py', ROOT / 'src/compressme/boltz2_io.py', ROOT / 'src/compressme/sharing.py'])
        return BackendSuite(reference, lambda d: load_boltz2(artifact, device=d), cases, metadata, tuple(source_files))
    raise ValueError(f'Unknown recipe: {recipe}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, help='cpu, mps, cuda, or cuda:N')
    parser.add_argument('--recipe', choices=['moljepa-smiles', 'moljepa-full', 'boltz2', 'state-st'])
    parser.add_argument('--provider', help='Trusted local Python path:function returning BackendSuite')
    parser.add_argument('--options-json', type=Path, help='Custom provider options; JSON object')
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--reference-weights', type=Path)
    parser.add_argument('--native-dir', type=Path)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--relative-l2-tolerance', type=float, default=1e-5)
    parser.add_argument('--require-bitwise', action='store_true')
    parser.add_argument('--determinism', choices=['inherit', 'strict', 'off'], default='inherit')
    parser.add_argument('--tf32', action='store_true', help='Explicitly allow TF32; disabled by default')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('Output exists; use a new report path to preserve earlier evidence')
    if args.probe_only:
        available = device_available(args.device)
        report = {'device': args.device, 'hardware': hardware_inventory(), 'status': 'available_not_validated' if available else 'unavailable',
                  'accepted': False, 'model_validation_ran': False, 'reason': 'Hardware inventory only; CUDA/model success is not inferred'}
    else:
        options = json.loads(args.options_json.read_text()) if args.options_json else {}
        if not isinstance(options, dict): parser.error('Provider options must be a JSON object')
        if args.provider:
            if args.recipe: parser.error('Choose --provider or --recipe')
            path, name = args.provider.rsplit(':', 1)
            # Expose dataclasses consistently to a provider importing this script.
            sys.modules.setdefault('validate_backends', sys.modules[__name__])
            provider = getattr(import_local(path, '_trusted_backend_provider'), name)
        else:
            if not args.recipe or not args.artifact or not args.reference_weights:
                parser.error('A built-in recipe requires --recipe, --artifact and --reference-weights')
            options.update(recipe=args.recipe, artifact=str(args.artifact), reference_weights=str(args.reference_weights),
                           native_dir=str(args.native_dir) if args.native_dir else None)
            provider = builtin_provider
        report = run_backend(provider, device=args.device, options=options, atol=args.atol,
                             rtol=args.relative_l2_tolerance, require_bitwise=args.require_bitwise,
                             determinism=args.determinism, tf32=args.tf32, threads=args.threads)
        if args.provider: report['source_hashes'][str(Path(path).resolve())] = sha256(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('device', 'status', 'accepted', 'accepted_tolerance', 'bitwise_identical', 'error')}, indent=2))
    return 0 if report.get('accepted') else 2 if report['status'] == 'unavailable' else 1


if __name__ == '__main__':
    raise SystemExit(main())
