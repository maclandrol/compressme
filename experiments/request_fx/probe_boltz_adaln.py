"""Small actual-weight CPU probe; no complete Boltz model is constructed."""
import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from request_fx import partial_evaluate


def bits(a, b):
    return torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


def timed(call, rounds=25):
    for _ in range(5):
        call()
    elapsed = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        call()
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-state', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from boltz.model.modules.transformersv2 import AdaLN
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('highest')
    manifest = json.loads((args.shared_state / 'manifest.json').read_text())
    prefix = 'structure_module.score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.0.adaln.'
    bindings = manifest['models']['boltz2_conf.ckpt']['bindings']
    model = AdaLN(128, 128).eval().requires_grad_(False)
    with safe_open(args.shared_state / manifest['tensor_file'], framework='pt', device='cpu') as archive:
        state = {name.removeprefix(prefix): archive.get_tensor(binding['tensor']).reshape(binding['shape'])
                 for name, binding in bindings.items() if name.startswith(prefix)}
        model.load_state_dict(state, strict=True)
    graph = torch.fx.symbolic_trace(model).eval()
    fixture = load_file(args.fixture)
    checks, budgets, timings = [], [], []
    for multiplicity in (1, 3):
        c = fixture['c'].repeat_interleave(multiplicity, 0)
        s = c.view(-1, 32, 128)
        with partial_evaluate(graph, {'s': s}) as prepared:
            for seed in range(4):
                a = torch.randn(s.shape, generator=torch.Generator().manual_seed(seed))
                expected, actual = model(a, s), prepared(a)
                checks.append({'multiplicity': multiplicity, 'seed': seed, 'bitwise_equal': bits(expected, actual), 'max_abs': float((expected - actual).abs().max())})
            budgets.append(dict(prepared.report))
            timings.append({'multiplicity': multiplicity, 'unit': 'microseconds',
                            'original_median': timed(lambda: model(a, s)),
                            'guarded_prepared_median': timed(lambda: prepared(a)),
                            'unguarded_dynamic_graph_diagnostic_median': timed(lambda: prepared.dynamic_graph(a)),
                            'timing_scope': 'Small CPU operator microbenchmark; unguarded graph is diagnostic, not a public safe execution route'})
    report = {'scope': 'One original checkpoint AdaLN; no full sampler or model claim', 'torch': torch.__version__,
              'checkpoint_revision': manifest['revision'], 'source_prefix': prefix,
              'request_fx_sha256': hashlib.sha256(Path(__file__).with_name('request_fx.py').read_bytes()).hexdigest(),
              'boltz_transformersv2_sha256': hashlib.sha256(Path(inspect.getsourcefile(AdaLN)).read_bytes()).hexdigest(),
              'all_bitwise_equal': all(item['bitwise_equal'] for item in checks),
              'checks': checks, 'budgets': budgets, 'timings': timings,
              'input_state_trust': 'Reads only selected tensors from the already separately verified joint safetensors audit file'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_bitwise_equal': report['all_bitwise_equal'], 'checks': len(checks), 'timings': timings}, indent=2))


if __name__ == '__main__':
    main()
