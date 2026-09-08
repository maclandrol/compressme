"""Compare all original/portable State ST outputs as bytes on one backend.

Uses the same four numerical cases as the original artifact build. This is a
stronger equality measurement of those cases, not a biological benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from build_state_st import probes
from compressme.state_io import load_state_st, state_factory
from compressme.validation import compare_outputs, tensor_leaves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    expected = '121ff54db0e6f9f53b2777bbf45139757367ed827a111824213e6bf3777b2618'
    if digest.hexdigest() != expected:
        raise ValueError('Expected the pinned ST-HVG-Replogle K562 checkpoint')
    torch.set_num_threads(4)
    original = state_factory(args.artifact / 'architecture')
    checkpoint = torch.load(args.checkpoint, weights_only=True, mmap=True, map_location='cpu')
    original.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    with patch('torch.load', side_effect=AssertionError('Original checkpoint forbidden at portable reload')):
        candidate = load_state_st(args.artifact, device=args.device)
    cases = list(probes(original))
    original.to(args.device).eval()
    report = {'checkpoint_sha256': expected, 'device': args.device, 'dtype': 'float32',
              'torch_version': str(torch.__version__),
              'validation': 'Actual pretrained weights; synthetic numerical API cases',
              'equality_method': 'Explicit contiguous logical tensor value bytes; signed zero distinguished',
              'portable_reload_forbids_torch_load': True, 'cases': []}
    with torch.inference_mode():
        for batch, case in cases:
            batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            left = original.predict_step(batch, 0, padded=case['padded'])
            right = candidate.predict_step(batch, 0, padded=case['padded'])
            metrics = compare_outputs(left, right)
            assert metrics and all(item['bitwise'] for item in metrics.values()), case
            size = sum(value.numel() * value.element_size() for value in tensor_leaves(left).values()
                       if isinstance(value, torch.Tensor))
            report['cases'].append({**case, 'metrics': metrics, 'tensor_value_bytes_compared': size,
                                    'all_outputs_bitwise_equal': True})
        indices = torch.arange(32000, device=args.device)
        left = original.transformer_backbone.embed_tokens(indices)
        right = candidate.transformer_backbone.embed_tokens(indices)
        assert compare_outputs(left, right)['output']['bitwise']
        report['all_32000_token_rows_bitwise_equal'] = True
        report['token_table_value_bytes_compared'] = left.numel() * left.element_size()
    report['all_outputs_bitwise_equal'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
