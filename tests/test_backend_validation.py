"""Failure-path tests for the portable full-output backend runner."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


PATH = Path(__file__).resolve().parents[1] / 'examples/validate_backends.py'
spec = importlib.util.spec_from_file_location('_backend_runner_test', PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Output(nn.Module):
    def __init__(self, sign=1., label='kept'):
        super().__init__()
        self.register_buffer('value', torch.tensor([0., 1.]))
        self.sign, self.label = sign, label

    def forward(self):
        return {'tensor': self.value * self.sign, 'nested': (torch.tensor([1, 2]), self.label), 'none': None}


def provider(ref=lambda: Output(), cand=lambda: Output()):
    return lambda d, options: runner.BackendSuite(lambda d: ref().to(d), lambda d: cand().to(d),
                lambda model: [runner.BackendCase('complete', lambda model, d: model())])


def test_complete_outputs_and_source_repeat():
    result = runner.run_backend(provider(), require_bitwise=True)
    assert result['accepted'] and result['bitwise_identical']
    assert result['candidate_tensor_comparisons'] == 2
    assert result['cases'][0]['source_self_repeat']['bitwise_identical']
    assert result['cases'][0]['candidate']['non_tensor_leaves'] == 2


def test_unavailable_device_never_calls_provider(monkeypatch):
    monkeypatch.setattr(runner, 'device_available', lambda device: False)
    def forbidden(*args): raise AssertionError('No model factory should run')
    result = runner.run_backend(forbidden, device='cuda')
    assert result['status'] == 'unavailable'
    assert not result['accepted'] and not result['cases']


@pytest.mark.parametrize('candidate', [lambda: Output(label='changed'), lambda: Output(sign=-1.)])
def test_changed_complete_leaf_fails(candidate):
    result = runner.run_backend(provider(cand=candidate))
    assert result['status'] == 'failed' and not result['accepted']


def test_signed_zero_is_numeric_equal_but_not_byte_equal():
    class Zeros(nn.Module):
        def __init__(self, negative=False):
            super().__init__(); self.negative = negative
        def forward(self):
            return torch.tensor([-0. if self.negative else 0.])
    build = provider(lambda: Zeros(), lambda: Zeros(True))
    loose = runner.run_backend(build)
    assert loose['accepted_tolerance'] and loose['accepted'] and not loose['bitwise_identical']
    strict = runner.run_backend(build, require_bitwise=True)
    assert strict['accepted_tolerance'] and not strict['accepted']


def test_self_repeat_drift_is_not_hidden_by_candidate_match():
    class Counter(nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer('value', torch.zeros(1))
        def forward(self):
            self.value.add_(1)
            return self.value  # Snapshot must own bytes before the next mutation.
    result = runner.run_backend(provider(Counter, Counter), atol=0., rtol=0.)
    assert result['cases'][0]['candidate']['bitwise_identical']
    assert not result['cases'][0]['source_self_repeat']['accepted_tolerance']
    assert not result['accepted']


def test_nonfinite_and_empty_outputs_fail():
    class Bad(nn.Module):
        def forward(self): return {'x': torch.tensor([float('nan')])}
    class Empty(nn.Module):
        def forward(self): return {'no_tensor': True}
    for cls in (Bad, Empty):
        result = runner.run_backend(provider(cls, cls))
        assert not result['accepted'] and result['status'] == 'failed'


def test_shape_and_discrete_changes_fail():
    a = runner.snapshot_output({'a': torch.tensor([1, 2])})
    for value in (torch.tensor([1, 3]), torch.tensor([[1, 2]])):
        with pytest.raises(ValueError):
            runner.compare_snapshots(a, runner.snapshot_output({'a': value}), 1., 1.)


def test_wrong_device_rejected():
    suite = lambda d, options: runner.BackendSuite(lambda d: nn.Linear(1, 1, device='meta'),
           lambda d: nn.Linear(1, 1), lambda model: [])
    report = runner.run_backend(suite)
    assert not report['accepted'] and 'requested device' in report['error']


def test_settings_restored_even_after_failure():
    before = runner.settings()
    runner.run_backend(lambda d, options: (_ for _ in ()).throw(ValueError('factory failure')),
                       determinism='strict', tf32=False)
    assert runner.settings() == before


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_invalid_tolerances_are_explicit_valid_json_failures(value):
    result = runner.run_backend(provider(), atol=value)
    assert not result['accepted'] and result['status'] == 'failed'
    json.dumps(result, allow_nan=False)


def test_inventory_is_not_a_model_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'device_available', lambda d: False)
    target = tmp_path / 'missing.json'
    assert runner.main(['--device', 'cuda', '--probe-only', '--output', str(target)]) == 2
    report = json.loads(target.read_text())
    assert report['status'] == 'unavailable' and not report['accepted']
    assert not report['model_validation_ran']
    with pytest.raises(SystemExit):
        runner.main(['--device', 'cuda', '--probe-only', '--output', str(target)])


def test_custom_provider_cli(tmp_path):
    source = tmp_path / 'provider.py'
    source.write_text('''from validate_backends import BackendCase, BackendSuite
from torch import nn
import torch
def build(device, options):
    return BackendSuite(lambda d: nn.Identity().to(d), lambda d: nn.Identity().to(d),
                        lambda m: [BackendCase("identity", lambda m,d: m(torch.ones(3,device=d)))])
''')
    report = tmp_path / 'result.json'
    assert runner.main(['--device', 'cpu', '--provider', str(source) + ':build',
                        '--require-bitwise', '--output', str(report)]) == 0
    assert json.loads(report.read_text())['accepted']


def test_gpu_synchronization_dispatch_is_explicit(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda d: calls.append(('cuda', str(d))))
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: calls.append(('mps', None)))
    runner.synchronize(torch.device('cpu'))
    runner.synchronize(torch.device('cuda:2'))
    runner.synchronize(torch.device('mps'))
    assert calls == [('cuda', 'cuda:2'), ('mps', None)]


def test_cuda_index_guard_with_mocked_inventory(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert runner.device_available('cuda') and runner.device_available('cuda:1')
    assert not runner.device_available('cuda:2')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_packed_host_buffers_are_narrowly_allowed():
    from compressme.packed_embedding import PackedFrozenEmbedding
    source = nn.Embedding(12, 4).eval().requires_grad_(False)
    packed = PackedFrozenEmbedding(source).to('mps')
    model = nn.Sequential(packed, nn.Linear(4, 2).to('mps'))
    info = runner.model_info(model, torch.device('mps'))
    assert info['permitted_cpu_codec_buffers'] == ['0.offsets', '0.payload']
    assert info['registered_storage_bytes_all_devices'] >= packed.payload.numel()
    model.register_buffer('ordinary_cpu', torch.ones(1))
    with pytest.raises(ValueError, match='not on requested device'):
        runner.model_info(model, torch.device('mps'))
    del model.ordinary_cpu
    model.register_buffer('ordinary_alias', packed.payload)
    with pytest.raises(ValueError, match='not on requested device'):
        runner.model_info(model, torch.device('mps'))
    del model.ordinary_alias
    packed._anchor = torch.empty(0)
    with pytest.raises(ValueError, match='not on requested device'):
        runner.model_info(model, torch.device('mps'))
