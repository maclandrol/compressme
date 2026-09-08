"""Lightweight Nesso script protocol checks; no pretrained model execution."""
import pytest
import torch


@pytest.mark.parametrize('mode,accepted', [('paired', True), ('original-only', False), ('final-only', False)])
def test_benchmark_single_path_is_not_mislabelled_as_comparison(tmp_path, monkeypatch, mode, accepted):
    pytest.importorskip('nesso')
    pytest.importorskip('transformers')
    import importlib.util, json, sys
    from pathlib import Path
    from transformers import EsmConfig, EsmForMaskedLM
    path = Path(__file__).resolve().parents[1] / 'examples/nesso/benchmark_esm.py'
    spec = importlib.util.spec_from_file_location('_esm_benchmark_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_dir = tmp_path/'model'; model_dir.mkdir()
    (model_dir/'model.safetensors').write_bytes(b'constructed fake checkpoint')
    (model_dir/'config.json').write_text('{}')
    sequence = tmp_path/'sample.yaml'
    sequence.write_text('sequences:\n- protein:\n    id: A\n    sequence: AC\n')
    real_hash = module.sha256
    monkeypatch.setattr(module, 'sha256', lambda p: module.PINNED_WEIGHTS_SHA256 if Path(p).name=='model.safetensors' else real_hash(p))
    def tiny_model(*args, **kwargs):
        config = EsmConfig(vocab_size=33, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=16, pad_token_id=1, mask_token_id=32,
                           position_embedding_type='rotary', token_dropout=True)
        return EsmForMaskedLM(config).eval()
    class Tokenizer:
        def __call__(self, sequence, return_tensors='pt'):
            return {'input_ids':torch.tensor([[0,4,5,2]]), 'attention_mask':torch.ones(1,4,dtype=torch.long)}
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    monkeypatch.setattr(AutoModelForMaskedLM, 'from_pretrained', tiny_model)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: Tokenizer())
    output = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', [str(path), '--model-dir', str(model_dir), '--sequence-yaml', str(sequence),
                                    '--device', 'cpu', '--mode', mode, '--rounds', '1', '--warmup', '0',
                                    '--threads', '1', '--output', str(output)])
    assert module.main()==0
    result = json.loads(output.read_text())
    assert result['accepted'] is accepted
    assert result['original_vs_final_comparison_ran'] is accepted
    assert result['pretrained_embedding_gate_passed'] is accepted
    assert result['full_nesso_validation'] is False


@pytest.mark.parametrize('script', ['download.py', 'prepare.py', 'benchmark.py', 'benchmark_esm.py'])
def test_script_help_needs_no_optional_packages(script):
    import subprocess
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples' / 'nesso' / script
    result = subprocess.run([sys.executable, '-S', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_preparation_case_listing_needs_no_optional_packages():
    import json
    import subprocess
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples/nesso/prepare.py'
    result = subprocess.run([sys.executable, '-S', str(path), '--list-cases'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cases = json.loads(result.stdout)
    assert {name: case['protein_residues'] for name, case in cases.items()} == {
        'tiny20': 20, 'fragment130': 130, 'tutorial384': 384}


@pytest.mark.parametrize('script,required', [
    ('benchmark.py', ['--checkpoint', 'missing', '--fixtures', 'missing']),
    ('benchmark_esm.py', ['--model-dir', 'missing', '--sequence-yaml', 'missing']),
])
def test_benchmark_preserves_existing_report_before_model_imports(tmp_path, script, required):
    import subprocess
    import sys
    from pathlib import Path
    output = tmp_path / 'existing.json'
    output.write_bytes(b'preserved earlier evidence')
    path = Path(__file__).resolve().parents[1] / 'examples/nesso' / script
    result = subprocess.run([sys.executable, '-S', str(path), *required, '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Output exists' in result.stderr
    assert output.read_bytes() == b'preserved earlier evidence'


def test_native_benchmark_requires_local_checkpoint_before_optional_imports(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples/nesso/benchmark.py'
    output = tmp_path / 'result.json'
    result = subprocess.run([sys.executable, '-S', str(path), '--checkpoint', str(tmp_path / 'missing'),
                             '--fixtures', 'missing', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'does not download weights' in result.stderr
    assert not output.exists()


@pytest.fixture
def fake_native_benchmark(tmp_path, monkeypatch):
    """Exercise the real report/control flow with a deliberately tiny native stub."""
    import importlib.metadata
    import importlib.util
    import sys
    from contextlib import contextmanager
    from pathlib import Path
    from types import ModuleType
    import compressme.nesso_runtime as runtime
    path = Path(__file__).resolve().parents[1] / 'examples/nesso/benchmark.py'
    spec = importlib.util.spec_from_file_location('_nesso_benchmark_script_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'nesso'
    root.mkdir()
    (root / '__init__.py').write_text('# Constructed source fixture; no pretrained model.\n')
    for name in ('nesso', 'nesso.model', 'nesso.model.models', 'nesso.model.models.nesso1'):
        stub = ModuleType(name)
        stub.__file__ = str(root / '__init__.py')
        stub.__path__ = [str(root)]
        monkeypatch.setitem(sys.modules, name, stub)
    flags_seen = []
    state = {'fault': None}

    class FakeNesso(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.predict_args = {}
            self.candidate = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def forward(self, batch, **kwargs):
            value = self.weight + batch['x']
            metadata = {'distogram': torch.eye(2), 'indices': torch.tensor([0, 1]), 'label': 'kept'}
            output = {'affinity': value, 'metadata': metadata, 'exception': False}
            if self.candidate:
                fault = state['fault']
                if fault == 'shape': output['affinity'] = value.repeat(2)
                elif fault == 'dtype': output['affinity'] = value.double()
                elif fault == 'nonfinite': output['affinity'] = value * float('nan')
                elif fault == 'non_tensor': metadata['label'] = 'changed'
                elif fault == 'structure': del metadata['indices']
                elif fault == 'numerical': output['affinity'] = value + 0.25
            return output

        def predict_step(self, batch, index):
            return self(batch)

    monkeypatch.setattr(sys.modules['nesso.model.models.nesso1'], 'Nesso1', FakeNesso, raising=False)
    prepare = ModuleType('prepare')
    prepare.load_batch = lambda *a, **kw: {'x': torch.tensor([2.0]), 'token_pad_mask': torch.ones(1, 3)}
    monkeypatch.setitem(sys.modules, 'prepare', prepare)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'constructed-fixture' if name == 'nesso' else original_version(name))

    @contextmanager
    def fake_context(model, **flags):
        flags_seen.append(flags)
        model.candidate = True
        try:
            yield {'constructed_context': True}
        finally:
            model.candidate = False

    monkeypatch.setattr(runtime, 'nesso_inference_optimizations', fake_context)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'constructed tensor fixture')
    (checkpoint / 'hparams.json').write_text('{}')
    fixtures = tmp_path / 'fixtures'
    case = fixtures / 'tiny20'
    case.mkdir(parents=True)
    (case / 'batch.safetensors').write_bytes(b'constructed prepared tensor fixture')
    (case / 'batch.json').write_text('{"constructed_metadata": true}')
    output = tmp_path / 'result.json'

    def invoke(variants, fault=None):
        state['fault'] = fault
        monkeypatch.setattr(sys, 'argv', [str(path), '--checkpoint', str(checkpoint), '--fixtures', str(fixtures),
                                        '--cases', 'tiny20', '--variants', *variants, '--device', 'cpu',
                                        '--threads', '1', '--rounds', '1', '--output', str(output)])
        return module.main()

    return module, invoke, output, flags_seen, case


def test_native_benchmark_preserves_metadata_flags_and_reproducibility_hashes(fake_native_benchmark):
    import json
    module, invoke, output, flags_seen, case = fake_native_benchmark
    variants = ['guards', 'copy', 'conditioning', 'both', 'packed', 'all']
    invoke(variants)
    report = json.loads(output.read_text())
    assert report['status'] == 'accepted_byte_identical'
    assert report['cases'][0]['batch_metadata_sha256'] == module.sha256(case / 'batch.json')
    assert len(report['preparation_script_sha256']) == len(report['validation_script_sha256']) == 64
    assert report['parameter_dtypes'] == ['torch.float32']
    assert 'per-call adapter construction' in report['timing_scope']
    assert 'not a per-variant comparison' in report['peak_rss_scope']
    observed = {(f['single_chunk'], f['cache_esm'], f['pack_triangles']) for f in flags_seen}
    assert observed == {(False, False, False), (True, False, False), (False, True, False),
                        (True, True, False), (False, False, True), (True, True, True)}
    assert all(f['immutable_request'] is True for f in flags_seen)
    for result in report['cases'][0]['variants'].values():
        assert result['forward']['tensor_leaves'] == result['predict_step']['tensor_leaves'] == 3
        assert all(p['reference_gate']['bitwise_identical'] and p['candidate_gate']['bitwise_identical']
                   for p in result['pairs'])


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'nonfinite', 'non_tensor', 'structure'])
def test_native_benchmark_saves_validation_exceptions_then_reraises(fake_native_benchmark, fault):
    import json
    _, invoke, output, _, _ = fake_native_benchmark
    with pytest.raises(ValueError):
        invoke(['packed'], fault=fault)
    report = json.loads(output.read_text())
    assert report['status'] == 'validation_exception'
    assert report['failure']['case'] == 'tiny20'
    assert report['failure']['variant'] == 'packed'
    assert report['failure']['phase'] == 'candidate forward and predict validation'
    assert report['failure']['type'] == 'ValueError'
    assert report['failure']['message']
    assert report['cases'][0]['reference_predict_self_repeat']['bitwise_identical']
    assert report['cases'][0]['variants']['packed']['pairs'] == []


def test_native_benchmark_numerical_rejection_has_no_accepted_timing(fake_native_benchmark):
    import json
    _, invoke, output, _, _ = fake_native_benchmark
    with pytest.raises(SystemExit) as failure:
        invoke(['packed'], fault='numerical')
    assert failure.value.code == 2
    report = json.loads(output.read_text())
    assert report['status'] == 'candidate_rejected'
    result = report['cases'][0]['variants']['packed']
    assert result['status'] == 'rejected_output_mismatch'
    assert result['pairs'] == []
    assert result['predict_step']['max_abs'] == 0.25
