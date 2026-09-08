import sys
from types import SimpleNamespace
import copy
import hashlib
import json
import pickle
import torch
from torch import nn
import pytest
from safetensors.torch import save_file
from compressme.validation import Example
import compressme.hub as hub

SHA = '1234567890abcdef1234567890abcdef12345678'


class Small(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.expand = nn.Linear(3, 17, dtype=dtype)
        self.read = nn.Linear(17, 7, dtype=dtype)
    def forward(self, x):
        return {'embedding': self.read(self.expand(x))}


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)
        self.b.weight = self.a.weight
    def forward(self, x):
        return self.a(x) + self.b(x)


class UnsupportedObject:
    pass


class FrozenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 7, padding_idx=0)
        self.embedding.weight.requires_grad_(False)
    def forward(self, tokens):
        return self.embedding(tokens)


class MockHub:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.calls = []
        self.info = None
        monkeypatch.setattr(hub, '_hub', lambda: SimpleNamespace(HfApi=lambda: self,
                                                               hf_hub_download=self.download))
    def pin(self):
        files = []
        for path in self.root.rglob('*'):
            if path.is_file():
                data = path.read_bytes()
                files.append(SimpleNamespace(rfilename=path.relative_to(self.root).as_posix(),
                    size=len(data), lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())))
        self.info = SimpleNamespace(sha=SHA, siblings=files)
    def model_info(self, repo_id, *, revision, files_metadata, token):
        assert repo_id == 'test/model' and files_metadata is True
        self.calls.append(('info', revision))
        return self.info
    def download(self, *, repo_id, filename, revision, **kwargs):
        assert repo_id == 'test/model' and revision == SHA
        assert not filename.endswith('.py')
        self.calls.append(('download', filename, revision))
        return str(self.root / filename)


def example():
    return [Example(args=(torch.randn(5, 3),))]


def write_state(root, state, filename='model.safetensors'):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.clone() for key, value in state.items()}, str(path))
    return path


def test_inspect_is_pinned_and_does_not_download_weights_or_code(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    write_state(tmp_path, Small().state_dict())
    (tmp_path / 'modeling_custom.py').write_text('raise RuntimeError("do not execute")')
    network.pin()
    report = hub.inspect_huggingface('test/model', revision='release')
    assert report['revision'] == SHA and report['selected'] == 'model.safetensors'
    assert report['remote_code_executed'] is False
    assert network.calls == [('info', 'release')]


def test_single_checkpoint_exact_compression_with_provenance(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    original = Small().eval()
    write_state(tmp_path, original.state_dict()); network.pin()
    samples = example()
    result = hub.compress_huggingface('test/model', Small, validation=samples)
    assert result.report['parameters_after'] < result.report['parameters_before']
    assert result.report['validation']['accepted']
    torch.testing.assert_close(samples[0].call(original)['embedding'], samples[0].call(result.model)['embedding'])
    provenance = result.report['huggingface']
    assert provenance['revision'] == SHA and len(provenance['files'][0]['sha256']) == 64
    assert network.calls == [('info', 'main'), ('download', 'model.safetensors', SHA)]


@pytest.mark.parametrize('subfolder', ['', 'sub/'])
def test_sharded_index_loading(tmp_path, monkeypatch, subfolder):
    network = MockHub(tmp_path, monkeypatch)
    state = Small().state_dict()
    part1 = {k: v for k, v in state.items() if k.startswith('expand')}
    part2 = {k: v for k, v in state.items() if k.startswith('read')}
    write_state(tmp_path, part1, subfolder + 'one.safetensors')
    write_state(tmp_path, part2, subfolder + 'two.safetensors')
    index_name = subfolder + 'model.safetensors.index.json'
    (tmp_path / index_name).write_text(json.dumps({'weight_map': {
        **{k: 'one.safetensors' for k in part1}, **{k: 'two.safetensors' for k in part2}}}))
    network.pin()
    report = hub.inspect_huggingface('test/model')
    assert report['candidates'] == [index_name]
    assert all(call[1].endswith('.json') for call in network.calls if call[0] == 'download')
    result = hub.compress_huggingface('test/model', Small, validation=example())
    assert result.report['validation']['accepted']
    assert len(result.report['huggingface']['files']) == 3


def test_ambiguous_weight_files_require_explicit_name(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    write_state(tmp_path, Small().state_dict(), 'a.safetensors')
    write_state(tmp_path, Small().state_dict(), 'b.safetensors'); network.pin()
    assert hub.inspect_huggingface('test/model')['selected'] is None
    with pytest.raises(ValueError, match='Choose filename explicitly'):
        hub.compress_huggingface('test/model', Small, validation=example())
    assert not any(c[0] == 'download' for c in network.calls)
    assert hub.compress_huggingface('test/model', Small, filename='b.safetensors', validation=example()).report['validation']['accepted']


@pytest.mark.parametrize('damage,match', [('missing', 'keys mismatch'), ('extra', 'keys mismatch'),
                                       ('dtype', 'shape/dtype mismatch'), ('shape', 'shape/dtype mismatch')])
def test_strict_state_rejections(tmp_path, monkeypatch, damage, match):
    network = MockHub(tmp_path, monkeypatch)
    state = dict(Small().state_dict())
    if damage == 'missing': del state['read.bias']
    if damage == 'extra': state['alien'] = torch.zeros(1)
    if damage == 'dtype': state['read.bias'] = state['read.bias'].double()
    if damage == 'shape': state['read.bias'] = torch.zeros(5)
    write_state(tmp_path, state); network.pin()
    with pytest.raises(ValueError, match=match):
        hub.compress_huggingface('test/model', Small, validation=example())


def test_ties_fill_only_declared_alias_and_preserve_identity(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    state = Tied().state_dict(); del state['b.weight']
    write_state(tmp_path, state); network.pin()
    result = hub.compress_huggingface('test/model', Tied, validation=example())
    assert result.model.a.weight is result.model.b.weight
    assert result.report['huggingface']['loading']['filled_declared_tied_keys'] == {'b.weight': 'a.weight'}


def test_ties_conflicting_values_refused(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    state = dict(Tied().state_dict()); state['b.weight'] = state['b.weight'] + 1
    write_state(tmp_path, state); network.pin()
    with pytest.raises(ValueError, match='Conflicting checkpoint'):
        hub.compress_huggingface('test/model', Tied, validation=example())


def test_explicit_ckpt_only_and_weights_only_loader(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    torch.save({'state_dict': Small().state_dict(), 'epoch': 2}, tmp_path / 'weights.ckpt')
    network.pin()
    assert hub.inspect_huggingface('test/model')['selected'] is None
    result = hub.compress_huggingface('test/model', Small, filename='weights.ckpt', validation=example())
    assert result.report['validation']['accepted']
    torch.save({'state_dict': Small().state_dict(), 'unsafe': UnsupportedObject()}, tmp_path / 'weights.ckpt')
    network.pin()
    with pytest.raises(pickle.UnpicklingError):
        hub.compress_huggingface('test/model', Small, filename='weights.ckpt', validation=example())


def test_hash_mismatch_refuses_modified_file(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    path = write_state(tmp_path, Small().state_dict()); network.pin()
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='hash differs'):
        hub.compress_huggingface('test/model', Small, validation=example())


@pytest.mark.parametrize('kind', ['traversal', 'wrong_shard', 'missing_tensor'])
def test_bad_index_refused(tmp_path, monkeypatch, kind):
    network = MockHub(tmp_path, monkeypatch)
    state = Small().state_dict(); write_state(tmp_path, state, 'shard.safetensors')
    mapping = {key: 'shard.safetensors' for key in state}
    if kind == 'traversal': mapping['read.weight'] = '../other.safetensors'
    if kind == 'wrong_shard':
        write_state(tmp_path, {'alien': torch.ones(1)}, 'extra.safetensors')
        mapping['read.weight'] = 'extra.safetensors'
    if kind == 'missing_tensor': mapping['missing'] = 'shard.safetensors'
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping})); network.pin()
    with pytest.raises(ValueError):
        hub.compress_huggingface('test/model', Small, validation=example(), filename='model.safetensors.index.json')


def test_validation_mandatory_before_network(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    for validation in (None, [], [torch.zeros(1)]):
        with pytest.raises(ValueError):
            hub.compress_huggingface('test/model', Small, validation=validation)
    assert network.calls == []


def test_failed_output_gate_returns_original_weights(tmp_path, monkeypatch):
    import compressme.affine
    network = MockHub(tmp_path, monkeypatch)
    original = Small().eval(); write_state(tmp_path, original.state_dict()); network.pin()
    compile_original = compressme.affine.compile_affine
    def incorrect(model):
        result = compile_original(model)
        with torch.no_grad(): next(result.model.parameters()).add_(2)
        return result
    monkeypatch.setattr(compressme.affine, 'compile_affine', incorrect)
    result = hub.compress_huggingface('test/model', Small, validation=example())
    assert result.report['status'] == 'rejected_and_rolled_back'
    assert not result.report['validation']['accepted']
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, result.model.state_dict()[name], rtol=0, atol=0)


def test_factory_dtype_is_explicit_and_linear_still_validates(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    original = nn.Linear(3, 5, dtype=torch.float64)
    write_state(tmp_path, original.state_dict()); network.pin()
    samples = [Example(args=(torch.randn(2, 3, dtype=torch.float64),))]
    result = hub.compress_huggingface('test/model', lambda: nn.Linear(3, 5, dtype=torch.float64), validation=samples)
    assert result.report['validation']['examples'] == 1
    assert result.report['status'] == 'accepted_on_validation_examples'


def test_distinct_shared_storage_is_refused():
    model = Small()
    model.read.bias = nn.Parameter(model.expand.bias[:7])
    with pytest.raises(ValueError, match='share storage'):
        hub._strict_load(model, {key: value.clone() for key, value in model.state_dict().items()})


def test_duplicate_json_index_keys_refused(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    write_state(tmp_path, Small().state_dict(), 'shard.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(
        '{"weight_map":{"a":"shard.safetensors","a":"shard.safetensors"}}')
    network.pin()
    with pytest.raises(ValueError, match='Duplicate key'):
        hub.inspect_huggingface('test/model')


def test_tied_signed_zero_conflicts_are_refused():
    model = Tied()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state['a.weight'][0, 0] = 0.0
    state['b.weight'][0, 0] = -0.0
    with pytest.raises(ValueError, match='Conflicting checkpoint'):
        hub._strict_load(model, state)


def test_loaded_signed_zero_bits_are_checked():
    class ChangesZero(nn.Linear):
        def load_state_dict(self, state, **kwargs):
            result = super().load_state_dict(state, **kwargs)
            with torch.no_grad(): self.weight.abs_()
            return result
    model = ChangesZero(1, 1, bias=False)
    with pytest.raises(ValueError, match='modified checkpoint tensor'):
        hub._strict_load(model, {'weight': torch.tensor([[-0.0]])})
    assert hub._same_tensor_bits(torch.tensor(0.0), torch.tensor(0.0))
    assert not hub._same_tensor_bits(torch.tensor(0.0), torch.tensor(-0.0))


def test_constant_embeddings_hub_pass_and_serialization(tmp_path, monkeypatch):
    from compressme.constant_embeddings import ConstantRowEmbedding
    from compressme.serialization import load
    network = MockHub(tmp_path, monkeypatch)
    original = FrozenEmbedding().eval()
    with torch.no_grad():
        row = torch.tensor([[0.0, -0.0, 1.5, -2.0, 3.0, -4.0, 0.125]])
        original.embedding.weight.copy_(row.expand_as(original.embedding.weight))
    write_state(tmp_path, original.state_dict()); network.pin()
    samples = [Example(args=(torch.tensor([[0, 1, 22], [5, 9, 10]]),))]
    result = hub.compress_huggingface('test/model', FrozenEmbedding, method='constant_embeddings', validation=samples)
    assert result.report['validation']['accepted']
    assert result.report['parameters_before'] == 161 and result.report['parameters_after'] == 7
    assert isinstance(result.model.embedding, ConstantRowEmbedding)
    assert hub._same_tensor_bits(samples[0].call(original), samples[0].call(result.model))
    export = tmp_path / 'artifact'
    result.save(export)
    replay = load(FrozenEmbedding, export)
    assert replay.report['huggingface']['revision'] == SHA
    assert isinstance(replay.model.embedding, ConstantRowEmbedding)
    assert hub._same_tensor_bits(samples[0].call(result.model), samples[0].call(replay.model))


def test_constant_embedding_method_still_rolls_back_failed_gate(tmp_path, monkeypatch):
    import compressme.constant_embeddings
    network = MockHub(tmp_path, monkeypatch)
    original = FrozenEmbedding().eval()
    with torch.no_grad(): original.embedding.weight.zero_()
    write_state(tmp_path, original.state_dict()); network.pin()
    real = compressme.constant_embeddings.deduplicate_embeddings
    def broken(model):
        candidate, report = real(model)
        with torch.no_grad(): candidate.embedding._row.add_(2)
        return candidate, report
    monkeypatch.setattr(compressme.constant_embeddings, 'deduplicate_embeddings', broken)
    result = hub.compress_huggingface('test/model', FrozenEmbedding, method='constant_embeddings',
                                    validation=[Example(args=(torch.tensor([0, 22]),))])
    assert result.report['status'] == 'rejected_and_rolled_back'
    assert type(result.model.embedding) is nn.Embedding
    assert hub._same_tensor_bits(result.model.embedding.weight, original.embedding.weight)


class ClosedTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Sequential(nn.Embedding(23, 32), nn.Linear(32, 7), nn.Tanh())
        self.readout = nn.Linear(7, 2)
    def forward(self, ids):
        rows = self.tokens(ids)
        return {"rows": rows, "readout": self.readout(rows), "ids": ids}


def test_finite_block_hub_route_and_portable_replay(tmp_path, monkeypatch):
    from compressme import load
    from compressme.finite_lookup import FiniteTokenLookup
    network = MockHub(tmp_path, monkeypatch)
    original = ClosedTokenModel().eval()
    write_state(tmp_path, original.state_dict()); network.pin()
    samples = [Example((torch.arange(23),))]
    result = hub.compress_huggingface('test/model', ClosedTokenModel, method='finite_lookup',
        finite_input_contract='token_indices_only', finite_paths=['tokens'], validation=samples)
    assert result.report['validation']['accepted']
    assert type(result.model) is ClosedTokenModel
    assert type(result.model.tokens) is FiniteTokenLookup
    result.save(tmp_path/'export')
    restored = load(ClosedTokenModel, tmp_path/'export').model
    for key, value in result.model(torch.arange(23)).items():
        assert torch.equal(value, restored(torch.arange(23))[key])
    assert result.report['huggingface']['revision'] == SHA


def test_hub_finite_contract_required_before_network(tmp_path, monkeypatch):
    network = MockHub(tmp_path, monkeypatch)
    for kwargs in ({'method': 'finite_lookup'},
                   {'method':'finite_lookup','finite_input_contract':'token_indices_only','finite_paths':'tokens'},
                   {'method':'affine','finite_input_contract':'token_indices_only'}):
        with pytest.raises((ValueError, TypeError)):
            hub.compress_huggingface('test/model', ClosedTokenModel,
                validation=[Example((torch.arange(23),))], **kwargs)
    assert network.calls == []
