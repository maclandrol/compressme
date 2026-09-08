import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from compressme import Example, compile_finite_blocks, load
from compressme.finite_fanout import FiniteTokenFanout, FiniteFanoutLookup, Float32RMSNorm
from compressme.finite_lookup import FiniteTokenLookup


def fanout(dtype=torch.float64):
    return FiniteTokenFanout(
        nn.Embedding(7, 32, dtype=dtype),
        nn.Sequential(Float32RMSNorm(torch.ones(32, dtype=dtype), eps=1e-6)),
        {"query": nn.Linear(32, 12, dtype=dtype),
         "key": nn.Linear(32, 12, dtype=dtype),
         "value": nn.Linear(32, 8, dtype=dtype)},
        residual_key="residual",
    )


class FanoutModel(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.stem = fanout(dtype)
        self.readout = nn.Linear(12, 3, dtype=dtype)
        self.other = nn.Sequential(nn.Embedding(7, 24, dtype=dtype),
                                   nn.Linear(24, 5, dtype=dtype), nn.SiLU())

    def forward(self, ids):
        branches = self.stem(ids)
        return {"branches": branches,
                "score": self.readout(branches["query"] + branches["key"]),
                "other": self.other(ids), "ids": ids, "metadata": "complete"}

    def helper(self, ids):
        return self.forward(ids)


def examples():
    return [Example((torch.arange(7),)),
            Example((torch.tensor([[0, 6, 2], [4, 4, 1]], dtype=torch.int32),)),
            Example((torch.tensor(5),)),
            Example((torch.empty((2, 0), dtype=torch.long),))]


def assert_tree_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_tree_same(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize("paths", [None, [""]])
def test_root_fanout_keeps_every_named_output_and_residual_then_reloads(tmp_path, paths):
    source = fanout().eval()
    source_bytes = sum(t.numel() * t.element_size() for t in source.state_dict().values())
    result = compile_finite_blocks(source, paths=paths, input_contract="token_indices_only",
                                   validation=examples())
    assert type(result.model) is FiniteFanoutLookup
    assert result.report["rewrites"] == [""]
    assert result.report["status"] == "accepted_on_validation_examples"
    assert result.report["blocks"][0]["residual_embedding_values_included"]
    assert result.report["storage_saving"]
    assert result.report["tensor_bytes_after"] < source_bytes
    assert result.report["resident_storage_bytes_after"] < result.report["resident_storage_bytes_before"]
    for sample in examples():
        before, after = sample.call(source), sample.call(result.model)
        assert set(after) == {"query", "key", "value", "residual"}
        torch.testing.assert_close(before, after)
        assert torch.equal(before["residual"], after["residual"])
    result.save(tmp_path, packing=True)
    replay = load(fanout, tmp_path).model
    assert type(replay) is FiniteFanoutLookup
    for sample in examples():
        assert_tree_same(sample.call(result.model), sample.call(replay))


@pytest.mark.parametrize("paths,rewrites", [(None, ["stem", "other"]), (["stem"], ["stem"])])
def test_nested_fanout_and_sequential_mix_preserves_enclosing_api_and_replay(tmp_path, paths, rewrites):
    source = FanoutModel().eval()
    result = compile_finite_blocks(source, paths=paths, input_contract="token_indices_only",
                                   validation=examples())
    assert type(result.model) is FanoutModel
    assert type(result.model.stem) is FiniteFanoutLookup
    assert type(source.stem) is FiniteTokenFanout
    assert result.report["rewrites"] == rewrites
    if paths is None:
        assert type(result.model.other) is FiniteTokenLookup
        assert result.report["selection"] == "registered_finite_block_discovery"
    else:
        assert type(result.model.other) is nn.Sequential
        assert result.report["selection"] == "explicit_paths"
    assert result.report["validation"]["accepted"]
    result.save(tmp_path, packing=True)
    replay = load(FanoutModel, tmp_path).model
    assert type(replay.stem) is FiniteFanoutLookup
    for sample in examples():
        assert_tree_same(sample.call(result.model), replay.helper(sample.args[0]))


@pytest.mark.parametrize("alias", ["module", "parameter", "buffer_view"])
def test_fanout_external_owner_alias_is_audited_before_deepcopy(alias):
    source = FanoutModel().eval()
    if alias == "module":
        source.external = source.stem.embedding
    elif alias == "parameter":
        source.external = nn.Parameter(source.stem.embedding.weight.detach())
    else:
        source.register_buffer("external", source.stem.embedding.weight.detach()[:2])
    result = compile_finite_blocks(source, paths=["stem"], input_contract="token_indices_only",
                                   validation=examples())
    assert result.report["rewrites"] == []
    assert result.report["status"] == "retained_no_accepted_rewrites"
    assert result.report["blocks"][0]["status"] == "retained_unsupported"
    assert "external" in result.report["blocks"][0]["reason"].lower()
    assert type(result.model.stem) is FiniteTokenFanout
    if alias == "module":
        assert result.model.external is result.model.stem.embedding


def test_bad_fanout_candidate_rolls_back_every_rewrite(monkeypatch):
    import compressme.finite_blocks as implementation
    original = implementation.compile_finite_fanout
    source = FanoutModel().eval()
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.report["status"] == "accepted_on_full_domain_validation"
        with torch.no_grad():
            next(iter(result.model.tables.values())).add_(0.5)
        return result
    monkeypatch.setattr(implementation, "compile_finite_fanout", damaged)
    result = compile_finite_blocks(source, input_contract="token_indices_only", validation=examples())
    assert result.report["status"] == "rejected_and_rolled_back"
    assert result.report["rewrites"] == []
    assert result.report["proposed_rewrites"] == ["stem", "other"]
    assert not result.report["validation"]["accepted"]
    assert type(result.model.stem) is FiniteTokenFanout
    assert type(result.model.other) is nn.Sequential
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, result.model.state_dict()[name])


def test_unprofitable_fanout_is_retained_with_full_output_gate():
    source = FiniteTokenFanout(nn.Embedding(101, 3), nn.Sequential(),
                              {"wide": nn.Linear(3, 20)}, residual_key="residual").eval()
    result = compile_finite_blocks(source, input_contract="token_indices_only",
                                   validation=[Example((torch.arange(101),))])
    assert result.report["status"] == "retained_no_accepted_rewrites"
    assert result.report["blocks"][0]["status"] == "no_storage_saving"
    assert result.report["validation"]["accepted"]
    assert type(result.model) is FiniteTokenFanout


def test_sequential_report_label_stays_compatible():
    source = nn.Sequential(nn.Embedding(7, 24), nn.Linear(24, 5)).eval()
    result = compile_finite_blocks(source, input_contract="token_indices_only", validation=examples())
    assert result.report["selection"] == "registered_sequential_discovery"
    assert type(result.model) is FiniteTokenLookup


@pytest.mark.parametrize('profiles', [None, [{'max_rows': 1, 'evaluation_rows': 1}]])
def test_hub_finite_lookup_discovers_fanout_via_local_factory(tmp_path, monkeypatch, profiles):
    import compressme.hub as hub
    source = FanoutModel().eval()
    weights = tmp_path / "model.safetensors"
    save_file({name: tensor.clone() for name, tensor in source.state_dict().items()}, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    revision = "a" * 40
    calls = []
    def info(repo_id, *, revision, files_metadata, token):
        calls.append(("metadata", repo_id, revision))
        return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(
            rfilename="model.safetensors", size=weights.stat().st_size,
            lfs=SimpleNamespace(sha256=digest))])
    def download(*, repo_id, filename, revision, **kwargs):
        calls.append(("download", filename, revision))
        assert filename == "model.safetensors" and revision == "a" * 40
        return str(weights)
    monkeypatch.setattr(hub, "_hub", lambda: SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=info), hf_hub_download=download))
    result = hub.compress_huggingface("test/fanout", FanoutModel,
        method="finite_lookup", finite_input_contract="token_indices_only", validation=examples(),
        finite_row_count_profiles=profiles)
    assert type(result.model.stem) is FiniteFanoutLookup
    assert result.report["rewrites"] == ["stem", "other"]
    assert result.report["validation"]["accepted"]
    assert result.report["huggingface"]["revision"] == revision
    assert result.report["huggingface"]["remote_code_executed"] is False
    assert result.report['fanout_row_count_profiles'] == (profiles or [])
    assert result.model.stem.recipe()['version'] == (2 if profiles else 1)
    assert calls == [("metadata", "test/fanout", "main"),
                     ("download", "model.safetensors", revision)]
    result.save(tmp_path / "compressed", packing=True)
    replay = load(FanoutModel, tmp_path / "compressed").model
    for sample in examples():
        assert_tree_same(sample.call(result.model), sample.call(replay))


@pytest.mark.parametrize('method,profiles', [
    ('finite_lookup', [{'max_rows': 0, 'evaluation_rows': 1}]),
    ('finite_lookup', [{'max_rows': 1, 'evaluation_rows': True}]),
    ('affine', [{'max_rows': 1, 'evaluation_rows': 1}]),
])
def test_hub_profile_options_are_checked_before_network(monkeypatch, method, profiles):
    import compressme.hub as hub
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid options must not access Hugging Face')
    monkeypatch.setattr(hub, '_hub', forbidden)
    with pytest.raises(ValueError):
        hub.compress_huggingface('test/fanout', FanoutModel, method=method,
            finite_input_contract='token_indices_only' if method == 'finite_lookup' else None,
            finite_row_count_profiles=profiles, validation=examples())


def test_fanout_original_audit_precedes_copy(monkeypatch):
    import compressme.finite_blocks as implementation
    source = FanoutModel().eval()
    original_audit = implementation._audit_source
    original_copy = implementation._copy_with_compiler_metadata
    audited = []
    def audit(block, owner_model):
        assert block is source.stem and owner_model is source
        audited.append(block)
        return original_audit(block, owner_model)
    def copied(model):
        assert audited == [source.stem]
        return original_copy(model)
    monkeypatch.setattr(implementation, "_audit_source", audit)
    monkeypatch.setattr(implementation, "_copy_with_compiler_metadata", copied)
    result = compile_finite_blocks(source, paths=["stem"], input_contract="token_indices_only",
                                   validation=examples())
    assert result.report["rewrites"] == ["stem"]
