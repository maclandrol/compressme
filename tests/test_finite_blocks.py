import copy
import pytest
import torch
from torch import nn
from compressme import Example, compile_finite_blocks, load
from compressme.finite_lookup import FiniteTokenLookup


class TokenModel(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Embedding(19, 32, dtype=dtype),
            nn.Linear(32, 7, dtype=dtype), nn.LayerNorm(7, dtype=dtype), nn.SiLU())
        self.head = nn.Linear(7, 3, dtype=dtype)
    def forward(self, ids):
        encoded = self.encoder(ids)
        return {"tokens": encoded, "prediction": self.head(encoded), "ids": ids, "meta": "all_outputs"}
    def helper(self, ids):
        return self.forward(ids)


def examples():
    return [Example((torch.tensor([[0, 4, 18], [3, 3, 1]]),)),
            Example((torch.arange(19),)), Example((torch.empty(0, dtype=torch.long),))]


@pytest.mark.parametrize("paths", [None, ["encoder"]])
def test_closed_block_preserves_model_and_all_outputs_then_reloads(tmp_path, paths):
    source = TokenModel().eval()
    result = compile_finite_blocks(source, paths=paths, input_contract="token_indices_only", validation=examples())
    assert type(result.model) is TokenModel
    assert result.report["rewrites"] == ["encoder"]
    assert result.report["parameters_after"] < result.report["parameters_before"]
    assert result.report["validation"]["accepted"]
    assert type(source.encoder) is nn.Sequential
    result.save(tmp_path, packing=True)
    replay = load(TokenModel, tmp_path).model
    assert type(replay.encoder) is FiniteTokenLookup
    for example in examples():
        before, after = example.call(result.model), example.call(replay)
        for key in ("tokens", "prediction", "ids"):
            assert torch.equal(before[key], after[key])
        assert replay.helper(example.args[0])["meta"] == "all_outputs"


def test_nested_candidates_are_subsumed_only_after_parent_acceptance():
    source = nn.Sequential(nn.Sequential(nn.Embedding(13, 48), nn.Linear(48, 9)), nn.Tanh()).eval()
    result = compile_finite_blocks(source, input_contract="token_indices_only", validation=[Example((torch.arange(13),))])
    assert result.report["rewrites"] == [""]
    assert type(result.model) is FiniteTokenLookup
    assert any(row["status"] == "subsumed_by_accepted_parent" for row in result.report["blocks"])


def test_external_registered_alias_prevents_discarding_shared_source():
    source = TokenModel().eval()
    source.other_consumer = source.encoder[0]
    result = compile_finite_blocks(source, paths=["encoder"], input_contract="token_indices_only", validation=examples())
    assert result.report["status"] == "retained_no_accepted_rewrites"
    assert "external" in result.report["blocks"][0]["reason"]
    assert result.model.other_consumer is result.model.encoder[0]


def test_failed_whole_output_rolls_back_all_blocks(monkeypatch):
    import compressme.finite_blocks as implementation
    source = TokenModel().eval()
    real = implementation.compile_finite_lookup
    def broken(*args, **kwargs):
        result = real(*args, **kwargs)
        with torch.no_grad():
            result.model._rows.add_(0.5)
        return result
    monkeypatch.setattr(implementation, "compile_finite_lookup", broken)
    result = compile_finite_blocks(source, input_contract="token_indices_only", validation=examples())
    assert result.report["status"] == "rejected_and_rolled_back"
    assert not result.report["validation"]["accepted"]
    assert result.report["rewrites"] == [] and result.report["proposed_rewrites"] == ["encoder"]
    assert type(result.model.encoder) is nn.Sequential
    for name, value in source.state_dict().items():
        assert torch.equal(value, result.model.state_dict()[name])


def test_unsupported_and_unprofitable_blocks_are_reported():
    source = nn.Sequential(nn.Embedding(13, 5), nn.Linear(5, 64)).eval()
    result = compile_finite_blocks(source, input_contract="token_indices_only", validation=[Example((torch.arange(13),))])
    assert result.report["rewrites"] == []
    assert result.report["blocks"][0]["status"] == "retained_no_byte_saving"
    assert result.report["validation"]["accepted"]


@pytest.mark.parametrize("paths,error", [("encoder", TypeError), (["unknown"], ValueError),
    (["encoder", "encoder"], ValueError), (["", "encoder"], ValueError), ([], ValueError)])
def test_invalid_selection_is_explicit(paths, error):
    with pytest.raises(error):
        compile_finite_blocks(TokenModel().eval(), paths=paths,
            input_contract="token_indices_only", validation=examples())


def test_contract_validation_and_eval_are_required():
    for kwargs in ({"input_contract": "raw_vectors", "validation": examples()},
                   {"input_contract": "token_indices_only", "validation": []}):
        with pytest.raises(ValueError):
            compile_finite_blocks(TokenModel().eval(), **kwargs)
    with pytest.raises(ValueError, match="eval"):
        compile_finite_blocks(TokenModel(), input_contract="token_indices_only", validation=examples())



def test_original_distinct_storage_alias_is_seen_before_deepcopy():
    source = TokenModel().eval()
    source.register_buffer("external_view", source.encoder[0].weight.detach()[:2])
    result = compile_finite_blocks(source, paths=["encoder"], input_contract="token_indices_only", validation=examples())
    assert result.report["rewrites"] == []
    assert "external aliases" in result.report["blocks"][0]["reason"]
