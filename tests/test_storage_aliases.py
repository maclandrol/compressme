"""Lossless file deduplication must not conflate different storage views."""
import json
import pytest
import torch
from torch import nn
from safetensors.torch import load_file
from compressme import CompressionResult, load
from compressme.serialization import _storage_aliases


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 16)
        self.gene_encoder = self.encoder
        self.register_buffer("signs", torch.tensor([0., -0.]))
        self.register_buffer("same_signs", self.signs)

    def forward(self, x):
        return {"main": self.encoder(x), "alias": self.gene_encoder(x),
                "signs": self.signs}


@pytest.mark.parametrize("packing", [False, True])
def test_shared_state_roundtrip_preserves_bytes_and_ties(tmp_path, packing):
    model = TiedModel().eval()
    CompressionResult(model, {}).save(tmp_path, packing=packing)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tensor_aliases"] == {"same_signs": "signs",
        "gene_encoder.weight": "encoder.weight", "gene_encoder.bias": "encoder.bias"}
    if not packing:
        assert len(load_file(str(tmp_path / "model.safetensors"))) == 3
    restored = load(TiedModel, tmp_path).model
    assert restored.encoder is restored.gene_encoder
    assert restored.signs is restored.same_signs
    for name, expected in model.state_dict().items():
        assert torch.equal(expected.view(torch.uint8), restored.state_dict()[name].view(torch.uint8))
    x = torch.randn(9, 32)
    for name, expected in model(x).items():
        assert torch.equal(expected, restored(x)[name])


def test_different_views_and_equal_independent_weights_not_deduplicated():
    weight = torch.arange(16.).reshape(4, 4)
    state = {"base": weight, "alias": weight.view(4, 4), "transpose": weight.T,
             "offset": weight[1:], "independent": weight.clone(),
             "empty": weight[:0], "other_empty": weight[:0]}
    stored, aliases = _storage_aliases(state)
    assert aliases == {"alias": "base"}
    for name, value in stored.items():
        assert torch.equal(value, state[name])


@pytest.mark.parametrize("bad", [
    {"gene_encoder.weight": "missing"},
    {"gene_encoder.weight": "gene_encoder.weight"},
    {"gene_encoder.weight": "gene_encoder.bias", "gene_encoder.bias": "gene_encoder.weight"},
    {"encoder.weight": "encoder.bias"},
    {"gene_encoder.weight": 12},
    ["gene_encoder.weight"],
])
def test_corrupt_alias_recipes_are_rejected(tmp_path, bad):
    CompressionResult(TiedModel().eval(), {}).save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tensor_aliases"] = bad
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor alias"):
        load(TiedModel, tmp_path)


def test_compiled_lookup_shared_at_two_module_paths_roundtrips(tmp_path):
    from compressme import compile_finite_lookup
    def factory():
        block = nn.Sequential(nn.Embedding(11, 32), nn.Linear(32, 5)).eval()
        return nn.ModuleDict({"a": block, "b": block}).eval()
    source = factory()
    compiled = compile_finite_lookup(source["a"], input_contract="token_indices_only").model
    candidate = nn.ModuleDict({"a": compiled, "b": compiled}).eval()
    CompressionResult(candidate, {}).save(tmp_path)
    restored = load(factory, tmp_path).model
    assert restored["a"] is restored["b"]
    for key in ("a", "b"):
        assert torch.equal(candidate[key](torch.arange(11)), restored[key](torch.arange(11)))


@pytest.mark.parametrize("bad", [{"gene_encoder": "missing"}, {"encoder": "encoder.weight"},
                                 {"gene_encoder": "encoder", "encoder": "gene_encoder"},
                                 {"": "encoder"}, ["encoder"]])
def test_invalid_module_alias_recipe_rejected(tmp_path, bad):
    CompressionResult(TiedModel().eval(), {}).save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["module_aliases"] = bad
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Mm]odule alias"):
        load(TiedModel, tmp_path)


@pytest.mark.parametrize("mutate", [False, True])
def test_finite_export_validation_distinguishes_mutated_weights(tmp_path, mutate):
    from compressme import compile_finite_lookup
    block = nn.Sequential(nn.Embedding(11, 32), nn.Linear(32, 5)).eval()
    result = compile_finite_lookup(block, input_contract="token_indices_only")
    assert result.model.validation_is_current()
    if mutate:
        with torch.no_grad():
            result.model._rows.add_(100)
    result.save(tmp_path)
    report = json.loads((tmp_path / "manifest.json").read_text())["report"]
    assert report["finite_lookup_validation_at_export"][0]["state"] == (
        "historical_or_invalidated" if mutate else "current_at_export")
    if mutate:
        assert not report["validation"]["accepted"]
        assert report["historical_validation"]["accepted"]
    else:
        assert report["validation"]["accepted"]
    # Export does not rewrite the caller's historical in-memory report.
    assert result.report["status"] == "accepted_on_full_domain_validation"
