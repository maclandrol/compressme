import torch
from torch import nn
from compressme import (CompressionResult, Example, compile_affine,
    deduplicate_embeddings, load, validate, compile_finite_lookup, build_projected_lookup)
from compressme.finite_lookup import DualModeEmbeddingProjection


def factory():
    # Fresh factory rows need not be constant: replay uses the stored row.
    return nn.Sequential(nn.Embedding(37,3),nn.Linear(3,17),nn.Linear(17,7)).eval()


def test_constant_embedding_and_affine_export_replay(tmp_path):
    original = factory()
    original[0].weight.requires_grad_(False)
    original[0].weight.data.copy_(torch.tensor([[1.,-0.,2.]]).expand_as(original[0].weight))
    dedup, _ = deduplicate_embeddings(original)
    result = compile_affine(dedup)
    result.save(tmp_path)
    recovered = load(factory,tmp_path).model
    examples = [Example((torch.tensor([0,3,36]),))]
    assert validate(original,recovered,examples,absolute_tolerance=1e-5,relative_tolerance=1e-5)["accepted"]
    assert sum(p.numel() for p in recovered.parameters()) < sum(p.numel() for p in dedup.parameters())
    torch.testing.assert_close(recovered.get_submodule("0")._row, dedup[0]._row)


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Sequential(nn.Embedding(101,64),nn.Linear(64,8),nn.LayerNorm(8),nn.SiLU())
        self.read = nn.Sequential(nn.Linear(8,32),nn.Linear(32,3))
        self.eval()
    def forward(self,indices):
        return self.read(self.tokens(indices))


def test_finite_lookup_affine_and_portable_original_factory(tmp_path):
    original = TokenModel()
    candidate = TokenModel()
    candidate.load_state_dict(original.state_dict())
    candidate.tokens = compile_finite_lookup(candidate.tokens,input_contract="token_indices_only").model
    result = compile_affine(candidate)
    result.save(tmp_path)
    recovered = load(TokenModel,tmp_path).model
    examples = [Example((torch.arange(101),)),Example((torch.tensor([[1,3],[9,100]]),))]
    assert validate(original,recovered,examples,absolute_tolerance=1e-5,relative_tolerance=1e-5)["accepted"]
    assert recovered.tokens._rows.shape == (101,8)


def test_dual_projection_portable_both_branches(tmp_path):
    def original_factory():
        return DualModeEmbeddingProjection(nn.Embedding(37,32).eval(),nn.Linear(32,7).eval(),1e-12)
    source = original_factory()
    result = build_projected_lookup(source.embedding,source.linear,
        input_contract="raw_or_l2_normalized_token_indices")
    assert result.report["validation"]["accepted"]
    result.save(tmp_path)
    recovered = load(original_factory,tmp_path).model
    ids = torch.arange(37)
    for normalized in (False,True):
        torch.testing.assert_close(source(ids,normalized),recovered(ids,normalized),atol=1e-5,rtol=1e-5)
