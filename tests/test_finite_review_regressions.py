import torch
from torch import nn

from compressme import Example, compile_affine, compile_finite_blocks, load
from compressme.compiler import state_bytes
from compressme.finite_lookup import (
    _resident_storage_bytes, build_projected_lookup, compile_finite_lookup,
)


def repeated_source():
    shared = nn.Linear(3, 3, dtype=torch.float64)
    with torch.no_grad():
        shared.weight.copy_(torch.eye(3, dtype=torch.float64))
        shared.bias.zero_()
    return nn.Sequential(nn.Embedding(1000, 2, dtype=torch.float64),
        nn.Linear(2, 3, dtype=torch.float64), *[shared] * 100).eval()


def test_shared_modules_cannot_create_a_false_finite_storage_saving():
    source = repeated_source()
    result = compile_finite_lookup(source, input_contract="token_indices_only")
    assert result.report["status"] == "retained_no_byte_saving"
    assert result.report["parameters_before"] == result.report["parameters_after"] == 2021
    assert result.report["resident_storage_bytes_before"] == 2021 * 8
    assert result.report["proposed_resident_storage_bytes"] == 3000 * 8
    # Public logical state bytes still include all repeated state_dict entries.
    assert state_bytes(source) == 3209 * 8
    assert result.model[2] is result.model[101]
    assert torch.equal(source(torch.arange(1000)), result.model(torch.arange(1000)))


def test_model_wide_compiler_retains_shared_source_without_growing_it():
    source = repeated_source()
    result = compile_finite_blocks(source, input_contract="token_indices_only",
        validation=[Example((torch.arange(1000),))])
    assert result.report["status"] == "retained_no_accepted_rewrites"
    assert result.report["rewrites"] == []
    assert result.report["validation"]["accepted"]
    assert _resident_storage_bytes(result.model) == _resident_storage_bytes(source)


def test_projected_lookup_counts_a_tied_weight_only_once():
    embedding = nn.Embedding(4, 4, dtype=torch.float64).eval()
    linear = nn.Linear(4, 4, bias=False, dtype=torch.float64).eval()
    linear.weight = embedding.weight
    result = build_projected_lookup(embedding, linear,
        input_contract="raw_or_l2_normalized_token_indices")
    assert result.report["status"] == "retained_no_byte_saving"
    assert result.report["resident_storage_bytes_before"] == 16 * 8
    assert result.report["proposed_resident_storage_bytes"] == 20 * 8
    assert result.model.embedding.weight is result.model.linear.weight


def test_resident_storage_counts_views_and_nonpersistent_buffers():
    source = nn.Module()
    storage = torch.arange(100, dtype=torch.float32)
    source.register_buffer("a", storage[:5])
    source.register_buffer("b", storage[50:60], persistent=False)
    source.register_parameter("p", nn.Parameter(storage[10:20]))
    assert _resident_storage_bytes(source) == 400
    assert state_bytes(source) == (5 + 10) * 4


class AffineSource(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.expand = nn.Linear(2, 32, dtype=dtype)
        self.project = nn.Linear(32, 3, dtype=dtype)

    def forward(self, x):
        return {"prediction": self.project(self.expand(x)), "meta": "complete"}


def assert_replay_matches(source, result, directory):
    assert result.model._compressme_affine_fx_spec == source._compressme_affine_fx_spec
    assert result.model._compressme_source_tensor_dtypes == source._compressme_source_tensor_dtypes
    assert "expand.weight" in result.model._compressme_source_tensor_dtypes
    result.save(directory)
    replay = load(AffineSource, directory).model
    x = torch.randn(5, 2, dtype=torch.float64)
    assert torch.equal(source(x)["prediction"], replay(x)["prediction"])
    assert replay(x)["meta"] == "complete"


def test_noop_after_affine_retains_portable_recipes_and_original_dtypes(tmp_path):
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    result = compile_finite_blocks(source, input_contract="token_indices_only",
        validation=[Example((torch.randn(5, 2, dtype=torch.float64),))])
    assert result.report["status"] == "retained_no_accepted_rewrites"
    assert_replay_matches(source, result, tmp_path)


def test_rollback_after_affine_retains_portable_recipes_and_original_dtypes(tmp_path, monkeypatch):
    import compressme.finite_blocks as implementation
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    monkeypatch.setattr(implementation, "validate", lambda *args, **kwargs: {"accepted": False})
    result = compile_finite_blocks(source, input_contract="token_indices_only",
        validation=[Example((torch.randn(5, 2, dtype=torch.float64),))])
    assert result.report["status"] == "rejected_and_rolled_back"
    assert_replay_matches(source, result, tmp_path)


def test_accepted_child_preserves_original_compiler_dtype_metadata():
    source = nn.ModuleDict({"tokens": nn.Sequential(nn.Embedding(7, 16), nn.Linear(16, 2))}).eval()
    # A custom enclosing class is needed for an actual forward; keep the dtype
    # provenance check separate from FX's flattening of Sequential children.
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = source["tokens"]
        def forward(self, ids):
            return self.tokens(ids)
    model = Model().eval()
    original = {"original_removed_producer.weight": "float64"}
    model._compressme_source_tensor_dtypes = original
    result = compile_finite_blocks(model, input_contract="token_indices_only",
        validation=[Example((torch.arange(7),))])
    assert result.report["status"] == "accepted_on_validation_examples"
    assert result.model._compressme_source_tensor_dtypes == original
    assert result.model._compressme_source_tensor_dtypes is not original


def test_noop_operator_compression_keeps_affine_replay(tmp_path):
    from compressme import compress
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    result = compress(source, min_parameters=999999)
    assert_replay_matches(source, result, tmp_path)


def test_operator_rollback_keeps_affine_replay(tmp_path, monkeypatch):
    import compressme.compiler as implementation
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    monkeypatch.setattr(implementation, "validate", lambda *args, **kwargs: {"accepted": False})
    result = implementation.compress(source, min_parameters=999999,
        validation=[Example((torch.randn(5, 2, dtype=torch.float64),))])
    assert result.report["status"] == "rejected_and_rolled_back"
    assert_replay_matches(source, result, tmp_path)


def test_noop_constant_embedding_pass_keeps_affine_replay(tmp_path):
    from compressme.compiler import CompressionResult
    from compressme.constant_embeddings import deduplicate_embeddings
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    candidate, report = deduplicate_embeddings(source)
    assert_replay_matches(source, CompressionResult(candidate, report), tmp_path)


def test_same_filter_affine_rerun_is_portable(tmp_path):
    source = compile_affine(AffineSource(torch.float64).eval(), include=["project"]).model.eval()
    result = compile_affine(source, include=["project"])
    assert result.report["rewrites"] == []
    assert_replay_matches(source, result, tmp_path)


def test_affine_rollback_keeps_prior_replay(tmp_path, monkeypatch):
    import compressme.validation as implementation
    source = compile_affine(AffineSource(torch.float64).eval()).model.eval()
    monkeypatch.setattr(implementation, "validate", lambda *args, **kwargs: {"accepted": False})
    result = compile_affine(source, validation=[Example((torch.randn(5, 2, dtype=torch.float64),))])
    assert result.report["status"] == "rejected_and_rolled_back"
    assert_replay_matches(source, result, tmp_path)


class TwoAffineBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_expand = nn.Linear(2, 32, dtype=torch.float64)
        self.a_project = nn.Linear(32, 3, dtype=torch.float64)
        self.b_expand = nn.Linear(2, 32, dtype=torch.float64)
        self.b_project = nn.Linear(32, 3, dtype=torch.float64)
    def forward(self, x):
        return {"a": self.a_project(self.a_expand(x)), "b": self.b_project(self.b_expand(x))}


def test_changed_affine_filters_retain_existing_portable_graph(tmp_path):
    source = compile_affine(TwoAffineBranches().eval(), include=["a_project"]).model.eval()
    samples = [Example((torch.randn(5, 2, dtype=torch.float64),))]
    result = compile_affine(source, include=["b_project"], validation=samples)
    assert result.report["status"] == "retained_existing_affine_recipe"
    assert result.report["validation"]["accepted"]
    assert result.model._compressme_affine_fx_spec["include"] == ["a_project"]
    assert result.model.a_project.in_features == 2
    assert result.model.b_project.in_features == 32
    result.save(tmp_path)
    replay = load(TwoAffineBranches, tmp_path).model
    x = samples[0].args[0]
    assert all(torch.equal(source(x)[key], replay(x)[key]) for key in ("a", "b"))
