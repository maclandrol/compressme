import copy
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from compressme import load
from compressme.finite_fanout import FiniteTokenFanout, FiniteFanoutLookup, compile_finite_fanout
from compressme.finite_lookup import _resident_storage_bytes

PROFILES = [{"max_rows": 1, "evaluation_rows": 1},
            {"max_rows": 15, "evaluation_rows": 2}]


def source():
    torch.manual_seed(42)
    return FiniteTokenFanout(nn.Embedding(7, 128), nn.Sequential(),
        {"q": nn.Linear(128, 128, bias=False), "k": nn.Linear(128, 128, bias=False)},
        residual_key="residual").eval()


def compile_source(model, **kwargs):
    return compile_finite_fanout(model, input_contract="token_indices_only", **kwargs)


def assert_outputs_equal(a, b):
    assert list(a) == list(b)
    for name in a:
        assert torch.equal(a[name], b[name])


def test_default_remains_version_one_and_replays():
    result = compile_source(source())
    spec = result.model.recipe()
    assert spec["version"] == 1 and "row_count_profiles" not in spec
    rebuilt = FiniteFanoutLookup.from_recipe(json.loads(json.dumps(spec)))
    rebuilt.load_state_dict(result.model.state_dict())
    assert_outputs_equal(result.model(torch.arange(7)), rebuilt(torch.arange(7)))


def test_profiles_keep_every_output_all_shapes_and_count_shared_storage():
    original = source()
    original.branches["k"] = original.branches["q"]
    result = compile_source(original, row_count_profiles=PROFILES)
    assert result.report["status"] == "accepted_on_full_domain_validation"
    candidate = result.model
    assert candidate.recipe()["version"] == 2
    assert candidate.validation_is_current()
    assert result.report["row_count_profiles"] == PROFILES
    assert result.report["profile_validation_sizes"] == [1, 2, 14, 15, 16]
    assert result.report["validation_cases"] > 100
    assert result.report["tensor_bytes_after"] == sum(t.numel() * t.element_size() for t in candidate.state_dict().values())
    assert result.report["candidate_resident_storage_bytes"] == _resident_storage_bytes(candidate)
    assert result.report["candidate_stored_values"] == sum(p.numel() for p in candidate.parameters())
    assert result.report["resident_storage_bytes_after"] < result.report["resident_storage_bytes_before"]
    # Identical q/k and every profile's unchanged residual refer to the same
    # stored locations; alternate floating-point q tables remain counted.
    descriptors = [candidate.outputs, *(item[2] for item in candidate.row_count_profiles)]
    residuals = []
    for outputs in descriptors:
        by_name = {item[0]: item[1:] for item in outputs}
        assert by_name["q"] == by_name["k"]
        residuals.append(by_name["residual"])
    assert len(set(residuals)) == 1
    source_storages = {p.untyped_storage()._cdata for p in original.parameters()}
    assert not source_storages.intersection(p.untyped_storage()._cdata for p in candidate.parameters())
    for shape in [(), (1,), (1, 1), (2,), (1, 14), (3, 5), (2, 8), (2, 3, 7), (0,), (3, 0, 2)]:
        ids = torch.arange(torch.tensor(shape).prod().item() if shape else 1).remainder(7).long().reshape(shape)
        left, right = original(ids), candidate(ids)
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        assert torch.equal(left["residual"], right["residual"])
        if right["q"].numel():
            saved = right["k"].clone()
            right["q"].add_(1)
            assert torch.equal(saved, right["k"])
    invalid = torch.tensor([[7]])
    with pytest.raises((IndexError, RuntimeError)):
        candidate(invalid)


def test_actual_cpu_shape_rounding_can_be_repaired_without_relaxing_gate():
    # Real BLAS reduction choices differ across platforms. This local hardware
    # example is evidence, not a claim that every backend needs this profile.
    if sys.platform != "darwin":
        pytest.skip("This regression records the verified Apple CPU BLAS example")
    original = source()
    with torch.no_grad():
        for branch in original.branches.values():
            branch.weight.mul_(50)
    plain = compile_source(original)
    profiled = compile_source(original, row_count_profiles=[PROFILES[0]])
    assert plain.report["status"] == "rejected_numerical_validation"
    assert type(plain.model) is FiniteTokenFanout
    assert profiled.report["status"] == "accepted_on_full_domain_validation"
    assert all(item["bitwise_identical"] for item in profiled.report["numerical"].values())
    assert plain.report["absolute_tolerance"] == profiled.report["absolute_tolerance"] == 1e-5
    assert profiled.report["candidate_stored_values"] > plain.report["candidate_stored_values"]


def test_profile_dispatch_repair_under_a_simulated_shape_rounding_backend(monkeypatch):
    # Fault injection models two deterministic, shape-dependent kernel answers.
    # It does not expand the production grammar to arbitrary Python modules.
    real_linear = F.linear
    def alternate_kernel(values, weight, bias=None):
        output = real_linear(values, weight, bias)
        if values.numel() // values.shape[-1] == 1:
            output = output + 1e-3
        return output
    monkeypatch.setattr(F, "linear", alternate_kernel)
    original = source()
    plain = compile_source(original)
    assert plain.report["status"] == "rejected_numerical_validation"
    profiled = compile_source(original, row_count_profiles=[PROFILES[0]])
    assert profiled.report["status"] == "accepted_on_full_domain_validation"
    assert profiled.report["numerical"]["q"]["max_abs"] <= 1e-5


def test_profile_rejects_position_dependent_outputs_for_identical_ids(monkeypatch):
    real_linear = F.linear
    def position_dependent(values, weight, bias=None):
        result = real_linear(values, weight, bias)
        if values.ndim == 3 and values.shape[1] == 2:
            result[:, 1] += 0.01
        return result
    monkeypatch.setattr(F, "linear", position_dependent)
    with pytest.raises(ValueError, match="unequal outputs at identical"):
        compile_source(source(), row_count_profiles=PROFILES)


def test_mixed_token_gate_rejects_data_dependent_shape_proposal(monkeypatch):
    real_linear = F.linear
    def composition_dependent(values, weight, bias=None):
        result = real_linear(values, weight, bias)
        flat = values.reshape(-1, values.shape[-1])
        if flat.shape[0] == 2 and not torch.equal(flat[0], flat[1]):
            result = result + 0.01
        return result
    monkeypatch.setattr(F, "linear", composition_dependent)
    result = compile_source(source(), row_count_profiles=PROFILES)
    assert result.report["status"] == "rejected_numerical_validation"
    assert type(result.model) is FiniteTokenFanout


@pytest.mark.parametrize("profiles", [[], "automatic", [{}], [{"max_rows": 1}],
    [{"max_rows": True, "evaluation_rows": 1}], [{"max_rows": 1, "evaluation_rows": 0}],
    [{"max_rows": 2, "evaluation_rows": 1}, {"max_rows": 1, "evaluation_rows": 1}],
    [{"max_rows": 1, "evaluation_rows": 1, "backend": "guess"}]])
def test_profiles_require_a_strict_explicit_schema(profiles):
    with pytest.raises(ValueError):
        compile_source(source(), row_count_profiles=profiles)


def test_profile_recipe_replay_packed_save_and_historical_validation(tmp_path):
    result = compile_source(source(), row_count_profiles=PROFILES)
    assert result.model.validation_is_current()
    result.save(tmp_path, packing=True)
    replay = load(source, tmp_path).model
    assert replay.recipe() == result.model.recipe()
    for shape in [(), (1, 1), (2,), (3, 5), (2, 8), (2, 3, 7), (0, 2)]:
        ids = torch.randint(0, 7, shape)
        assert_outputs_equal(result.model(ids), replay(ids))
    spec = json.loads(json.dumps(result.model.recipe()))
    with torch.inference_mode():
        rebuilt = FiniteFanoutLookup.from_recipe(spec)
    rebuilt.load_state_dict(result.model.state_dict(), strict=True)
    assert not rebuilt.validation_is_current()
    assert all(not torch.is_inference(p) for p in rebuilt.parameters())
    maximum, evaluation, outputs = result.model.row_count_profiles[0]
    result.model.row_count_profiles = ((maximum + 1, evaluation, outputs), *result.model.row_count_profiles[1:])
    assert not result.model.validation_is_current()
    result.save(tmp_path / "changed", packing=True)
    report = json.loads((tmp_path / "changed" / "manifest.json").read_text())
    assert report["report"]["finite_lookup_validation_at_export"][0]["state"] == "historical_or_invalidated"
    assert report["report"]["validation"]["accepted"] is False


@pytest.mark.parametrize("damage", ["unordered", "missing", "wrong_output", "wrong_width", "version1_profiles"])
def test_version_two_recipe_refuses_malformed_dispatch(damage):
    result = compile_source(source(), row_count_profiles=PROFILES)
    spec = copy.deepcopy(result.model.recipe())
    if damage == "unordered":
        spec["row_count_profiles"].reverse()
    elif damage == "missing":
        del spec["row_count_profiles"]
    elif damage == "wrong_output":
        spec["row_count_profiles"][0]["outputs"][0][0] = "alien"
    elif damage == "wrong_width":
        spec["row_count_profiles"][0]["outputs"][0][3] -= 1
    else:
        spec["version"] = 1
    with pytest.raises(ValueError):
        FiniteFanoutLookup.from_recipe(spec)


def test_every_added_profile_table_is_subject_to_the_storage_gate(monkeypatch):
    real_linear = F.linear
    def distinct_shape(values, weight, bias=None):
        result = real_linear(values, weight, bias)
        return result + (values.numel() // values.shape[-1]) * 1e-7
    monkeypatch.setattr(F, "linear", distinct_shape)
    torch.manual_seed(3)
    original = FiniteTokenFanout(nn.Embedding(9, 12), nn.Sequential(),
        {"q": nn.Linear(12, 8, bias=False)}, residual_key="residual").eval()
    plain = compile_source(original)
    assert plain.report["status"] == "accepted_on_full_domain_validation"
    profiled = compile_source(original, row_count_profiles=PROFILES)
    assert profiled.report["status"] == "no_storage_saving"
    assert type(profiled.model) is FiniteTokenFanout
    assert profiled.report["proposal_tensor_bytes_after"] >= profiled.report["tensor_bytes_before"]
    assert profiled.report["parameters_after"] == profiled.report["parameters_before"]


def test_forward_gathers_only_active_profile_columns_and_one_shared_residual(monkeypatch):
    real_linear = F.linear
    def alternate_shape(values, weight, bias=None):
        result = real_linear(values, weight, bias)
        rows = values.numel() // values.shape[-1]
        return result + (1e-7 if rows == 1 else 2e-7 if rows < 16 else 3e-7)
    monkeypatch.setattr(F, "linear", alternate_shape)
    original = source()
    result = compile_source(original, row_count_profiles=PROFILES)
    assert result.report["status"] == "accepted_on_full_domain_validation"
    candidate = result.model
    calls = []
    real_embedding = F.embedding
    def observed(ids, weight, *args, **kwargs):
        calls.append(tuple(weight.shape))
        return real_embedding(ids, weight, *args, **kwargs)
    monkeypatch.setattr(F, "embedding", observed)
    for count in (1, 2, 15, 16, 23):
        calls.clear()
        candidate(torch.arange(count).remainder(7))
        # q+k together form one 256-wide row, the unchanged residual one128.
        # No row contains inactive q/k profile columns.
        assert sorted(width for _, width in calls) == [128, 256]
        assert len(calls) == 2
    routes = [candidate.outputs, *(item[2] for item in candidate.row_count_profiles)]
    residual_descriptors = [{row[0]: row[1:] for row in route}["residual"] for route in routes]
    assert len(set(residual_descriptors)) == 1
