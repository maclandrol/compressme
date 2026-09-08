import copy

import pytest
import torch
from torch import nn

from compressme.nesso_runtime import _state_signature, _tensor_signature as tensor_signature


def old_signature(model):
    modules = tuple((name, id(module), type(module), module.training)
                    for name, module in model.named_modules(remove_duplicate=False))
    tensors = tuple((name, tensor_signature(tensor)) for name, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))))
    return modules, tensors


def canonical(new):
    modules, parameters, buffers = new
    return modules, tuple((f"{path}.{key}" if path else key, value)
                          for path, key, value in parameters + buffers)


def model():
    root = nn.Sequential(nn.Linear(3, 3), nn.Sequential(nn.Linear(3, 3)))
    root.alias = root[0]
    root[1][0].weight = root[0].weight
    root.register_buffer("root_buffer", torch.ones(1))
    root[0].register_buffer("child_buffer", torch.ones(2))
    root[0].register_buffer("none_buffer", None)
    root[0].register_parameter("none_parameter", None)
    root.add_module("none_child", None)
    return root.eval().requires_grad_(False)


def test_exact_old_signature_contents_with_aliases_buffers_and_none_slots():
    root = model()
    old = old_signature(root)
    new = _state_signature(root)
    assert canonical(new) == old
    assert [entry[0] for entry in new[0]] == ["", "0", "1", "1.0", "alias"]
    assert len(new[1]) == 6  # Repeated owner paths and tied parameter slots survive.
    assert len(new[2]) == 3


@pytest.mark.parametrize("change", ["alias_rebind", "new_module", "remove_module", "order", "parameter", "buffer", "version", "dtype", "training", "none"])
def test_matches_old_change_detection(change):
    root = model()
    old, new = old_signature(root), _state_signature(root)
    with torch.no_grad():
        if change == "alias_rebind": root.alias = copy.deepcopy(root[0])
        elif change == "new_module": root.new = nn.Identity().eval()
        elif change == "remove_module": del root.alias
        elif change == "order": root._modules["0"] = root._modules.pop("0")
        elif change == "parameter": root[0].weight = nn.Parameter(root[0].weight.clone(), requires_grad=False)
        elif change == "buffer": root[0].child_buffer = root[0].child_buffer.clone()
        elif change == "version": root[0].weight.add_(1)
        elif change == "dtype": root.double()
        elif change == "training": root[0].train()
        elif change == "none": root.register_buffer("another_none", None)
    old_after = old_signature(root)
    new_after = _state_signature(root)
    assert canonical(new_after) == old_after
    assert (old_after != old) == (new_after != new)
    assert (old_after == old) is (change == "none")


def test_uses_one_module_walk_and_no_named_tensor_walks(monkeypatch):
    root = model()
    old = old_signature(root)
    calls = []
    named_modules = root.named_modules
    def walk(*args, **kwargs):
        calls.append(1)
        yield from named_modules(*args, **kwargs)
    monkeypatch.setattr(root, "named_modules", walk)
    monkeypatch.setattr(root, "named_parameters", lambda **kwargs: pytest.fail("Extra parameter traversal"))
    monkeypatch.setattr(root, "named_buffers", lambda **kwargs: pytest.fail("Extra buffer traversal"))
    assert canonical(_state_signature(root)) == old
    assert len(calls) == 1
