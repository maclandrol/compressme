import copy
import json

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from compressme import share_frozen_parameters, load
from compressme.sharing import parameter_aliases, parameter_storage_aliases, restore_parameter_aliases
from compressme.validation import compare_outputs


def bundle():
    one = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    two = copy.deepcopy(one)
    with torch.no_grad():
        two[2].bias.add_(1)
    return nn.ModuleDict({'one': one, 'two': two}).eval().requires_grad_(False)


def test_shares_equal_values_and_preserves_different_head_and_all_outputs():
    source = bundle()
    result = share_frozen_parameters(source)
    candidate = result.model
    assert candidate['one'][0].weight is not candidate['two'][0].weight
    assert candidate['one'][0].weight.untyped_storage()._cdata == candidate['two'][0].weight.untyped_storage()._cdata
    assert candidate['one'][2].bias is not candidate['two'][2].bias
    assert source['one'][0].weight is not source['two'][0].weight
    assert candidate['one'][0].weight is not source['one'][0].weight
    assert result.report['saved_resident_storage_bytes'] == (32 + 8 + 24) * 4
    assert result.report['parameters_before'] - result.report['parameters_after'] == 0
    for shape in ((4,), (2, 4), (2, 3, 4), (0, 4)):
        x = torch.randn(shape)
        for name in source:
            assert compare_outputs(source[name](x), candidate[name](x))['output']['bitwise']


def test_retains_signed_zero_trainable_parameters_buffers_and_nonfinite_values():
    source = nn.Module()
    for name, values, requires in [('plus', [0., 0.], False), ('minus', [-0., 0.], False),
                                   ('trainable', [0., 0.], True), ('nan', [float('nan'), 0.], False)]:
        source.register_parameter(name, nn.Parameter(torch.tensor(values), requires_grad=requires))
    source.register_buffer('buffer', torch.zeros(2))
    result = share_frozen_parameters(source.eval())
    assert result.model.plus is not result.model.minus
    assert result.model.plus is not result.model.trainable
    assert result.model.buffer.untyped_storage()._cdata != result.model.plus.untyped_storage()._cdata
    assert result.report['parameter_replacements'] == []
    assert set(result.report['retained_ineligible_parameters']) == {'trainable', 'nan'}


def test_preserves_preexisting_parameter_aliases_when_merging_duplicate():
    source = bundle()
    source['two'].alias = source['two'][0].weight
    result = share_frozen_parameters(source)
    assert result.model['two'].alias is result.model['two'][0].weight
    assert result.model['two'].alias is not result.model['one'][0].weight
    assert result.model['two'].alias.untyped_storage()._cdata == result.model['one'][0].weight.untyped_storage()._cdata


def test_hooks_or_training_are_refused():
    source = bundle().train()
    with pytest.raises(ValueError, match='evaluation'):
        share_frozen_parameters(source)
    source.eval()
    handle = source['one'][0].register_forward_hook(lambda module, inputs, result: result)
    with pytest.raises(ValueError, match='hook-free'):
        share_frozen_parameters(source)
    handle.remove()


def test_portable_reload_restores_resident_ties(tmp_path):
    source = bundle()
    result = share_frozen_parameters(source)
    result.save(tmp_path)
    restored = load(bundle, tmp_path).model
    assert restored['one'][0].weight is not restored['two'][0].weight
    assert restored['one'][0].weight.untyped_storage()._cdata == restored['two'][0].weight.untyped_storage()._cdata
    assert parameter_storage_aliases(restored) == parameter_storage_aliases(result.model)
    assert restored['one'][2].bias is not restored['two'][2].bias
    assert parameter_aliases(restored) == parameter_aliases(result.model)
    x = torch.randn(5, 4)
    for name in restored:
        assert compare_outputs(result.model[name](x), restored[name](x))['output']['bitwise']


def test_alias_restore_rejects_conflicting_payload_before_mutating():
    source = bundle()
    state = {name: value.clone() for name, value in source.state_dict().items()}
    state['two.0.weight'][0, 0].add_(1)
    with pytest.raises(ValueError, match='Conflicting'):
        restore_parameter_aliases(source, {'two.0.weight': 'one.0.weight'}, state)
    assert source['one'][0].weight is not source['two'][0].weight
    with pytest.raises(ValueError, match='Invalid'):
        restore_parameter_aliases(source, {'two.0.weight': 'one.0.weight', 'one.0.weight': 'two.0.weight'}, state)


def test_hash_collisions_do_not_merge_different_bytes(monkeypatch):
    import compressme.sharing as sharing
    monkeypatch.setattr(sharing, '_digest', lambda value: b'constant')
    source = nn.ParameterDict({'a': nn.Parameter(torch.ones(4), requires_grad=False),
                               'b': nn.Parameter(torch.zeros(4), requires_grad=False)}).eval()
    result = share_frozen_parameters(source)
    assert result.model.a is not result.model.b
    assert result.report['saved_resident_storage_bytes'] == 0


def test_shared_source_backing_is_not_double_counted_as_saved_memory():
    values = torch.ones(8)
    source = nn.ParameterDict({'a': nn.Parameter(values, requires_grad=False),
                               'b': nn.Parameter(values, requires_grad=False)}).eval()
    result = share_frozen_parameters(source)
    assert result.report['resident_storage_bytes_before'] == 32
    assert result.report['resident_storage_bytes_after'] == 32
    assert result.report['saved_resident_storage_bytes'] == 0


def test_parameter_enumeration_and_identity_are_preserved_including_inplace():
    class Enumerates(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.ones(3), requires_grad=False)
            self.b = nn.Parameter(torch.ones(3), requires_grad=False)
        def forward(self, x):
            return x + sum(p.sum() for p in self.parameters())
    source = Enumerates().eval()
    old_ids = [id(p) for p in source.parameters()]
    expected = source(torch.tensor(0.))
    result = share_frozen_parameters(source, inplace=True)
    assert result.model is source
    assert [id(p) for p in source.parameters()] == old_ids
    assert source.a is not source.b
    assert compare_outputs(expected, source(torch.tensor(0.)))['output']['bitwise']
    assert result.report['saved_resident_storage_bytes'] == 12
    assert result.report['parameters_before'] == result.report['parameters_after'] == 6


def test_distinct_views_and_buffer_aliases_are_refused_before_copying():
    values = torch.arange(16.).reshape(4, 4)
    source = nn.ParameterDict({'a': nn.Parameter(values, requires_grad=False),
                              'b': nn.Parameter(values.t(), requires_grad=False)}).eval()
    with pytest.raises(ValueError, match='storage aliases'):
        share_frozen_parameters(source)
    source = nn.Module()
    source.register_parameter('a', nn.Parameter(values, requires_grad=False))
    source.register_buffer('b', values)
    with pytest.raises(ValueError, match='storage aliases'):
        share_frozen_parameters(source.eval())


def test_parameter_subclasses_are_retained():
    class CustomParameter(nn.Parameter):
        pass
    source = nn.ParameterDict({'a': nn.Parameter(torch.ones(3), requires_grad=False),
                              'b': CustomParameter(torch.ones(3), requires_grad=False)}).eval()
    result = share_frozen_parameters(source)
    assert type(result.model.b) is CustomParameter
    assert result.model.a.untyped_storage()._cdata != result.model.b.untyped_storage()._cdata
    assert result.report['retained_ineligible_parameters'] == ['b']
