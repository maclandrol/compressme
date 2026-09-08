import operator
import weakref

import pytest
import torch
from torch import fx, nn

from request_fx import partial_evaluate


class Ada(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.a_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.s_norm = nn.LayerNorm(width)
        self.s_scale = nn.Linear(width, width)
        self.s_bias = nn.Linear(width, width, bias=False)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        return torch.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


class Prefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.linear = nn.Linear(8, 8)

    def forward(self, times, trunk, inputs):
        prefix = self.linear(self.norm(torch.cat((trunk, inputs), dim=-1)))
        return prefix + times, prefix


def trace(module):
    return fx.symbolic_trace(module.eval().requires_grad_(False)).eval()


def bits(a, b):
    return torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('shape', [(2, 3, 8), (0, 3, 8), (1, 8)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_adaln_discovers_invariants_and_preserves_multiple_dynamic_inputs(shape, dtype):
    graph = trace(Ada().to(dtype=dtype))
    s = torch.randn(shape, dtype=dtype)
    source_ids = {name: id(value) for name, value in graph.named_parameters()}
    with torch.no_grad(), partial_evaluate(graph, {'s': s}) as prepared:
        assert prepared.report['evaluated_nodes'] == ['s_norm', 's_scale', 'sigmoid', 's_bias']
        assert prepared.report['boundary_nodes'] == ['sigmoid', 's_bias']
        assert prepared.report['retained_parameter_values'] == prepared.report['source_parameter_values']
        for _ in range(3):
            a = torch.randn(shape, dtype=dtype)
            assert bits(graph(a, s), prepared(a, s=s))
        assert source_ids == {name: id(value) for name, value in graph.named_parameters()}
        ref = weakref.ref(prepared.dynamic_graph._constant_0)
    assert prepared.report['closed'] and not list(prepared.parameters()) and not list(prepared.buffers())
    assert ref() is None
    with torch.no_grad(), pytest.raises(RuntimeError, match='closed'):
        prepared(torch.zeros(shape, dtype=dtype))


def test_noncontiguous_static_input_and_multiple_outputs():
    graph = trace(Prefix())
    trunk = torch.randn(2, 4, 3).transpose(1, 2)
    inputs = torch.randn(2, 4, 3).transpose(1, 2)
    assert not trunk.is_contiguous()
    with torch.no_grad(), partial_evaluate(graph, {'trunk': trunk, 'inputs': inputs}) as prepared:
        assert prepared.report['boundary_nodes'] == ['linear']
        for scale in (0.0, 1.0, -3.0):
            times = torch.full((2, 3, 8), scale)
            expected, actual = graph(times, trunk, inputs), prepared(times)
            assert all(bits(a, b) for a, b in zip(expected, actual))


def test_all_static_gate_runs_only_during_preparation():
    graph = trace(nn.Sequential(nn.Linear(8, 8), nn.Sigmoid()))
    s = torch.randn(3, 8)
    with torch.no_grad(), partial_evaluate(graph, {'input': s}) as prepared:
        assert prepared.report['dynamic_inputs'] == []
        assert not any(node.op == 'call_module' for node in prepared.dynamic_graph.graph.nodes)
        assert bits(graph(s), prepared())


@pytest.mark.parametrize('change', ['inplace_input', 'replace_input', 'inplace_parameter', 'replace_parameter', 'parameter_data_storage', 'eps', 'graph', 'new_hook'])
def test_detectable_mutations_refused(change):
    graph = trace(Ada())
    s, a = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    with torch.no_grad(), partial_evaluate(graph, {'s': s}) as prepared:
        if change == 'inplace_input':
            s.add_(1)
        elif change == 'replace_input':
            with pytest.raises(ValueError, match='differs'):
                prepared(a, s=s.clone())
            return
        elif change == 'inplace_parameter':
            graph.s_scale.bias.add_(1)
        elif change == 'replace_parameter':
            graph.s_scale.bias = nn.Parameter(graph.s_scale.bias.clone() + 1, requires_grad=False)
        elif change == 'parameter_data_storage':
            graph.s_scale.bias.data = graph.s_scale.bias.clone() + 1
        elif change == 'eps':
            graph.s_norm.eps = 0.1
        elif change == 'graph':
            graph.graph.nodes.__iter__().__next__().name = 'changed_placeholder_name'
        else:
            graph.s_scale.register_forward_hook(lambda m, a, result: result)
        with pytest.raises(ValueError):
            prepared(a)


def test_cached_output_mutation_detected():
    graph = trace(nn.Sequential(nn.Linear(8, 8), nn.Sigmoid()))
    with torch.no_grad(), partial_evaluate(graph, {'input': torch.randn(2, 8)}) as prepared:
        prepared().add_(1)
        with pytest.raises(ValueError, match='constant'):
            prepared()


def test_no_grad_training_autocast_and_conversion_guards():
    graph, s, a = trace(Ada()), torch.randn(2, 8), torch.randn(2, 8)
    with pytest.raises(ValueError, match='no-grad'):
        with partial_evaluate(graph, {'s': s}):
            pass
    with torch.no_grad(), partial_evaluate(graph, {'s': s}) as prepared:
        with torch.enable_grad(), pytest.raises(ValueError, match='no-grad'):
            prepared(a)
        with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='Autocast'):
            prepared(a)
        with pytest.raises(ValueError, match='Move source'):
            prepared.to(dtype=torch.float64)
        graph.s_norm.train()
        with pytest.raises(ValueError, match='evaluation'):
            prepared(a)


@pytest.mark.parametrize('bad', [nn.Dropout(0.2), nn.ReLU(inplace=True), nn.BatchNorm1d(8)])
def test_nonpure_or_unsupported_modules_refused(bad):
    graph = trace(nn.Sequential(bad))
    with torch.no_grad(), pytest.raises(ValueError, match='grammar'):
        with partial_evaluate(graph, {'input': torch.randn(2, 8)}):
            pass


def test_random_and_inplace_fx_nodes_refused():
    for target in (torch.rand_like, torch.relu_):
        graph = fx.Graph()
        x = graph.placeholder('x')
        graph.output(graph.call_function(target, (x,)))
        module = fx.GraphModule(nn.Module(), graph).eval()
        with torch.no_grad(), pytest.raises(ValueError, match='grammar'):
            with partial_evaluate(module, {'x': torch.randn(2, 8)}):
                pass

    graph = fx.Graph()
    x = graph.placeholder('x')
    graph.output(graph.call_function(torch.nn.functional.relu, (x, True)))
    with torch.no_grad(), pytest.raises(ValueError, match='inplace'):
        with partial_evaluate(fx.GraphModule(nn.Module(), graph).eval(), {'x': torch.randn(2, 8)}):
            pass


def test_inference_tensor_explicit_ownership_and_cleanup_on_exception():
    graph = trace(Ada())
    with torch.inference_mode():
        s, a = torch.randn(2, 8), torch.randn(2, 8)
        with pytest.raises(ValueError, match='Unversioned'):
            with partial_evaluate(graph, {'s': s}):
                pass
        with pytest.raises(RuntimeError, match='intentional'):
            with partial_evaluate(graph, {'s': s}, allow_unversioned_tensors=True) as prepared:
                assert bits(graph(a, s), prepared(a))
                raise RuntimeError('intentional')
        assert prepared.report['closed'] and not list(prepared.buffers())


def test_source_storage_aliases_are_counted_once():
    graph = trace(Ada())
    graph.s_bias.weight = graph.s_scale.weight
    s = torch.randn(3, 8)
    with torch.no_grad(), partial_evaluate(graph, {'s': s}) as prepared:
        assert prepared.report['retained_parameter_values'] == sum(p.numel() for p in graph.parameters())
        assert prepared.report['source_registered_storage_bytes'] == sum(p.numel() * p.element_size() for p in graph.parameters())
        assert prepared.report['retained_registered_storage_bytes'] == (prepared.report['source_registered_storage_bytes'] + s.numel() * s.element_size() + 2 * s.numel() * s.element_size())


def test_dynamic_tensor_subclass_and_custom_object_refused():
    class CustomTensor(torch.Tensor):
        pass
    graph, s = trace(Ada()), torch.randn(2, 8)
    with torch.no_grad(), partial_evaluate(graph, {'s': s}) as prepared:
        with pytest.raises(ValueError, match='ordinary'):
            prepared(torch.randn(2, 8).as_subclass(CustomTensor))
        with pytest.raises(ValueError, match='literal'):
            prepared(object())
