import copy
import pytest
import torch
from torch import nn
from compressme import ShapeMatchedConstantRows


def setup():
    torch.manual_seed(82)
    source = nn.Sequential(nn.Linear(17, 32), nn.LayerNorm(32), nn.SiLU()).eval()
    a, b = nn.Parameter(torch.randn(17)), nn.Parameter(torch.randn(1, 1, 17))
    return source, a, b


def test_known_rows_match_real_inputs_at_identical_shapes_and_cache():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows(max_entries=2)
    x = torch.randn(2, 11, 17)
    pairs = [(0, a), (10, b), (11, a), (21, b)]
    ids = torch.tensor([p[0] for p in pairs])
    with torch.inference_mode():
        x.reshape(-1, 17)[ids] = torch.stack([p[1].flatten() for p in pairs])
        expected = source(x).reshape(-1, 32)[ids]
        actual = cache(source, x.shape, pairs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        again = cache(source, x.shape, pairs)
        torch.testing.assert_close(again, expected, rtol=0, atol=0)
        assert cache.hits == 1 and cache.misses == 1
        again.add_(10)
        torch.testing.assert_close(cache(source, x.shape, pairs), expected, rtol=0, atol=0)
        assert cache._bytes == expected.numel() * expected.element_size()


def test_weight_constant_setting_and_dtype_changes_invalidate():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    with torch.inference_mode():
        first = cache(source, (3, 17), [(0, a)])
        source[0].bias.add_(.5)
        second = cache(source, (3, 17), [(0, a)])
        assert cache.misses == 2
        a.add_(1)
        cache(source, (3, 17), [(0, a)])
        assert cache.misses == 3
        source[1].eps = .2
        cache(source, (3, 17), [(0, a)])
        assert cache.misses == 4
        source.double(); a.data = a.data.double()
        cache(source, (3, 17), [(0, a)])
        assert cache.misses == 5


def test_lru_and_deepcopy_clear_only_parameter_derived_outputs():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows(max_entries=1)
    with torch.inference_mode():
        cache(source, (3, 17), [(0, a)])
        cache(source, (4, 17), [(0, a)])
        assert len(cache._cache) == 1
        cache(source, (3, 17), [(0, a)])
        assert cache.misses == 3
    clone = copy.deepcopy(cache)
    assert len(clone._cache) == clone._bytes == clone.hits == clone.misses == 0


def test_calls_needing_gradients_and_unversioned_tensors_are_never_cached():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    output = cache(source, (3, 17), [(0, a)])
    output.sum().backward()
    assert a.grad is not None and source[0].weight.grad is not None and not cache._cache
    with torch.inference_mode():
        inference_constant = torch.randn(17)
        cache(source, (3, 17), [(0, inference_constant)])
        cache(source, (3, 17), [(0, inference_constant)])
        assert not cache._cache and cache.hits == 0


def test_fully_frozen_plain_calls_cache_and_unfreezing_preserves_gradients():
    source, a, b = setup()
    source.requires_grad_(False)
    a.requires_grad_(False)
    cache = ShapeMatchedConstantRows()
    assert torch.is_grad_enabled()
    first = cache(source, (3, 17), [(0, a)])
    second = cache(source, (3, 17), [(0, a)])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not second.requires_grad and cache.hits == 1 and cache.misses == 1

    # A previously cached frozen result must never hide a newly requested
    # gradient, whether the trainable tensor is a constant or source weight.
    a.requires_grad_(True)
    cache(source, (3, 17), [(0, a)]).sum().backward()
    assert a.grad is not None and cache.hits == 1 and cache.misses == 2
    a.requires_grad_(False)
    source[0].weight.requires_grad_(True)
    cache(source, (3, 17), [(0, a)]).sum().backward()
    assert source[0].weight.grad is not None and cache.hits == 1 and cache.misses == 3

    source.requires_grad_(False)
    with torch.inference_mode():
        inference_constant = torch.randn(17)
    cache(source, (3, 17), [(0, inference_constant)])
    cache(source, (3, 17), [(0, inference_constant)])
    assert cache.hits == 1 and cache.misses == 5


def test_cross_token_ops_training_and_custom_forwards_refused():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    with pytest.raises(ValueError):
        cache(nn.Sequential(nn.Softmax(dim=0)).eval(), (3, 17), [(0, a)])
    with pytest.raises(ValueError):
        cache(source.train(), (3, 17), [(0, a)])
    source.eval(); source[0].forward = lambda x: x
    with pytest.raises(ValueError):
        cache(source, (3, 17), [(0, a)])


@pytest.mark.parametrize('shape,positions', [((17,), (0,)), ((2, 3, 5, 17), (0, 14, 29))])
def test_arbitrary_leading_dimensions_and_original_operation_shape(shape, positions, monkeypatch):
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    constants = [a if i % 2 == 0 else b for i in range(len(positions))]
    assignments = list(zip(positions, constants))
    x = torch.randn(shape)
    with torch.inference_mode():
        flat = x.reshape(-1, 17)
        flat[torch.tensor(positions)] = torch.stack([value.flatten() for value in constants])
        expected = source(x).reshape(-1, 32)[torch.tensor(positions)]
        observed = []
        original = torch.nn.functional.linear
        def record_linear(value, *args, **kwargs):
            observed.append(tuple(value.shape))
            return original(value, *args, **kwargs)
        monkeypatch.setattr(torch.nn.functional, 'linear', record_linear)
        actual = cache(source, shape, assignments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert observed == [shape]
        cache(source, shape, assignments)
        assert observed == [shape]
        assert cache._bytes == len(positions) * 32 * 4


def test_replacing_source_or_parameter_cannot_hit_old_cache():
    source, a, b = setup()
    replacement = copy.deepcopy(source)
    cache = ShapeMatchedConstantRows()
    with torch.inference_mode():
        cache(source, (3, 17), [(0, a)])
        cache(replacement, (3, 17), [(0, a)])
        assert cache.misses == 2 and cache.hits == 0
        source[0].weight = nn.Parameter(source[0].weight.clone() + .125)
        actual = cache(source, (3, 17), [(0, a)])
        template = torch.zeros(3, 17)
        template[0] = a
        torch.testing.assert_close(actual, source(template)[:1], rtol=0, atol=0)
        assert cache.misses == 3


def test_gradient_path_matches_known_rows_in_real_input():
    source, a, b = setup()
    reference = copy.deepcopy(source)
    left_a = a.detach().clone().requires_grad_()
    left_b = b.detach().clone().requires_grad_()
    right_a = a.detach().clone().requires_grad_()
    right_b = b.detach().clone().requires_grad_()
    positions = torch.tensor([0, 2, 5])
    x = torch.randn(2, 3, 17)
    x.reshape(-1, 17).index_copy_(0, positions,
        torch.stack((left_a.flatten(), left_b.flatten(), left_a.flatten())))
    expected = reference(x).reshape(-1, 32).index_select(0, positions)
    cache = ShapeMatchedConstantRows()
    actual = cache(source, x.shape, [(0, right_a), (2, right_b), (5, right_a)])
    loss_weights = torch.randn_like(expected)
    (expected * loss_weights).sum().backward()
    (actual * loss_weights).sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left_a.grad, right_a.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left_b.grad, right_b.grad, atol=1e-6, rtol=1e-6)
    for left, right in zip(reference.parameters(), source.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-6)
    assert cache._bytes == 0 and not cache._cache


def test_dtype_mismatches_autocast_and_invalid_shape_are_explicitly_refused():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    with pytest.raises(ValueError, match='same explicit dtype'):
        cache(source, (3, 17), [(0, a.double())])
    with pytest.raises(ValueError, match='share an explicit device/dtype'):
        cache(source, (3, 17), [(0, a), (1, b.double())])
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='autocast'):
        cache(source, (3, 17), [(0, a)])
    for shape, assignments in [((0, 17), [(0, a)]), ((3, 16), [(0, a)]),
                               ((3, 17), [(0, a), (0, b)]), ((3, 17), [(3, a)])]:
        with pytest.raises(ValueError):
            cache(source, shape, assignments)


def test_byte_limit_bounds_selected_rows_and_clear_releases_entries():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows(max_entries=10, max_bytes=256)
    with torch.inference_mode():
        for count in (3, 4, 5):
            cache(source, (count, 17), [(0, a)])
        assert len(cache._cache) == 2 and cache._bytes == 256
        cache(source, (8, 17), [(0, a), (1, b), (2, a)])
        assert len(cache._cache) == 2 and cache._bytes == 256
    cache.clear()
    assert len(cache._cache) == cache._bytes == cache.hits == cache.misses == 0


def test_execution_precision_setting_is_part_of_cache_identity():
    source, a, b = setup()
    cache = ShapeMatchedConstantRows()
    old = torch.get_float32_matmul_precision()
    try:
        with torch.inference_mode():
            torch.set_float32_matmul_precision('highest')
            cache(source, (3, 17), [(0, a)])
            torch.set_float32_matmul_precision('high')
            cache(source, (3, 17), [(0, a)])
            assert cache.misses == 2 and cache.hits == 0
    finally:
        torch.set_float32_matmul_precision(old)
