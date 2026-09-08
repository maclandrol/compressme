"""Independent small routing/error tests; native numerical gates are separate."""
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

import boltz2_runtime as candidate
from test_scope_guards import fixture, local_forward, DiffusionModule, AtomDiffusion


@contextmanager
def harness(body):
    with torch.no_grad():
        model, inputs = fixture()
        with patch.object(candidate, '_verify_source_and_hierarchy', lambda d: d.score_model), \
             patch.object(DiffusionModule, 'forward', local_forward), \
             patch.object(AtomDiffusion, 'sample', body):
            yield model, inputs


def test_fallback_really_uses_new_condition_values():
    def body(model, **inputs):
        model.score_model(**inputs, times=torch.zeros(1))
        other = dict(inputs, s_trunk=inputs['s_trunk'] + torch.tensor([1.0, 0.0]))
        actual = model.score_model(**other, times=torch.zeros(1))
        assert not isinstance(model.score_model.single_conditioner, candidate.PreparedSingleConditioningPrefix)
        expected = local_forward(model.score_model, **other, times=torch.zeros(1))
        assert torch.equal(actual, expected)
        return actual
    with harness(body) as (model, inputs):
        with candidate.boltz2_invariant_sampling(model, immutable_request=True) as stats:
            model.sample(**inputs)
        assert stats['requests'][0]['fallback_score_calls'] == 1


def test_autocast_fallback_value_and_layout_reentry():
    def body(model, **inputs):
        model.score_model(**inputs, times=torch.zeros(1))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model.score_model(**inputs, times=torch.zeros(1))
            expected = local_forward(model.score_model, **inputs, times=torch.zeros(1))
            assert actual.dtype == expected.dtype and torch.equal(actual, expected)
        return [model.score_model(**inputs, times=torch.zeros(m), multiplicity=m) for m in (3, 1)]
    with harness(body) as (model, inputs):
        with candidate.boltz2_invariant_sampling(model, immutable_request=True) as stats:
            outputs = model.sample(**inputs)
        for output, mult in zip(outputs, (3, 1)):
            assert torch.equal(output, local_forward(model.score_model, **inputs, times=torch.zeros(mult), multiplicity=mult))
        assert len(stats['requests'][0]['prepared_layouts']) == 2


def test_partial_decoder_install_failure_restores_every_module():
    def body(model, **inputs):
        return model.score_model(**inputs, times=torch.zeros(1))
    with harness(body) as (model, inputs):
        original = {name: id(module) for name, module in model.named_modules()}
        activate = candidate.PreparedAtomConditioning.activate
        calls = 0
        @contextmanager
        def fail_second(prepared):
            nonlocal calls
            calls += 1
            with activate(prepared):
                if calls == 2:
                    raise RuntimeError('decoder activation failed')
                yield
        with patch.object(candidate.PreparedAtomConditioning, 'activate', fail_second):
            with pytest.raises(RuntimeError, match='decoder activation'):
                with candidate.boltz2_invariant_sampling(model, immutable_request=True) as stats:
                    model.sample(**inputs)
        assert original == {name: id(module) for name, module in model.named_modules()}
        assert 'forward' not in model.score_model.__dict__ and 'sample' not in model.__dict__
        assert stats['requests'][0]['cache_cleared']


def test_bad_input_does_not_poison_next_request():
    def body(model, **inputs):
        # This control-flow fixture ignores c, just as the unchanged native
        # fallback must be allowed to decide whether its own inputs are valid.
        return model.score_model(**inputs, times=torch.zeros(1))
    with harness(body) as (model, inputs):
        bad = dict(inputs, diffusion_conditioning={'c': inputs['diffusion_conditioning']['c'].to_sparse()})
        with candidate.boltz2_invariant_sampling(model, immutable_request=True):
            try:
                model.sample(**bad)
            except RuntimeError:
                pass
            result = model.sample(**inputs)
        assert torch.equal(result, local_forward(model.score_model, **inputs, times=torch.zeros(1)))


def test_between_request_submodule_change_rejected():
    def body(model, **inputs):
        return model.score_model(**inputs, times=torch.zeros(1))
    with harness(body) as (model, inputs):
        with candidate.boltz2_invariant_sampling(model, immutable_request=True):
            model.sample(**inputs)
            source = model.score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[0]
            source.adaln = type(source.adaln)(4, 4).eval().requires_grad_(False)
            with pytest.raises(ValueError, match='outer scope'):
                model.sample(**inputs)
