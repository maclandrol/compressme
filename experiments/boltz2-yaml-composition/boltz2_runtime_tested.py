"""Opt-in invariant conditioning for an unmodified, pinned Boltz-2 sampler.

Only boltz2_invariant_sampling is public. Internal prepared modules rely on its
source-specific immutable-input proof and must not be used as arbitrary-input
module replacements. All cached tensors are discarded after each request.
"""
from contextlib import contextmanager, ExitStack
import torch
from torch import nn

__all__ = ['boltz2_invariant_sampling']

def signature(t):return tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.layout

def require_frozen_plain(module):
    if any(type(p) is not nn.Parameter for p in module.parameters()) or any(type(b) is not torch.Tensor for b in module.buffers()):
        raise ValueError('Tensor/Parameter subclasses are outside the plain-state contract')
    if torch.is_grad_enabled():raise ValueError('No-grad or inference execution is required')
    if any(child.training for child in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise ValueError('Every descendant must be frozen and in eval mode')
    if any(buffer.requires_grad for buffer in module.buffers()):
        raise ValueError('Gradient-bearing buffers are unsupported')
    from torch.nn.modules import module as module_api
    for name in ('_global_forward_hooks','_global_forward_pre_hooks','_global_backward_hooks'):
        if getattr(module_api,name,{}):raise ValueError('Global module hooks are unsupported')
    for child in module.modules():
        if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
            raise ValueError('Module hooks are outside this prototype contract')
        if 'forward' in child.__dict__ or getattr(child,'_compiled_call_impl',None) is not None:
            raise ValueError('Instance forward overrides and compiled wrappers are unsupported')

def autocast_state(t):
    enabled=torch.is_autocast_enabled(t.device.type)
    return (enabled,torch.get_autocast_dtype(t.device.type) if enabled else None)

class PreparedAdaLN(nn.Module):
    def __init__(self,source,s):
        super().__init__();self.a_norm=source.a_norm
        normalized=source.s_norm(s)
        # Preserve original operation order/dtype/shapes. No matrix folding.
        self.register_buffer('scale',torch.sigmoid(source.s_scale(normalized)),persistent=False)
        self.register_buffer('bias',source.s_bias(normalized),persistent=False)
        self.eval()
    def forward(self,a,s):
        # Internal pinned-owner wrapper: the score controller validates its static inputs.
        return self.scale*self.a_norm(a)+self.bias

class PreparedOutputGate(nn.Module):
    def __init__(self,source,s):
        super().__init__()
        self.register_buffer('value',source(s),persistent=False);self.eval()
    def forward(self,s):
        return self.value

class PreparedSingleConditioningPrefix(nn.Module):
    """Same time-dependent branch with an exact request-constant prefix.

    strunk/sinputs are the exact already-repeated source tensors, not narrower
    representatives; preserving batch shape avoids changing GEMM rounding.
    All weights and these declared request inputs must stay immutable.
    """
    def __init__(self,source,s_trunk,s_inputs):
        super().__init__();require_frozen_plain(source);self.source=source
        prefix=source.single_embed(source.norm_single(torch.cat((s_trunk,s_inputs),dim=-1)))
        self.register_buffer('prefix',prefix,persistent=False);self.eval()
    def forward(self,times,s_trunk,s_inputs):
        # Exact source callsite owns the repeated layouts; no per-node revalidation.
        s=self.prefix
        if not self.source.disable_times:
            fourier_embed=self.source.fourier_embed(times)
            normed_fourier=self.source.norm_fourier(fourier_embed)
            fourier_to_single=self.source.fourier_to_single(normed_fourier)
            from einops import rearrange
            s=rearrange(fourier_to_single,'b d -> b 1 d')+s
        for transition in self.source.transitions:s=transition(s)+s
        return s,normed_fourier if not self.source.disable_times else None


class PreparedAtomConditioning:
    """Reusable values for one exact layout, activatable only within its request."""
    def __init__(self,atom_transformer,c_repeated):
        require_frozen_plain(atom_transformer)
        W=atom_transformer.attn_window_queries;B,N,D=c_repeated.shape
        s=c_repeated.view(B*(N//W),W,-1);self.entries=[];self.values=0
        for layer in atom_transformer.diffusion_transformer.layers:
            for owner,name in [(layer,'adaln'),(layer.transition,'adaln')]:
                original=getattr(owner,name);replacement=PreparedAdaLN(original,s)
                self.entries.append((owner,name,original,replacement))
                self.values+=replacement.scale.numel()+replacement.bias.numel()
            for owner,name in [(layer,'output_projection'),(layer.transition,'output_projection')]:
                original=getattr(owner,name);replacement=PreparedOutputGate(original,s)
                self.entries.append((owner,name,original,replacement));self.values+=replacement.value.numel()
        self.bytes=self.values*c_repeated.element_size()
    @contextmanager
    def activate(self):
        installed=[]
        try:
            for owner,name,original,replacement in self.entries:
                if getattr(owner,name) is not original:raise ValueError('Unexpected module change inside request')
                setattr(owner,name,replacement);installed.append((owner,name,original))
            yield
        finally:
            for owner,name,original in reversed(installed):setattr(owner,name,original)

def version(t):
    try:return t._version
    except RuntimeError:return None

def state_token(t):
    return id(t),signature(t),t.data_ptr(),t.storage_offset(),version(t)

PINNED_BOLTZ_REVISION = 'b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc'
PINNED_SOURCE_SHA256 = {
    'boltz.model.modules.diffusionv2': 'e68643da577decddaef25e97313fa0d014d5653752d0a97282ec1a78665d697a',
    'boltz.model.modules.encodersv2': '2938af17252607d056106fc04a3afa0ea96b145fa7485764ebf33d646c99e0d4',
    'boltz.model.modules.transformersv2': '4fbecbe835ec2d7286d5a46e408cb9134b2378bce8dd443df58b80392736f982',
}


def _verify_source_and_hierarchy(diffusion):
    """Validate the source-dependent proof boundaries before preparing any values.

    Exact source classes are required; monkeypatching class methods or replacing
    their globals is outside the explicitly unmodified-source contract.
    """
    import hashlib
    import importlib
    from pathlib import Path
    modules = {}
    for name, digest in PINNED_SOURCE_SHA256.items():
        source = importlib.import_module(name)
        if hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unaudited Boltz source: {name}')
        modules[name] = source
    d = modules['boltz.model.modules.diffusionv2']
    e = modules['boltz.model.modules.encodersv2']
    t = modules['boltz.model.modules.transformersv2']

    def exact(obj, cls, path):
        if type(obj) is not cls:
            raise ValueError(f'Expected unmodified exact {cls.__name__} at {path}')

    exact(diffusion, d.AtomDiffusion, 'structure_module')
    score = diffusion.score_model
    exact(score, d.DiffusionModule, 'score_model')
    exact(score.single_conditioner, e.SingleConditioning, 'single_conditioner')
    for name in (('norm_single', 'norm_fourier') if not score.single_conditioner.disable_times else ('norm_single',)):
        exact(getattr(score.single_conditioner, name), nn.LayerNorm, 'single_conditioner.' + name)
    exact(score.single_conditioner.single_embed, nn.Linear, 'single_conditioner.single_embed')
    for outer_name, inner_name, outer_cls in (
        ('atom_attention_encoder', 'atom_encoder', e.AtomAttentionEncoder),
        ('atom_attention_decoder', 'atom_decoder', e.AtomAttentionDecoder),
    ):
        outer = getattr(score, outer_name)
        exact(outer, outer_cls, outer_name)
        atom = getattr(outer, inner_name)
        exact(atom, t.AtomTransformer, outer_name + '.' + inner_name)
        block = atom.diffusion_transformer
        exact(block, t.DiffusionTransformer, 'diffusion_transformer')
        exact(block.layers, nn.ModuleList, 'diffusion_transformer.layers')
        for layer in block.layers:
            exact(layer, t.DiffusionTransformerLayer, 'diffusion_transformer.layer')
            exact(layer.transition, t.ConditionedTransitionBlock, 'transition')
            for owner in (layer, layer.transition):
                exact(owner.adaln, t.AdaLN, 'adaln')
                for attr in ('a_norm', 's_norm'):
                    exact(getattr(owner.adaln, attr), nn.LayerNorm, 'adaln.' + attr)
                for attr in ('s_scale', 's_bias'):
                    exact(getattr(owner.adaln, attr), nn.Linear, 'adaln.' + attr)
                exact(owner.output_projection, nn.Sequential, 'output_projection')
                if len(owner.output_projection) != 2:
                    raise ValueError('Modified condition output projection')
                exact(owner.output_projection[0], nn.Linear, 'output_projection.0')
                exact(owner.output_projection[1], nn.Sigmoid, 'output_projection.1')
    return score


def _boundary_signature(module):
    """Resolve current bindings, rather than only tracking formerly bound tensors."""
    rows = []
    for name, child in module.named_modules():
        config = tuple((key, value) for key, value in child.__dict__.items()
                       if not key.startswith('_') and (isinstance(value, (bool, int, float, str, type(None)))
                           or isinstance(value, tuple) and all(isinstance(v, (bool, int, float, str, type(None))) for v in value)))
        bindings = tuple((key, state_token(value) if value is not None else None)
                         for key, value in list(child._parameters.items()) + list(child._buffers.items()))
        rows.append((name, id(child), type(child), child.training, config, bindings))
    return tuple(rows)


@contextmanager
def boltz2_invariant_sampling(model, *, immutable_request):
    """Reuse declared-invariant conditioning only within each Boltz sampling call.

    This opt-in adapter accepts a model returned by ``load_boltz2(...)["confidence"]``
    or ``["affinity"]``, or its native AtomDiffusion. It requires the pinned source,
    exact affected classes, frozen eval descendants, no hooks/instance overrides,
    no-grad execution and exclusive ownership of the model for the entire scope.

    ``immutable_request=True`` is an explicit contract: model state, settings,
    source methods and conditioning data may not mutate during native sampling.
    Model state and hierarchy must also remain unchanged between requests inside
    the outer context; open a new context after any deliberate model edit.
    Original weight/module bindings and versions are checked at each request's
    entry and exit; a detected mutation rejects the request before a normal
    return. Static tensor identity, layout and tracked versions are also checked
    per score call. Unsafe writes bypassing tensor versions and temporary changes
    reversed before the boundary remain prohibited, rather than claimed detected.
    Autocast-enabled or unsupported score layouts run the original path.

    Prepared wrappers stay installed for consecutive score calls with the same
    actual multiplicity and are restored on a layout switch or request exit.
    Cached tensors contain only request conditioning computations, never model
    predictions; all are discarded in ``finally``. Parameters and the original
    dynamic operations, tensor shapes, RNG sequence and sampling schedule remain.
    Concurrent/reentrant requests and training are unsupported.
    """
    if immutable_request is not True:
        raise ValueError('This adapter requires the explicit immutable-request contract')
    diffusion = model.structure_module if hasattr(model, 'structure_module') else model
    score = _verify_source_and_hierarchy(diffusion)
    require_frozen_plain(diffusion)
    if any(name in diffusion.__dict__ for name in ('sample', 'preconditioned_network_forward', 'sample_schedule')):
        raise ValueError('Existing sampler method overrides are unsupported')
    original_sample = diffusion.sample
    original_forward = score.forward
    original_single = score.single_conditioner
    outer_state = _boundary_signature(diffusion)
    active = False
    statistics = {'requests': [], 'source_revision': PINNED_BOLTZ_REVISION,
                  'contract': 'exclusive_immutable_native_request'}

    def sample_with_constants(*sample_args, **sample_kwargs):
        nonlocal active
        if active:
            raise RuntimeError('Concurrent/reentrant sampling is unsupported')
        require_frozen_plain(score)
        if score.single_conditioner is not original_single:
            raise ValueError('Model hierarchy changed during the outer scope')
        before = _boundary_signature(diffusion)
        if before != outer_state:
            raise ValueError('Model hierarchy or state changed during the outer scope')
        memo = {}
        installed = ExitStack()
        installed_mult = None
        report = {'score_calls': 0, 'optimized_score_calls': 0, 'fallback_score_calls': 0,
                  'prepared_layouts': [], 'cached_bytes': 0, 'cache_cleared': False,
                  'boundary_state_unchanged': False, 'wrapper_installations': 0}
        conditioning = sample_kwargs.get('diffusion_conditioning')
        trunk = sample_kwargs.get('s_trunk')
        inputs = sample_kwargs.get('s_inputs')
        eligible = (type(conditioning) is dict and type(conditioning.get('c')) is torch.Tensor
                    and type(trunk) is torch.Tensor and type(inputs) is torch.Tensor)
        c = conditioning['c'] if eligible else None
        if eligible and any(t.layout is not torch.strided or t.device.type not in ('cpu', 'mps', 'cuda') for t in (c, trunk, inputs)):
            eligible = False
        if eligible and any(t.requires_grad for t in (c, trunk, inputs)):
            eligible = False
        initial_inputs = tuple(state_token(t) for t in (c, trunk, inputs)) if eligible else None

        def restore_prepared():
            nonlocal installed, installed_mult
            score.single_conditioner = original_single
            installed.close()
            installed = ExitStack()
            installed_mult = None

        def score_with_constants(*score_args, **score_kwargs):
            nonlocal installed_mult
            report['score_calls'] += 1
            mult = score_kwargs.get('multiplicity', 1)
            same = (eligible and not score_args and type(mult) is int and mult > 0
                    and score_kwargs.get('diffusion_conditioning') is conditioning
                    and conditioning.get('c') is c and score_kwargs.get('s_trunk') is trunk
                    and score_kwargs.get('s_inputs') is inputs)
            if same and initial_inputs != tuple(state_token(t) for t in (c, trunk, inputs)):
                raise RuntimeError('Immutable conditioning changed during sampling')
            same = same and not torch.is_grad_enabled() and not autocast_state(c)[0]
            if not same:
                restore_prepared()
                report['fallback_score_calls'] += 1
                return original_forward(*score_args, **score_kwargs)
            if mult != installed_mult:
                restore_prepared()
                if mult not in memo:
                    repeated_c = c.float().repeat_interleave(mult, 0)
                    encoder = PreparedAtomConditioning(score.atom_attention_encoder.atom_encoder, repeated_c)
                    decoder = PreparedAtomConditioning(score.atom_attention_decoder.atom_decoder, repeated_c)
                    single = PreparedSingleConditioningPrefix(original_single, trunk.repeat_interleave(mult, 0), inputs.repeat_interleave(mult, 0))
                    memo[mult] = encoder, decoder, single
                    memory = encoder.bytes + decoder.bytes + single.prefix.numel() * single.prefix.element_size()
                    report['prepared_layouts'].append({'multiplicity': mult, 'c_shape': list(repeated_c.shape),
                                                      'prefix_shape': list(single.prefix.shape), 'cached_bytes': memory})
                    report['cached_bytes'] += memory
                encoder, decoder, single = memo[mult]
                installed.enter_context(encoder.activate())
                installed.enter_context(decoder.activate())
                score.single_conditioner = single
                installed_mult = mult
                report['wrapper_installations'] += 1
            report['optimized_score_calls'] += 1
            return original_forward(*score_args, **score_kwargs)

        score.forward = score_with_constants
        active = True
        try:
            return original_sample(*sample_args, **sample_kwargs)
        finally:
            restore_prepared()
            del score.forward
            memo.clear()
            report['cache_cleared'] = True
            report['boundary_state_unchanged'] = before == _boundary_signature(diffusion)
            statistics['requests'].append(report)
            active = False
            require_frozen_plain(score)
            if not report['boundary_state_unchanged']:
                raise RuntimeError('Immutable model state changed during sampling; request rejected')
    diffusion.sample = sample_with_constants
    try:
        yield statistics
    finally:
        del diffusion.sample
        if 'forward' in score.__dict__:
            del score.forward


# Compatibility for the independent native verification harness.
@contextmanager
def constant_conditioning_sampling_scope(model):
    """Research-harness compatibility; public callers must state the contract."""
    with boltz2_invariant_sampling(model, immutable_request=True) as stats:
        yield stats
