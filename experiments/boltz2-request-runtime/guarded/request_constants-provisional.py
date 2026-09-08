"""Experimental request-scoped partial evaluation for pinned Boltz subgraphs.

This is a declared-static-input primitive, not a standalone arbitrary-input
model replacement. Its owner must prove that every s passed during the scope
is the same request conditioning, with the prepared layout. Boltz's sample
route has that property for c/c_skip. No cross-request cache is retained.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from einops import rearrange

def signature(t):return tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.layout

def require_frozen_plain(module):
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

def check_layout(x,expected):
    if signature(x)!=expected:raise ValueError('Static input layout changed')

class PreparedAdaLN(nn.Module):
    def __init__(self,source,s):
        super().__init__();self.a_norm=source.a_norm;self.expected=signature(s)
        normalized=source.s_norm(s)
        # Preserve original operation order/dtype/shapes. No matrix folding.
        self.register_buffer('scale',torch.sigmoid(source.s_scale(normalized)),persistent=False)
        self.register_buffer('bias',source.s_bias(normalized),persistent=False)
        self.eval()
    def forward(self,a,s):
        check_layout(s,self.expected)
        return self.scale*self.a_norm(a)+self.bias

class PreparedOutputGate(nn.Module):
    def __init__(self,source,s):
        super().__init__();self.expected=signature(s)
        self.register_buffer('value',source(s),persistent=False);self.eval()
    def forward(self,s):
        check_layout(s,self.expected);return self.value

@contextmanager
def static_atom_conditioning_scope(atom_transformer,c_repeated):
    """Temporarily prepare condition-only nodes; restore every module on exit.

    c_repeated must already have the exact source repeat_interleave layout.
    c must remain request-constant and weights immutable within this scope.
    The caller needs exclusive ownership: concurrent calls are unsupported.
    Original attention, transitions, masks, to_keys and residuals are untouched.
    """
    require_frozen_plain(atom_transformer)
    W=atom_transformer.attn_window_queries;B,N,D=c_repeated.shape
    s=c_repeated.view(B*(N//W),W,-1)
    saved=[];prepared_values=0
    try:
        for layer in atom_transformer.diffusion_transformer.layers:
            for owner,name in [(layer,'adaln'),(layer.transition,'adaln')]:
                original=getattr(owner,name);replacement=PreparedAdaLN(original,s)
                saved.append((owner,name,original));setattr(owner,name,replacement)
                prepared_values+=replacement.scale.numel()+replacement.bias.numel()
            for owner,name in [(layer,'output_projection'),(layer.transition,'output_projection')]:
                original=getattr(owner,name);replacement=PreparedOutputGate(original,s)
                saved.append((owner,name,original));setattr(owner,name,replacement)
                prepared_values+=replacement.value.numel()
        yield {'prepared_values':prepared_values,'prepared_bytes':prepared_values*c_repeated.element_size(),'layers':len(atom_transformer.diffusion_transformer.layers),'source_condition_shape':list(s.shape),'condition_linear_calls_per_original_forward':6*len(atom_transformer.diffusion_transformer.layers),'condition_layernorm_calls_per_original_forward':2*len(atom_transformer.diffusion_transformer.layers),'condition_sigmoid_calls_per_original_forward':4*len(atom_transformer.diffusion_transformer.layers)}
    finally:
        for owner,name,original in reversed(saved):setattr(owner,name,original)
        saved.clear()

class PreparedSingleConditioningPrefix(nn.Module):
    """Same time-dependent branch with an exact request-constant prefix.

    strunk/sinputs are the exact already-repeated source tensors, not narrower
    representatives; preserving batch shape avoids changing GEMM rounding.
    All weights and these declared request inputs must stay immutable.
    """
    def __init__(self,source,s_trunk,s_inputs):
        super().__init__();require_frozen_plain(source);self.source=source
        self.trunk_signature=signature(s_trunk);self.input_signature=signature(s_inputs)
        prefix=source.single_embed(source.norm_single(torch.cat((s_trunk,s_inputs),dim=-1)))
        self.register_buffer('prefix',prefix,persistent=False);self.eval()
    def forward(self,times,s_trunk,s_inputs):
        check_layout(s_trunk,self.trunk_signature);check_layout(s_inputs,self.input_signature)
        s=self.prefix
        if not self.source.disable_times:
            fourier_embed=self.source.fourier_embed(times)
            normed_fourier=self.source.norm_fourier(fourier_embed)
            fourier_to_single=self.source.fourier_to_single(normed_fourier)
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

@contextmanager
def constant_conditioning_sampling_scope(model):
    """Enable invariant partial evaluation only inside each native sample call.

    Prototype requirements: pinned uncompiled Boltz modules, frozen eval weights,
    no module hooks and exclusive per-model request ownership. The original
    sampler, score forward, attention, residuals and nonlinear transitions run.
    Caches are private to one sample invocation and cleared in finally. This
    adapter never keeps predictions or caches values across requests.

    Parameter/input mutation during sampling is outside the pure-input contract.
    Tracked Tensor versions are checked before use without reading GPU data;
    an unexpected static-input change falls back to the original score call.
    """
    from boltz.model.modules.diffusionv2 import AtomDiffusion,DiffusionModule
    diffusion=model.structure_module if hasattr(model,'structure_module') else model
    if type(diffusion) is not AtomDiffusion or type(diffusion.score_model) is not DiffusionModule:
        raise ValueError('Only the audited uncompiled Boltz sampler/score classes are supported')
    score=diffusion.score_model;require_frozen_plain(score)
    if 'forward' in score.__dict__ or 'sample' in diffusion.__dict__:
        raise ValueError('Existing instance forward/sample overrides are unsupported')
    original_sample=diffusion.sample;original_forward=score.forward;original_single=score.single_conditioner
    active=False;statistics={'requests':[]}
    def sample_with_constants(*sample_args,**sample_kwargs):
        nonlocal active
        if active:raise RuntimeError('Concurrent/reentrant sampling is unsupported by this prototype')
        active=True;memo={};report={'score_calls':0,'optimized_score_calls':0,'fallback_score_calls':0,'prepared_layouts':[],'cached_bytes':0,'cache_cleared':False}
        static_conditioning=sample_kwargs.get('diffusion_conditioning');static_trunk=sample_kwargs.get('s_trunk');static_inputs=sample_kwargs.get('s_inputs')
        eligible=isinstance(static_conditioning,dict) and isinstance(static_conditioning.get('c'),torch.Tensor) and isinstance(static_trunk,torch.Tensor) and isinstance(static_inputs,torch.Tensor)
        if eligible and any(t.requires_grad for t in (static_conditioning['c'],static_trunk,static_inputs)):eligible=False
        static_c=static_conditioning['c'] if eligible else None
        input_versions=[version(t) for t in (static_c,static_trunk,static_inputs)] if eligible else []
        parameters=list(score.parameters())+list(score.buffers());parameter_versions=[version(p) for p in parameters]
        def score_with_constants(*score_args,**score_kwargs):
            report['score_calls']+=1
            mult=score_kwargs.get('multiplicity',1)
            same=eligible and not score_args and type(mult) is int and mult>0
            same=same and score_kwargs.get('diffusion_conditioning') is static_conditioning and static_conditioning.get('c') is static_c and score_kwargs.get('s_trunk') is static_trunk and score_kwargs.get('s_inputs') is static_inputs
            if same:
                same=input_versions==[version(t) for t in (static_c,static_trunk,static_inputs)] and parameter_versions==[version(p) for p in parameters]
            if not same:
                report['fallback_score_calls']+=1;return original_forward(*score_args,**score_kwargs)
            if mult not in memo:
                c=static_c.float().repeat_interleave(mult,0)
                encoder=PreparedAtomConditioning(score.atom_attention_encoder.atom_encoder,c)
                decoder=PreparedAtomConditioning(score.atom_attention_decoder.atom_decoder,c)
                single=PreparedSingleConditioningPrefix(original_single,static_trunk.repeat_interleave(mult,0),static_inputs.repeat_interleave(mult,0))
                memo[mult]=(encoder,decoder,single)
                memory=encoder.bytes+decoder.bytes+single.prefix.numel()*single.prefix.element_size()
                report['prepared_layouts'].append({'multiplicity':mult,'c_shape':list(c.shape),'prefix_shape':list(single.prefix.shape),'cached_bytes':memory})
                report['cached_bytes']+=memory
            encoder,decoder,single=memo[mult]
            report['optimized_score_calls']+=1
            with encoder.activate(),decoder.activate():
                if score.single_conditioner is not original_single:raise ValueError('Single conditioner changed during request')
                score.single_conditioner=single
                try:return original_forward(*score_args,**score_kwargs)
                finally:score.single_conditioner=original_single
        score.forward=score_with_constants
        try:return original_sample(*sample_args,**sample_kwargs)
        finally:
            del score.forward
            memo.clear();report['cache_cleared']=True;statistics['requests'].append(report);active=False
    diffusion.sample=sample_with_constants
    try:yield statistics
    finally:
        del diffusion.sample
        if 'forward' in score.__dict__:del score.forward
