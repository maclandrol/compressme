"""Research-only first-layer lookup inside native Llama 4.46.2.

This explicitly token-only adapter preserves the native computation after the
first projections. It is not a released compressor or an inputs_embeds adapter.
The context holds temporary projections for one call and is always reset.
"""
import contextvars
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

_CURRENT = contextvars.ContextVar("compressme_token_qkv", default=None)
CHECKSUM = "3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699"
CONFIG_CHECKSUM = "f3cc487205aa417f4786f50d49a4120097baaffd80a9510ccfe5c37aa4ed3a34"
SOURCE_CHECKSUM = "733e7625fec5abcfa416ab9b866cb91875d41007f2e31c3e90d2ae0111ae98f7"
FANOUT_CHECKSUM = "b83130353cf74b89c6a274259b84f17824041349f07576ae7acdff0315343fbb"

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _TokenProjection(nn.Module):
    def __init__(self, branch, input_width, output_width):
        super().__init__()
        self.branch = branch
        self.in_features, self.out_features = input_width, output_width

    def forward(self, value):
        outputs = _CURRENT.get()
        if outputs is None:
            raise ValueError("Call the enclosing token-only model, not this internal projection")
        result = outputs[self.branch]
        if result.shape[:-1] != value.shape[:-1] or result.device != value.device:
            raise ValueError("The token lookup context does not match the original projection input")
        return result


class TokenLookupCausalLM(LlamaForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None, num_logits_to_keep=0, **loss_kwargs):
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("Research contract is token IDs only; arbitrary inputs_embeds requires original weights")
        if self.training or any(p.dtype != torch.float32 for p in self._first_qkv_lookup.parameters()):
            raise ValueError("Research adapter requires eval and the original float32 dtype")
        if input_ids.ndim != 2:
            raise ValueError("Research contract requires two-dimensional token IDs")
        outputs = self._first_qkv_lookup(input_ids)
        handle = _CURRENT.set(tuple(outputs[name] for name in ('q', 'k', 'v')))
        try:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=None, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep, **loss_kwargs)
        finally:
            _CURRENT.reset(handle)


def load_reference(checkpoint, config, *, device="cpu", backend="eager"):
    import inspect
    import transformers
    if transformers.__version__ != "4.46.2":
        raise ValueError("Use the isolated pinned Transformers 4.46.2 runtime")
    if file_sha256(checkpoint) != CHECKSUM:
        raise ValueError("Expected the pinned 32M AtomWise checkpoint")
    if file_sha256(config) != CONFIG_CHECKSUM:
        raise ValueError("Expected the pinned native 32M AtomWise config")
    if file_sha256(inspect.getfile(LlamaForCausalLM)) != SOURCE_CHECKSUM:
        raise ValueError("Expected the audited native Transformers 4.46.2 model source")
    cfg = LlamaConfig.from_json_file(config)
    cfg._attn_implementation = backend
    with torch.random.fork_rng(devices=[]):
        model = LlamaForCausalLM(cfg).eval()
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model.requires_grad_(False).to(device)


def make_candidate(source, *, table_device="cpu", lookup_artifact=None):
    import inspect
    from compressme.finite_fanout import FiniteTokenFanout, Float32RMSNorm, compile_finite_fanout
    from compressme.serialization import save, load
    if file_sha256(inspect.getfile(compile_finite_fanout)) != FANOUT_CHECKSUM:
        raise ValueError('Expected the audited general finite-fanout implementation; revalidate changed code')
    if source.config.pretraining_tp != 1:
        raise ValueError("This prototype does not support tensor-parallel weight reads")
    source_layer = source.model.layers[0]
    source_device = source.model.embed_tokens.weight.device
    declared = FiniteTokenFanout(
        copy.deepcopy(source.model.embed_tokens),
        nn.Sequential(Float32RMSNorm(copy.deepcopy(source_layer.input_layernorm.weight),
                                    eps=source_layer.input_layernorm.variance_epsilon)),
        {name: copy.deepcopy(getattr(source_layer.self_attn, name+'_proj')) for name in ('q','k','v')},
        residual_key=None).eval().to(table_device).requires_grad_(False)
    profiles=[{'max_rows':1,'evaluation_rows':1}]
    if torch.device(table_device).type == 'mps':profiles.append({'max_rows':15,'evaluation_rows':2})
    result=compile_finite_fanout(declared,input_contract='token_indices_only',row_count_profiles=profiles)
    if result.report['status'] != 'accepted_on_full_domain_validation':
        raise RuntimeError('General fanout rejected: '+repr(result.report))
    if lookup_artifact is not None:
        save(result,lookup_artifact)
        result=load(lambda:copy.deepcopy(declared).cpu(),lookup_artifact)
    lookup=result.model.to(source_device)
    candidate = copy.deepcopy(source)
    candidate.__class__ = TokenLookupCausalLM
    candidate._first_qkv_lookup = lookup
    candidate._first_qkv_compilation_report = result.report
    candidate._first_qkv_compilation_report['implementation_source_sha256'] = FANOUT_CHECKSUM
    candidate._first_qkv_widths = tuple(getattr(source_layer.self_attn,name+'_proj').out_features for name in ('q','k','v'))
    first = candidate.model.layers[0]
    first.input_layernorm = nn.Identity().eval()
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        setattr(first.self_attn, name, _TokenProjection(index, source.config.hidden_size, candidate._first_qkv_widths[index]).eval())
    return candidate.eval()
