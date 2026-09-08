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
        if self.training or self._first_qkv_rows.dtype != torch.float32:
            raise ValueError("Research adapter requires eval and the original float32 dtype")
        if input_ids.ndim != 2:
            raise ValueError("Research contract requires two-dimensional token IDs")
        rows = self._first_qkv_singleton_rows if input_ids.numel() == 1 else self._first_qkv_rows
        packed = F.embedding(input_ids, rows)
        handle = _CURRENT.set(packed.split(self._first_qkv_widths, dim=-1))
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


def make_candidate(source, *, table_device="cpu"):
    if source.config.pretraining_tp != 1:
        raise ValueError("This prototype does not support tensor-parallel weight reads")
    source_layer = source.model.layers[0]
    source_device = source.model.embed_tokens.weight.device
    # Construct tables under the explicitly recorded backend; retain dtype.
    encoder = copy.deepcopy(source_layer.input_layernorm).to(table_device)
    embeddings = source.model.embed_tokens.weight.detach().to(table_device)
    widths, tables, singleton_tables = [], [], []
    with torch.inference_mode(False), torch.no_grad():
        encoded = encoder(embeddings)
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = copy.deepcopy(getattr(source_layer.self_attn, name)).to(table_device)
            widths.append(projection.out_features)
            tables.append(projection(encoded).to(source_device))
            singleton_tables.append(torch.cat([
                projection(encoder(embeddings[token].reshape(1, 1, -1))).reshape(1, -1)
                for token in range(embeddings.shape[0])
            ], dim=0).to(source_device))
        rows = torch.cat(tables, dim=-1).clone()
        singleton_rows = torch.cat(singleton_tables, dim=-1).clone()
    candidate = copy.deepcopy(source)
    candidate.__class__ = TokenLookupCausalLM
    candidate._first_qkv_rows = nn.Parameter(rows, requires_grad=False)
    candidate._first_qkv_singleton_rows = nn.Parameter(singleton_rows, requires_grad=False)
    candidate._first_qkv_widths = tuple(widths)
    first = candidate.model.layers[0]
    first.input_layernorm = nn.Identity().eval()
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        setattr(first.self_attn, name, _TokenProjection(index, source.config.hidden_size, widths[index]).eval())
    return candidate.eval()
