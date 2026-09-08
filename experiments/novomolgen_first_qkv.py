# The AuditedLlamaRMSNorm forward below is adapted from Transformers v4.46.2:
# https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adaptation: rename the class and accept a frozen checkpoint scale in its
# constructor. The normalization forward is preserved for a local numerical
# audit. The surrounding token-lookup candidate and audit code are separate.
# The pinned upstream license is retained in the accompanying audit bundle.

"""CPU audit of the first token-only Q/K/V span; no full-model replacement.

Use: python probe_first_qkv.py CHECKPOINT.safetensors --output RESULTS.json
No remote code is imported, no training occurs, and no weights are downloaded.
The small RMSNorm forward below reproduces the pinned upstream 4.46.2 code.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

REVISION = "dcd3f59261bebf84142c13617d4e129b4b0d0fdc"
SHA256 = "3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699"
UPSTREAM = "https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/llama/modeling_llama.py#L60"


class AuditedLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TokenOnlyQKV(nn.Module):
    """Local span only: preserve residual embeddings and pre-RoPE Q/K/V.

    This deliberately accepts token IDs only. It is not a LlamaForCausalLM
    substitute and does not accept arbitrary inputs_embeds or manage caches.
    """
    def __init__(self, embedding, q, k, v):
        super().__init__()
        self.embedding = nn.Parameter(embedding.clone(), requires_grad=False)
        self.rows = nn.Parameter(torch.cat((q, k, v), dim=-1), requires_grad=False)
        self.width = embedding.shape[-1]
        self.eval()

    def forward(self, ids):
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Local test contract: two-dimensional token IDs only")
        q, k, v = F.embedding(ids, self.rows).split(self.width, dim=-1)
        return {"q": q, "k": k, "v": v, "residual": F.embedding(ids, self.embedding)}


def metric(reference, candidate):
    if reference.shape != candidate.shape or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite output or changed shape")
    if not reference.numel():
        return {"max_abs": 0.0, "relative_l2": 0.0, "bitwise": True, "passed": True}
    left, right = reference.detach().cpu().double(), candidate.detach().cpu().double()
    delta = left - right
    absolute = delta.abs().max().item()
    relative = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(left).clamp_min(1e-30)).item()
    return {"max_abs": absolute, "relative_l2": relative,
            "bitwise": torch.equal(reference, candidate),
            "passed": absolute <= 1e-5 and relative <= 1e-5}


def run(checkpoint):
    checksum = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    if checksum.hexdigest() != SHA256:
        raise ValueError("This bounded audit requires the exact published checkpoint")
    state = load_file(str(checkpoint), device="cpu")
    dtype_counts = {}
    for value in state.values():
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + value.numel()
    E = state["model.embed_tokens.weight"]
    gamma = state["model.layers.0.input_layernorm.weight"]
    weights = [state[f"model.layers.0.self_attn.{name}_proj.weight"] for name in ("q", "k", "v")]
    if E.shape != (84, 512) or gamma.shape != (512,) or any(W.shape != (512, 512) for W in weights):
        raise ValueError("Checkpoint shape differs from the audited variant")
    if any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in [E, gamma, *weights]):
        raise ValueError("First-layer source is not finite float32")
    norm = AuditedLlamaRMSNorm(gamma).eval()
    torch.manual_seed(543)
    with torch.inference_mode():
        source_rows = norm(E)
        tables = [F.linear(source_rows, W) for W in weights]
    # Normal tensors preserve portable mutation/version semantics.
    with torch.inference_mode(False), torch.no_grad():
        candidate = TokenOnlyQKV(E, *[x.clone() for x in tables])
    cases = [("all_tokens", torch.arange(84).reshape(1, 84)),
             ("all_tokens_reversed", torch.arange(83, -1, -1).reshape(2, 42))]
    for token in range(84):
        cases.append((f"singleton_{token}", torch.tensor([[token]])))
    for batch, length in [(1, 2), (4, 17), (4, 64), (1, 512), (2, 2048), (0, 5), (3, 0)]:
        cases.append((f"random_{batch}x{length}", torch.randint(0, 84, (batch, length))))
    checks = []
    with torch.inference_mode():
        for label, ids in cases:
            embedded = F.embedding(ids, E)
            normalized = norm(embedded)
            reference = {name: F.linear(normalized, W) for name, W in zip(("q", "k", "v"), weights)}
            reference["residual"] = embedded
            output = candidate(ids)
            checks.append({"case": label, "shape": list(ids.shape),
                           "outputs": {name: metric(value, output[name]) for name, value in reference.items()}})
    entries = [entry for case in checks for entry in case["outputs"].values()]
    source_span = E.numel() + gamma.numel() + sum(W.numel() for W in weights)
    candidate_span = sum(value.numel() for value in candidate.state_dict().values())
    total = sum(value.numel() for value in state.values())
    return {"audit_date": "2026-09-07", "repository": "chandar-lab/NovoMolGen_32M_SMILES_AtomWise",
            "revision": REVISION, "checkpoint_sha256_verified": checksum.hexdigest(),
            "checkpoint_bytes": Path(checkpoint).stat().st_size,
            "checkpoint_tensor_count": len(state), "dtype_values": dtype_counts,
            "normalization": "custom LlamaRMSNorm, not nn.RMSNorm", "normalization_source": UPSTREAM,
            "runtime": {"torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
            "scope": "First token-only embedding/RMSNorm/QKV span; no RoPE/attention/cache/full-model execution",
            "contract": "Explicit two-dimensional token IDs; arbitrary inputs_embeds not supported by this local candidate",
            "source_span_values": source_span, "candidate_span_values": candidate_span,
            "saved_values": source_span - candidate_span, "saved_float32_bytes": 4 * (source_span - candidate_span),
            "candidate_stores_original_residual_embeddings": True,
            "candidate_keeps_original_projection_weights": False,
            "source_total_values": total,
            "estimated_whole_model_values_if_integrated": total - source_span + candidate_span,
            "nominal_dense_qkv_multiply_accumulates_per_token_removed": 3 * 512 * 512,
            "speed_measured": False, "timing_claim": None,
            "numerical_gate": {"absolute": 1e-5, "relative_l2": 1e-5},
            "cases": checks, "output_comparisons": len(entries),
            "failed_comparisons": sum(not entry["passed"] for entry in entries),
            "max_abs": max(entry["max_abs"] for entry in entries),
            "maximum_relative_l2": max(entry["relative_l2"] for entry in entries),
            "whole_model_validated": False, "macos_mps_validated": False,
            "compression_package_support": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["checkpoint_sha256_verified", "source_span_values", "candidate_span_values", "saved_values", "output_comparisons", "failed_comparisons", "max_abs", "maximum_relative_l2"]}, indent=2))
