# NovoMolGen source audit, 7 September 2026

The official collection contains six public SMILES models: 32M, 157M and 300M, each with AtomWise or BPE tokenization. Each has an official `hf-checkpoint` revision containing native `LlamaForCausalLM` safetensors. That is the clearest route for a Mac experiment. The 32M AtomWise checkpoint now has complete token-only compression experiments on Mac CPU and MPS; the other five variants remain metadata-audited candidates. There is no production NovoMolGen adapter or preservation of its arbitrary `inputs_embeds` route. The initial metadata audit downloaded only headers and text; the later 32M experiments verified the complete weight payload. [Collection](https://huggingface.co/collections/chandar-lab/novomolgen), [official loading instructions](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/9c5692e454e87c2298d0e213453e09c248ce686f/README.md).

The paper also studies SELFIES, SAFE and DeepSMILES; those are not additional model variants in this collection. Four dataset entries and the paper complete its 11 items. [Paper, version 2](https://arxiv.org/html/2508.13408v2).

| Variant | Layers / width / MLP width / heads | Vocabulary | Stored values | Native safetensors bytes | Native revision prefix |
|---|---|---:|---:|---:|---|
| 32M AtomWise | 12 / 512 / 1,024 / 8 | 84 | 31,556,096 | 126,236,496 | `dcd3f59261be` |
| 32M BPE | 12 / 512 / 1,024 / 8 | 500 | 31,982,080 | 127,940,448 | `e1cab1287c39` |
| 157M AtomWise | 24 / 640 / 2,560 / 10 | 84 | 157,425,280 | 629,725,448 | `9a9a78c46a8a` |
| 157M BPE | 24 / 640 / 2,560 / 10 | 500 | 157,957,760 | 631,855,376 | `e133b9792d10` |
| 300M AtomWise | 32 / 768 / 3,072 / 12 | 84 | 302,168,832 | 1,208,707,848 | `2be770a56c42` |
| 300M BPE | 32 / 768 / 3,072 / 12 | 500 | 302,807,808 | 1,211,263,760 | `e63d6e00dec3` |

All native tensors are **F32**, verified through exact HTTP byte-range reads of safetensors headers. Combined headers required 137,952 bytes, including length prefixes. Full commits, LFS SHA-256 hashes, header hashes, original `main` checkpoint sizes and downloaded text-file hashes are in [the complete metadata record](../benchmarks/novomolgen_metadata.json). The original `main` weights are `.bin`; their dtype was not read and their configs omit `torch_dtype`. AtomWise repositories include earlier checkpoints, making filename/revision selection necessary. The 32M AtomWise payload hash has since been recomputed locally. The other five weight hashes remain publisher metadata.

Native configs specify head width 64, full multi-head attention, bias-free projections, SiLU-gated MLPs, RoPE theta 10,000, RMS epsilon `1e-6`, and untied input/output embeddings. The position limit is 2,048; the custom sampler defaults to length 64. Normalization is **custom `LlamaRMSNorm`, not `torch.nn.RMSNorm`**, and computes variance in float32. [32M config](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/dcd3f59261bebf84142c13617d4e129b4b0d0fdc/config.json), [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/llama/modeling_llama.py).

The custom `NovoMolGen.forward` takes token IDs, labels, positions, FlashAttention inference state and a last-token count. It returns loss, logits and a final hidden-state tensor, and explicitly ignores `attention_mask`. Its `.sample` returns decoded `SMILES` and token `sequences`, with temperature, top-k/top-p and EOS filtering. Native Llama offers token-ID or arbitrary `inputs_embeds` paths, masks, positions, KV caches, optional attentions, tuples of layer hidden states and `.generate`. These interfaces must remain distinct. [Custom source](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/9c5692e454e87c2298d0e213453e09c248ce686f/modeling_novomolgen.py).

The repository's `prepare_hf_model` attaches `.sample` to native Llama, but uses `max_new_tokens` and different defaults. `generate_valid_smiles` adds optional canonicalization and uniqueness filtering. A separate reward wrapper applies an MLP to token hidden states and pools at the last non-padding token; no reward checkpoint is listed in the collection. No dedicated trained molecule-pooling head was found. BPE tokenizer JSON retains dropout `0.1`: encode once and replay token IDs during comparison. Disabling this silently would change preprocessing. [Sampling helpers](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/src/models/modeling_utils.py), [reward wrapper](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/src/models/model_with_value_head.py).

The main class inherits FlashAttention's `GPTLMHeadModel`. Without FlashAttention, its imported base is `None`; disabling a config flag alone cannot fix that import. Sampling defaults to CUDA, and the Pixi environment targets `linux-64`. The native branch avoids the custom class; a minimal CPU/MPS experiment can use safetensors, native Llama, explicit float32 and a suitable PyTorch attention backend. The native 32M model has since completed original-API smoke checks on CPU and MPS, using both eager attention and SDPA under Transformers 4.46.2. This does not establish equivalence with the original custom FlashAttention class. [Environment](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/pixi.toml), [requirements](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/requirements.txt).

The strongest bounded weight candidate is the first block's finite token path:

\[
Q[v]=W_Q\operatorname{RMSNorm}(E[v]),\quad
K[v]=W_K\operatorname{RMSNorm}(E[v]),\quad
V[v]=W_V\operatorname{RMSNorm}(E[v]).
\]

Apply RoPE after lookup. Keep the original embedding table for residuals and returned hidden states. Three square projections become three vocabulary tables, saving \(3d(d-|\mathcal V|)\) values: 657,408, 1,067,520 and 1,575,936 for AtomWise, about 2.08%, 0.68% and 0.52% of total weights. BPE saves 18,432, 268,800 and 617,472. Removing an unused first RMS scale could save another \(d\); that is excluded from these totals.

This exact-real rewrite requires an **explicit token-ID-only contract**. Arbitrary `inputs_embeds` needs retained original projections, which removes the storage saving. The package now provides an explicit `Float32RMSNorm` formula and `FiniteTokenFanout` graph declaration. Replacing arbitrary custom code still requires a justified adapter; class-name matching is insufficient. Later layers depend on sequence context and cannot be enumerated as token-only tables. Rounded tables still require complete numerical validation across sequence shapes and caches.

Other candidates are smaller: fold internal RMS scales into all their linear consumers, or pack Q/K/V and gate/up projections to reduce kernel launches. Packing alone saves no weights. Untied embeddings cannot be deduplicated without inspecting actual values. Generic QK contraction is unattractive when input width exceeds head width, and RoPE adds position dependence. No quantization, distillation or approximate low-rank truncation is proposed.

The experiments below use the 32M AtomWise native checkpoint at the registry's full SHA. They compare token rows, teacher-forced logits/loss, requested hidden states and attentions, masks/positions, cached decoding, and generation with identical token inputs and sampling settings. Similar logits do not guarantee identical sampled molecules. No supported-target flag should change until numerical tests and a real Mac run pass.

## Actual first-layer probe

The first token-only RMSNorm/QKV rewrite passed **372 CPU output comparisons** on the real published 32M AtomWise weights. Maximum absolute error was `5.7220458984375e-6`; maximum relative L2 error was `5.38022524153663e-7`. Both separate limits remained `1e-5`.

This initial result covered the local span only. The subsequent complete-model experiments below include RoPE, attention, caches, decoder layers, logits, generation and MPS. NovoMolGen is still a research target rather than a production compressed adapter.

The checkpoint was loaded with safetensors after verifying all 126,236,496 bytes against SHA-256 `3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699`, at revision `dcd3f59261bebf84142c13617d4e129b4b0d0fdc`. All 111 stored tensors are float32, totaling 31,556,096 values. The payload remains outside the package and audit directory. [Pinned checkpoint](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/tree/dcd3f59261bebf84142c13617d4e129b4b0d0fdc).

The reference computes the audited upstream custom `LlamaRMSNorm` formula, followed by the actual first Q/K/V projection matrices. It is not `torch.nn.RMSNorm`. The candidate stores precomputed Q/K/V rows for all 84 tokens and retains the original embedding rows for the residual. It does not retain the source normalization scale or projection matrices. Tests include every token individually, the complete vocabulary in two shapes, random batches through length 2,048, and empty dimensions. Each case compares Q, K, V and the residual embedding.

Local storage is 829,952 values before and 172,032 after, including residual embeddings in both counts. This saves 657,920 float32 values or 2,631,680 bytes. The corresponding whole-model size would be 30,898,176 values **if** this span were integrated successfully and all other parameters retained. That single-table whole-model proposal subsequently failed the strict output gate; the smaller accepted reductions below include the extra tables needed to match numerical execution. The lookup also avoids 786,432 nominal dense Q/K/V multiply-accumulates per token, but no timing or speed claim was measured.

The explicit input contract is two-dimensional integer token IDs. Native Llama's arbitrary `inputs_embeds` route would require an original-weight fallback, removing this storage benefit. Even a complete token-row check cannot bound error amplification in later layers or guarantee identical sampled molecules.

Initial local results: [novomolgen32_first_qkv_cpu.json](../benchmarks/novomolgen32_first_qkv_cpu.json). Reproduction: [novomolgen_first_qkv.py](../experiments/novomolgen_first_qkv.py). With PyTorch and safetensors installed, run `python experiments/novomolgen_first_qkv.py /path/to/model.safetensors --output results.json`. The script checks the exact checkpoint hash, imports no remote model code, and downloads nothing. It reproduces the pinned normalization implementation directly for this bounded span.

The normalization forward is adapted from the pinned Transformers implementation, copyright 2022 EleutherAI and the HuggingFace Inc. team, under Apache-2.0. Its attribution remains in the reproduction script; the complete pinned [license text](../vendor/novomolgen_audit/sources/TRANSFORMERS_LICENSE) and [download provenance](../vendor/novomolgen_audit/sources/TRANSFORMERS_LICENSE_PROVENANCE.json) are retained in this audit. The adapted constructor accepts the frozen checkpoint scale.


## Complete model: preserve the floating-point execution regimes

A single vocabulary table is exact in real arithmetic but failed the complete
CPU output gate on all 84 singleton-token cases. Small local differences reached
`1.16348e-4` in later outputs. The separate absolute and relative L2 limits stayed
at `1e-5`; generation agreement alone did not qualify the candidate.

The successful CPU prototype stores two tables: one built by executing the
original normalization and projections separately on each `(1, 1)` token input,
and one built in bulk. Calls containing exactly one token select the first.
The 123 main cases plus 134 additional boundary/composite/strided cases,
comprising 12,599 tensor comparisons, were **numerically identical** to the original CPU eager model. Explicit byte equality is checked separately in the generic replay below. This includes every singleton token,
lengths through 2,048, loss/logits, all requested hidden states and attentions,
legacy/Dynamic/Static caches, tuple output, left padding, positions, and greedy,
beam and seeded sampled generation. Generation scores retain exact finite and
infinity masks as well as their numerical values.

CPU-built tables failed the MPS gate. Tables built on MPS needed a third regime:
exactly one token, 2–15 tokens, and 16 or more tokens in a call. The small-batch
table evaluates each token in two identical positions. These choices were found
from the original kernel behavior and validated, not inferred from an algebraic
rounding theorem. All 123 main MPS cases and 134 additional boundary, composite
batch/length and strided-input cases had zero numerical error: **12,599 tensor comparisons**. The later generic replay adds explicit byte comparisons.
Each compressed call was compared with the original using the same attention
backend, cache route, tensor shape and output flags. Original cached and uncached
calls can themselves round differently.

| Native 32M AtomWise representation | Stored float32 parameters | Reduction |
|---|---:|---:|
| Original | 31,556,096 | — |
| CPU: singleton and bulk tables | 31,027,200 | 528,896 / 1.676% |
| MPS: singleton, small-batch and bulk tables | 31,156,224 | 399,872 / 1.267% |

The counts include every retained table and the original input embeddings used
by residuals and returned hidden states. No original first-layer normalization
scale or Q/K/V projection matrix remains. The rest of the decoder is unchanged.
The reduction is modest because only the first layer has a context-independent
finite-token input; later hidden states depend on the sequence.

This is a token-ID-only research representation. Native Llama also accepts
arbitrary continuous `inputs_embeds`, which would require the original projection
weights. No broad native-API replacement, cross-device bitwise guarantee, proof
over every sequence, or molecular-quality benchmark is claimed. The numerical
result is specific to the recorded checkpoint, float32, PyTorch 2.14.0 and pinned
Transformers 4.46.2. Backend changes require a fresh complete-output comparison.


Nine randomized interleaved timing rounds, after three warmups, found essentially
neutral end-to-end speed. Across eight API workloads, CPU median ratios ranged
from 0.979–1.029× and MPS from 0.997–1.052×. The largest observed MPS ratio was
four 32-token prefills, 8.761→8.331 ms; greedy generation of twelve new tokens
was 101.284→100.458 ms (1.008×). These short measurements do not establish a
material speedup. Each benchmark workload also passed complete-output comparison
with exact numerical equality. CPU used four threads; MPS timing synchronized GPU work.


## General compiler integration

The same whole-model experiment now uses `compile_finite_fanout` to construct
its tables, then the ordinary package `save`/`load` path to restore them. The
explicit declaration contains the copied embedding, `Float32RMSNorm` and named
Q/K/V branches. The native model keeps its original embedding separately; that
live residual consumer is included in the whole-model counts above. No native
model-name rule is used inside the compiler.

The generic local gate is bitwise on all 370 recorded CPU and 748 MPS cases.
The restored compiled lookup also passes all 257 complete-model cases and
12,599 tensor comparisons on each backend bitwise. Packing groups tables by
which execution routes use them, so selecting one profile does not gather the
other profiles' unused columns. Shared outputs remain deduplicated.

This demonstrates the reusable operator within an actual model. The small
Python adapter that supplies token IDs to the first-layer projections remains
research code specific to the audited native interface. The package does not
claim to infer that interface or preserve its arbitrary continuous-embedding
route automatically. The earlier timing table describes the manual layout;
generic-layout timing is recorded separately in the experiment archive.


Reproduction and retained failures are in the
[complete experiment archive](../experiments/novomolgen_execution/README.md).
The [portable original-model smoke](../experiments/novomolgen_execution/original_native/README.md)
checks the published reference independently. The
[manual candidate evidence](../experiments/novomolgen_execution/manual/CANDIDATE_EVIDENCE.md)
retains every timing sample and the rejected numerical proposals.


The final generic rerun explicitly compares tensor value bytes after conversion
to contiguous CPU storage, rather than inferring byte equality from zero error
or `torch.equal`. It found no byte differences across 12,599 tensor leaves and
314,076,156 logical value bytes per backend: 12,439 float32 leaves, 16 int64
leaves and 144 Boolean leaves. This check includes signed zero. The original
numerical reports and timing gates remain historical and are not relabeled as
byte checks. The package now reports numerical `exact` and tensor `bitwise`
equality separately; its configured numerical tolerances retain their meaning.
