# NovoMolGen: modest storage savings on Mac

**Original work:** [NovoMolGen paper, version 2](https://arxiv.org/html/2508.13408v2) · [official model collection](https://huggingface.co/collections/chandar-lab/novomolgen).
{ .original-work }

The 32M AtomWise model uses 1.676% fewer stored parameters on CPU and 1.267%
fewer on MPS when its first normalization and Q/K/V projections are replaced by
token lookup tables. The restored generic implementation matches 12,599 output
tensors per backend byte for byte. End-to-end speed is essentially unchanged.

This is a token-ID-only research adapter. It does not preserve arbitrary
`inputs_embeds`, and there is no production NovoMolGen adapter. The other five
published variants have only a metadata audit.

## Complete outputs and runtime

The algebraic rewrite needs different tables for different execution shapes.
A single vocabulary table failed the complete CPU gate on all 84 singleton-token
cases: small local differences grew to `1.16348e-4` in later outputs. Separate
absolute and relative L2 limits remained `1e-5`. Generation agreement alone was
insufficient.

The successful CPU prototype stores two tables: one built by executing the
original normalization and projections separately on each `(1, 1)` token input,
and one built in bulk. Calls containing exactly one token select the first.
The 123 main cases plus 134 additional boundary/composite/strided cases,
comprising 12,599 tensor comparisons, matched the original CPU eager model
numerically. These cases include every singleton token,
lengths through 2,048, loss/logits, all requested hidden states and attentions,
legacy/Dynamic/Static caches, tuple output, left padding, positions, and greedy,
beam and seeded sampled generation. Generation scores retain exact finite and
infinity masks as well as their numerical values.

CPU-built tables failed the MPS gate. Tables built on MPS needed a third regime:
exactly one token, 2–15 tokens, and 16 or more tokens in a call. The small-batch
table evaluates each token in two identical positions. These choices were found
from the original kernel behavior and validated, not inferred from an algebraic
rounding theorem. All 123 main MPS cases and 134 additional boundary, composite
batch/length and strided-input cases had zero numerical error across 12,599
tensor comparisons. The generic replay below checks bytes separately.
Each compressed call was compared with the original using the same attention
backend, cache route, tensor shape and output flags. Original cached and uncached
calls can themselves round differently.

| Native 32M AtomWise representation | Stored float32 parameters | Reduction |
|---|---:|---:|
| Original | 31,556,096 | Baseline |
| CPU: singleton and bulk tables | 31,027,200 | 528,896 / 1.676% |
| MPS: singleton, small-batch and bulk tables | 31,156,224 | 399,872 / 1.267% |

The counts include every retained table and the original input embeddings used
by residuals and returned hidden states. No original first-layer normalization
scale or Q/K/V projection matrix remains. The rest of the decoder is unchanged.
The reduction is modest because only the first layer has a context-independent
finite-token input; later hidden states depend on the sequence.

Retaining arbitrary continuous `inputs_embeds` would require the original
projection weights and remove the storage benefit. The validated representation
uses token IDs, float32, PyTorch 2.14.0 and pinned Transformers 4.46.2. Its byte
comparisons hold within each backend; they establish neither cross-device
equality, a proof over every sequence nor molecular quality. A backend change
requires a fresh complete-output comparison.

Nine randomized interleaved timing rounds, after three warmups, found essentially
neutral end-to-end speed. Across eight API workloads, CPU median ratios ranged
from 0.979–1.029× and MPS from 0.997–1.052×. The largest observed MPS ratio was
four 32-token prefills, 8.761→8.331 ms; greedy generation of twelve new tokens
was 101.284→100.458 ms (1.008×). These short measurements do not establish a
material speedup. Each benchmark workload also passed complete-output comparison
with exact numerical equality. CPU used four threads; MPS timing synchronized GPU work.

## Build the tables with the general compiler

`compile_finite_fanout` constructs the tables from an explicit declaration of
the copied embedding, `Float32RMSNorm` and named Q/K/V branches. The ordinary
package `save`/`load` path restores them. The native model retains its original
embedding for residuals, included in the counts above. The compiler uses the
declared graph without a NovoMolGen-specific rule.

The generic local gate is bitwise on all 370 recorded CPU and 748 MPS cases.
The restored compiled lookup also passes all 257 complete-model cases and
12,599 tensor comparisons on each backend bitwise. Packing groups tables by
which execution routes use them, so selecting one profile does not gather the
other profiles' unused columns. Shared outputs remain deduplicated.

A small Python adapter supplies token IDs to the first-layer projections. That
interface mapping remains audited research code; the compiler does not infer it
or restore arbitrary continuous-embedding inputs. The timings above describe the
manual layout. Generic-layout timing is recorded separately in the experiment
archive.

Reproduction and retained failures are in the
[complete experiment archive](../experiments/novomolgen_execution/README.md).
The [portable original-model smoke](../experiments/novomolgen_execution/original_native/README.md)
checks the published reference independently. The
[manual candidate evidence](../experiments/novomolgen_execution/manual/CANDIDATE_EVIDENCE.md)
retains every timing sample and the rejected numerical proposals.

The generic replay compares value bytes in contiguous CPU storage, including
signed zero. All 12,599 tensor leaves and 314,076,156 logical value bytes match
per backend: 12,439 float32 leaves, 16 int64 leaves and 144 Boolean leaves.
The earlier reports and timing gates checked numerical equality; zero error or
`torch.equal` alone does not establish byte equality. Those records keep their
original scope. The package reports numerical `exact` and tensor `bitwise`
equality separately and retains its configured numerical tolerances.

## Why the first layer can use a lookup

The first block's projections have a finite token input:

\[
Q[v]=W_Q\operatorname{RMSNorm}(E[v]),\quad
K[v]=W_K\operatorname{RMSNorm}(E[v]),\quad
V[v]=W_V\operatorname{RMSNorm}(E[v]).
\]

Apply RoPE after lookup. Keep the original embedding table for residuals and returned hidden states. Three square projections become three vocabulary tables, saving \(3d(d-|\mathcal V|)\) values: 657,408, 1,067,520 and 1,575,936 for AtomWise, about 2.08%, 0.68% and 0.52% of total weights. BPE saves 18,432, 268,800 and 617,472. Removing an unused first RMS scale could save another \(d\); that is excluded from these totals.

The rewrite is exact in real arithmetic under an explicit token-ID-only contract.
`Float32RMSNorm` supplies the normalization formula and `FiniteTokenFanout` the
graph declaration. An adapter must justify the mapping from custom code; a class
name is insufficient. Later layers depend on sequence context and cannot use
token-only enumeration. Rounded tables need complete-output validation across
sequence shapes and caches, as the singleton failures show.

Other candidates are smaller: fold internal RMS scales into all their linear consumers, or pack Q/K/V and gate/up projections to reduce kernel launches. Packing alone saves no weights. Untied embeddings cannot be deduplicated without inspecting actual values. Generic QK contraction is unattractive when input width exceeds head width, and RoPE adds position dependence. No quantization, distillation or approximate low-rank truncation is proposed.

The 32M experiments use the native checkpoint at the registry's full SHA.
Comparisons hold token inputs and sampling settings fixed and cover token rows,
teacher-forced logits/loss, requested hidden states and attentions, masks and
positions, cached decoding and generation. Similar logits alone cannot guarantee
identical sampled molecules. A supported-target flag requires both numerical
tests and a real Mac run.

## The local comparison and its limits

The single-table RMSNorm/QKV rewrite passed 372 local CPU output comparisons
on the published 32M AtomWise weights. Maximum absolute error was
`5.7220458984375e-6`; maximum relative L2 error was `5.38022524153663e-7`,
both below their separate `1e-5` limits. This local pass did not predict the
complete-model failure described above, which includes RoPE, attention, caches,
decoder layers, logits and generation. MPS also needs its own complete gate.

The checkpoint was loaded with safetensors after verifying all 126,236,496 bytes against SHA-256 `3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699`, at revision `dcd3f59261bebf84142c13617d4e129b4b0d0fdc`. All 111 stored tensors are float32, totaling 31,556,096 values. The payload remains outside the package and audit directory. [Pinned checkpoint](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/tree/dcd3f59261bebf84142c13617d4e129b4b0d0fdc).

The reference applies the audited `LlamaRMSNorm` formula and actual first Q/K/V
matrices. The candidate stores Q/K/V rows for all 84 tokens plus original residual
embeddings, removing the source scale and projection matrices. Tests compare Q,
K, V and residuals for every token individually, the vocabulary in two shapes,
random batches through length 2,048 and empty dimensions.

Local storage is 829,952 values before and 172,032 after, including residual embeddings in both counts. This saves 657,920 float32 values or 2,631,680 bytes. A single-table whole model retaining all other parameters would contain
30,898,176 values, but failed the strict output gate. The accepted representations
above include extra tables to match numerical execution. Lookup avoids 786,432
nominal dense Q/K/V multiply-accumulates per token; this local experiment did
not measure timing or speed.

The local input contract is two-dimensional integer token IDs. A complete
token-row comparison cannot bound error amplification through the remaining
layers or guarantee identical sampled molecules.

Initial local results: [novomolgen32_first_qkv_cpu.json](../benchmarks/novomolgen32_first_qkv_cpu.json). Reproduction: [novomolgen_first_qkv.py](../experiments/novomolgen_first_qkv.py). With PyTorch and safetensors installed, run `python experiments/novomolgen_first_qkv.py /path/to/model.safetensors --output results.json`. The script checks the exact checkpoint hash, imports no remote model code, and downloads nothing. It reproduces the pinned normalization implementation directly for this bounded span.

The normalization forward is adapted from the pinned Transformers implementation, copyright 2022 EleutherAI and the HuggingFace Inc. team, under Apache-2.0. Its attribution remains in the reproduction script; the complete pinned [license text](../vendor/novomolgen_audit/sources/TRANSFORMERS_LICENSE) and [download provenance](../vendor/novomolgen_audit/sources/TRANSFORMERS_LICENSE_PROVENANCE.json) are retained in this audit. The adapted constructor accepts the frozen checkpoint scale.

## Source, checkpoints and API

The 7 September 2026 source audit covers six public SMILES models: 32M, 157M
and 300M, each with AtomWise or BPE tokenization. All have an official
`hf-checkpoint` revision with native `LlamaForCausalLM` safetensors, used for the
Mac experiments. The metadata audit read headers and text only; the 32M AtomWise
experiments verified its complete weight payload. [Collection](https://huggingface.co/collections/chandar-lab/novomolgen), [official loading instructions](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/9c5692e454e87c2298d0e213453e09c248ce686f/README.md).

The paper also studies SELFIES, SAFE and DeepSMILES; those are not additional model variants in this collection. Four dataset entries and the paper complete its 11 items. [Paper, version 2](https://arxiv.org/html/2508.13408v2).

| Variant | Layers / width / MLP width / heads | Vocabulary | Stored values | Native safetensors bytes | Native revision prefix |
|---|---|---:|---:|---:|---|
| 32M AtomWise | 12 / 512 / 1,024 / 8 | 84 | 31,556,096 | 126,236,496 | `dcd3f59261be` |
| 32M BPE | 12 / 512 / 1,024 / 8 | 500 | 31,982,080 | 127,940,448 | `e1cab1287c39` |
| 157M AtomWise | 24 / 640 / 2,560 / 10 | 84 | 157,425,280 | 629,725,448 | `9a9a78c46a8a` |
| 157M BPE | 24 / 640 / 2,560 / 10 | 500 | 157,957,760 | 631,855,376 | `e133b9792d10` |
| 300M AtomWise | 32 / 768 / 3,072 / 12 | 84 | 302,168,832 | 1,208,707,848 | `2be770a56c42` |
| 300M BPE | 32 / 768 / 3,072 / 12 | 500 | 302,807,808 | 1,211,263,760 | `e63d6e00dec3` |

All native tensors are F32, verified by exact HTTP byte-range reads of the
safetensors headers. Combined headers required 137,952 bytes, including length prefixes. Full commits, LFS SHA-256 hashes, header hashes, original `main` checkpoint sizes and downloaded text-file hashes are in [the complete metadata record](../benchmarks/novomolgen_metadata.json). The original `main` weights are `.bin`; their dtype was not read and their configs omit `torch_dtype`. AtomWise repositories include earlier checkpoints, making filename/revision selection necessary. The 32M AtomWise payload hash was also recomputed locally. The other five weight hashes remain publisher metadata.

Native configs specify head width 64, full multi-head attention, bias-free projections, SiLU-gated MLPs, RoPE theta 10,000, RMS epsilon `1e-6`, and untied input/output embeddings. The position limit is 2,048; the custom sampler defaults to length 64. Normalization uses custom `LlamaRMSNorm`, which computes variance in float32;
substituting `torch.nn.RMSNorm` would change the reference. [32M config](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/dcd3f59261bebf84142c13617d4e129b4b0d0fdc/config.json), [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/llama/modeling_llama.py).

The custom `NovoMolGen.forward` takes token IDs, labels, positions, FlashAttention inference state and a last-token count. It returns loss, logits and a final hidden-state tensor, and explicitly ignores `attention_mask`. Its `.sample` returns decoded `SMILES` and token `sequences`, with temperature, top-k/top-p and EOS filtering. Native Llama offers token-ID or arbitrary `inputs_embeds` paths, masks, positions, KV caches, optional attentions, tuples of layer hidden states and `.generate`. These interfaces must remain distinct. [Custom source](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/blob/9c5692e454e87c2298d0e213453e09c248ce686f/modeling_novomolgen.py).

The repository's `prepare_hf_model` attaches `.sample` to native Llama, but uses `max_new_tokens` and different defaults. `generate_valid_smiles` adds optional canonicalization and uniqueness filtering. A separate reward wrapper applies an MLP to token hidden states and pools at the last non-padding token; no reward checkpoint is listed in the collection. No dedicated trained molecule-pooling head was found. BPE tokenizer JSON retains dropout `0.1`: encode once and replay token IDs during comparison. Disabling this silently would change preprocessing. [Sampling helpers](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/src/models/modeling_utils.py), [reward wrapper](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/src/models/model_with_value_head.py).

The main class inherits FlashAttention's `GPTLMHeadModel`. Without FlashAttention, its imported base is `None`; disabling a config flag alone cannot fix that import. Sampling defaults to CUDA, and the Pixi environment targets `linux-64`. The native branch avoids this custom class. The 32M native model passes
original-API smoke checks on CPU and MPS with safetensors, explicit float32,
eager attention and SDPA under Transformers 4.46.2. Equivalence to the custom
FlashAttention class remains untested. [Environment](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/pixi.toml), [requirements](https://github.com/chandar-lab/NovoMolGen/blob/d7c520674701533e2b5e18a8ace826aed611e7b0/requirements.txt).
