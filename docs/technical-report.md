# Removing avoidable inference cost from Mol-JEPA

**Original models:** [Mol-JEPA — Rottach et al.](https://arxiv.org/abs/2608.22642) · [Boltz-2 — Passaro et al.](https://doi.org/10.1101/2025.06.14.659707) · [STATE — Arc Institute](https://arcinstitute.org/manuscripts/State).
{ .original-work }

**compressme technical report · 8 September 2026**

For SMILES-only inference, the tested Mol-JEPA checkpoint can retain all its
embedding outputs with **56.47% fewer parameters** and approximately **2× faster
complete calls on an Apple GPU**. The reductions use input specialisation,
algebraic composition and sparse execution. No retraining, distillation or
precision reduction was used.

## What changed

The original multimodal checkpoint has 45,406,721 float32 parameters. Our declared
input contract accepts SMILES with `embeddings_data=None`; additional modality
inputs are rejected. Outputs retain all 12 predicted modality embeddings, the
CLS vector, all 13 latent embeddings and requested attention maps.

![Three selected Mol-JEPA rewrites, showing the original and retained computation](figures/moljepa-rewrites.png)

*Figure 1. The input contract removes unused encoders. Algebraic rewrites replace
oversized intermediate projections with the combinations consumed downstream.
The diagram is schematic; residuals, biases, edge terms and output branches remain.*

Three reductions account for the complete parameter change:

| Transformation | Parameters removed | Why it applies |
| --- | ---: | --- |
| Omit encoders for absent input modalities | 15,462,401 | Those branches are unreachable under the declared SMILES-only contract. They already performed no inference work on this route. |
| Compose the narrow atom projection with its first-layer consumers | 5,504,000 | `W(Ax+a)+b = (WA)x+Wa+b`; consumers can operate on the original 82-feature input. The original expansion remains where the residual needs it. |
| Store eligible query/key interactions as bilinear maps | 4,677,160 | The attention score observes `QᵀK`, allowing a smaller interaction representation while preserving each head and its edge terms. |
| **Total** | **25,643,561** | **45,406,721 → 19,763,160 retained parameters** |

Sparse bond-only graph preparation, graph kernels that avoid large edge-message
intermediates, fewer device synchronisations and batched readouts then improve
execution. These changes retain all output heads and add no further parameter
reduction. The measured speedup combines these changes; it cannot be attributed
to the parameter count alone. [Derivations and prior art](theory.md).

## Measured result

![Parameter waterfall and original versus compressed MPS latency with empirical run intervals](figures/moljepa-results.png)

*Figure 2. Left: exact parameter accounting. Right: complete-call median latency;
error bars show empirical p10–p90, not confidence intervals. The original and
compressed models receive the same SMILES inputs. [Editable SVG](figures/moljepa-results.svg) · [Data provenance](figures/provenance.json).*

The reference timing run on 7 September used an Apple M5 MacBook Air, 16 GB unified memory, Python 3.12
and PyTorch 2.14. Each workload had 10 warmups and 20 shuffled, synchronised
rounds. Timings include fresh SMILES parsing, graph construction, device transfers
and every prediction head; loading and first-call shader compilation are excluded.

| Molecules per call | Original median | Compressed median | Speed ratio |
| --- | ---: | ---: | ---: |
| 1 | 14.47 ms | 6.56 ms | 2.21× |
| 4 | 22.07 ms | 11.27 ms | 1.96× |
| 32 | 64.59 ms | 32.42 ms | 1.99× |

All 64 validation SMILES passed complete output and requested-attention checks.
The largest absolute difference was **2.15×10⁻⁶ on MPS** and **4.05×10⁻⁶ on CPU**.
These are measured errors on those inputs. The algebra is exact over real
numbers, but floating-point reassociation means the results are not universally
byte-identical. [Full timing samples and output metrics](../benchmarks/moljepa_runtime_macos.json).

A fresh MPS revalidation on 8 September passed all 33 SMILES-suite cases and
133 output-tensor comparisons, with the same 2.15×10⁻⁶ maximum error. The original
model's own repeated execution varied by up to 1.59×10⁻⁶, reinforcing the need
to distinguish numerical tolerance from byte identity.
[Fresh GPU checks](../benchmarks/gpu_validation_2026-09-08_summary.json).

## A second substantial result: Boltz-2 storage

Boltz-2's confidence and affinity checkpoints have 5,019 common named tensors
with identical bytes. Sharing immutable parameter storage reduces the jointly
loaded models from **4.087 to 2.062 GB of registered weights, a 49.55% saving**.
Logical parameter enumeration and original arithmetic remain unchanged.

All 48 returned tensor outputs matched the original bytes on CPU and MPS,
including fresh reloads, for one 20-residue protein/ethanol fixture with the
standard 200-step schedules. Sequentially unloading one model has a different
memory baseline. This result does not establish a single-model speedup;
optional conditioning reuse gave no reliable MPS gain.
[Boltz evidence and limits](boltz2.md).

## Transferability and limits

The package recognises structural opportunities rather than deleting tensors
by model name. Its reusable passes include affine composition, normalisation
statistics, fixed-vocabulary evaluation and immutable storage sharing. An
incompressible or unsupported model is a legitimate unchanged result. Local
rewrites still need complete-output checks in their enclosing model.

**GPU evidence currently means Apple MPS. NVIDIA CUDA is unverified because no
NVIDIA GPU is available.** STATE SE's 28.67% parameter reduction remains CPU-only.
A separate lossless original-table representation now preserves all tested output
bytes on CPU/MPS after reload, saving 6.31% of registered state. Its MPS decoding
cost makes the tested calls about five times slower, so it is an optional storage
tradeoff rather than a speed result. [STATE details](state.md).
Backend-specific kernels and precision settings can change
floating-point results, so CUDA requires its own comparisons against an original
CUDA reference. [GPU validation guide](gpu-validation.md) ·
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

These tests assess preservation of pretrained outputs, not downstream biological
accuracy or unrestricted fine-tuning equivalence. The [supporting research notes](research-notes.md)
retain smaller gains and negative experiments. The [CLI](cli.md) can inspect and
pack weights without biology dependencies; semantic rewrites require a trusted
local model constructor and representative inputs.
