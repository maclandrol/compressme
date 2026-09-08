# Historical macOS experiments

Earlier stages of the work, retained for provenance. See macos.md and runtime.md for current results.

# macOS runtime evidence

Measured 2026-09-07 on Apple M5 MacBook Air, 16 GB unified memory, macOS 26.6.1, arm64, Python 3.12.14, PyTorch 2.14.0. These measurements establish portable factor execution; they are **not results for compressing trained Boltz-2 or Mol-JEPA weights**.

## Environment

An isolated environment was created at `/private/tmp/compressme-runtime`, without changing the global Python environment. Python executable: `/private/tmp/compressme-runtime/bin/python`. Cache: `/private/tmp/compressme-uv-cache`.

```sh
rtk proxy /Users/manu/.local/bin/uv venv --cache-dir /private/tmp/compressme-uv-cache --python /Users/manu/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3 /private/tmp/compressme-runtime
rtk proxy /Users/manu/.local/bin/uv pip install --cache-dir /private/tmp/compressme-uv-cache --python /private/tmp/compressme-runtime/bin/python torch numpy pytest safetensors
rtk proxy /Users/manu/.local/bin/uv pip install --cache-dir /private/tmp/compressme-uv-cache --python /private/tmp/compressme-runtime/bin/python transformers torch-geometric rdkit molfeat
rtk proxy /private/tmp/compressme-runtime/bin/python /private/tmp/compressme_runtime_bench.py
```

Installed inference dependencies include torch 2.14.0, numpy 2.5.3, transformers 5.16.1, torch-geometric 2.8.0.post1, rdkit 2026.3.6, molfeat 0.11.0, safetensors 0.8.0. Future reproductions should pin versions; these commands document the environment setup performed in this session.

The restricted execution sandbox reported `torch.backends.mps.is_available() == False`. The same interpreter with approved host execution access reported `True`, and both forward and backward matrix operations ran successfully on the actual Metal GPU. This is an execution-permission difference, not evidence that this Mac lacks MPS support.

## Synthetic matrix benchmark

Weights were **constructed exactly as** `W = B @ A`, with random float32 factors. Dense execution was `linear(x, W, bias)` and factored execution was `linear(linear(x, A), B, bias)`. There is no low-rank truncation in this experiment. Floating-point reassociation explains the nonzero differences.

Four CPU threads; ten warm-up calls; median of five rounds, each containing 100 calls. GPU rounds explicitly synchronised. Values measure repeated-call throughput, not cold-start latency or complete model inference. Other concurrent activity and power/thermal conditions were not controlled. A dedicated repeated benchmark is required before claiming production speedups.

| Device | Rows | Input → output | Rank | Dense ms | Factored ms | Speed ratio | Weight ratio | Relative output error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 768 → 3072 | 64 | 0.04135 | 0.00882 | 4.69 | 9.6 | 1.51e-7 |
| CPU | 32 | 768 → 3072 | 64 | 0.19270 | 0.04472 | 4.31 | 9.6 | 7.41e-7 |
| CPU | 256 | 768 → 3072 | 64 | 0.74292 | 0.17728 | 4.19 | 9.6 | 7.31e-7 |
| CPU | 256 | 768 → 3072 | 256 | 0.72071 | 0.41877 | 1.72 | 2.4 | 8.11e-7 |
| CPU | 1 | 2048 → 2048 | 64 | 0.13499 | 0.00758 | 17.82 | 16.0 | 1.79e-7 |
| CPU | 256 | 2048 → 2048 | 64 | 1.40027 | 0.16815 | 8.33 | 16.0 | 1.16e-6 |
| MPS | 1 | 768 → 3072 | 64 | 0.10190 | 0.03087 | 3.30 | 9.6 | 1.34e-7 |
| MPS | 32 | 768 → 3072 | 64 | 0.09878 | 0.04429 | 2.23 | 9.6 | 5.21e-7 |
| MPS | 256 | 768 → 3072 | 64 | 0.45099 | 0.08987 | 5.02 | 9.6 | 5.15e-7 |
| MPS | 256 | 768 → 3072 | 256 | 0.45267 | 0.21608 | 2.09 | 2.4 | 5.78e-7 |
| MPS | 1 | 2048 → 2048 | 64 | 0.17096 | 0.03063 | 5.58 | 16.0 | 1.56e-7 |
| MPS | 256 | 2048 → 2048 | 64 | 0.81786 | 0.09585 | 8.53 | 16.0 | 8.28e-7 |

Both factor gradients were finite after an actual backward pass on CPU and MPS. No fine-tuning convergence equivalence was tested.

Raw results: `/private/tmp/compressme_runtime_bench.json`. Reproduction script: `/private/tmp/compressme_runtime_bench.py`.

## Real Mol-JEPA checkpoint: exact affine input composition

The published Flogrammer/Mol-JEPA checkpoint at revision `4c912b450175f31b5ba913a5dc921c03b27b985a` was loaded locally. The original graph atom projection expands 82 features to 512, followed immediately by the first graph convolution's query/key/value/skip affine maps. Composing those maps back onto 82 input features removes algebraic redundancy while retaining the original 512-dimensional projection for the external residual. Edge processing, softmax, normalization, nonlinearities, later convolutions, and all optional modality APIs are preserved. No low-rank truncation, distillation, or datatype reduction is used.

Total parameters: **45,406,721 → 39,902,721**, a reduction of **5,504,000 (12.12%)**. All inference uses original float32; composition is prepared on CPU float64 and cast once to float32. Equivalence is in real arithmetic; floating-point bit identity is not claimed.

The actual 512×82 input-projection matrix has numerical rank 82 in CPU float64, singular values from 0.54937 to 6.69035, and condition number 12.1783. The compression exploits the known 82-dimensional input domain, not a claim that arbitrary downstream weight matrices have low rank. For a full-column-rank input map A, any composed consumer M can be represented as W A by choosing W=M A⁺. With independently adjustable biases, this retains the corresponding affine maps while removing parameters that cannot be observed through A. Fine-tuning in the composed coordinates changes optimisation trajectories and regularisation; gradient or training-outcome equivalence was not tested.

Four-SMILES batch: ethanol, aspirin, caffeine, ibuprofen. Compared the full published forward API and all three returned tensor fields on MPS. No unsupported operation or CPU fallback was encountered.

| Output field | Relative Frobenius error | Maximum absolute error |
|---|---:|---:|
| predictions | 2.32e-7 | 1.19e-6 |
| cls | 3.74e-7 | 1.79e-7 |
| embeddings | 3.81e-7 | 5.36e-7 |

| MPS scope | Original ms | Rewritten ms | Speed ratio |
|---|---:|---:|---:|
| Complete graph encoder on precomputed graph | 6.371 | 5.862 | 1.087 |
| Complete public SMILES API, including featurisation | 13.236 | 12.724 | 1.040 |

Three warm-up calls; median of five rounds, each with ten graph calls or three public-API calls; GPU synchronization after each timed round. This small smoke batch establishes the rewrite runs and preserves numerical outputs. It is not a statistical molecular benchmark; the modest timing difference needs repeated isolated measurement before a production speed claim.

The latest transformers `from_pretrained` path encountered a model-constructor meta/CPU buffer mismatch. The audited local loading route was `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=True)`, `AutoModel.from_config(...)`, followed by strict loading with `safetensors.torch.load_file`. HuggingFace module/cache directories were redirected to `/private/tmp`.

Adapter: `/private/tmp/compressme_moljepa_adapter.py`; benchmark: `/private/tmp/compressme_moljepa_exact_bench.py`; results: `/private/tmp/compressme_moljepa_exact_bench.json`.

## Stronger real-checkpoint result: bilinear graph-attention contraction

The subsequent graph-attention rewrite contracts the query/key maps into their bilinear score operator, including the edge-dependent terms and softmax row-shift invariance. In combination with the input-affine rewrite, it reduces the complete model from **45,406,721 to 35,225,561 parameters (22.42%)**, with the full optional-modality input API retained. A separate, explicitly restricted SMILES-only variant has 19,763,160 parameters; it rejects extra embeddings rather than silently ignoring them.

The initial sequential benchmark showed material timing drift, so it was superseded by five warm-up calls per model and **20 shuffled, interleaved timing rounds**, synchronizing before and after each complete four-SMILES call. These are same-run median comparisons, including public-API featurisation:

| MPS model | Median ms | Range ms | Speed ratio vs original |
|---|---:|---:|---:|
| Original | 15.809 | 14.318–21.715 | 1.000 |
| Input-affine rewrite | 15.367 | 13.515–19.395 | 1.029 |
| Input-affine + bilinear scores | 13.024 | 11.281–16.851 | 1.214 |
| Input-affine + bilinear + aggregate before value projection | 14.254 | 13.093–20.699 | 1.109 |

The bilinear score rewrite is preferable on this tested MPS workload. Moving aggregation before value projection is algebraically valid for the audited conditions, but its runtime benefit is backend-dependent.

The preferred bilinear variant's relative output errors were 2.36e-7 (predictions), 4.08e-7 (cls) and 4.12e-7 (embeddings); maximum absolute errors were 1.19e-6, 2.16e-7 and 5.66e-7 respectively. Repeating the original itself gave 1.93e-7 prediction-relative variation on MPS. These are numerical equivalence checks on four molecules, not empirical validation of chemical prediction quality or a bitwise guarantee.

Warmed benchmark source: `/private/tmp/compressme_bilinear_interleaved.py`. Full per-call timings and output errors: `/private/tmp/compressme_bilinear_interleaved_mps.json`. The earlier sequential timings should not be used for a speed claim.

An additional exact simplification skipped score computation for receiving atoms with a single incoming neighbour, whose softmax attention is identically one. On the same four-SMILES input, a separate warmed 20-round interleaved comparison found **10.831 ms without skipping versus 13.727 ms with skipping**, approximately 27% slower. It should remain disabled on this MPS workload; extra dynamic indexing outweighed arithmetic savings. Output errors remained at the same small numerical scale. These absolute timings must not be compared across runs; only their paired ratio is meaningful here. Evidence: `/private/tmp/compressme_bilinear_skip_interleaved_mps.json`.

## Final package verification

A persistent editable environment was created at `/Users/manu/Code/compressme/.venv` from the existing downloaded cache. It contains the core package and its `test,molecules` optional dependencies. Direct import resolves to the requested project source. The current complete package suite passed **31 tests in 6.47 seconds**, including actual Metal access.

Independent review found and then verified fixes for: unsupported/incorrect combined Metal float64 transfers; complex weight truncation; frozen parameters becoming trainable; stale numerical bounds being revalidated after export; and mixed CPU/MPS restored models. The regression suite now includes actual MPS low-rank, covariance, LayerNorm, and bound-provenance cases.

## Backend and API recommendations

Use ordinary PyTorch modules and dense factors first. MPS retains the existing Python model code and supports both inference and autograd. MLX is a promising later backend, but its distinct module/array API means an arbitrary PyTorch checkpoint does not become an MLX model without graph or source conversion. MLX's unified memory and lazy execution are useful runtime properties; neither is a compression guarantee.

Prepare decompositions and audit residual bounds on CPU float64; return factors to the original model dtype/device for execution. On this host, MPS float32 SVD ran, while converting a tensor to MPS float64 raised the explicit unsupported-dtype error. Changes of dtype and floating-point execution need separate numerical allowances beyond real-arithmetic approximation bounds.

Select compressed execution only when `r*(input+output) < input*output`, and benchmark candidate ranks on representative shapes. Two matrix multiplies have an intermediate tensor and extra launch overhead. GPU is not automatically best for small inputs: the measured batch-one 768→3072 factored operation was faster on CPU than MPS.

Preserving the forward input/output API does not preserve original state-dictionary keys, module types, optimiser states, or direct `.weight` access. Store an explicit transformation manifest, rebuild transformed modules before loading factor tensors, preserve tied/shared parameters deliberately, and construct a new optimiser after module replacement. Do not silently reconstruct a dense `.weight` on every access; that discards the intended computation saving. Keep ordinary registered Parameters and buffers so `.to()`, `.eval()`, `.train()` and serialization work normally.

Do not enable fast-math or CPU fallback silently while advertising output fidelity or MPS speed. PyTorch documents both settings; fallback changes where unsupported operations execute and must be included in whole-model timings.

## Primary sources

- [Apple: Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)
- [PyTorch Module API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
- [Apple: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [MLX unified memory documentation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst)
