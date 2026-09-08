# Running compressme on macOS

The prepared environments use Python 3.12.14 and PyTorch 2.14.0 on an Apple M5 MacBook Air with 16 GB unified memory. Both CPU and MPS have been exercised with actual trained Mol-JEPA and State ST checkpoints. State SE is additionally validated on CPU float32; its MPS candidates failed the output gates.

## Prepared environments

- `.venv`: Mol-JEPA, the general compressor, Hugging Face inspection and core tests.
- `.venv-state`: the portable State ST and SE loaders, with Transformers 4.52.3 required by the ST checkpoint.
- `.venv-boltz`: the verified Boltz-2 source, native YAML/FASTA example and lossless packing dependencies.

The core package and CLI have a separate, lightweight installation; these are
optional development environments for the actual model integrations. See
[installation](installation.md). Apple MPS verification does not establish
NVIDIA CUDA verification; the [backend guide](gpu-validation.md) records both
the tested paths and hardware that is unavailable.

No NVIDIA GPU is required for these validated paths. The Mol-JEPA loader enables sparse input processing, packed graph projections and the verified float32 Metal kernels. All requested embedding and attention outputs remain available. Older Torch versions without `torch.mps.compile_shader` use the ordinary PyTorch path.

```python
from compressme import load_moljepa
model = load_moljepa("artifacts/moljepa-smiles", device="mps")
output = model(["CCO", "c1ccccc1"], return_attn=True)
```

Run State in its separate environment and call `load_state_st` as documented in [state.md](state.md). CPU is also supported. The main package does not change global Python settings or install CUDA compatibility shims.

## Evidence and limits

The main Mol-JEPA runtime is about 2× faster than the original in the recorded warmed full-API comparisons, including fresh SMILES parsing and transfers. Its 19,763,160 parameters retain float32 precision. All 64 validation SMILES and requested attention outputs passed the 1e-5 numerical gates. Details and exact comparison workloads are in [runtime.md](runtime.md) and the benchmark JSON files.

State ST uses a generic frozen-table storage reduction. All tested numerical batch outputs, decoded counts and token lookup were bitwise identical on CPU and MPS. It saves 21.25% of resident weights but does not remove active inference work on the expression route.

The generic tensor transfer utility coalesces multiple independent CPU tensors before a device copy. Its separate transport measurements improved substantially, but whole-model timings were mixed; it remains opt-in. Residual/LayerNorm fusion did not show a dependable additional gain and was not enabled.

First-call compilation, loading and dependency imports are outside warmed inference timings. Background activity affects absolute timings; comparisons use synchronized, interleaved calls and preserve raw samples. Do not multiply speedups obtained in different runs.

## Development caveats established on this host

MPS supports the tested float32 paths. Prepare double-precision decompositions on CPU, then convert dtype and device in separate operations. This environment exposed incorrect combined CPU-double/MPS-float32 transfer behaviour, now covered by regressions. No lower precision is silently introduced.

A restricted automation sandbox can report MPS unavailable even when host execution on the same Mac works. Check the actual execution environment before concluding that the hardware is unsupported.

The generic package tests currently cover operator algebra, full-output validation, replay, weight aliases, invalidation, finite-domain compilation and actual Metal operations. Historical early experiments, including constructed low-rank matrices, are preserved in [macos-history.md](macos-history.md); their timings and old suite counts are not the current headline results.

References: [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), [Apple PyTorch support](https://developer.apple.com/metal/pytorch/).
