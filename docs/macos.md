# Running compressme on macOS

Mol-JEPA and STATE ST run on CPU and Apple MPS with the trained checkpoints used here. The prepared environments use Python 3.12.14 and PyTorch 2.14.0 on an Apple M5 MacBook Air with 16 GB unified memory.

STATE SE offers a smaller CPU-only finite-table representation, reducing parameters by 28.67%, and a lossless original-table mode that passes byte-identical CPU/MPS checks. The latter saves 6.31% of registered state but adds substantial decoding time. See [STATE modes and timings](state.md).

## Prepared environments

- `.venv`: Mol-JEPA, the general compressor, Hugging Face inspection and core tests.
- `.venv-state`: the portable STATE ST and SE loaders, with Transformers 4.52.3 required by the ST checkpoint.
- `.venv-boltz`: the verified Boltz-2 source, native YAML/FASTA example and lossless packing dependencies.

These optional model environments are separate from the lightweight core package
and CLI installation. See
[installation](installation.md). Apple MPS verification does not establish
NVIDIA CUDA verification; the [backend guide](gpu-validation.md) records both
the tested paths and hardware that is unavailable.

These validated paths run without an NVIDIA GPU. On MPS, the Mol-JEPA loader enables sparse input processing, packed graph projections and the verified float32 Metal kernels. All requested embedding and attention outputs remain available. Older Torch versions without `torch.mps.compile_shader` use the ordinary PyTorch path.

```python
from compressme import load_moljepa
model = load_moljepa("artifacts/moljepa-smiles", device="mps")
output = model(["CCO", "c1ccccc1"], return_attn=True)
```

Run STATE in its separate environment and call `load_state_st` as documented in [state.md](state.md). CPU is also supported. The main package does not change global Python settings or install CUDA compatibility shims.

## Measured behaviour

The main Mol-JEPA runtime is about 2× faster than the original in the recorded warmed full-API comparisons, including fresh SMILES parsing and transfers. Its 19,763,160 parameters retain float32 precision. All 64 validation SMILES and requested attention outputs passed the 1e-5 numerical gates. Details and exact comparison workloads are in [runtime.md](runtime.md) and the benchmark JSON files.

STATE ST uses a generic frozen-table storage reduction. All tested numerical batch outputs, decoded counts and token lookup were bitwise identical on CPU and MPS. It saves 21.25% of resident weights but does not remove active inference work on the expression route.

Coalescing independent CPU tensors before a device copy improved the transfer component, but complete-model timings were mixed, so the option remains off by default. Residual/LayerNorm fusion gave no dependable additional gain and was not enabled.

Warmed inference timings exclude first-call compilation, loading and dependency imports. Comparisons use synchronized, interleaved calls and retain the raw samples because background activity affects absolute timings. Speedups measured in separate runs cannot be multiplied.

## Backend behaviour seen on this host

MPS supports the tested float32 paths. Prepare double-precision decompositions on CPU, then convert dtype and device in separate operations. A combined CPU-double/MPS-float32 transfer produced incorrect results in this environment and has a regression test. No lower precision is silently introduced.

A restricted automation sandbox can report MPS unavailable even when host execution on the same Mac works. Check the actual execution environment before concluding that the hardware is unsupported.

Package tests cover operator algebra, full-output validation, replay, weight aliases, invalidation, finite-domain compilation and actual Metal operations. The [early macOS experiments](macos-history.md) retain constructed low-rank tests and historical suite counts separately from the later model results.

References: [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), [Apple PyTorch support](https://developer.apple.com/metal/pytorch/).
