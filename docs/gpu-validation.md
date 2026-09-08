# GPU validation: Apple MPS measured, NVIDIA CUDA not run

**Original models:** [Mol-JEPA — Rottach et al.](https://arxiv.org/abs/2608.22642) · [Boltz-2 — Passaro et al.](https://doi.org/10.1101/2025.06.14.659707) · [STATE — Arc Institute](https://arcinstitute.org/manuscripts/State).
{ .original-work }

The current results include actual Apple GPU inference. They do not establish
NVIDIA CUDA equivalence. On 8 September 2026 the available machine was an Apple
M5 MacBook Air with 16 GB unified memory, macOS 26.6.1, Python 3.12.14 and
PyTorch 2.14.0. MPS was available. PyTorch reported no CUDA build, no CUDA
devices, and no NVIDIA driver utility. No NVIDIA machine was available and no
cloud machine was provisioned.

The [hardware report](../benchmarks/gpu_validation_2026-09-08_cuda_unavailable.json)
records `status: unavailable`, `accepted: false`, and that no model validation
ran. A portable CUDA command is supplied below; it is a future validation job,
not a successful CUDA result.

## Fresh complete-output checks

The [new runner](../examples/validate_backends.py) loaded the original reference
weights and each compressed artifact independently, ran reference self-repeats,
then compared every returned tensor and non-tensor leaf on the same backend.
These were actual checkpoints and complete prediction calls, rather than only
synthetic linear operators. The [summary](../benchmarks/gpu_validation_2026-09-08_summary.json)
links the detailed metrics and report hashes.

| Artifact, on MPS | Cases | Candidate tensor comparisons | Maximum absolute difference | Result |
|---|---:|---:|---:|---|
| Mol-JEPA SMILES runtime | 33 | 133 | 2.146e-6 | Numerical gate passed |
| Mol-JEPA full modalities | 34 | 138 | 2.146e-6 | Numerical gate passed |
| STATE ST HVG K562 | 5 | 17 | 0 | Byte-identical |
| Boltz-2 confidence and affinity | 2 | 48 | 0 | Byte-identical |

Mol-JEPA covers all 64 verification SMILES, with attention both requested and
omitted, plus isolated helium. The full artifact additionally checks a supplied
example for each optional modality. Its supplied optional vectors are numerical
API probes, not biological measurements. Source self-repeats also had small MPS
differences: at most 1.594e-6 for the SMILES run and 1.788e-6 for the full run.
These results therefore establish numerical agreement on the cases, not byte
equality for Mol-JEPA or an absence of transformation error.

STATE ST checks every `predict_step` field for 1, 7, 64 and 128 cells, including
padded/unpadded and integer/one-hot batch labels, plus all 32,000 token IDs.
Inputs are synthetic numerical probes on trained weights. Boltz uses the native
20-residue protein/ethanol fixture, all structure/confidence/affinity outputs,
200 diffusion steps, 3 confidence recycles/1 sample, and 5 affinity recycles/3
samples. Both models' source self-repeats were also byte-identical. The optional
Boltz request-conditioning runtime was not enabled in this new storage-artifact
check; its separate results remain in [the Boltz documentation](boltz2.md).

STATE SE now has two distinct artifacts. The **28.67% parameter reduction remains
CPU-only**: its precomputed encoder tables failed the complete MPS numerical gate.
The [guard check](../benchmarks/gpu_validation_2026-09-08_state_se_mps_rejected.json)
continues to apply to that artifact. A new **lossless original-table mode** restores
the original gene vectors before running the original computation. Fresh portable
reloads passed **377 tensor byte comparisons on each of CPU and MPS**, including
all 19,790 original gene vectors, all tested output heads and arbitrary raw-vector
inputs. It reduces registered state from 848,155,296 to 794,613,047 bytes (6.31%).
It does not remove arithmetic or change precision; its small MPS timing check was
4.89–5.44 times slower because of decoding. AnnData export remains CPU-only.
See the [CPU reload report](../experiments/state-se-mps-fix/portable-reload-cpu.json),
[MPS reload report](../experiments/state-se-mps-fix/portable-reload-mps.json) and
[STATE guide](state.md). CUDA is unvalidated for both representations.

None of these checks measures biological accuracy, throughput, large-complex
memory requirements, or equality between CPU, Apple GPU and NVIDIA GPU. PyTorch
does not promise identical results across devices or releases even with matched
seeds; each backend needs its own reference comparison. See the official
[reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## What the general runner checks

`examples/validate_backends.py` supports `cpu`, `mps`, `cuda` and `cuda:N`.
It requires the PyTorch dependencies; model-specific imports occur only for the
selected recipe. It never downloads weights or source code.

- Reference and candidate factories run sequentially to avoid keeping both
  large models resident. Reference outputs are copied into independent CPU
  snapshots before the next call, so reused output buffers cannot erase errors.
- Ordinary model state and runtime output anchors must be on the requested
  device. Only the exact `PackedFrozenEmbedding` type may keep its encoded byte
  payload and offsets on CPU; those buffers are included in state-byte accounting.
  Complete output container types, keys, shapes and dtypes must match. Discrete
  and non-tensor values must match.
- For every floating tensor, both maximum absolute error and relative L2 error
  must be at most `1e-5` by default. This is **not** an `allclose` mixed tolerance.
  Empty tensor sets, nonfinite values and unsupported output types fail explicitly.
- Byte equality is a separate metric, using logical contiguous tensor value
  bytes, including signed zero. `--require-bitwise` makes it an acceptance
  requirement. Byte equality does not mean physical strides or allocator layout
  must match.
- The original model runs twice with the same seed. Source self-repeat must pass
  the same requested numerical/byte gate as the candidate. All complete outputs
  are checked; the runner has no output-selector option.
- Python, NumPy and PyTorch RNG states are controlled. Models run frozen in eval
  mode under inference mode with autocast disabled. Custom generators must be
  controlled by the trusted case provider.
- MPS/CUDA synchronization is explicit. TF32 is off by default; matmul precision,
  TF32, cuDNN, deterministic-algorithm settings, environment flags, hardware,
  reference hashes and relevant source hashes are recorded. Settings are restored
  afterwards. The fresh MPS runs inherited deterministic-algorithm mode (off);
  they did not silently promise deterministic kernels.
- Unavailable devices return `status: unavailable` and a nonzero exit code.
  Numerical/runtime failures return `status: failed`; partial checks and an error
  are retained. Existing report paths are refused to preserve earlier evidence.

Exit status is 0 only for an accepted model gate, 2 for an unavailable device,
and 1 for a failed gate. `--probe-only` does no model validation, so even an
available-device inventory is not a successful model gate.

## Reproduce with local assets

Run from the checkout. Substitute your trusted original checkpoint paths. The
artifacts are read-only inputs. Keep the separate tested environments because
Mol-JEPA and STATE use different Transformers releases.

```bash
.venv/bin/python examples/validate_backends.py --device mps \
  --recipe moljepa-smiles --artifact artifacts/moljepa-smiles \
  --reference-weights /path/to/Mol-JEPA/model.safetensors \
  --output /tmp/mol-smiles-mps.json

.venv/bin/python examples/validate_backends.py --device mps \
  --recipe moljepa-full --artifact artifacts/moljepa-full \
  --reference-weights /path/to/Mol-JEPA/model.safetensors \
  --output /tmp/mol-full-mps.json

.venv-state/bin/python examples/validate_backends.py --device mps \
  --recipe state-st --artifact artifacts/state-st-hvg-k562 \
  --reference-weights /path/to/state_st_k562_final.ckpt --require-bitwise \
  --output /tmp/state-st-mps.json

.venv-boltz/bin/python examples/validate_backends.py --device mps \
  --recipe boltz2 --artifact artifacts/boltz2-shared \
  --reference-weights /path/to/boltz2/shared-state --require-bitwise \
  --output /tmp/boltz2-mps.json
```

For Boltz, `--reference-weights` names the verified original tensor-only state
bank produced by [the checkpoint audit](../experiments/boltz2-weight-audit/README.md).
The runner strictly loads those original state names into the unmodified native
classes. It does not validate the shared artifact against itself. `--native-dir`
can point to another copy of the native fixture evidence; its default is the
checkout's `experiments/boltz2-runtime`. That small chemistry directory supports
this fixture, not arbitrary molecules. STATE ST uses `torch.load(weights_only=True)`
for its verified original checkpoint; portable candidate loading is separate.

## Portable NVIDIA job — not yet run on NVIDIA

On an existing NVIDIA machine, install a CUDA-enabled PyTorch build appropriate
for that system using [PyTorch's official installer](https://pytorch.org/get-started/locally/),
then the applicable model dependencies. Do not copy the Mac virtual environment.
Start with a fresh report path and original FP32 arithmetic:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python examples/validate_backends.py \
  --device cuda:0 --determinism strict \
  --recipe moljepa-smiles --artifact artifacts/moljepa-smiles \
  --reference-weights /path/to/Mol-JEPA/model.safetensors \
  --output /tmp/mol-smiles-cuda-strict.json
```

Repeat the STATE ST and Boltz commands with `--device cuda:0 --determinism strict`
and preserve their `--require-bitwise` gate. Mol-JEPA's Metal shaders are specific
to MPS; the loader chooses the PyTorch path on CUDA. This is a backend selection,
not evidence that CUDA numerics or performance have passed.

Strict determinism can reject operators with no deterministic implementation.
That is a recorded failure, not a reason to report success or silently widen a
tolerance. A separately justified production-settings trial may use
`--determinism inherit` with a new report, retaining the strict failure and the
same numerical thresholds. Backend settings and TF32 can change numerical
behaviour; see [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
No NVIDIA success or speed result is currently claimed.
The command's unavailable-hardware path was actually exercised locally and
returned exit status 2; [its report](../benchmarks/gpu_validation_2026-09-08_cuda_job_unavailable.json)
contains no candidate comparisons. Unit tests cover CUDA device-index checks
and synchronization dispatch with mocks; those are not NVIDIA numerical tests.

## A trusted local provider for another model

Create a local Python file importing `BackendCase` and `BackendSuite` from
`validate_backends`. Define `build(device, options)` returning a suite with:

```python
from validate_backends import BackendCase, BackendSuite

def build(device, options):
    return BackendSuite(
        reference_factory=load_original,       # callable(device) -> nn.Module
        candidate_factory=load_compressed,     # callable(device) -> nn.Module
        cases=lambda reference: [
            BackendCase("complete_prediction", run_complete_prediction, seed=1729)
        ],                                    # case.call(model, device)
        metadata={"model": "my_model"},        # JSON-serializable description
        source_files=("/path/to/model.py",),
    )
```

Then run:

```bash
python examples/validate_backends.py --device cuda:0 \
  --provider /path/provider.py:build --output /tmp/custom-cuda.json
```

`--options-json` supplies a JSON object to the provider. A provider executes local Python and is therefore
trusted code. It must return the complete model outputs and supply equivalent,
seed-controlled cases; the runner cannot prove that an arbitrary provider has
not omitted an API branch. Its reports cover the supplied cases only.
