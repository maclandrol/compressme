# GPU validation: Apple MPS measured, NVIDIA CUDA not run

**Original models:** [Mol-JEPA, Rottach et al.](https://arxiv.org/abs/2608.22642) · [Boltz-2, Passaro et al.](https://doi.org/10.1101/2025.06.14.659707) · [STATE, Arc Institute](https://arcinstitute.org/manuscripts/State).
{ .original-work }

The trained models below ran on Apple MPS. CUDA remains untested because no
NVIDIA hardware was available. On 8 September 2026, the host was an Apple M5
MacBook Air with 16 GB unified memory, macOS 26.6.1, Python 3.12.14 and
PyTorch 2.14.0. MPS was available; PyTorch reported no CUDA build or CUDA devices,
and no NVIDIA driver utility was present. No cloud machine was provisioned.

The [hardware report](../benchmarks/gpu_validation_2026-09-08_cuda_unavailable.json)
records `status: unavailable`, `accepted: false` and no model validation. The
CUDA command below is ready to run when hardware is available; it has no CUDA
numerical result yet.

## Complete-output checks on trained checkpoints

The [runner](../examples/validate_backends.py) independently loaded the original
reference weights and each compressed artifact. It repeated the reference calls,
then compared every returned tensor and non-tensor leaf from complete predictions
on the same backend. The [summary](../benchmarks/gpu_validation_2026-09-08_summary.json)
links the metrics and report hashes.

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
Mol-JEPA met the numerical thresholds on these cases. The self-repeat differences
do not establish byte equality or rule out error introduced by the rewrite.

STATE ST checks every `predict_step` field for 1, 7, 64 and 128 cells, including
padded/unpadded and integer/one-hot batch labels, plus all 32,000 token IDs.
Inputs are synthetic numerical probes on trained weights. Boltz uses the native
20-residue protein/ethanol fixture, all structure/confidence/affinity outputs,
200 diffusion steps, 3 confidence recycles/1 sample, and 5 affinity recycles/3
samples. Both models' source self-repeats were also byte-identical. The optional
Boltz request-conditioning runtime was not enabled in this storage-artifact
check; its separate results remain in [the Boltz documentation](boltz2.md).

STATE SE has two representations with different tradeoffs. The 28.67%
parameter reduction remains CPU-only because its precomputed encoder tables
failed the complete MPS numerical gate.
The [guard check](../benchmarks/gpu_validation_2026-09-08_state_se_mps_rejected.json)
applies to that artifact. A separate lossless original-table mode restores the
original gene vectors before running the original computation. Fresh portable
reloads passed 377 tensor byte comparisons on each of CPU and MPS, including
all 19,790 original gene vectors, all tested output heads and arbitrary raw-vector
inputs. It reduces registered state from 848,155,296 to 794,613,047 bytes (6.31%).
It does not remove arithmetic or change precision; its small MPS timing check was
4.89–5.44 times slower because of decoding. AnnData export remains CPU-only.
See the [CPU reload report](../experiments/state-se-mps-fix/portable-reload-cpu.json),
[MPS reload report](../experiments/state-se-mps-fix/portable-reload-mps.json) and
[STATE guide](state.md). CUDA is unvalidated for both representations.

These output checks do not measure biological accuracy, throughput or
large-complex memory requirements. They compare each candidate with its reference
on the same backend. Equality across CPU, Apple GPU and NVIDIA GPU has not been
established, and PyTorch does not promise it across devices or releases even
with matched seeds. See the official
[reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Validation protocol

`examples/validate_backends.py` supports `cpu`, `mps`, `cuda` and `cuda:N`.
It requires the PyTorch dependencies; model-specific imports occur only for the
selected recipe. It never downloads weights or source code.

Reference and candidate factories run sequentially so both large models need
not remain resident. Each reference output is copied into an independent CPU
snapshot before the next call, preventing reused output buffers from erasing an
error. Ordinary model state and runtime output anchors must be on the requested
device. The exact `PackedFrozenEmbedding` type may keep its encoded payload and
offsets on CPU; those buffers still count towards state bytes.

Complete output container types, keys, shapes and dtypes must match, as must
discrete and non-tensor values. For each floating tensor, maximum absolute error
and relative L2 error must both be at most `1e-5` by default. These are separate
thresholds, not an `allclose` mixed tolerance. Empty tensor sets, nonfinite values
and unsupported output types fail. Byte equality is measured separately on
logical contiguous tensor values, including signed zero. `--require-bitwise`
makes it an acceptance requirement; physical strides and allocator layout need
not match.

The original model runs twice with the same seed and must pass the same numerical
or byte gate as the candidate. The runner checks every output and has no
output-selector option. It controls Python, NumPy and PyTorch RNG states and
runs frozen models in eval and inference mode, with autocast disabled. The
trusted case provider must control any custom generators.

MPS/CUDA calls synchronize explicitly. TF32 is off by default. Reports record
matmul precision, TF32, cuDNN, deterministic-algorithm settings, environment
flags, hardware, reference hashes and relevant source hashes; settings are
restored afterwards. The recorded MPS runs inherited deterministic-algorithm
mode, which was off, so they did not require deterministic kernels.

Unavailable devices return `status: unavailable` and a nonzero exit code.
Numerical or runtime failures return `status: failed` with the partial checks
and error retained. The runner refuses existing report paths to preserve earlier
evidence.

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

## Portable NVIDIA job, awaiting hardware

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
to MPS, so its loader selects the PyTorch path on CUDA. That path still needs
CUDA numerical and performance checks.

Strict determinism can reject operators that have no deterministic
implementation. Preserve that failure. A separately justified trial of
production settings may use `--determinism inherit` in a new report, with
the same numerical thresholds. Backend settings and TF32 can change numerical
behaviour; see [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
No NVIDIA numerical or speed result is available. The local
unavailable-hardware check returned exit status 2; [its report](../benchmarks/gpu_validation_2026-09-08_cuda_job_unavailable.json)
contains no candidate comparisons. Unit tests check CUDA device indices and synchronization dispatch with mocks;
they establish only the runner's control flow.

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

`--options-json` supplies a JSON object to the provider, which executes trusted
local Python. The provider must return complete outputs for equivalent,
seed-controlled cases. Reports cover only those cases: the runner cannot detect
an API branch that the provider omitted.
