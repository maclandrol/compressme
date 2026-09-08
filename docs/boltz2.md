# Boltz-2: complete models with shared frozen storage

**Original work:** Passaro et al., [Boltz-2 (2025)](https://doi.org/10.1101/2025.06.14.659707) · [original code](https://github.com/jwohlwend/boltz) · [upstream checkpoints](https://huggingface.co/boltz-community/boltz-2).
{ .original-work }

The portable bundle contains the original Boltz-2 confidence and affinity models.
Its general compression pass stores byte-identical, immutable parameters once
across the pair. It uses no distillation, rank truncation or precision change.
Both original native batch APIs and all returned tensors remain available.

## Measured result

| Quantity | Original pair | Shared pair |
| --- | ---: | ---: |
| Unique registered tensor storage | 4,087,121,944 B | 2,061,868,568 B |
| Logical Parameter values | 1,021,780,232 | 1,021,780,232 |
| Weight precision | FP32 | FP32 |

The registered storage saving is **49.55%**. This is a comparison with both
models loaded, including their original internal aliases. A workflow that
unloads one model before loading the other has a different memory baseline.
Process RSS, allocator reservations, activations and temporary load memory are
not the same as registered tensor storage. The saved safetensors file is
2,062,669,224 bytes, with a 5,429,443-byte rewrite manifest and small JSON
architecture files. Training checkpoint and optimizer overhead are excluded
from the fair inference-state comparison.

All 5,019 common named tensors in the published checkpoints are byte-identical.
The general pass uses hashes to identify candidates, then compares their actual
bytes, shape, dtype and layout. It preserves Parameter objects and enumeration;
only backing storage changes. See [the transferable pass and its contract](sharing.md).

## Load on a Mac

The project now has a prepared `.venv-boltz` environment with the verified
source and tested dependency versions. To reproduce it in a new checkout,
use an isolated Python 3.12 environment with the pinned upstream source:

```bash
cd ~/Code/compressme
uv venv --python 3.12 .venv-boltz
uv pip install --python .venv-boltz/bin/python -e . \
  'boltz @ git+https://github.com/jwohlwend/boltz.git@b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc' \
  'torch==2.14.0' 'numpy==1.26.4' 'pytorch-lightning==2.5.0'
```

The tested dependency inventory is in
[the runtime environment](../experiments/boltz2-runtime/environment.json), with
[the complete pinned requirements](../experiments/boltz2-runtime/requirements-tested.txt).
Boltz source verification checks all 107 Python files, not only the package
version. A different source revision needs its own compatibility validation.
The loader imports Boltz lazily; other compressme models do not need it installed.

```python
import torch
from compressme import load_boltz2

bundle = load_boltz2("artifacts/boltz2-shared", device="mps")  # or "cpu"
with torch.inference_mode():
    # Prepare a native Boltz batch on the selected device first.
    result = bundle["confidence"].predict_step(batch, 0)
```

`bundle["affinity"]` exposes the original affinity model and its native API.
Affinity input preparation still uses the predicted structure, as upstream does.
The loader needs neither original checkpoint: it reads safetensors and JSON,
constructs the locally installed original classes, and restores shared storage
after final device placement. Its ordinary constructors allocate both full
models first, so cold-load peak memory is higher than final resident weights.
The weights are frozen and must remain immutable. Moving the returned bundle
again can split shared allocations; load directly onto the intended device.

Native preprocessing, molecular reference assets, MSA preparation and output
writers remain upstream components. The small reference molecule directory in
the evidence is sufficient only for the supplied protein/ethanol fixture; it is
not a complete chemistry cache. Arbitrary native inputs need the appropriate
upstream assets. This loader accepts the usual native batches unchanged and
does not replace the entire Boltz command-line application.

## Run an unchanged native input file

The example accepts one native Boltz YAML/FASTA complex and uses the original
parser, data module and output writers. Affinity is run when requested in the
YAML. It saves every returned tensor as well as the usual mmCIF and score files.
The included numerical fixture can run entirely from local assets:

```bash
.venv-boltz/bin/python examples/boltz2.py \
  experiments/boltz2-runtime/tiny_complex.yaml \
  --molecules experiments/boltz2-runtime/mols \
  --output /tmp/compressme-boltz-example --device mps
```

Choose a new output directory for each run. For other inputs, supply their native
MSA files and the required molecular references. The example does not submit
sequences to an MSA server. The small included reference directory is not a full
Boltz chemistry cache. The file-based workflow was also checked with optional
request-invariant computation enabled; all 48 tensor outputs matched the original
MPS bytes. The final runtime source hash is recorded correctly in
[the final composition report](../benchmarks/boltz2_yaml_mps_final.json), with
complete outputs and tested source in
[the composition evidence](../experiments/boltz2-yaml-composition/README.md).

## What was verified

An Apple M5 MacBook Air with 16 GB unified memory ran both native models on CPU
and MPS, in FP32 with CUDA-specific kernels disabled and no Boltz source patch.
The fixture is a 20-residue protein plus ethanol, empty MSA, 23 tokens and 160
padded atoms. Confidence used 200 diffusion steps, 3 recycles and 1 sample;
affinity used 200 steps, 5 recycles and 3 samples.

Every returned tensor was compared, including structure, confidence and affinity
outputs: **48 tensor leaves / 639,024 value bytes per backend**. Original
self-repeats, shared execution and a fresh-process portable reload all matched
bytes, including signed zeros. Fresh reload tests forbade `torch.load`. The public `compressme.load_boltz2`
entry point was independently rerun on both devices with the same complete
byte agreement; reproduce it with `examples/verify_boltz2_reload.py`.
Native writers also produced mmCIF, confidence, pLDDT, PAE, PDE and affinity
outputs. Comparisons use the same backend and both the preprocessing and
sampling random seeds. CPU and MPS are not claimed to produce identical values
to each other. The upstream mixed-BF16 CLI mode was not this FP32 reference.

Run the complete public-loader check in the prepared project environment:

```bash
.venv-boltz/bin/python examples/verify_boltz2_reload.py --device mps \
  --output /tmp/boltz2-reload-mps.json
```

The [runtime evidence](../experiments/boltz2-runtime/README.md) includes complete
reports, saved original output tensors, reproduction scripts, hashes and the
retained failed reload comparison caused by mismatched preprocessing seeds.
The [checkpoint audit](../experiments/boltz2-weight-audit/README.md) documents
safe static extraction, publisher checksums and the complete tensor equality
analysis. No checkpoint pickle was executed during extraction or reload.

These are numerical execution and preservation tests on one small fixture.
They do not establish biological accuracy, large-complex memory requirements,
a universal floating-point certificate or a speedup. Storage sharing leaves the
original arithmetic unchanged. The optional compute experiment below is not
enabled by this storage loader.

## Lossless file packing

The general streaming codec reduces the already shared safetensors file from
**2,062,669,224 to 1,759,531,547 bytes**, saving another **14.696%** in storage
or transfer. Every reconstructed byte was compared to the complete source.
The standalone process peaked at **53.75 MiB RSS**, including Python, NumPy and
Zstandard workspace, without importing Torch. The manifest and architecture
are additional small files. [The full report](../benchmarks/boltz2_streaming_pack.json)
records hashes, sizes and diagnostic pack/unpack times.

`pack_file` and `unpack_file` work on any ordinary file; no model adapter is
required. Their [API and output limits](packing.md#stream-large-files) are explicit.
The supplied Boltz loader continues to read plain safetensors: unpack a transport
copy before loading. Keeping both copies consumes more disk. This codec saves
disk/transport bytes and adds no change to the reconstructed model's values,
resident weight size or prediction computation.

The ready-to-load bundle is `artifacts/boltz2-shared`. A complete packed transport
copy is supplied separately at
[`artifacts/boltz2-shared-transport`](../artifacts/boltz2-shared-transport/README.md),
with the original manifest, architecture, hashes and restoration instructions.

## Optional reuse within a prediction

Some diffusion conditioning depends only on fixed request inputs and frozen
weights. The optional adapter computes these branches once with their original
operators, dtype and full execution shapes, then reuses their values throughout
the 200 score calls. It leaves every timestep and time-dependent branch intact.
All 48 output tensor leaves pass byte comparisons on both CPU and MPS, including
original self-repeat and post-restoration controls.

The model must be frozen, unmodified, exclusively owned by the request and used
without gradients. Model state and conditioning must remain immutable throughout the
scope. Entry/exit audits and per-call checks reject detectable violations; writes
that bypass tensor version tracking remain prohibited by this explicit contract.
This is not a training adapter. Its caches are cleared after each request and
retain 3.02 MB for confidence or 9.06 MB for affinity on this fixture, in addition
to other runtime memory. The [implementation and evidence](../experiments/boltz2-request-runtime/lean/README.md)
describe the complete contract and limits.

Enable it explicitly with `--invariant-conditioning` in `examples/boltz2.py`,
or use `boltz2_invariant_sampling(model, immutable_request=True)` from
`compressme.boltz2_runtime` inside a `torch.no_grad()` scope.

Three warmed, interleaved timing pairs per model/backend include context setup,
native prediction and cleanup; they exclude unchanged preprocessing and loading.
All 12 timed pairs passed full output byte checks.

| Backend | Model | Original median | Candidate median | Median paired speed ratio | Paired range |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU, 4 threads | Confidence | 11.365 s | 10.888 s | 0.980× | 0.957–1.044× |
| CPU, 4 threads | Affinity | 21.742 s | 20.995 s | 1.031× | 1.014–1.064× |
| MPS | Confidence | 8.177 s | 8.340 s | 1.067× | 0.830–1.094× |
| MPS | Affinity | 10.880 s | 10.965 s | 1.000× | 0.992–1.007× |

A speed ratio above one is faster. The median paired ratio need not equal the
ratio of the two independent medians when timings drift. Confidence is
inconsistent; CPU affinity shows a small benefit in three pairs, and MPS has no
reliable gain. This evidence does **not** justify enabling the adapter by default
or claiming general acceleration. The transferable FX discovery prototype is
[a separate research component](../experiments/request_fx/README.md); this native
Boltz integration uses verified source-specific invariant analysis.

## Pinned original files

- Source: [Boltz b1ebfc4](https://github.com/jwohlwend/boltz/tree/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc).
- Weights: [boltz-community/boltz-2, revision 6fdef46](https://huggingface.co/boltz-community/boltz-2/tree/6fdef46d763fee7fbb83ca5501ccceff43b85607).
- `boltz2_conf.ckpt`: SHA256 `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1`.
- `boltz2_aff.ckpt`: SHA256 `dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e`.
- Portable tensor file: SHA256 `1e6904266eccc8826225e828159eaea252888cdd6ea787fac79611f4e013d434`.

Upstream attribution and license are included with the architecture and audit.
