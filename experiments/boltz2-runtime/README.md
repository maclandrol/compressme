# Native Boltz-2 runtime and frozen storage sharing

Pinned upstream source: `jwohlwend/boltz`, revision
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` (Boltz 2.2.1).
The native classes and inference operations were not modified. Tests used
Python 3.12.14, PyTorch 2.14.0, Lightning 2.5.0 and NumPy 1.26.4 on an Apple
M5 MacBook Air. `environment.json` records the complete isolated environment.

The input is a 20-residue protein plus ethanol, with an explicitly empty MSA.
It is an execution and output-preservation fixture, not a binding-affinity or
structure-quality benchmark. Its native features contain 23 tokens and 160
padded atoms. No output or input caching was added.

## Native execution

Both complete native models strictly loaded every original named tensor from
the separately audited, checksum-verified pure safetensors input bank. The
constructor filters obsolete checkpoint arguments exactly as Lightning does,
and applies upstream CLI inference overrides; see `construct.py`.

All native `predict_step` outputs were finite, and the native writers produced
mmCIF, pLDDT, PAE, PDE, confidence JSON and the separate affinity JSON on CPU
and MPS. Full confidence runs used 200 diffusion steps, 3 recycles and 1 sample;
affinity used 200 steps, 5 recycles and 3 samples. Runs used FP32, CUDA-specific
kernels off and no source patches. The upstream CLI enables BF16 mixed
precision instead; these FP32 tests do not establish equivalence to that mode.
Pinned Lightning accepts both `accelerator='mps'` and the original CLI's
`accelerator='gpu'` on this Mac: the latter selects MPS automatically.

The first two-step runs are retained as diagnostic history. They demonstrate
execution only and produced unfinished diffusion coordinates. The full-schedule
runs supersede them as the practical native execution check. Recorded durations
are single-run diagnostics, not a benchmark or a compression speedup claim.
CPU and MPS outputs are not expected to match across backends: both floating
arithmetic and random-number generation can differ. Compression comparisons
always use the same backend, same original inputs and same random seed.

## Generic sharing result

`sharing_probe.py` places both ordinary native models in `nn.ModuleDict`, calls
`.eval().requires_grad_(False).to(device)`, and then applies the package's
general `share_frozen_parameters(..., inplace=True)` pass. It changes only
byte-identical parameter storage. Every Parameter object, enumeration entry,
shape, dtype, value and forward operation is retained. Shared parameters are
immutable under this inference contract. Their frozen status is explicit;
no claim is made for fine-tuning either model independently after sharing.

Registered unique tensor storage changes from **4,087,121,944 bytes to
2,061,868,568 bytes**, saving 2,025,253,376 bytes (49.55%). Logical parameter
count remains 1,021,780,232. There are 5,018 newly shared parameter storage
bindings. This saves resident registered weights, not model arithmetic or
activation memory. Process RSS and GPU allocator reservations are separate.

Both CPU and MPS passed explicit byte comparisons for all 48 returned native
tensor leaves: 21 confidence-model outputs and 27 affinity-model outputs,
639,024 logical value bytes per backend. Comparison includes tensor dtype,
shape and contiguous uint8 bytes, so signed zeros are checked. All original
self-repeat controls also matched byte for byte. Full reports preserve every
leaf metric and the tested sharing implementation's source hash. This is a
complete-output check on this one fixture, not a biological-quality study.

The general serializer saved a plain safetensors artifact: 2,062,669,224 bytes
for the tensor file, plus a 5,429,443-byte alias/architecture manifest and small
pinned factory data. Its ordinary ModuleDict keys are `confidence` and
`affinity`. The original paired
tensor bank is needed only for these research reference tests, not for loading
the standard compiled artifact once its pinned factory is supplied.

Fresh-process general artifact reload passed all 48 explicit output byte checks
on both CPU and MPS, with `torch.load` forbidden throughout. The portable loader
verifies the installed pinned Boltz source, uses the general safetensors loader,
then shares storage again after moving to the final device. The first CPU reload
probe compared differently augmented inputs and failed; its report is retained
as `reload-cpu-harness-rng-mismatch.json`. Upstream
`data/feature/featurizerv2.py:1494–1500` randomly rotates and translates reference
conformers during featurization, before diffusion. Matching both seeds fixed
the comparison without changing the package or artifact. The corrected reports
are `reload-cpu.json` and `reload-mps.json`.

## Reproduction

Install the pinned upstream source in an isolated Python 3.12 environment and
install `compressme` from its tested source. Paths are configurable in all model
execution scripts. `fetch_molecules.py` downloads the pinned official molecular
archive, verifies its published SHA256, and extracts only the 21 canonical
protein reference molecules. Native processing also generates ethanol from the
SMILES in `tiny_complex.yaml`. The archive and environments are not part of the
small evidence bundle. Its extracted `mols` directory contains only the
canonical protein references needed here; arbitrary CCD ligand or nucleic-acid
input preparation needs the full upstream molecular reference collection.

Run `native_smoke.py` with `--device cpu` or `--device mps`,
`--sampling-steps 200 --recycling-steps 3 --work <confidence-directory>`.
Then use `--kind aff --sampling-steps 200 --recycling-steps 5
--diffusion-samples 3 --structure-dir <confidence-directory>/predictions
--work <affinity-directory>`. The scripts expose `--shared-state`,
`--safe-loader-dir`, `--molecules` and `--input` for portable local paths.

`sharing_probe.py` runs the seeded before/after and self-repeat checks;
`build_artifact.py` creates the general package artifact only after accepted
CPU and MPS reports match the current sharing implementation hash.

`reload_probe.py` compares a fresh standard artifact load to the saved original
native tensor files. Supply `--artifact`, `--loader-dir` and `--native-dir` for
the artifact, trusted portable loader module and this evidence directory.
