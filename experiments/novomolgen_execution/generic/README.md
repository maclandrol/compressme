# General finite-fanout compiler inside native NovoMolGen

The reusable `compressme.compile_finite_fanout` compiler replaces the first
token embedding → float32 RMS normalization → Q/K/V projection fan-out inside
the real NovoMolGen 32M AtomWise native checkpoint. The complete downstream
native Llama model remains. Its original token embedding is retained for the
residual path and counted separately. The compiled lookup stores no source
weights or calibration examples.

This directory contains portable audit scripts and reports only. It does not
ship a dependency overlay, the compressme package, a model checkpoint, or a
whole-model production artifact. The scripts use installed `compressme`.

## Exact scope and evidence

The research wrapper accepts two-dimensional integer token IDs in frozen
float32 evaluation. It rejects training and `inputs_embeds`. It does not claim
equivalence to the separate custom NovoMolGen API or preserve arbitrary
continuous-embedding inputs. No quantization or distillation is used.

Compilation explicitly declares the source with `FiniteTokenFanout`, copied
ordinary token embedding and Q/K/V linears, `Float32RMSNorm`, and
`residual_key=None`. Floating-point shape profiles are supplied explicitly:

- CPU: one singleton profile (`max_rows=1`, `evaluation_rows=1`) plus bulk.
- MPS: singleton, then a small-matrix profile (`max_rows=15`,
  `evaluation_rows=2`), plus bulk.

These are empirical backend choices for the pinned implementation, not a
universal guarantee for other hardware, software, dtypes or untested shapes.
Local exhaustive token/shape comparisons pass bitwise: 370 cases on CPU and
748 on MPS. A local pass alone is not used to claim whole-model equivalence.

For **each backend**, the complete model passes 123 main cases / 5,899 tensor
comparisons and 134 additional shape cases / 6,700 comparisons. A separate
recursive byte audit verifies **257 cases / 12,599 tensor leaves /
314,076,156 logical value bytes per backend**, with zero byte differences and
zero numerical error. It checks shape and dtype, then compares CPU contiguous
`reshape(-1).view(torch.uint8)` tensors. This includes 12,439 float32 leaves,
16 int64 leaves and 144 boolean leaves, and distinguishes positive/negative zero.
Backing-storage padding and alias identities are outside this value-byte check.

The new `reports/*-bytes.json` files contain these explicit byte results. The
earlier reports without that suffix are preserved as numerical-equality
evidence: `torch.equal` and zero maximum error alone do not establish identical
floating-point bytes. The package-independent helper and its five regression
tests are included. Reconstructing finite generation scores with `where`
preserves the signs of finite zeros; infinity masks remain separate observables.

The main gate saves and reloads the compiled lookup through the ordinary
package serializer before attaching it to the model. This validates portable
lookup replay; the surrounding research wrapper is not a serialized native
model adapter.

The complete gate covers every singleton token, lengths through 2,048, loss,
tuple/dictionary results, hidden states, attention maps, legacy/Dynamic/Static
caches, B1/B2 cached decoding, and greedy/beam/seeded sampling at B1/B3. Generation
uses left-padding and one BOS token. Long probes above 128 tokens retain all
hidden states and omit attention maps. Exact token IDs, output keys and sampled
score infinity masks are also checked. Boundary cases include 13/14/15/16,
multiple factorisations into batch/length, and strided IDs and positions.

| Model | Resident parameter values | Reduction from original |
|---|---:|---:|
| Original | 31,556,096 | — |
| Generic CPU lookup | 31,027,200 | 528,896 / 1.676% |
| Generic MPS lookup | 31,156,224 | 399,872 / 1.267% |

Both retain the original 43,008-value token embedding and 416 buffer values.
CPU lookup tables contain 258,048 float32 values; MPS contains 387,072. These
counts include every profile table. They describe resident registered state,
not compressed disk size or process peak RAM.

## Runtime is essentially unchanged

Nine randomized interleaved rounds after three warmup pairs compare the same
token inputs, cache route, dtype and output settings. CPU uses four threads;
MPS is synchronized. Inputs are already on the device; tokenization and input
transfer are excluded. Every timed case first passes a numerical-equality gate
with zero numerical error. Timings were not rerun for the byte audit and their
stored gates are not presented as independent byte-equality checks.

| Native API call | CPU original → generic (ms) | MPS original → generic (ms) |
|---|---:|---:|
| Prefill B1×32 | 5.410 → 5.344 | 3.708 → 3.707 |
| Prefill B1×32, hidden states/attentions | 5.479 → 5.409 | 3.692 → 3.539 |
| Prefill B1×512 | 50.434 → 49.702 | 23.584 → 23.273 |
| One cached token, prefix32 | 2.621 → 2.694 | 4.201 → 4.205 |
| Greedy, up to12 new tokens | 34.313 → 33.916 | 72.344 → 72.100 |

The table shows medians. Reports retain every sample, means and sample standard
deviations. The initial MPS B4×32 case had high variability: medians 9.446 versus
10.821 ms, sample SD 2.062 versus 1.979 ms. It is retained, not discarded. A
separate 25-round / 10-warmup recheck against installed package code gives
5.986 versus 5.977 ms (1.001×). These measurements do not establish a material
speed improvement. The general compiler's named outputs are cloned to preserve
independent output semantics; no manual-table timing is attributed to it.

## Reproduction

Use Python 3.12.14, PyTorch 2.14.0, Transformers 4.46.2 and tokenizers 0.20.3,
plus the installed audited compressme package. The native runtime audit
documents the isolated pinned Transformers environment. The scripts stream
file hashes and do not use remote Python code or unsafe checkpoint loading.
The hashing implementation supports Python 3.10; actual inference was tested
only with the versions just listed.

```sh
rtk proxy python probe.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/native/config.json \
  --device cpu --table-device cpu --backend eager \
  --lookup-artifact /tmp/novo-lookup-cpu \
  --output /tmp/novo-generic-cpu.json

rtk proxy python boundary_probe.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/native/config.json \
  --device mps --output /tmp/novo-generic-boundaries-mps.json

rtk proxy python benchmark.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/native/config.json \
  --device mps --output /tmp/novo-generic-benchmark-mps.json
```

MPS execution needs native host GPU access. The scripts deliberately pin both
config/checkpoint and installed native/fan-out source hashes, so an updated
implementation requires new validation rather than silently inheriting results.

| Input | SHA256 |
|---|---|
| Native checkpoint | `3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699` |
| Native config | `f3cc487205aa417f4786f50d49a4120097baaffd80a9510ccfe5c37aa4ed3a34` |
| Transformers native Llama source | `733e7625fec5abcfa416ab9b866cb91875d41007f2e31c3e90d2ae0111ae98f7` |
| General finite-fanout source | `b83130353cf74b89c6a274259b84f17824041349f07576ae7acdff0315343fbb` |

The native checkpoint is the official `hf-checkpoint` revision
`dcd3f59261bebf84142c13617d4e129b4b0d0fdc` of
`chandar-lab/NovoMolGen_32M_SMILES_AtomWise`. The earlier manual-table failures
and original four-backend runtime smoke remain in their separate archived
evidence. `EVIDENCE_FILES.json` identifies the scripts and reports in this bundle.
