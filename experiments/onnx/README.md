# Mol-JEPA with ONNX Runtime on CPU

On the recorded four-molecule workload, the structurally compressed core remained
smaller and faster after ONNX export. With the same fresh original SMILES
preprocessing included in every call, original ORT took **25.71 ms** and compressed
ORT **15.68 ms**. The median of the ten same-round speed ratios was **1.626×**
(range **1.557–1.675×**). This is a matched CPU comparison with all five output
arrays, not an MPS result or a biological accuracy benchmark.

Original work: [Rottach et al., Mol-JEPA](https://arxiv.org/abs/2608.22642),
[authors' code](https://github.com/Boehringer-Ingelheim/mol-jepa), and the
[pinned author-linked checkpoint](https://huggingface.co/Flogrammer/Mol-JEPA/tree/4c912b450175f31b5ba913a5dc921c03b27b985a).
Upstream source and weights retain CC BY-NC 4.0.

## Final comparison

The host was an Apple M5 CPU on macOS 26.6.1. Python 3.12.14, PyTorch 2.14.0,
ONNX 1.22.0, ONNX Runtime 1.29.0 and ONNX Script 0.7.1 were used. The exporter
used Dynamo and opset 18. ORT used only `CPUExecutionProvider`, `ORT_ENABLE_ALL`,
sequential execution, four intra-op threads and one inter-op thread. Both idle
spinning settings were zero. PyTorch also used four intra-op threads and one
inter-op thread, with float32 matmul precision `highest`; no quantization,
precision conversion or fitting was used.

| Arm | Tensor core median | Common preprocessing + core median | Complete-call paired speedup vs original PyTorch |
|---|---:|---:|---:|
| Original / PyTorch | 19.76 ms | 27.02 ms | 1.000× |
| Structurally compressed / PyTorch | 13.87 ms | 21.79 ms | 1.245× |
| Final runtime core / PyTorch | 11.03 ms | 18.29 ms | 1.491× |
| Original / ONNX Runtime | 16.89 ms | 25.71 ms | 1.051× |
| Structurally compressed / ONNX Runtime | 8.41 ms | 15.68 ms | 1.724× |

Core-only original-ORT/compressed-ORT paired speedup was **2.067×**
(range **1.853–2.273×**). Each scope has three warmups per arm and ten rounds;
arm order rotates and reverses. Ratios use calls from the same round, so their
medians need not equal the ratio of the displayed latency medians. The
[final raw report](moljepa-final-2026-09-08.json) retains every call and gate;
the [summary](moljepa-final-summary.json) derives the paired ratios and storage
counts. Their timing scopes must not be combined with older Mol-JEPA benchmarks,
which used different molecules or omitted requested attention.

## Outputs and export boundary

The four SMILES are hexane, trimethylamine, indole and the recorded stereochemical
sugar string, given exactly in the report. Their tensor batch has 73 nodes,
144 directed edges, 82 node features and 17 edge features. The graph pointers
have shape `[5]`. This export fixes those shapes and the four-graph batching
contract. RDKit/molfeat/PyG preprocessing stays in Python; it runs afresh and
identically in every complete-call arm. `embeddings_data` must be `None`.
Variable graph sizes, other batches and optional modality inputs were not tested.

Outputs are predictions `[4,12,512]`, CLS `[4,512]`, latent embeddings
`[4,13,512]`, and both transformer attention arrays `[4,4,13,13]`. All are
float32. The native-to-core wrapper checks were byte-identical for the original,
portable compressed model and final runtime. Export also passed changed node
features, reordered edges and node relabelling at the same shapes. Those are
numerical probes, not additional measured molecules, and do not prove every
possible same-shape input is supported.

All **660 recorded tensor comparisons passed** the separate maximum-absolute
and relative-L2 thresholds of `1e-5`. The maximum absolute difference was
**3.814697265625e-6**, and maximum relative L2 difference **7.430159244162351e-7**.
Of these comparisons, 170 were byte-identical, including source and ORT
self-repeats. Candidate/source agreement is numerical, not a general byte
identity claim. The count includes repeated timing gates, not 660 independent
molecular examples.

## Storage: credit export for removing inactive branches

| Emitted ONNX graph | FLOAT initializer values | INT64 values | BOOL values | Graph + external data bytes |
|---|---:|---:|---:|---:|
| Original tensor core | 29,996,607 | 2,119 | 8 | 120,777,202 |
| Compressed tensor core | 19,815,447 | 2,119 | 8 | 80,062,191 |

FLOAT is float32. These are all initializer values, including constants and
buffers, rather than a count of trainable parameters. The files include ONNX
graph metadata and external data; they have not been losslessly packed here.
The compressed export has **10,181,160 fewer float initializer values (33.94%)**
and **33.71% fewer file bytes** than the original export.

The original PyTorch model still has 45,406,721 parameters, including inactive
optional encoders. Exporting the SMILES route already omits those inactive
branches; that saving belongs to export and the declared input contract. It
must not be attributed to the affine/bilinear rewrites. The compressed PyTorch
model has 19,763,160 parameters. Neither file sizes nor these initializer counts
measure peak process memory, activation storage or ORT prepacking allocations.

## Initial failures and compatibility change

The [unadjusted attempt](moljepa-2026-09-08.json) produced graph files with both
exporters, and the default ONNX checker returned without error. ORT session
loading then failed, before numerical validation or timing:

| Exporter, both model cores | Graph export | Default ONNX checker | ORT session load |
|---|---|---|---|
| Dynamo | Completed | Passed | `Concat` at `node_stack`: int64 and float type mismatch |
| Legacy TorchScript | Completed | Passed | First graph convolution `Sub`: incompatible dimensions |

The mixed types came from the native active-mask construction: graph pointer
differences are int64, while absent-modality zeros are float32. PyTorch's
`stack` promotes them to float32. Making that existing promotion explicit in
the export wrapper allowed Dynamo graphs to pass checker, ORT loading and all
output gates. No model weight, attention operation or requested output was
changed. Exporter warnings about native module bookkeeping are retained; the
changed-input and complete-output probes bound what was checked.

The [first accepted run](moljepa-mask-promotion-2026-09-08.json) retained ORT's
default idle spinning. It is preliminary: two resident sessions could consume
CPU while another arm was timed. The final run disables spinning consistently.
This scheduling choice follows the documented
[ORT thread controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
not a change to model arithmetic. Historical reports retain their original
script hashes. The current script can reproduce the unadjusted boundary by
omitting `--explicit-mask-promotion`, and the preliminary setting with
`--ort-spinning on`.

## Reproduction and provenance

Use the [short reproduction recipe](../../examples/onnx/README.md), which starts
from the existing Mol-JEPA tutorial and adds only isolated optional ONNX wheels.
The final benchmark script SHA256 is
`333d1cf2d9b5f2af51f4c5ad4756d5f351e0cdfddfb6eb83a548b457aa1860e3`.
The final report SHA256 is
`6d25a2d444a88593dc8b83c5066eabcdc416aaf7629c82675d4c15ab8084edb4`.
Reports record checkpoint, artifact, architecture, script and runtime-module
hashes. The supplementary [file ledger](provenance.json) hashes the evidence and
runtime source at handoff. Exported model binaries stay under ignored `work/`.

The exporter and runtime configuration are described in the official
[PyTorch ONNX documentation](https://docs.pytorch.org/docs/2.14/onnx_export.html) and
[ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
