# Comparing with ONNX Runtime

ONNX Runtime is a sensible first baseline for deployment. In a small
[Mol-JEPA](https://arxiv.org/abs/2608.22642) CPU test, Compress Me's structural
rewrites still helped after ONNX export and runtime optimisation: the rewritten
model was **1.63× faster than the original ONNX model** by median paired timing.
This result applies to the fixed-shape workload below.

[ONNX](https://onnx.ai/onnx/intro/concepts.html) is a format for describing a
model's graph and weights. **ONNX Runtime** executes and optimises that graph.
Exporting to ONNX and measuring ONNX Runtime are separate steps.

## A small Mol-JEPA comparison

The test uses four molecules, 73 atoms and 144 directed edges on the Apple M5,
with FP32, PyTorch 2.14.0, ONNX 1.22.0 and ONNX Runtime 1.29.0. Each call includes
fresh original SMILES/RDKit graph preparation and the full numerical core.
All configurations return 12 predicted embeddings, CLS, all 13 latent embeddings
and both requested attention arrays.

| Configuration | Median call time | Observed range |
| --- | ---: | ---: |
| Original, PyTorch | 27.02 ms | 26.74–27.52 ms |
| Structural rewrites, PyTorch | 21.79 ms | 21.49–22.18 ms |
| Structural rewrites plus runtime changes, PyTorch | 18.29 ms | 17.90–19.33 ms |
| Original, ONNX Runtime | 25.71 ms | 25.00–26.41 ms |
| Structural rewrites, ONNX Runtime | **15.68 ms** | **15.36–16.06 ms** |

There are three warmups and ten interleaved rounds. Both runtimes use four CPU
threads; ONNX Runtime uses `ORT_ENABLE_ALL`, `CPUExecutionProvider` and disabled
idle thread spinning. Directly comparing the two ONNX configurations within
each round gives a median ratio of **1.626×**, ranging from **1.557 to 1.675×**.
The ratio of the two independent medians need not be identical.

Every arm uses the same original graph preparation. The third row therefore
tests the faster PyTorch numerical core with that common preparation, rather
than its separate sparse SMILES parser. Timings include NumPy/PyTorch input and
output views where needed, and exclude loading, export, session compilation,
warmup and validation. With precomputed graph tensors, ONNX Runtime core medians
were 16.89 → 8.41 ms; those narrower timings remain separate in the
[full report](../experiments/onnx/moljepa-final-2026-09-08.json).

All recorded output checks passed the existing `1e-5` maximum-absolute and
relative-L2 gates. The largest absolute difference was **3.815×10⁻⁶** across the
native-output, changed-input and timed comparisons. The exported outputs include
the full embedding and attention arrays; this is tolerance-based agreement,
not byte identity.

The ONNX export already omits the unused modality encoders. Exported initializer
counts, including constants, fall from 29,998,734 to 19,817,574 after the
structural rewrites. Graph plus external weight files occupy 120,777,202 and
80,062,191 bytes, respectively: **33.71% fewer file bytes**. These are export
sizes, not peak memory.

This is a bounded export: four graphs with fixed tensor shapes. Changed features,
edge order and node labels were also checked at those shapes; arbitrary graph
sizes and optional modalities remain outside this comparison. The first export
attempt failed at ONNX Runtime loading. Making PyTorch's implicit float32
promotion explicit in the mask stack fixed the Dynamo export, and the resulting
PyTorch wrappers still matched their native APIs byte for byte. No weights or
model equations changed. The [reproduction notes](../examples/onnx/README.md)
retain that compatibility change and the earlier failed attempts.

This CPU test shows that the structural rewrites and ONNX Runtime can help
together on this workload. The previously reported approximately 2× MPS gain
compares with original PyTorch execution on a different protocol. Core ML,
NVIDIA CUDA and a broad dynamic-shape ONNX deployment remain untested here.

## Where the approaches overlap

ONNX Runtime already performs constant folding, redundant-node elimination,
operator fusion and some layout optimisations. It can save an optimised graph
for later loading. These are established compiler techniques, also relevant to
the reductions used here. [ONNX Runtime graph optimisations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).

| Operation | Relationship to Compress Me |
| --- | --- |
| Remove computation outside the exported input/output path | An export specialised to SMILES can omit inactive modality branches. Those savings belong in the deployment baseline. |
| Fold constants and fuse operators | This overlaps with some local rewrites. Whether an exporter or runtime applies a particular affine, attention or normalisation rewrite must be checked in the resulting graph. |
| Share weights across models | ONNX Runtime explicitly supports shared initializers and prepacked weights. Boltz-2 sharing applies that established principle while retaining the native PyTorch models. |
| Improve execution layout and kernels | Both approaches can reduce execution cost without removing parameters. The measured device and complete workload determine the benefit. |

The shared-weight precedent is particularly direct: ONNX Runtime describes
models that use the same weights except in their last layers. Its
`AddInitializer` and shared prepacked-weight APIs address that case.
[ONNX Runtime session sharing](https://onnxruntime.ai/docs/get-started/with-c.html).

Fusion alone does not establish a smaller parameter representation. For
example, finding a fused LayerNorm node in an exported graph would not show
that the wide affine expansion has disappeared. Our [normalisation pass](normalization-statistic.md)
instead represents its downstream numerator and input-dependent denominator
from the narrow input. The resulting shapes and weights are what need comparing.

The current Compress Me passes also make explicit choices about the supported
API: for example, SMILES-only Mol-JEPA still returns every embedding, while a
finite gene lookup cannot replace unrestricted access to raw protein vectors.
Those choices determine which reductions are valid. A suitable rewrite can
precede ONNX export; successful export and numerical validation are still
required. The two tools can therefore be used together.

## What makes the comparison fair

The comparison above runs the original and rewritten computation in both
PyTorch and ONNX Runtime. That separates the effect of the structural rewrites
from the effect of changing runtimes.

The original full checkpoint has 45,406,721 parameters. Removing only inactive
input encoders leaves 29,944,320. The structural rewrites reduce this to
19,763,160, **34.00% below the already specialised model**. An exporter that
omits those inactive branches must receive the same credit. Count the actual
exported initializers and all weight files, including external data.

For Boltz-2, the memory comparison should also include sequential loading and
ONNX Runtime's shared-weight setup where export is supported. The existing
49.55% result compares two jointly resident PyTorch models with shared versus
separate registered storage. It does not measure those other deployments.

## Export and Apple hardware

PyTorch's recommended ONNX exporter captures tensor computation and records
shape constraints. Arbitrary Python preprocessing does not become an ONNX
graph merely by passing a model to the exporter. In a SMILES workflow, RDKit
preparation can remain outside the exported numerical core and still be counted
in complete-call latency. Unsupported operators or data-dependent control flow
may require an export adapter. [PyTorch ONNX exporter](https://docs.pytorch.org/docs/2.14/onnx_export.html).

On Apple hardware, ONNX Runtime's Core ML execution provider offers CPU, GPU
and Neural Engine choices, subject to supported operators and shapes. This is
a different execution path from PyTorch MPS. A CPU ONNX result cannot establish
Core ML or MPS performance; record the actual providers and any CPU fallback.
[Core ML provider options and operator support](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

The [methods notes](theory.md#relationship-to-existing-compression-work) cover
the closest prior work on the individual rewrites.
