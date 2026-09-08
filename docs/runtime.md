# Mol-JEPA inference on macOS

**Original work:** Rottach et al., [Mol-JEPA (2026)](https://arxiv.org/abs/2608.22642) · [authors’ code](https://github.com/Boehringer-Ingelheim/mol-jepa) · [author-linked checkpoint](https://huggingface.co/Flogrammer/Mol-JEPA).
{ .original-work }

Removing inactive encoders reduced Mol-JEPA's resident weights. Making a SMILES
request faster required changing the work it actually executes: graph
construction, attention, readouts and pooling.

1. **Construct only observed graph features.** The original feature code builds
   an atom-pair table and computes graph distances, then selects adjacent pairs.
   On selected bonds the distance is exactly one. The sparse implementation uses
   the same atom and bond descriptors, ring definition, hydrogen policy and atom
   ordering. It constructs only those directed edges, in the same order.
2. **Evaluate graph attention without the expanded messages.** A CSR adjacency
   layout groups each destination's incoming edges. Two Metal kernels compute
   the original bilinear scores and softmax, then accumulate the weighted values.
   The large edge-by-head-by-channel message tensor is never materialized.
   Every edge and head contributes. Degree above 32 uses the PyTorch formula;
   neighbours are never truncated.
3. **Batch independent affine work.** Query-interaction, value and skip maps
   share one packed projection. The thirteen output heads use a batched matrix
   multiplication. These operators store the complete trained maps once.
4. **Carry known metadata.** Batch size is already known from the SMILES list.
   Passing it to pooling avoids querying the GPU to rediscover it. The neighbour
   layout is constructed for each fresh batch and bound to that graph tensor.

The sparse descriptors and batches are bitwise equal to upstream in the tested
cases. The graph and readout algebra is unchanged in real arithmetic; its
floating-point reductions can differ. Validation covers complete outputs,
including requested attention tensors and the CLS vector.

The Metal implementation uses float32 and supports inference. CPU, gradients,
unsupported graph degrees and older Torch builds use ordinary PyTorch operations.
Autocast is rejected by the exact graph operators. Fused parameter updates can
change a training trajectory; no faster-fine-tuning claim is made here.

Run the reproducible complete-pipeline comparison from the project root:

```bash
.venv/bin/python examples/benchmark_moljepa_runtime.py \
  --checkpoint /path/to/original/model.safetensors
```

The measurements and output errors are recorded in
[`benchmarks/moljepa_runtime_macos.json`](../benchmarks/moljepa_runtime_macos.json).
The component profile is in `benchmarks/moljepa_runtime_profile.json`; the exact
feature audit is in `benchmarks/moljepa_sparse_features.json`.

The graph kernel can be reused for the supported contracted PyG attention
operator. Its input schema is established here by the SMILES feature adapter;
other architectures need their own execution-contract checks. These conditions
do not imply a universal compression ratio.

## Optional transfer coalescing

`transfer_tensors(tensors, device)` packs independent CPU tensor payloads into one
aligned byte buffer, performs one device copy, then exposes typed views. Tests
preserve float bits, large integer IDs, booleans, scalars and empty shapes. The
result preserves values and shapes, not original strides or storage aliases.
Gradient-carrying tensors require their original transfer path.

Local MPS probes made the transfer component 4–5 times faster. Complete
Mol-JEPA timing was mixed: one final run improved batches 1 and 32 while slowing
batch 4. Coalescing therefore remains optional: set
`accelerate_smiles(model, coalesced_transfer=True)`, or set the same attribute on
an already accelerated model. The default remains false. The complete outputs
and attention tensors were bitwise equal with this option on versus off. See
`benchmarks/coalesced_transfer_final.json` for the samples. The complete-call
samples determine the speedup; multiplying component gains would overstate it.
