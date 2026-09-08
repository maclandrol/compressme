# Draft posts

## LinkedIn

One thing that has bothered me recently is how much baggage some bio/chem models carry into inference. We download a large checkpoint, install the research codebase, and often assume that this is what running the model has to cost.

So I asked Astra to optimise Mol-JEPA for a common use case: SMILES in, all embeddings out. No retraining, no distillation, no lower precision.

It found unused input encoders, large projections that could be composed algebraically, and query/key projections whose interaction could be stored more compactly. The parameter count fell from 45.4M to 19.8M, a 56% reduction, with every embedding output retained.

Sparse graph execution and batched readouts then brought complete inference to roughly 2× the original speed on my Mac GPU, including parsing and transfers. Across 64 validation SMILES, output and attention differences stayed below 1e-5. That is measured numerical agreement, not a guarantee over every molecule.

Boltz-2 offered a different gain: its confidence and affinity models share thousands of identical tensors. Sharing their storage cut the jointly loaded weights by 49.6%, with identical tested outputs. It did not deliver a reliable GPU speedup.

The ideas are mostly classical. The work is establishing when each one applies and checking the complete outputs afterwards. I have wrapped the reusable passes into compressme, with separate checks for storage, output agreement and runtime. GPU validation so far is on Apple MPS; NVIDIA CUDA remains untested.

## X

I asked Astra to audit bio/chem models for inference waste, without retraining
or lower precision. SMILES-only Mol-JEPA: 45.4M → 19.8M parameters, all embedding
outputs kept, ~2× faster on my Mac GPU. Tested output error <1e-5. Reusable passes
are now in compressme.

## Evidence for the draft

These are drafts only; nothing has been published. The quantitative claims come
from the [technical report](technical-report.md) and its linked local benchmark
records. “All outputs” is scoped to each declared input contract; numerical
agreement is not a downstream biological quality benchmark. The Mol-JEPA error
bound describes the measured validation inputs, not every possible molecule.
The Boltz number is joint weight storage, not a single-model parameter reduction
or a speedup. No CUDA result is implied.
