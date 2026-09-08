# Nesso-1: modest CPU gains and lossless packing

**Original work:** Shenoy et al., [Nesso-1 (2026)](https://doi.org/10.64898/2026.08.01.742196) · [code](https://github.com/recursionpharma/nesso) · [weights](https://huggingface.co/recursionpharma/nesso).
{ .original-work }

For Nesso-1, the useful runtime changes were better matrix layouts and less Python bookkeeping. On two CPU examples, the final adapter gave median paired speed ratios of **1.047× and 1.108×**, with every tested output byte unchanged. The same matrix-layout change was slower on Apple MPS, so I would leave it disabled there.

Lossless packing also reduced the checkpoint from **165.4 MB to 140.8 MB**, a **14.91%** saving. The model still has 41,223,928 parameters and 164,895,712 bytes of registered tensor storage. Packing reduces the file we store or transfer; loading it restores the original weights.

## Results on a Mac

These measurements used an Apple M5 MacBook Air with 16 GB unified memory, macOS 26.6.1, PyTorch 2.14.0, FP32 and four CPU threads. Each row contains three warmed, alternating reference/candidate pairs. Both paths ran on the same backend with seed 42.

| Backend and input | Original median | Optimised median | Median paired speed ratio | Paired range |
| --- | ---: | ---: | ---: | ---: |
| CPU, 23 tokens | 0.865 s | 0.833 s | 1.047× | 1.038–1.086× |
| CPU, 143 tokens | 26.687 s | 25.044 s | 1.108× | 1.008–1.121× |
| MPS, 23 tokens | 0.277 s | 0.299 s | 0.925× | 0.772–0.940× |
| MPS, 143 tokens | 16.519 s | 17.531 s | 0.942× | 0.915–0.959× |

A ratio above one is faster. The median paired ratio can differ from the ratio of the two latency medians. The spread matters here: the larger CPU example ranges from almost neutral to 1.12×. An earlier runtime revision gave CPU medians of 1.096× and 1.072×. I would treat this as a modest, workload-dependent improvement, not a general speed guarantee. [Final CPU report](../experiments/nesso/reports/final_onepass_cpu.json), [small MPS report](../experiments/nesso/reports/final_onepass_tiny_mps.json), [larger MPS report](../experiments/nesso/reports/final_onepass_fragment_mps.json).

The timer includes adapter setup, state checks, cleanup and the complete native `predict_step`, including metadata and pocket-selection transfers. It excludes checkpoint loading, ESM extraction, YAML/RDKit preparation, external input/output transfers, RNG reset and validation. The [tutorial](tutorials/nesso.md) reproduces those boundaries.

## What changed

Nesso repeatedly contracts token-pair features independently for each channel. One direction computes

\[
C_{bijd}=\sum_k A_{bikd}B_{bjkd}.
\]

The [general contraction helper](../src/compressme/contractions.py) packs each operand into contiguous matrices, performs the batched matrix multiplication, then restores the axes. The Nesso adapter uses it for 608 triangle contractions per prediction. All surrounding projections, normalisation, masks and gates stay in their original order. The amount of arithmetic is unchanged; the aim is to avoid repeated handling of strided matrices. Layout changes can select different floating-point kernels, which is why the complete output comparison remains necessary.

There was also avoidable Python work. The adapter originally walked the module tree three times to inspect modules, parameters and buffers. It now inspects registered state in one pass, retaining every alias path and the same tensor identity, storage, shape, dtype, layout and version checks. Across 40 paired metadata measurements, one check fell from **4.724 to 2.914 ms**. That saves about 5.43 ms over the three checks in a fresh scope; it is not a 1.62× model speedup. A more elaborate cached checker cost more to construct than it saved and was discarded. [Python benchmark](../experiments/nesso/reports/python_checks.json).

File packing uses the existing general byte-shuffle/Zstandard codec. The exact size change was 165,426,752 → 140,754,378 bytes, and the restored file matched every original byte. It adds no prediction error and changes neither resident weight size nor computation. [Packing report](../experiments/nesso/reports/transport_roundtrip.json).

## What the comparisons establish

The inputs were a synthetic 20-residue peptide with ethanol, and the original Nesso test fragment of 130 residues with tyrosine: 23 and 143 tokens after ligand inclusion. Both used genuine ESM-2 features and unchanged native featurisation. Inference retained five recycling steps, refinement with the original 22 Å / 256-token crop, the 15 Å affinity cutoff and both affinity heads.

The checks cover all **11 native `forward` tensors and 21 `predict_step` tensors**, including affinities, raw pair representations, distograms, crop indices, entropies and metadata. Container structure, dtypes and scalar fields also have to agree. Original self-repeats and every timed prediction passed byte comparisons. The native bfloat16 cast in metadata remains; the raw FP32 representation is checked separately.

These are numerical preservation tests on two inputs, with no experimental affinity labels. They do not establish biological accuracy across a dataset. The complete 397-token tutorial prediction and NVIDIA CUDA remain untested. CPU and MPS are each compared with their own reference, not with each other.

## Other candidates

Removing a single-chunk attention copy and reusing fixed ESM conditioning preserved the tested outputs, but gave no reliable MPS gain. A fused-attention substitution changed output bytes: the largest raw-forward difference was 0.00244 on the small MPS example. It is excluded from the strict preservation path.

Running only ESM's final-embedding backbone avoided an unused language-model head and retention of intermediate layer outputs. The complete saved embeddings matched, but CPU speed ratios were 1.007× and 0.990× on the 130- and 384-residue sequences. That is too small to recommend as a speed optimisation. ESM remains a separate 650M-parameter preprocessing model with a 2.61 GB checkpoint; it is not included in the 165 MB Nesso file.

The transferable pieces are the contraction layout, single-pass state inspection, explicit final-embedding contract and lossless file codec. The Nesso integration remains an opt-in adapter for the pinned upstream implementation. It requires frozen, exclusively owned modules and immutable inputs under no-grad with autocast disabled. Its checks cannot detect every unsafe `.data` write or mutation of an unversioned inference tensor.

[Reproduce the results](tutorials/nesso.md) · [Implementation and experimental records](../experiments/nesso/README.md).
