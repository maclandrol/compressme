# State ST and SE: general passes on pretrained models

The selected checkpoint is
[`arcinstitute/ST-HVG-Replogle`, K562](https://huggingface.co/arcinstitute/ST-HVG-Replogle/tree/bb6a9562cbbf1fd152df14cc53b4cc7517c77175),
with the published State source pinned to
[`9bbfe78`](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add).

Its 32,000 × 328 token table is frozen and entirely zero. The ordinary expression
path bypasses token lookup. The general constant-row embedding pass stores one
row and preserves both expression inference and ordinary token IDs. It reduces
49,396,728 parameters to 38,901,056, and raw tensor storage from 197,586,912 to
155,604,224 bytes: **21.25% less resident weight storage**.

All outputs of `predict_step`, including decoded gene counts and metadata, were
bitwise identical on CPU and MPS for 1, 7, 64 and 128 cells, padded/unpadded
inputs, and integer/one-hot batch labels. These are constructed numerical probes
with actual trained weights, not a held-out biological benchmark. Token lookup
at multiple vocabulary positions also remains identical.

This does **not** save active FLOPs on expression prediction. It also does not
improve an already losslessly packed file: the unchanged baseline packs to
132,137,704 bytes and the new layout to 132,148,096 bytes, about 0.008% larger.
The zero table already compresses almost for free. The benefit is resident
weights and unpacked storage. The original 471.7 MB training checkpoint contains
additional training state and is not the fair inference-weight denominator.

The artifact, including local architecture/configuration and original licenses,
is prepared in `artifacts/state-st-hvg-k562`. Use the separate State environment:

```bash
.venv-state/bin/python
```

```python
from compressme import load_state_st

model = load_state_st("artifacts/state-st-hvg-k562", device="mps")
# Supply the same correctly ordered numerical batch as the original State API.
output = model.predict_step(batch, batch_idx=0, padded=False)
```

The batch uses the original `ctrl_cell_emb`, `pert_emb` and `batch` fields,
with the checkpoint's gene, perturbation and batch ordering. Original
preprocessing remains necessary. The compressed numerical model retains the
original `forward` and `predict_step`; this is not a replacement AnnData CLI.

For a new machine, install `.[state,packing,hub]` in a separate environment.
The published checkpoint requires Transformers 4.52.3 for its explicit
head-dimension configuration. The tested Mol-JEPA environment remains separate.
The local State loader defers an unused legacy VCI decoder import that otherwise
references an obsolete namespace; requesting that branch still fails explicitly.
Every class used by this checkpoint is loaded from the unchanged, hashed source.

Portable reload was tested in a fresh process with `torch.load` forbidden. It
reads the compressed safetensors payload and replay recipe; the original
training checkpoint is unnecessary. Evidence is in `benchmarks/state_st_export.json`,
`state_st_reload.json`, `state_st_cpu.json` and `state_st_mps.json`.

The later `state_st_cpu_bytes.json` and `state_st_mps_bytes.json` explicitly
compare logical tensor bytes, including the sign of zero, for all four complete
output cases and all 32,000 token rows. Both pass after portable reload. Earlier
reports used numeric equality while calling it bitwise equality; the new reports
supply the stronger measurement. Reproduce with `examples/verify_state_st_bytes.py`
and explicit `--checkpoint`, `--artifact`, `--device` and `--output` paths.

The transferable operation is `deduplicate_embeddings(model)`, also available
through `compress_huggingface(..., method="constant_embeddings")`. It is not
specific to gene models or zero values. It accepts any eligible frozen table
whose rows are bitwise identical, and retains tables that fail its conditions.

## What the State SE 5,120-dimensional vectors mean

Each known gene has a fixed vector of 5,120 floating-point features derived from
its protein sequence by ESM2. These are protein features, not 5,120 different
genes and not a cell's expression measurements. In an ordinary workflow, gene
names select these vectors and expression counts provide separate cell-specific
information. State projects the protein features to 1,024 internal features
before processing the cell.

For 19,790 entries, the original float32 protein table occupies about 405.30 MB.
Keeping the original 5,120→1,024 encoder costs about 21 MB, so it is possible to
retain arbitrary raw-vector inputs while replacing the much larger known-gene
table. Removing that raw-vector API is not necessary for the current hybrid.

The CPU adapter stores two 1,024-feature tables: one for the raw-gene
encoder branch and one for the normalized gene-sentence branch. A smaller shared
projection-plus-norm representation was rejected after its classifier outputs
failed the numerical checks. Full-vocabulary evaluation across the nonlinear
encoder, followed by shape-matched constant rows, passes the extended CPU gate.
See [finite-domain compilation](finite-domains.md) for the transferable method.

The strict scope matters: numeric gene-ID batch calls and original raw-vector
forward calls are validated. The AnnData ingestion/export workflow additionally
passes seven constructed datasets, as detailed below. The original `StateEmbeddingModel.get_gene_embedding`
helper tests a sum over raw protein vectors; preserving that exact helper needs
the original protein dictionary supplied separately. Direct access to the removed
raw `pe_embedding.weight` is not reconstructed by the compressed lookup.

## Run the portable State SE numerical adapter

Use the prepared `.venv-state` environment. The artifact includes the smaller
float32 tables, all retained numerical heads, the original raw-vector encoder,
19,790 ordered gene names, original architecture source and licenses. Reload does
not need the original 869 MB checkpoint or the raw protein dictionary.

```python
import torch
from compressme import load_state_se

model = load_state_se("artifacts/state-se-100m-cpu", device="cpu")
# Follow the original sentence convention: position zero is the CLS token.
ids = torch.tensor([[model.cfg.dataset.cls_token_idx,
                     model.gene_to_index["TP53"], model.gene_to_index["KRAS"]]])
mask = torch.zeros_like(ids, dtype=torch.bool)
with torch.inference_mode():
    token_outputs, cell_embedding, dataset_embedding = model.forward_gene_ids(
        ids, mask, counts=None)
```

The original numeric `_compute_embedding_for_batch` entry point is also retained,
as are the callable raw `forward` and `gene_embedding_layer`. `gene_names` and
`gene_to_index` provide known vocabulary metadata; unknown-gene behavior is not
invented. Supply `protein_embeddings=...` only when using the original model's
raw-vector gene-name helper. That dictionary's extra memory is separate from the
compressed model's 151,243,944 parameters.

The weight file is **511,190,195 bytes**. Extended checks compare 105 numerical
outputs over seven cases, including 2,048-gene inputs; maximum observed absolute
error is **6.68e-6** at `atol=rtol=1e-5`. Portable reload reproduces the compressed
outputs bit for bit. CPU float32 is the validated execution mode. Both loading on
MPS and using the token route after a manual device change are rejected because
the MPS candidates failed the numerical gates.

Reproduce portable reload independently:

```bash
.venv-state/bin/python examples/verify_state_se.py
```

Rebuild from the pinned original checkpoint with
`examples/state_se/build.py --checkpoint /path/to/model.safetensors` in the same
environment. The compiler's general operators are documented separately from
this architecture adapter.

A short CPU timing comparison used fresh gene/count batches and five shuffled,
interleaved rounds after warmup on the Apple M5, with four threads:

| Cells × gene tokens | Original median | Compressed median | Speed ratio |
|---|---:|---:|---:|
| 1 × 32 | 15.58 ms | 11.20 ms | 1.39× |
| 1 × 2,048 | 412.31 ms | 372.40 ms | 1.11× |

These timings cover the original numerical batch entry point. Loading,
compilation and optional downstream decoder calls are separate. The longer
case spends more time in the unchanged transformer, so the same weight reduction
does not imply the same runtime gain. First calls at a new shape and all raw
samples are retained in `benchmarks/state_se_cpu_runtime.json`.

## Lossless original State SE on Apple GPU

A separate artifact, `artifacts/state-se-100m-lossless`, now passes the original
State SE model on CPU and Apple MPS. It keeps the complete original encoder and
all heads. The 19,790 × 5,120 gene table is stored as independently compressed
16-row blocks. Each request reconstructs the selected original float32 bytes,
then executes the original normalization, projection and transformer operations
at their original shapes. There are no shape profiles, rounded values or cached
cell predictions. This avoids the GPU rounding mismatch of the CPU finite tables.

```python
import torch
from compressme import load_state_se

model = load_state_se("artifacts/state-se-100m-lossless", device="mps")
with torch.inference_mode():
    result = model._compute_embedding_for_batch(batch)  # unchanged native batch
    gene_features = model.get_gene_embedding(["TP53", "KRAS"])
```

The original raw 5,120-dimensional vector `forward` and `gene_embedding_layer` remain intact.
The original gene-name helper receives exact decoded protein rows, including its
raw zero-sum guard, without a second protein dictionary. Reading
`model.pe_embedding.weight` materializes a complete exact snapshot; modifying
that snapshot does not modify this frozen representation. Training and mutable
weight aliases are outside its contract.

Registered model storage is **848,155,296 → 794,613,047 bytes: 6.31% less**.
The candidate total includes **351,756,951 bytes of CPU compressed payload and
index metadata**, which count toward the Mac's unified memory. Parameter counts
alone are misleading here because the compressed table is represented by byte
buffers. There is no persistent decoded-row cache. Requested row tensors,
CPU-to-GPU copies, decompression workspace, activations and allocator overhead
are additional; peak request memory has not been measured. A single decoded
block contains at most 327,680 bytes, but this is not a peak-memory bound.

Fresh portable reload passes **377 tensor byte comparisons per device** on CPU
and MPS, with byte-identical source self-repeats. This includes 25 complete model
cases, 1–2,048 gene tokens, varied batch sizes, noncontiguous IDs, count inputs,
all token/cell/dataset outputs and decoder heads, arbitrary raw vectors, the
original name helper, and all 101,324,800 original embedding values. Maximum
observed output difference is zero. The source is the same pinned SE-100M
checkpoint; these are constructed numerical checks, not biological benchmarks
or NVIDIA CUDA validation.

This storage path adds CPU decoding and transfers and is considerably slower
in the measured requests. Three warmed, interleaved pairs on the Apple M5 used
fresh gene/count values and the original numerical batch entry point. Every
paired output remained byte-identical.

| Backend | Cells × tokens | Original median | Lossless median | Latency multiplier |
|---|---:|---:|---:|---:|
| CPU, four threads | 1 × 32 | 20.43 ms | 79.01 ms | 3.87× |
| CPU, four threads | 1 × 2,048 | 859.50 ms | 1,612.39 ms | 1.88× |
| Apple MPS | 1 × 32 | 14.22 ms | 77.38 ms | 5.44× |
| Apple MPS | 1 × 2,048 | 233.41 ms | 1,141.89 ms | 4.89× |

These are same-run ratios; the short sample is not a general performance bound.
Internal decoding and device transfers are included, while model loading and
external input construction are excluded. Optional downstream decoder calls
were validated separately and are not included in this timing scope. Raw pairs
are retained in `experiments/state-se-mps-fix/latency-cpu.json` and
`latency-mps.json`. This path is an explicit memory option, with no acceleration
claim.
The CPU finite-table artifact remains the smaller, faster CPU option; its MPS
rejection remains in force. The lossless artifact is selected by its manifest,
so `load_state_se` does not silently change an existing artifact's representation.
AnnData validation below still refers to the CPU finite-table workflow.

Rebuild the lossless artifact using `examples/state_se/build_lossless.py` with
explicit `--checkpoint`, `--architecture` and `--output` paths. Validate on either
backend using `examples/validate_backends.py --provider
examples/state_se/lossless_backend.py:provider --device mps --require-bitwise`.
Supply `--options-json` containing `artifact` and `checkpoint` paths and a new
`--output` report path. All original weights are loaded with safetensors; portable
reload explicitly forbids `torch.load`. The transferable operator is
`compressme.packed_embedding.PackedFrozenEmbedding`, which can replace another
eligible frozen float32 embedding and uses the general serializer.

See `experiments/state-se-mps-fix/portable-reload-cpu.json` and
`portable-reload-mps.json` for complete output metrics and exact source hashes.
The new backend runner explicitly permits only the packed embedding's CPU codec
buffers; its output anchor and all ordinary model state must be on the requested
device. Earlier backend reports retain their original runner hashes.

## Encode AnnData with the portable model

The reusable loader bundles the original pinned dataset and collator with the
compressed model. It uses the artifact's ordered gene vocabulary and does not
load the original protein dictionary. In a Python script, retain the main guard
because the original collator runs in a worker process on macOS:

```python
from compressme import load_state_se_encoder

if __name__ == "__main__":
    encoder = load_state_se_encoder("artifacts/state-se-100m-cpu")
    embeddings = encoder.encode_adata(
        "cells.h5ad", "cells-embedded.h5ad", emb_key="X_emb",
        batch_size=2, seed=42,
    )
    print(embeddings.shape)  # (number_of_cells, 1034)
```

This preserves the upstream 1,024 cell features plus 10 dataset features.
The output path is optional; existing files are never overwritten. Explicit
seeding reproduces upstream sampling and restores the caller's Python, NumPy
and CPU PyTorch random state. Without a seed, the original ambient random
behavior is retained. The model does not cache cell inputs or predictions.

Seven synthetic h5ad variants, using actual pretrained weights, cover dense,
CSR and CSC matrices, count handling, gene-column fallback, partial index
precedence, duplicate gene names and zero counts. All nine preprocessing batch
fields and all 1,034 output features match the original bitwise. Metadata,
existing obsm values and cell ordering survive export. Original edge-case
failures, including empty data and missing padding candidates, are retained.
Fresh-process reload also passed with torch.load forbidden in parent and workers.

These checks cover CPU float32 h5ad embedding. They do not validate an unseen
biological dataset, the expression decoder, LanceDB ingestion or every option of
the original Inference class. Evidence is in `benchmarks/state_se_anndata.json`
and `benchmarks/state_se_anndata_portable.json`; the reproducible public loader
check is `examples/verify_state_se_anndata.py`.

`benchmarks/state_se_anndata_bytes.json` repeats all seven original-versus-portable
cases with explicit byte comparison for the nine preprocessing fields and all
1,034 output features. All pass. This strengthens the older checks that used
numeric equality under a bitwise label; it does not add a biological dataset or
an Apple GPU validation claim.
