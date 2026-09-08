# STATE: storage savings, CPU speed and GPU limits

**Original work:** [STATE paper](https://doi.org/10.1101/2025.06.26.661135) · [Arc Institute project](https://arcinstitute.org/manuscripts/State) · [original code](https://github.com/ArcInstitute/state) · [ST checkpoint](https://huggingface.co/arcinstitute/ST-HVG-Replogle) · [SE checkpoint](https://huggingface.co/arcinstitute/SE-100M).
{ .original-work }

STATE offers three different tradeoffs. ST uses 21.25% less resident weight
storage with unchanged outputs on CPU and MPS. SE's finite tables are smaller
and faster in the measured CPU requests but fail the MPS numerical gate. A separate SE representation
preserves original bytes on both devices, saving 6.31% of registered storage at
the cost of slower inference.

## ST: remove a redundant frozen table

In [`arcinstitute/ST-HVG-Replogle`, K562](https://huggingface.co/arcinstitute/ST-HVG-Replogle/tree/bb6a9562cbbf1fd152df14cc53b4cc7517c77175),
the 32,000 × 328 token table is frozen and entirely zero. Storing one row preserves
ordinary token IDs and expression inference, which already bypasses the lookup.
This reduces 49,396,728 parameters to 38,901,056 and raw tensor storage from
197,586,912 to 155,604,224 bytes. The published STATE source is pinned to
[`9bbfe78`](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add).

All outputs of `predict_step`, including decoded gene counts and metadata, were
bitwise identical on CPU and MPS for 1, 7, 64 and 128 cells, padded/unpadded
inputs, and integer/one-hot batch labels. These are constructed numerical probes
with actual trained weights, not a held-out biological benchmark. Token lookup
at multiple vocabulary positions also remains identical.

Expression prediction performs the same FLOPs. Lossless file packing already
handles the zero table cheaply: the unchanged baseline packs to 132,137,704
bytes, versus 132,148,096 bytes for the new layout, about 0.008% larger. The
saving is in resident weights and unpacked storage. Comparing against the
original 471.7 MB training checkpoint would also count unrelated training state.

The artifact, including local architecture/configuration and original licenses,
is prepared in `artifacts/state-st-hvg-k562`. Use the separate STATE environment:

```bash
.venv-state/bin/python
```

```python
from compressme import load_state_st

model = load_state_st("artifacts/state-st-hvg-k562", device="mps")
# Supply the same correctly ordered numerical batch as the original STATE API.
output = model.predict_step(batch, batch_idx=0, padded=False)
```

Use the original `ctrl_cell_emb`, `pert_emb` and `batch` fields and the
checkpoint's gene, perturbation and batch ordering. The model retains `forward`
and `predict_step`; it still needs upstream preprocessing and does not provide
an ST AnnData CLI.

For a new machine, install `.[state,packing,hub]` in a separate environment.
The published checkpoint requires Transformers 4.52.3 for its explicit
head-dimension configuration. The tested Mol-JEPA environment remains separate.
The local STATE loader defers an unused legacy VCI decoder import that otherwise
references an obsolete namespace; requesting that branch still fails explicitly.
Every class used by this checkpoint is loaded from the unchanged, hashed source.

Portable reload was tested in a fresh process with `torch.load` forbidden. It
reads the compressed safetensors payload and replay recipe; the original
training checkpoint is unnecessary. Evidence is in `benchmarks/state_st_export.json`,
`state_st_reload.json`, `state_st_cpu.json` and `state_st_mps.json`.

`state_st_cpu_bytes.json` and `state_st_mps_bytes.json` check logical tensor
bytes, including signed zero, for all four complete output cases and all 32,000
token rows after portable reload. Both pass. Earlier reports called numeric
equality “bitwise”; only these explicit byte comparisons support that stronger
claim. Reproduce them with `examples/verify_state_st_bytes.py` and explicit
`--checkpoint`, `--artifact`, `--device` and `--output` paths.

`deduplicate_embeddings(model)` accepts any eligible frozen table with
bitwise-identical rows, including nonzero rows outside gene models. It retains
tables that fail its conditions and is also available through
`compress_huggingface(..., method="constant_embeddings")`.

## SE: compile known protein features for CPU

Each known gene has 5,120 fixed floating-point features derived from its protein
sequence by ESM2. Gene names select these protein vectors; expression counts
supply the separate cell-specific information. STATE projects each vector to
1,024 internal features before processing the cell. The 5,120 dimensions describe
one protein, rather than different genes or expression measurements.

For 19,790 entries, the original float32 protein table occupies about 405.30 MB.
Keeping the original 5,120→1,024 encoder costs about 21 MB. The hybrid replaces
the much larger known-gene table while retaining this encoder for arbitrary
raw-vector inputs.

The CPU adapter stores two 1,024-feature tables: one for the raw-gene
encoder branch and one for the normalized gene-sentence branch. A smaller shared
projection-plus-norm representation was rejected after its classifier outputs
failed the numerical checks. Full-vocabulary evaluation across the nonlinear
encoder, followed by shape-matched constant rows, passes the extended CPU gate.
See [finite-domain compilation](finite-domains.md) for the transferable method.

Validation covers numeric gene-ID batches, original raw-vector forward calls
and seven constructed AnnData ingestion/export datasets described below.
`StateEmbeddingModel.get_gene_embedding` tests a sum over raw protein vectors,
so this helper still needs the original protein dictionary supplied separately.
The compressed lookup does not reconstruct direct access to raw
`pe_embedding.weight`.

### Load and validate the CPU adapter

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

The adapter retains `_compute_embedding_for_batch`, raw `forward` and
`gene_embedding_layer`. `gene_names` and `gene_to_index` expose the known
vocabulary without adding an unknown-gene policy. Supply `protein_embeddings=...`
only for the original raw-vector gene-name helper; that dictionary's memory is
additional to the compressed model's 151,243,944 parameters.

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

## SE: preserve original bytes on CPU and Apple GPU

`artifacts/state-se-100m-lossless` saves 6.31% of registered storage and matches
original STATE SE outputs byte for byte on CPU and Apple MPS, but runs
considerably slower in the measured requests. It retains the complete encoder
and all heads, storing the 19,790 × 5,120 gene table in independently compressed
16-row blocks. Each request reconstructs the selected float32 bytes and runs the
original normalization, projection and transformer at their original shapes.
This avoids the finite tables' GPU rounding mismatch without shape profiles,
rounded values or cached cell predictions.

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

Registered storage falls from 848,155,296 to 794,613,047 bytes. The latter includes
351,756,951 bytes of CPU compressed payload and index metadata, which count
toward the Mac's unified memory. Counting parameters alone misses these byte
buffers. There is no persistent decoded-row cache; requested rows, CPU-to-GPU
copies, decompression workspace, activations and allocator overhead add to this
total. Peak request memory is unmeasured. The 327,680-byte maximum decoded block
size does not bound that peak.

Fresh portable reload passes **377 tensor byte comparisons per device** on CPU
and MPS, with byte-identical source self-repeats. This includes 25 complete model
cases, 1–2,048 gene tokens, varied batch sizes, noncontiguous IDs, count inputs,
all token/cell/dataset outputs and decoder heads, arbitrary raw vectors, the
original name helper, and all 101,324,800 original embedding values. Maximum
observed output difference is zero. The source is the same pinned SE-100M
checkpoint; these are constructed numerical checks, not biological benchmarks
or NVIDIA CUDA validation.

CPU decoding and transfers add latency. Three warmed, interleaved pairs on the
Apple M5 used fresh gene/count values and the original numerical batch entry
point. Every paired output remained byte-identical.

| Backend | Cells × tokens | Original median | Lossless median | Latency multiplier |
|---|---:|---:|---:|---:|
| CPU, four threads | 1 × 32 | 20.43 ms | 79.01 ms | 3.87× |
| CPU, four threads | 1 × 2,048 | 859.50 ms | 1,612.39 ms | 1.88× |
| Apple MPS | 1 × 32 | 14.22 ms | 77.38 ms | 5.44× |
| Apple MPS | 1 × 2,048 | 233.41 ms | 1,141.89 ms | 4.89× |

These same-run ratios cover internal decoding and device transfers, excluding
loading and external input construction. Optional downstream decoder calls were
validated separately and excluded from timing. Three pairs do not give a general
performance bound; raw samples are in
`experiments/state-se-mps-fix/latency-cpu.json` and `latency-mps.json`.

Choose this artifact for its memory tradeoff. The finite-table artifact remains
smaller and faster on CPU and is still rejected on MPS. `load_state_se` selects
the representation recorded in the manifest; it never silently substitutes one
for the other. AnnData validation below covers the CPU finite-table workflow.

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
The backend runner permits CPU codec buffers only for the packed embedding;
its output anchor and ordinary model state must be on the requested device.
Earlier reports retain their original runner hashes.

## Encode AnnData with the CPU finite tables

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

The explicit byte check in `benchmarks/state_se_anndata_bytes.json` passes all
seven original-versus-portable cases, including the nine preprocessing fields
and all 1,034 output features. The earlier `benchmarks/state_se_anndata.json`
and `benchmarks/state_se_anndata_portable.json` reports used numeric equality
under a bitwise label; they remain separate evidence. Reproduce the public loader
check with `examples/verify_state_se_anndata.py`.

This validates CPU float32 h5ad embedding on the constructed datasets. Unseen
biological datasets, the expression decoder, LanceDB ingestion, Apple GPU AnnData
execution and other options of the original Inference class remain untested.
