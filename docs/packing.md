# Lossless checkpoint packing

Packing reduces checkpoint storage and transfer size without rounding any values. It first rearranges bytes within fixed-size blocks, then applies Zstandard. Unpacking verifies the original length and SHA256 and returns exactly the original bytes, including tensor headers, signed zeros and NaN payload bits.

Byte shuffling followed by compression is a standard reversible storage
pipeline. [Blosc](https://github.com/Blosc/c-blosc/blob/main/README.md) combines
shuffle and bitshuffle filters with codecs including Zstandard. compressme uses
this approach in its checked checkpoint format; the measurements below report
the resulting file savings and decoding cost.

Install the optional dependencies with `pip install 'compressme[packing]'`. The core package can be imported without them.

```python
from compressme.packing import pack_bytes, unpack_bytes

packed = pack_bytes(checkpoint_bytes)
restored = unpack_bytes(packed, max_output_bytes=512 * 1024**2)
assert restored == checkpoint_bytes
```

The default decoder limit is 1 GiB; callers can pass a larger explicit limit. Passing `None` disables that limit. The current bytes API decodes a whole checkpoint into memory; it is not a lazy tensor loader. Use a packed artifact in place of its plain weight file to obtain the reported disk saving. Keeping both copies consumes more disk.

The decoder also checks empty frames: the tested Zstandard one-shot decoder
skips some integrity checks on zero-length content. Their history window is
capped at 1 MiB; the canonical `pack_bytes(b"")` frame needs no history.

Measured on the final algebraically rewritten Mol-JEPA snapshots using 1 MiB blocks, four-byte groups and Zstandard level 9:

| Artifact | Plain bytes | Packed bytes | Disk reduction | Median unpack time |
|---|---:|---:|---:|---:|
| Complete optional-input API | 140,926,608 | 120,213,589 | 14.70% | 0.156 s |
| Explicit SMILES-only API | 79,064,748 | 67,506,691 | 14.62% | 0.089 s |

Each result passed byte-for-byte equality and SHA256 checks over three pack/unpack runs. Packing times were 0.697 s and 0.387 s respectively. These are local serialization measurements, not prediction speedups. Once loaded, the tensor sizes and model computation are unchanged. Packing itself also needs temporary working memory.

Raw Zstandard saved approximately 7.2%, whereas byte shuffling saved about 14.7%. Bit-plane shuffling was slightly worse in size and substantially slower to decode, so it was rejected. The tensor audit found no duplicate tensors, no exactly zero matrix rows or columns, and only one zero scalar buffer; exact sparse storage and tensor deduplication offer no material additional saving on these snapshots.

A layer-by-layer loader could decode independent frames as they are needed and evict old weights to reduce peak residency. That would add repeated decoding work, leave activation memory unchanged and potentially slow inference. Such a loader is not implemented here.

The packing exactness claim concerns the supplied rewritten checkpoint. Algebraic model rewrites may already differ from the original model by floating-point reassociation; lossless packing adds no further difference.

Evidence: `compressme-packing-benchmark.json` records byte counts, timings, input/output hashes and snapshot paths. `compressme-lossless-audit.json` includes raw/byte/bit-shuffle comparisons and exact-zero/duplicate tensor checks.

The format is versioned and stores codec, transform, grouping/block parameters, original size and SHA256. Its decoder rejects unsupported metadata, mismatched frame size, truncation, trailing frames and checksum mismatch. SHA256 provides corruption detection, not author authentication. The size validation follows [python-zstandard's documented decompression behaviour](https://python-zstandard.readthedocs.io/en/latest/decompressor.html).

## Stream large files

The public file API keeps payload buffers bounded by the shuffle block, rather
than loading the entire checkpoint into Python bytes. It uses the same CMPRPACK
v1 format, compatible with `pack_bytes` / `unpack_bytes` in both directions.
It accepts ordinary files, so it works independently of model architecture.

```python
from compressme import pack_file, unpack_file

packed = pack_file("model.safetensors", "model.cmprpack")
restored = unpack_file(
    "model.cmprpack", "model.restored.safetensors",
    max_output_bytes=3 * 1024**3,
)
assert packed["sha256"] == restored["sha256"]
```

Source and destination must differ. Existing output files are preserved unless
`overwrite=True` is explicit. A private temporary file is published atomically
only after the operation completes; unpacking first verifies the exact single
frame, output size, checksum and whole-file SHA256. Input files must remain
immutable during processing. Before the atomic publication point, failure
preserves an existing destination. If deleting the private temporary link fails
after no-clobber publication, the error explicitly states that the complete,
verified destination was already committed.

`unpack_file` requires a finite output cap:
the default is 1 GiB, and `None` is refused. For the Boltz tensor file the example
uses an explicit 3 GiB cap. A separate `max_window_bytes` limit defaults to 64 MiB
and bounds decoder history memory. The default shuffle block is 1 MiB, limited
to 16 MiB. Compression workspace also depends on Zstandard level. Process memory also includes Python, NumPy and codec workspace.

The complete shared Boltz tensor file was packed and restored in a fresh
standalone process:

| Quantity | Measured result |
| --- | ---: |
| Original file | 2,062,669,224 B |
| Packed file | 1,759,531,547 B |
| Additional disk reduction | 14.696% |
| Peak process RSS | 56,360,960 B (53.75 MiB) |
| Restored bytes compared | 2,062,669,224, all equal |

Peak RSS includes Python, NumPy and codec workspace; Torch was not imported.
Packing took 12.14 s and unpacking 3.70 s in this diagnostic run. Other validation
could run concurrently, so these are not isolated throughput or inference
benchmarks. The [full report](../benchmarks/boltz2_streaming_pack.json) records
all hashes and stages; [reproduction and adversarial tests](../experiments/streaming-packing/README.md)
include empty-frame checksum/trailing-data cases, caps, truncation and atomic
publication failures. The exact final codec also passed ten independent
adversarial tests.

This file API does not change `save(packing=True)` into a streaming model export
or add a lazy model loader. Those older paths still materialize whole byte
payloads and have their own decode limits. Unpack a transported Boltz tensor file
to its original safetensors name before using `load_boltz2`.

## Shared state entries

Exports store repeated names of the same tensor storage view once and record
an alias map. Shape, dtype, stride, storage offset and lazy view flags must all
match. Equal-looking independent parameters are kept separate. Reload expands
the state names before strict loading; the architecture constructor restores its
original parameter ties. This saves file bytes only, because those parameters
already shared resident memory. Older artifacts without this optional manifest
field continue to load.

For shared transformed modules, an alias recipe restores the same module
instance at each recorded path before loading tensors. Tests
cover shared lookups, tied encoders, signed zero, and rejection of malformed
alias maps; unrelated views and independent equal weights remain separate unless
the explicit [frozen parameter sharing pass](sharing.md) has established and
recorded a new immutable storage alias.
