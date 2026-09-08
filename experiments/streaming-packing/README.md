# Streaming lossless checkpoint files

`packing_files.py` adds a file API without changing the existing byte API or CMPRPACK version. It performs no tensor interpretation, numeric quantization, model rewrite, resident-memory reduction, or inference acceleration.

```python
from packing_files import pack_file, unpack_file

packed = pack_file("shared.safetensors", "shared.cmprpack")
verified = unpack_file(
    "shared.cmprpack", "restored.safetensors",
    max_output_bytes=3 * (1 << 30),
)
assert packed["sha256"] == verified["sha256"]
```

Install only `numpy` and `zstandard` for the standalone codec. Tests additionally use the existing installed `compressme.packing` API for cross-compatibility:

```sh
python -m pip install numpy zstandard
python test_packing_files.py
```

The functions accept source/destination paths and return sizes plus the raw SHA256. Both require distinct regular source and destination files. A destination is not overwritten unless `overwrite=True` is explicit. The private temporary file is in the destination directory; success publishes it atomically using a no-clobber hard link, or an atomic replacement for explicit overwrite. Before that commit point, failure cleans the temporary file and preserves any existing destination. If a no-clobber hard link has already committed and deleting its private temporary link fails, the function raises an explicit committed-destination cleanup error; the verified complete destination remains. This is atomic publication, not a promise of rollback after commitment or directory durability after a power failure. Filesystems without the required atomic operation reject rather than silently degrading to a partial write.

The default shuffle block is 1 MiB, bounded at 16 MiB; Python payload buffers grow with this block, not checkpoint size. Zstandard additionally keeps compression workspace and decompression history. The codec reports the latter; `unpack_file` defaults to a separate 64 MiB window limit checked before decoding and enforced in the decoder. Its minimum is 1 KiB. A cached fixed 2 KiB probe distinguishes the installed binding's bytes/KiB argument semantics; unknown enforcement behavior is refused. A one-byte priming read ensures the library history-window check runs even for small whole-frame outputs. High compression levels may require more compression workspace and a larger explicitly allowed decode window. No whole-file RAM or resident-model savings are claimed. For scale context only, `ZstdCompressionParameters.from_level(9, source_size=2_000_000_000).estimated_compression_context_size()` reported 11,002,904 bytes with python-zstandard 0.25.0; this is a library estimate, not measured peak process RAM.

The mandatory output cap defaults to 1 GiB and accepts only a nonnegative integer. Larger model files require an explicit larger cap; `None` is refused. Before output creation, the decoder checks the CMPRPACK header against the Zstandard content size/window and walks the ordinary frame's block lengths to find its exact boundary. It rejects truncated blocks/checksums, trailing bytes, concatenated or skippable frames, and unsupported headers. Streaming decode then checks Zstandard payload integrity, exact output size and SHA256 of the complete reconstructed bytes before publication. SHA256 verifies reconstruction, not author identity.

The header, shuffle block boundaries, group sizes 1/2/4/8, incomplete final groups and single Zstandard frame match CMPRPACK v1. The compressed frame need not be byte-identical to `pack_bytes` because streaming compressor choices may differ; both APIs reconstruct identical input bytes in either direction. Original `packing.py` is untouched.

Source files must remain immutable during either operation. On POSIX, nonblocking opening plus a regular-file check rejects a FIFO without waiting for a writer. This is not an assurance about every possible operating-system device path. Changes visible through file size/timestamps are refused; this is not a snapshot or a guarantee against an adversary changing input concurrently and hiding those changes. Destinations remain transactional even when a codec or integrity check fails.

Twenty-five small tests pass. They cover 24 byte/file cross-compatibility combinations; a 2 MiB plus 19-byte arbitrary payload; raw, compressed and RLE blocks; odd lengths and special float payload bits as opaque bytes; every truncation position in a small frame; corrupted headers/hashes/checksums; appended frames/bytes; output/window caps; unknown content size; no-clobber races; same-file/hard-link refusal; codec failure cleanup; input mutation detection; one-block shuffle bounds; decoder enforcement even when the preflight check is bypassed; POSIX FIFO refusal; the explicit post-commit cleanup boundary; and empty-frame checksum/trailing-data rejection. The complete Boltz tensor-file audit below was subsequently completed; no inference timing is attributed to the codec.

Implementation references: [python-zstandard streaming compression](https://python-zstandard.readthedocs.io/en/latest/compressor.html), [streaming decompression](https://python-zstandard.readthedocs.io/en/latest/decompressor.html), and the [official Zstandard frame/block specification](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md).

The installed 0.25.0 binding forwards the window argument directly to libzstd despite a KiB docstring; see its [primary C implementation](https://github.com/indygreg/python-zstandard/blob/0.25.0/c-ext/decompressor.c). The codec uses a small behavioral check rather than relying on that contradictory wording.


## Complete Boltz tensor-file verification

The separately launched `full_checkpoint.py` process packed and restored the complete 2,062,669,224-byte tensor file. It compared **every** reconstructed byte against the unchanged source, in addition to independent raw-file SHA256 checks. Both hashes were `1e6904266eccc8826225e828159eaea252888cdd6ea787fac79611f4e013d434`.

- Packed size: **1,759,531,547 bytes**, saving **303,137,677 bytes (14.6964%)** from that already shared tensor file.
- Packed SHA256: `fe19430c29b952aa39c131ff70f2b194cacf62e206c8e1b1dc50dc7ef39a1674`.
- Peak process RSS: **56,360,960 bytes (53.75 MiB)**, including Python, NumPy and codec workspace; `torch` was never imported.
- Diagnostic wall times: packing 12.14 seconds, unpacking 3.70 seconds. Other model checks could run concurrently, so these are neither isolated throughput benchmarks nor inference timing claims.
- Decoder history window: 4 MiB. Packing level: 9. The original file's size/timestamps were unchanged.

The exact report and output hashes are in `full-checkpoint/full-result.json` and `full-checkpoint/hashes.json`. The codec hash used was `5fd6525b8546ac317d32ff89e3ff772b94de0a013d39509f7c2f4189cfeb4db5`.

To reproduce, install the two optional dependencies, obtain the verified source tensor file, and choose a new output directory with sufficient disk space for the packed and restored files:

```sh
python full_checkpoint.py \
  --source /path/to/model.safetensors \
  --codec-dir . \
  --codec-sha256 5fd6525b8546ac317d32ff89e3ff772b94de0a013d39509f7c2f4189cfeb4db5 \
  --expected-sha256 1e6904266eccc8826225e828159eaea252888cdd6ea787fac79611f4e013d434 \
  --expected-size 2062669224 \
  --output-dir /path/to/new-audit-directory
```

The original source stays untouched. The output directory must not already exist. These results concern disk/transport bytes only; the reconstructed model has exactly the original tensor file and requires its original runtime storage.
