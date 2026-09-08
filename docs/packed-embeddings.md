# Lossless storage of frozen embeddings

`PackedFrozenEmbedding` compresses an eligible original float32 embedding in
independent row blocks. Each request reconstructs the selected rows before
passing them to the original downstream model. The stored data are the trained
weights themselves, preserved without precision change or floating-point
approximation. No training, distillation or input/output cache is needed.

```python
from compressme.packed_embedding import PackedFrozenEmbedding

source = model.embedding.eval().requires_grad_(False)
model.embedding = PackedFrozenEmbedding(source, block_rows=16)
```

Install the optional packing dependencies with `pip install 'compressme[torch,packing]'`.
The source must be an ordinary frozen, hook-free evaluation `nn.Embedding`, with
float32 weights, no `max_norm` update and no sparse-training behavior. Replacing
the module does not remove copies retained through other aliases,
which must be included in the storage count. Smaller or incompressible tables
may grow after packing, so compare registered state bytes before accepting the
replacement.

Byte shuffle is a permutation; Zstandard reconstruction plus SHA256 verification
restores every original byte. The decoder uses integer indexing and copies
without floating-point arithmetic. The gathered result keeps its original
shape. Downstream operations therefore
receive the same row values at the same request shape, avoiding the
shape-dependent kernel rounding introduced by finite compilation. End-to-end
checks must still cover the particular model, backend and all requested outputs,
including any nondeterminism in the source.

The compressed payload and offsets remain on CPU. An empty output anchor follows
`.to(device)`; returned float32 rows are transferred to that device. All encoded
CPU buffers count toward registered model storage and the Mac's unified memory.
Reconstruction adds synchronization, decoding and transfer costs. The storage
saving provides neither inference acceleration nor a bound on peak request
memory. Decoded rows are not kept in a persistent cache. Reading `weight`
creates a complete independent snapshot that needs additional memory; modifying
it leaves packed weights unchanged. Training and mutable original weight aliases
are not retained.

The general artifact serializer records a closed versioned recipe and verifies
every compressed block at export and reload. Safetensors stores all payload and
index buffers. No original checkpoint is required to reload a complete artifact.
Tests cover arbitrary float32 bit patterns, signed zero, NaN payloads, empty and
strided indices, block boundaries, corruption, device movement and portable replay.

[STATE SE](state.md) is the actual-checkpoint example: 754 fresh CPU/MPS tensor
comparisons passed byte-for-byte, including all original weights and complete
outputs. Stored state falls 6.31%, while tested MPS calls take 4.89–5.44 times longer.
CPU finite-table compilation remains a separate, more compact option. CUDA code
paths have not been validated on NVIDIA hardware.
