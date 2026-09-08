# Lossless storage of frozen embeddings

`PackedFrozenEmbedding` stores an eligible original float32 embedding as
independently compressed row blocks. Requests reconstruct the selected rows and
then use the original downstream model. There is no precision change, training,
distillation, floating-point approximation or input/output cache.

```python
from compressme.packed_embedding import PackedFrozenEmbedding

source = model.embedding.eval().requires_grad_(False)
model.embedding = PackedFrozenEmbedding(source, block_rows=16)
```

Install the optional packing dependencies with `pip install 'compressme[torch,packing]'`.
The source must be an ordinary frozen, hook-free evaluation `nn.Embedding`, with
float32 weights, no `max_norm` update and no sparse-training behavior. This is
an explicit module replacement; callers must account for other aliases retaining
the original table. Smaller or incompressible tables may grow after packing;
compare registered state bytes before accepting a replacement.

Byte shuffle is a permutation; Zstandard reconstruction plus SHA256 verification
restores every original byte. The decoder uses integer indexing and copies
without floating-point arithmetic. The original shape of the gathered result is
preserved. Thus downstream operations receive the original row values at the
original request shape, avoiding a finite compiler's shape-dependent kernel
rounding. End-to-end numerical checks are still required for a particular model
and backend, including source nondeterminism and all requested outputs.

The compressed payload and offsets remain on CPU. An empty output anchor follows
`.to(device)`; returned float32 rows are transferred to that device. All encoded
CPU buffers count toward registered model storage and the Mac's unified memory.
This adds synchronization, decoding and transfer costs. It is a storage option,
not an inference acceleration method or a bound on peak request memory.
There is no persistent decoded cache. `weight` returns a complete independent
snapshot; modifying it leaves packed weights unchanged. That snapshot needs
additional memory. Training and mutable original weight aliases are not retained.

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
