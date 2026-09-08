# Share immutable model weights

When related checkpoints contain identical tensors, keeping a separate copy
for each model wastes memory. `share_frozen_parameters` lets frozen models
share that storage while executing their original operations. It works within
one ordinary model or across several models in an `nn.ModuleDict`.

```python
import torch
from compressme import share_frozen_parameters

models = torch.nn.ModuleDict({
    "structure": structure_model,
    "affinity": affinity_model,
}).eval().requires_grad_(False).to("mps")

result = share_frozen_parameters(models, inplace=True)
print(result.report["saved_resident_storage_bytes"])
# Both original model interfaces remain callable through result.model.
```

Apply the pass after moving models to their final device. A later device or
dtype conversion can allocate separate storage again. The default returns a
copy; use `inplace=True` to avoid temporarily copying a large model bundle.

The pass checks ordinary frozen, contiguous, finite real parameters. Shape,
stride, device, dtype and every value byte must match. SHA256 only finds
candidates; a separate byte comparison confirms every match. Buffers, trainable
parameters, custom Parameter subclasses and noncontiguous weights are retained.
Parameter/buffer storage aliases and distinct views of one allocation require a
separate preserving adapter and are refused before copying.

Sharing the data leaves the Parameter objects distinct. Iterating `parameters()`
still visits
the same number of parameters, so sums and other computations over that
sequence retain their values. Resident storage bytes decrease; logical parameter
count, precision and operation count remain unchanged.

The value-preservation argument is direct: if two immutable arrays contain
identical bits, both can read from one allocation without approximation. This
requires callers to leave shared weights unchanged and not depend on their
storage addresses. Separate Parameter objects have separate version counters,
so a write through one alias may escape a validation check on another. Earlier
compiler checks are therefore retained as historical evidence; complete-model
comparisons establish execution agreement after sharing.

`CompressionResult.save` records both existing Parameter-object aliases and
the new data-storage aliases. Reload checks that alias payloads agree before
restoring shared storage and strict-loading all tensor names. The weights then
stay shared during ordinary CPU reload. Apply the pass
again after moving the reloaded model to another device. The recipe works
independently of Boltz or any other model family.

For Boltz-2, the complete official structure/confidence and affinity checkpoints
have 5,019 common tensor names, all byte-identical. A separate safe tensor-bank
export reduces their combined inference distribution by 49.55%, relative to
independently deduplicated inference states. The Boltz evidence records the
model-level sharing and full-output execution
checks. The saving applies to joint storage: each model still performs its
original arithmetic. A workflow that unloads models sequentially has a different
resident-memory baseline.
