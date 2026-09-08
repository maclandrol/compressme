# Share immutable model weights

Related checkpoints can contain large, exactly identical tensors. For frozen
inference, `share_frozen_parameters` stores matching parameter data once while
each model continues to execute its original operations. It applies to an
ordinary model or to several models in an `nn.ModuleDict`.

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

Apply the pass **after moving models to their final device**. A later device or
dtype conversion can allocate separate storage again. The default returns a
copy; `inplace=True` avoids temporarily copying a large model bundle.

The pass checks ordinary frozen, contiguous, finite real parameters. Shape,
stride, device, dtype and every value byte must match. SHA256 only finds
candidates; a separate byte comparison confirms every match. Buffers, trainable
parameters, custom Parameter subclasses and noncontiguous weights are retained.
Parameter/buffer storage aliases and distinct views of one allocation require a
separate preserving adapter and are refused before copying.

Distinct Parameter objects remain distinct. Iterating `parameters()` therefore
still visits the same number of parameters; sums or other computations over
that sequence retain their values. The saving is **resident storage bytes**,
not fewer logical parameters, reduced precision or fewer operations.

If two immutable arrays contain identical bits, replacing their backing
allocations with one allocation preserves the values read by each operation.
This uses no numerical approximation. The contract excludes inspecting storage
addresses and mutating shared weights. Separate Parameter objects have separate
version counters, so a write through one alias need not invalidate a validation
seal observing another. Earlier compiler seals are explicitly made historical;
complete-model comparisons still establish the actual execution evidence.

`CompressionResult.save` records both existing Parameter-object aliases and
the new data-storage aliases. Reload checks that alias payloads agree before
restoring shared storage and strict-loading all tensor names. This avoids
silently expanding the shared weights during ordinary CPU reload. After moving
the reloaded model to another device, apply the pass again. The saved recipe is
independent of Boltz or any other model family.

For Boltz-2, the complete official structure/confidence and affinity checkpoints
have 5,019 common tensor names, all byte-identical. A separate safe tensor-bank
export reduces their combined inference distribution by 49.55%, relative to
independently deduplicated inference states. The actual model-level sharing and
full-output execution checks are recorded in the Boltz evidence. This joint
storage result does not halve the arithmetic of one model, and a workflow that
already unloads models sequentially has a different resident-memory baseline.
