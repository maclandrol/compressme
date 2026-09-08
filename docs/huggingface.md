# Start from a Hugging Face checkpoint

Install the optional Hub client with `pip install -e '.[hub]'`. Inspection pins
the requested branch or revision to a commit and lists complete checkpoint
candidates, including sharded safetensors. It does not load model weights.

```python
from compressme import inspect_huggingface, get_target

plan = inspect_huggingface("Flogrammer/Mol-JEPA")
print(plan["revision"], plan["selected"], plan["weight_bytes"])
print(get_target("state")["hf_id"])
```

The same inspection is available from the project environment:

```bash
.venv/bin/python -m compressme inspect Flogrammer/Mol-JEPA
.venv/bin/python -m compressme targets
```

Compression needs the local architecture and representative calls. The factory
requires the optional PyTorch runtime (`pip install -e '.[torch,hub]'`) and the
dependencies of the local architecture, installed separately. The factory
must construct the checkpoint's architecture with matching tensor dtypes. It
does not need the original trained values. For a model in a local codebase:

```python
from compressme import Example, compress_huggingface
from my_model import Model, make_validation_batch

result = compress_huggingface(
    "organisation/model",
    model_factory=lambda: Model(...).eval(),
    validation=[Example((make_validation_batch(),))],
    method="affine",
)
print(result.report)
result.save("compressed-model", packing=True)
```

`method="affine"` uses the existing affine and normalization-statistic compiler.
`method="constant_embeddings"` finds frozen embedding tables whose rows have
identical bits and stores each such row once. The latter preserves the original
Python model's methods and ordinary token lookup. Custom consumers that require
contiguous embedding weights need separate validation because the full-shaped
weight becomes an expanded view.

`method="finite_lookup"` discovers eligible registered Sequential token encoders
or selects `finite_paths=["token_encoder"]`. It requires
`finite_input_contract="token_indices_only"` for those blocks and performs
complete-output validation inside the general model-level compiler. Other inputs
of the enclosing model can remain available. The compiler refuses known external
weight aliases and only accepts rewrites that save both logical tensor bytes and
unique registered backing storage. See [finite-domain compilation](finite-domains.md).

Every proposal must pass the supplied complete-output examples. A failed gate
returns the loaded original model with a rejection report. An unchanged model
is a valid outcome when no exact condition applies. Dynamic Python forwards
can require a model-specific adapter instead of FX tracing.

If several checkpoints exist in a repository, provide `filename`. The selected
STATE release, for example, uses
`filename="fewshot/k562/checkpoints/final.ckpt"`. `.ckpt` is accepted only when
explicitly selected and loaded with `torch.load(weights_only=True)`; the loader
does not fall back to unrestricted pickle. Safetensors can be single or sharded.

Downloads use the pinned commit. The report records file hashes, exact state
key/shape/dtype checks and declared parameter aliases. Missing tied names can be
filled only from another name for the same local tensor object. Conflicting
aliases are rejected. No Hub Python is downloaded or executed by this workflow;
the architecture factory is ordinary caller-owned local Python.

Live inspection of Mol-JEPA, X-Cell and STATE is recorded in
`benchmarks/huggingface_live_inspection.json`. X-Cell resolves successfully but
has no selected weights, because none are published in the inspected revision.
Single/sharded loading, strict aliases, rollback and export/reload are also
covered by isolated network-fixture tests.

The loader follows the public [Hub metadata and download APIs](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
and the [safetensors PyTorch interface](https://huggingface.co/docs/safetensors/en/api/torch).
