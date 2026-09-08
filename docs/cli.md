# Command-line use

`compressme` and `python -m compressme` expose the same commands. The CLI separates checkpoint inspection, lossless file packing, and model rewrites that execute explicitly supplied local Python.

From this source checkout, install the optional dependencies for the commands you need. This release has not been published to PyPI:

```sh
python -m pip install .
python -m pip install '.[hub]'      # Hugging Face metadata/downloads
python -m pip install '.[packing]'  # Streaming file codec
python -m pip install '.[torch]'    # Model rewrites and safetensors
```

Local metadata inspection and `targets` require none of these extras. Model-specific dependencies remain the caller's responsibility. No command installs a source repository automatically.

## Inspect weights and optional source

```sh
compressme inspect weights.safetensors --report inventory.json
compressme inspect weights.safetensors --details --sha256
compressme inspect weights.ckpt --repo ./model-source
compressme inspect owner/model --revision COMMIT --filename model.safetensors
compressme inspect hf://owner/model --repo https://github.com/owner/model
compressme targets
compressme targets boltz2
```

A local safetensors inspection reads a bounded JSON header and checks tensor offsets, shapes, known dtype sizes and complete file coverage. `--details` includes every tensor entry; `--sha256` additionally reads the whole file to compute its hash. It does not load tensor values. For a PyTorch ZIP checkpoint, inspection lists bounded archive metadata and disassembles bounded pickle opcodes without executing them. It does not infer tensor shapes or a parameter count from those opcodes. Legacy non-ZIP pickle checkpoints remain opaque.

Hugging Face inspection uses the existing metadata client and records the resolved revision. An ambiguous repository needs an explicit `--filename`; a safetensors index can select its shards. Inspection does not execute remote model code.

`--repo` means **provenance only**. A URL is recorded without fetching or cloning it. A local directory receives a bounded Python AST inventory: file hashes, recognised constructor call sites and some potential API hazards. This is neither an execution sandbox nor proof that a model rewrite is valid. `--repo` does not become an import root. The trusted factory described below must import installed model dependencies, import sibling files, or use a caller-configured `PYTHONPATH`.

## Pack a file without changing any bytes

```sh
compressme pack weights.safetensors weights.cmprpack --report packed.json
compressme unpack weights.cmprpack restored.safetensors \
  --max-output-bytes 3000000000 --max-window-bytes 67108864 \
  --report restored.json
```

The streaming codec uses bounded blocks and verifies whole-file reconstruction hashes. Unpacking requires a finite output-size cap and rejects oversized, truncated, extra or corrupt data. Ordinary immutable source files are the supported input. Existing destinations are refused unless `--overwrite` is explicit; reports are never overwritten. Output publication uses a verified temporary file. A failure after the publication commit point is distinguished from a failure before publication.

Packing reduces disk and transfer bytes. It does not reduce resident model memory, tensor precision or inference arithmetic. Files that do not compress well can grow.

## Rewrite a model with an explicit validation contract

Weights alone cannot establish a model's forward graph or supported inputs. `compress` therefore requires a trusted local architecture factory and a trusted local validation factory. Specifying those files authorises executing that Python in the current process. Their exact source hashes are recorded; dependencies they import are not sandboxed or automatically audited.

This complete example creates a small model whose two consecutive affine maps can be composed. Save it as `factory.py`:

```python
import torch
from torch import nn
from compressme import Example


def make_model():
    return nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 3))


def make_examples():
    return [
        Example((torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10,)),
        Example((torch.tensor([[-2., 0., 1., 4.]]),)),
        Example((torch.zeros(0, 4),)),
    ]


if __name__ == "__main__":
    from safetensors.torch import save_file
    torch.manual_seed(7)
    save_file(make_model().state_dict(), "weights.safetensors")
```

Run it, then compile and check every returned output:

```sh
python factory.py
compressme compress weights.safetensors compact \
  --factory factory.py:make_model --validation factory.py:make_examples \
  --method affine --device cpu --report compression.json
```

Use a new destination and `--device mps` to run the rewrite and gates on Apple Metal, or `--device cuda` for a supported NVIDIA environment. The CLI fails if that device is unavailable. It moves tensors without changing their dtype. It records device, Torch version, dtype, matmul precision/TF32 settings and deterministic settings. These are checks on the selected device, not cross-device equivalence or a speed benchmark.

Local semantic input currently accepts one `.safetensors` file. Hugging Face input can select safetensors files or a safetensors index. `.ckpt`, `.pt`, `.pth` and other pickle-based semantic loads are refused; convert them separately using an explicitly trusted upstream procedure. The loaded tensor keys, shapes, dtypes and aliases must match the factory's architecture.

The CLI freezes parameters, switches to evaluation mode, runs original self-repeat checks with controlled seeds, applies one method, and compares the complete output structure and every tensor on the supplied examples. Validation factories return a nonempty iterable of `compressme.Example` objects containing tensors, plain containers and scalar inputs. No output selector is used. The default gate requires **both** relative L2 error at most `1e-5` and maximum absolute error at most `1e-5`; it is not elementwise `allclose`. Set explicit tolerances with `--relative-tolerance` and `--absolute-tolerance` when justified by your contract. A failed gate exits with status 2 and exports no model. Acceptance is evidence for those examples, not a universal accuracy guarantee.

Available methods:

- `affine`: the generic conservative affine compiler. Unsupported graphs or non-profitable proposals can leave the model unchanged.
- `constant_embeddings`: removes identical rows from eligible frozen embedding tables while preserving the full token-index domain.
- `share`: shares eligible equal frozen tensor storage while retaining distinct Parameters and the original computations. It requires **bitwise** self-repeat and candidate output agreement regardless of numerical tolerance flags. Logical parameter counts may stay unchanged while resident storage falls. Sharing must run on the final selected device; later device conversion can allocate separate storage again. It is not a training or inference-speed optimisation.

An unchanged result is explicitly labelled `accepted_no_reduction`. Other accepted exports are labelled `accepted_on_validation_examples`. JSON stdout gives a small summary; `--report` stores the complete provenance and per-output metrics. Existing semantic output directories and reports are refused. A full report must be outside the exported directory. The export includes a transformation recipe and tensor state; it does not bundle arbitrary factory code or dependencies.

Load the result with the same trusted original architecture factory:

```python
from compressme import load
from factory import make_model

model = load(make_model, "compact").model
```

This deliberately preserves an explicit architecture boundary: a repository URL is not treated as permission to import remote code, infer an API, or assume a biological model's validation contract.
