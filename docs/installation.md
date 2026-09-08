# Installation

Python 3.10 or newer is required. The base package has no third-party runtime dependencies. Importing `compressme` and listing its target metadata do not load Torch, chemistry software or model weights.

From a source checkout, choose the features you need:

```sh
python -m pip install .
python -m compressme targets
```

| Extra | Installs | Use |
| --- | --- | --- |
| `torch` | PyTorch, safetensors and NumPy | Generic model transformations, validation, tensor sharing and model serialization |
| `packing` | NumPy and Zstandard | Lossless byte/file packing and unpacking; works without Torch |
| `hub` | Hugging Face Hub client | Pinned checkpoint metadata inspection; works without Torch |
| `test` | pytest | Test runner; select other extras for the features being tested |
| `molecules` | Torch/safetensors, Transformers, PyG, RDKit, Molfeat and NumPy | The optional Mol-JEPA artifact loader and molecular execution |
| `state` | Torch/safetensors and STATE loader dependencies, including Transformers 4.52.3 | The optional audited STATE ST/SE artifact loaders |
| `boltz` | Torch/safetensors/NumPy and Boltz 2.2.1 | The optional Boltz-2 artifact loader; the installed source must also match the recorded revision |

For model compression using your own local PyTorch architecture:

```sh
python -m pip install '.[torch]'
```

For lightweight file compression, or metadata inspection:

```sh
python -m pip install '.[packing]'
python -m pip install '.[hub]'
```

Combine extras when needed, for example `'.[torch,hub,packing]'`. Hub model compression needs both `torch` and `hub`; Hub inspection alone needs only `hub`. The package never infers executable architecture code from tensor names or automatically installs a model's dependencies.

NumPy is included in the model runtime because the safetensors Torch serializer uses it. This requirement is separate from the optional Zstandard file codec.

All existing top-level Python APIs remain available. They load their implementation when requested. Requesting a Torch-based API without that extra raises an installation hint. Avoid `from compressme import *` in lightweight applications: it requests every model API and therefore requires the model runtime.

Use separate environments for model stacks with conflicting requirements. In particular, the audited STATE loaders use Transformers 4.52.3, while the Mol-JEPA research environment was validated separately. Model extras provide dependencies; they do not include checkpoints, data assets or exported artifact directories. Mol-JEPA and STATE artifacts contain separately supplied, trusted architecture files. The Boltz loader checks the installed source against revision `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`; a matching version string alone is insufficient.

For CPU or Apple MPS execution, install a PyTorch build supporting that backend. CUDA requires a suitable PyTorch build, driver and device. Installing an extra does not establish that a particular model, shape or backend has passed numerical validation.

The wheel contains the library and small Boltz provenance/configuration files. The source distribution additionally contains public tests, operational documentation and the small standalone backend-validation example used by those tests. Neither contains model weights, exported artifacts, vendored architecture archives or local research evidence. Those files remain separate from package installation.

The upstream-dependent sparse Mol-JEPA fixture stays in the repository and is excluded from the source distribution together with its vendored architecture. Generic tests and the standalone backend runner are included.

For contributor checks, install `'.[torch,packing,test]'` and run `python -m pytest`. Optional model tests may skip when their dependencies or architecture fixtures are absent. The sparse Mol-JEPA tests explicitly skip in a clean clone without its upstream architecture; set `COMPRESSME_MOLJEPA_SOURCE` to the source directory created by the [Mol-JEPA tutorial](tutorials/moljepa.md) to enable them. Documentation checks use the separate [docs build](building-docs.md).
