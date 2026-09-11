# compressme

compressme transforms trained models without distillation and provides lossless checkpoint utilities. Each transformation has a stated input contract and a numerical acceptance check. Compression, storage reduction and faster execution are measured separately.

Start with [installation](installation.md) and the [command-line guide](cli.md). For a model with a local PyTorch architecture, follow the [general workflow](general-workflow.md). The [technical report](technical-report.md) explains the measured results, rejected proposals and limits.

| Task | Guide |
| --- | --- |
| Inspect or losslessly pack a checkpoint | [Command-line guide](cli.md) |
| Reproduce the Boltz-2 shared bundle | [Boltz-2 tutorial](tutorials/boltz2.md) |
| Reproduce the Mol-JEPA compressed model | [Mol-JEPA tutorial](tutorials/moljepa.md) |
| Use STATE ST or SE artifacts | [STATE guide](state.md) |
| Check Nesso-1 runtime changes | [Nesso-1 results](nesso.md) and [tutorial](tutorials/nesso.md) |
| Validate a complete model on CPU, MPS or CUDA | [Backend validation](gpu-validation.md) |
| Understand what has actually been verified | [Results and evidence](technical-report.md) |
| Compare with a standard deployment baseline | [ONNX Runtime comparison](onnx.md) |

The documentation build downloads no checkpoints and imports no model frameworks. The reproduction tutorials explicitly install their own dependencies and download their pinned inputs when you run them.

These pages are published on [GitHub Pages](https://maclandrol.github.io/compressme/). See [building and reading the documentation](building-docs.md) for local preview, automatic publication and downloadable builds.
