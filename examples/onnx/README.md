# Mol-JEPA: a bounded ONNX Runtime comparison

This optional experiment compares the original and structurally compressed
Mol-JEPA tensor cores in PyTorch and ONNX Runtime on CPU, using float32 and four
threads. It retains all 12 predictions, CLS, all 13 latent embeddings and both
attention arrays returned by `return_attn=True`. It does not export RDKit or
SMILES strings, support optional modality inputs, or establish dynamic graph
sizes. The [results and failed attempts](../../experiments/onnx/README.md) give
the exact scope.

Start with steps 1–3 of the [Mol-JEPA tutorial](../../docs/tutorials/moljepa.md).
Those steps prepare the pinned original checkpoint at `work/moljepa-original`,
the checked artifact at `work/moljepa-smiles`, and an isolated Python environment.
Do not substitute an unverified source or checkpoint. The upstream model and
its derived files retain the CC BY-NC 4.0 licence.

Add the optional exporter wheels to a separate directory; this does not change
core package dependencies or the tutorial environment's installed packages:

```sh
python -m pip install --no-deps --target work/onnx-deps \
  -r examples/onnx/requirements.txt
```

The pins supplement the tutorial's tested Mol-JEPA dependencies, including
PyTorch 2.14.0 and NumPy 2.5.3. They were exercised on macOS arm64 with Python
3.12.14, not verified as a wheel lock for every operating system.

Run from the repository root with fresh output paths:

```sh
PYTHONPATH="$PWD/work/onnx-deps:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/onnx/moljepa.py \
  --checkpoint work/moljepa-original/model.safetensors \
  --artifact work/moljepa-smiles \
  --verification-smiles benchmarks/verification_smiles.json \
  --work work/onnx-moljepa \
  --output work/onnx-moljepa-report.json \
  --threads 4 --warmup 3 --rounds 10 \
  --exporters dynamo --explicit-mask-promotion --ort-spinning off
```

The explicit promotion reproduces the native `torch.stack` conversion from a
mixture of int64 graph counts and float32 zeros to float32. The tested Dynamo
exporter omitted that promotion without the flag, producing an ONNX graph that
ORT refused to load. No weights or model operators are replaced to work around
an unsupported operation. The wrapper is compared against the native Python API
before export, and ORT outputs are checked against both its source core and the
original model. Changed features, edge order and node indices are tested at the
same tensor shapes.

The final protocol disables idle ORT thread spinning in both sessions. The four
intra-op threads then belong to the active arm rather than letting a waiting
session consume another arm's CPU budget. All arms run serially, with three
warmups and ten rotated/reversed rounds. The two timed scopes are:

- Tensor core with prepared graph inputs, including NumPy/ORT output views.
- The same tensor core plus fresh original SMILES preprocessing in every arm.

The extra final-runtime PyTorch arm also uses the common original preprocessing;
it does not measure that runtime's faster sparse featurizer. Loading, export,
session creation, validation and warmup are excluded. There is no input/output
cache. CPU inputs and outputs need no asynchronous GPU synchronization.

Summarize the saved report and verify the exported file hashes without running
inference again:

```sh
PYTHONPATH="$PWD/work/onnx-deps${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/onnx/summarize.py \
  --report work/onnx-moljepa-report.json --models work/onnx-moljepa \
  --output work/onnx-moljepa-summary.json
```

To reproduce the unadjusted exporter attempts, omit `--explicit-mask-promotion`
and request `--exporters dynamo legacy`, with fresh work/report paths. To examine
the preliminary scheduling choice, use `--ort-spinning on`; those timings are
retained as preliminary evidence, not the final comparison. The benchmark never
downloads model assets or chooses CoreML/CUDA automatically. Both scripts offer
`--help` without importing the model dependencies.
