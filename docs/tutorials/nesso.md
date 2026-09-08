# Reproduce the Nesso-1 results

**Original work:** [Nesso-1 paper](https://doi.org/10.64898/2026.08.01.742196) · [code](https://github.com/recursionpharma/nesso) · [weights](https://huggingface.co/recursionpharma/nesso).
{ .original-work }

This workflow downloads the original models, prepares native inputs and compares every Nesso output before timing the candidate. Read the [results and limits](../nesso.md) before choosing a backend.

## Set up an isolated environment

Run these commands from a checkout of this repository. Nesso requires Python
3.10–3.13; Python 3.12 was used here. Its source and weights use Apache-2.0; ESM
retains its original upstream licence. No pre-existing `vendor/` or model artifact
is required.

```sh
python3.12 -m venv .venv-nesso
source .venv-nesso/bin/activate
python -m pip install --upgrade pip
mkdir -p work/nesso
git clone https://github.com/recursionpharma/nesso.git work/nesso/source
git -C work/nesso/source checkout 6c72f66720d9d3447fd73c515cda963e39128b1f
python -m pip install -c requirements/nesso.txt \
  -e '.[torch,hub,packing]' -e work/nesso/source
```

The [constraints](../../requirements/nesso.txt) record the tested macOS environment; they are not a universal platform lock. Keep the
Nesso environment separate from models that require different Transformers or
NumPy versions. The general package does not import Nesso unless its adapter is
requested.

### Download and prepare inputs

The explicit [download helper](../../examples/nesso/download.py) retrieves the
165 MB Nesso checkpoint, approximately 413 MB CCD reference asset and 2.61 GB
ESM-2 checkpoint. ESM lives in the shared Hugging Face cache; `download.json` records its snapshot path. ESM is a separate preprocessing model; its size must not be
hidden inside a claim about the much smaller Nesso weights.

```sh
python examples/nesso/download.py \
  --output work/nesso/upstream --with-preprocessing

NESSO_ESM_SNAPSHOT=$(python -c \
  'import json; print(json.load(open("work/nesso/upstream/download.json"))["esm"]["snapshot"])')

python examples/nesso/prepare.py --case all \
  --nesso-source work/nesso/source \
  --esm-snapshot "$NESSO_ESM_SNAPSHOT" \
  --ccd work/nesso/upstream/ccd.pkl \
  --ccd-sha256 7ed0ccd3903f19627926a5e41a5c0b5309c127a071cb9498dcae96926799ffd8 \
  --output work/nesso/fixtures --seed 42 --threads 4
```

The helper uses the original final-layer ESM extraction on CPU and unchanged
native parsing and featurisation. It verifies the pinned source, checks the CCD
hash **before** loading its publisher-trusted RDKit pickle, and loads ESM from
the supplied local safetensors snapshot without further downloads. The code and
CCD are trusted upstream inputs; this is not a loader for arbitrary pickle files.

| Fixture | Protein residues | Tokens including ligand | Meaning |
| --- | ---: | ---: | --- |
| `tiny20` | 20 | 23 | Synthetic peptide and ethanol; execution smoke |
| `fragment130` | 130 | 143 | Exact upstream test fragment and tyrosine |
| `tutorial384` | 384 | 397 | Unchanged full upstream tutorial and tyrosine |

Each fixture directory contains `batch.safetensors`, `batch.json` and the native
processed input files. The preparer verifies a byte-preserving batch reload and
records hashes, versions and preparation costs in `preparation.json`. None of
these fixtures has an experimental affinity label.

Native preprocessing includes random conformer generation and coordinate
augmentation. Seeds are recorded, but different RDKit versions/platforms can
produce different newly generated features. The benchmark reuses the exact saved
batch bytes for both models. Choose a new fixture output directory when preparing
again; existing directories are refused.

### Check and time the complete model

```sh
python examples/nesso/benchmark.py \
  --checkpoint work/nesso/upstream/v1.0.0 \
  --fixtures work/nesso/fixtures --cases tiny20 fragment130 \
  --device cpu --variants packed --rounds 3 --threads 4 \
  --output work/nesso/reproduced_cpu.json

python examples/nesso/benchmark.py \
  --checkpoint work/nesso/upstream/v1.0.0 \
  --fixtures work/nesso/fixtures --cases tiny20 fragment130 \
  --device mps --variants packed --rounds 3 --threads 4 \
  --output work/nesso/reproduced_mps.json
```

The [benchmark](../../examples/nesso/benchmark.py) loads the original safetensors,
freezes the model and applies the selected runtime option temporarily. It writes
all numerical metrics and per-pair timings. Use `--variants packed` to test the matrix-layout change on either backend. The 397-token `tutorial384` fixture is available for a larger experiment, but its complete Nesso predictions have not been benchmarked here. Use new report paths and avoid competing CPU/GPU workloads.

To apply the measured option to your own **already prepared native batch**:

```python
import torch
from compressme.nesso_runtime import nesso_inference_optimizations

# model: pinned native Nesso1, configured for prediction and on the batch device.
model.eval().requires_grad_(False)
model.use_kernels = False
with torch.no_grad():
    with nesso_inference_optimizations(
        model, immutable_request=True,
        single_chunk=False, cache_esm=False, pack_triangles=True,
    ):
        result = model.predict_step(batch, 0)
```

This runtime context does not replace Nesso's YAML/SMILES application. Native
preparation and writers remain upstream components. Validate your full outputs
before treating a different input, backend, precision or runtime revision as
supported. These checks establish numerical preservation on the stated fixtures,
not affinity accuracy or biological equivalence across a dataset.

## Check Python overhead separately

```sh
python examples/nesso/benchmark_python.py \
  --checkpoint work/nesso/upstream/v1.0.0 --rounds 40 \
  --output work/nesso/reproduced_python_checks.json
```

This loads the original checkpoint but times only the old and new state-signature checks. It runs no prediction. `--variants guards` in the full-model benchmark measures the complete adapter overhead with all numerical optimisations disabled.

## Check ESM preprocessing

```sh
python examples/nesso/benchmark_esm.py \
  --model-dir "$NESSO_ESM_SNAPSHOT" \
  --sequence-yaml examples/nesso/fragment130.yaml examples/nesso/tutorial384.yaml \
  --device cpu --mode paired --rounds 3 --warmup 2 --threads 4 \
  --output work/nesso/reproduced_esm_cpu.json
```

This compares the original ESM extractor with an explicit final-embedding wrapper. The wrapper does not return language-model logits or intermediate layer outputs. The recorded CPU speed difference was negligible. ESM extraction is measured separately from Nesso prediction.

## Pack and restore the checkpoint

```sh
python -m compressme pack \
  work/nesso/upstream/v1.0.0/model.safetensors work/nesso/model.cmprpack \
  --level 9 --group-size 4 --block-size 1048576 \
  --report work/nesso/pack.json

python -m compressme unpack \
  work/nesso/model.cmprpack work/nesso/restored.safetensors \
  --max-output-bytes 165426752 --report work/nesso/unpack.json
```

The codec verifies the reconstructed length and SHA256. To load a restored copy, name it `model.safetensors` and place it beside the original `hparams.json`. Keeping both the plain and packed copies consumes additional disk. Packing changes neither resident weights nor prediction computation.

## Pinned inputs

The [download manifest](../../experiments/nesso/reports/download.json) records the published files and hashes. The [preparation record](../../experiments/nesso/reports/preparation.json) records the source inventory, feature hashes and software versions used in the measurements.

- Nesso source: `6c72f66720d9d3447fd73c515cda963e39128b1f`.
- Nesso Hugging Face revision: `499ed12b0343918ab01b2519226390cf8eca038a`.
- ESM-2 Hugging Face revision: `08e4846e537177426273712802403f7ba8261b6c`.

Load Nesso from the local `v1.0.0/` directory. Its upstream Hub loader uses the revision as part of a filename prefix, so a commit hash alone does not resolve the published weight path.
