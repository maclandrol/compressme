# Reproduce Mol-JEPA compression on a Mac

**Original work:** Rottach et al., [Mol-JEPA (2026)](https://arxiv.org/abs/2608.22642) · [authors’ code](https://github.com/Boehringer-Ingelheim/mol-jepa) · [author-linked checkpoint](https://huggingface.co/Flogrammer/Mol-JEPA).
{ .original-work }

This recipe starts with a checkout of this private repository and the original published weights. It builds a new portable **SMILES-only** artifact, preserving predictions, CLS, all latent embeddings and requested attentions. It uses no fitting, distillation or lower-precision weights. The architecture-specific recipe composes affine projections, contracts eligible attention score maps and removes modality encoders unreachable when `embeddings_data=None`.

This recipe combines several transformations beyond the general CLI's `--method affine` pass. [General workflow](../general-workflow.md) explains that boundary.

## 1. Install in an isolated environment

Access to the private repository is required. Run the commands from its root:

```sh
git clone git@github.com:maclandrol/compressme.git
cd compressme
python3.12 -m venv .venv-moljepa
source .venv-moljepa/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/reproduce/moljepa-macos-tested.txt
python -m pip install -e '.[molecules,packing]'
mkdir -p work/cache/matplotlib
export MPLCONFIGDIR="$PWD/work/cache/matplotlib"
```

The tested package versions are recorded in [moljepa-macos-tested.txt](../../examples/reproduce/moljepa-macos-tested.txt). They document the successful environment; they are not a cross-platform wheel lock. The model licence is CC BY-NC 4.0. Its code and weights retain their upstream licence regardless of this package's licence.

## 2. Fetch the pinned original source and weights

```sh
python examples/reproduce/fetch.py moljepa --directory work/moljepa-original
```

The explicit fetch downloads 181,651,424 weight bytes plus six small source/configuration files from [Flogrammer/Mol-JEPA](https://huggingface.co/Flogrammer/Mol-JEPA/tree/4c912b450175f31b5ba913a5dc921c03b27b985a). It verifies each file against committed hashes. Existing correct files are reused; mismatched existing files are refused. It adds the small local package initializer, attribution/licence and a hash manifest needed by the loader. Fetching does not import that Python. Building and loading subsequently execute this pinned, trusted local architecture.

- HF revision: `4c912b450175f31b5ba913a5dc921c03b27b985a`.
- Original `model.safetensors` SHA256: `a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14`.
- `modeling_moljepa.py` SHA256: `89eed5f7ebd458b2d4e6a1e4ad90aed275b7b35bfaa88f4a85a59833e57a0d4d`.

All source hashes are in [moljepa-source.json](../../examples/reproduce/moljepa-source.json). There is no dependency on a pre-existing `vendor/` directory, artifact, Hugging Face code cache or `trust_remote_code=True` loader. Downloaded files contain executable Python, so only use this reviewed source revision and trusted artifact directories.

## 3. Build and check the portable artifact

```sh
python examples/reproduce/moljepa.py build \
  --original work/moljepa-original \
  --artifact work/moljepa-smiles --packing
```

The builder loads the original safetensors strictly, freezes the model, applies the recipe on CPU and compares all returned outputs on 64 committed SMILES, with `return_attn=False` and `True`. It then reconstructs the architecture, reloads the saved artifact without the original checkpoint and repeats the complete-output gate. Only a passing artifact is published to the new directory; existing directories are refused. `reproduction.json` records the detailed checks.

The expected parameter count is **45,406,721 → 19,763,160**. Algebraic equivalence is in real arithmetic; floating-point reassociation can change the last bits. The gate requires both maximum absolute error and relative L2 error at most `1e-5` on the supplied examples. This verifies this workload, not all possible molecules or downstream biological accuracy. `--packing` additionally applies a reversible file codec; it does not change resident parameters or arithmetic. The exact packed size can vary with serializer/codec versions.

The exported input contract explicitly rejects non-`None` `embeddings_data`. It retains **all output embeddings**; it does not turn the model into a prediction-only wrapper. Fine-tuning the reparameterised model is a different optimisation trajectory and is outside this inference recipe.

## 4. Verify from a fresh process on CPU and Metal

```sh
python examples/reproduce/moljepa.py verify \
  --original work/moljepa-original --artifact work/moljepa-smiles \
  --device cpu --report work/moljepa-cpu.json

python examples/reproduce/moljepa.py verify \
  --original work/moljepa-original --artifact work/moljepa-smiles \
  --device mps --accelerate --report work/moljepa-mps.json
```

The verifier checks original self-repeat and all candidate outputs on the same backend and dtype. `--accelerate` enables the separate validated SMILES runtime (including eligible Metal graph operations on MPS). It is an explicit option in this verifier so the portable graph can also be checked independently. A failing numerical gate writes a failure report and exits without benchmarking. MPS availability is checked rather than assumed; CUDA is an option for local testing but does not acquire validation evidence from an Apple test.

To reproduce the **recorded benchmark protocol**, run:

```sh
python examples/reproduce/moljepa.py benchmark \
  --original work/moljepa-original --artifact work/moljepa-smiles \
  --device mps --report work/moljepa-mps-benchmark.json
```

This uses [moljepa-benchmark.json](../../examples/reproduce/moljepa-benchmark.json), extracted from the committed historical report: its exact batches of 1, 4 and 32 SMILES, four variants (`original`, portable `compressed`, `fast` with Metal disabled, and `production` with Metal enabled), **10 warmups**, **20 synchronised interleaved rounds**, and the same per-batch shuffle seed `4561 + batch_size`. Timed calls use the original default **`return_attn=False`**. Before timing, all 64 molecules are checked in batches of four with both attention settings for every variant, including original self-repeat. The updated gate retains the `1e-5` maximum-absolute bound and additionally requires relative L2 error at most `1e-5`.

Each timing includes the complete fresh SMILES call and featurization; predictions are not cached. The report stores the workloads, per-call milliseconds, medians and percentiles. `--rounds` can shorten a development smoke, but a value other than 20 is explicitly labelled as a different round count. An alternate `verify --rounds N --batch-size B` mode remains available; it uses the first B verification molecules, attentions enabled and three warmups, and is labelled a **custom protocol**, not reproduction of the reported timing experiment.

Run without competing CPU/GPU jobs and record the hardware and environment. Identical workloads do not guarantee identical wall-clock numbers or a universal speedup. The [historical report](../technical-report.md) supplies the original results and limitations.

## 5. Use the artifact without the original weights

```python
from compressme import load_moljepa

model = load_moljepa("work/moljepa-smiles", device="mps")
output = model(["CCO", "c1ccccc1"], return_attn=True)
print({name: getattr(value, "shape", type(value)) for name, value in output.items()})
```

The artifact contains its tensor state, rewrite recipe and hashed architecture files. Keep the whole directory together and keep its optional Python dependencies installed. The original 182 MB checkpoint is needed only when reproducing the reference comparison, not for this prediction call.

The [tutorial smoke record](../../examples/reproduce/reproduction-smoke.json) records fresh CPU builds/reloads from the pinned original weights, including all 48 Boltz outputs and all requested Mol-JEPA output modes. The Mol-JEPA benchmark command also passed a one-round execution check. That tutorial check covered CPU builds and reloads; it did not repeat the full timing study or the MPS tutorial runs.
