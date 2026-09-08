# Reproduce complete Boltz-2 storage sharing on a Mac

**Original work:** Passaro et al., [Boltz-2 (2025)](https://doi.org/10.1101/2025.06.14.659707) · [original code](https://github.com/jwohlwend/boltz) · [upstream checkpoints](https://huggingface.co/boltz-community/boltz-2).
{ .original-work }

This tutorial builds the complete original Boltz-2 **confidence/structure and affinity pair** from pinned public checkpoints, then shares equal frozen parameter storage. All original computations and native outputs remain. This is a joint resident-storage and artifact-storage reduction; it does not halve either model's arithmetic or establish faster inference.

The generic engine is `share_frozen_parameters`. The recipe supplies the pinned architecture, strict original load order, native preprocessing and complete-output verification that a weights-only CLI cannot infer. [Storage sharing](../sharing.md) gives the general contract.

## 1. Check out and install

```sh
git clone git@github.com:maclandrol/compressme.git
cd compressme
python3.12 -m venv .venv-boltz
source .venv-boltz/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/reproduce/boltz-macos-tested.txt
python -m pip install 'boltz @ git+https://github.com/jwohlwend/boltz.git@b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc'
python -m pip install -e '.[torch,packing]'
mkdir -p work/cache/numba work/cache/matplotlib
export NUMBA_CACHE_DIR="$PWD/work/cache/numba"
export MPLCONFIGDIR="$PWD/work/cache/matplotlib"
```

The source installation executes the upstream package's installation code. It is an explicit trust decision, separate from checkpoint parsing. The loader checks the installed Boltz version **2.2.1** and the hashes of its complete Python source file set before importing the model. Installing another revision is refused even if it has a similar API.

[boltz-macos-tested.txt](../../examples/reproduce/boltz-macos-tested.txt) records the actual environment rather than promising future platform-independent resolution. The original source is [jwohlwend/boltz at the audited commit](https://github.com/jwohlwend/boltz/tree/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc). CPU and Apple Metal use original PyTorch operators with CUDA-specific kernels disabled. The experiment used an Apple M5 with 16 GB unified memory for a tiny complex; larger complexes and cold-load memory need separate assessment.

Allow roughly 13–16 GB of free disk if retaining both checkpoints, converted state, final artifact, molecular archive and a packed/reconstructed copy. Cold construction temporarily allocates both complete original architectures, exceeding the final shared resident state. No weights or chemistry pickles are committed in this repository.

## 2. Download the original checkpoints and convert without unpickling

```sh
python examples/reproduce/fetch.py boltz2 --directory work/boltz-original
python examples/reproduce/export_shared_tensors.py \
  --checkpoints work/boltz-original --output work/boltz-safe-state
```

The download uses [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2/tree/6fdef46d763fee7fbb83ca5501ccceff43b85607), revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`, and verifies these exact publisher files before conversion:

| Checkpoint | Bytes | SHA256 |
|---|---:|---|
| `boltz2_conf.ckpt` | 2,286,561,469 | `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` |
| `boltz2_aff.ckpt` | 2,062,139,170 | `dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e` |

The exporter first verifies the complete checkpoint hash, then interprets the observed pickle opcodes as **inert data records**. It never calls `torch.load`, `pickle.load`, checkpoint globals, reducers or constructors. It copies full contiguous FP32 ZIP storages into safetensors, compares every reused byte blob, and preserves original state names, shapes and pure-JSON hyperparameters in a manifest. Unsupported checkpoint versions/layouts are refused. The build helper also checks the resulting pinned mapping manifest SHA256 `3c855681aa05d0d18ebeeb2335fe5850a66a03ba881eada1afcedfb60e13258c`. This is a narrow reader for these two verified files, not a general safe-pickle implementation.

At this stage equal tensor blobs are deduplicated only in the converted **file**. Loading normally still copies tensors into separate model Parameters. The next build applies the general frozen-storage sharing pass on actual instantiated models.

## 3. Prepare the native chemistry assets and fixture

```sh
python examples/reproduce/fetch.py boltz2-molecules --directory work/boltz-chemistry
```

This explicitly downloads and checks the original 1,855,662,080-byte `mols.tar`, SHA256 `39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7`, from the same pinned HF revision. It extracts only the canonical molecule files by individually checked names and writes their hashes. Existing verified downloads are reused. No general tar extraction or downloaded Python execution occurs.

**Chemistry trust boundary:** the original Boltz parser subsequently unpickles these verified upstream RDKit molecule objects. Neural checkpoint pickle is never executed. A manifest generated by this recipe provides integrity, not a sandbox against replacing both the manifest and assets. Keep this local directory trusted.

The included [tiny_complex.yaml](../../examples/reproduce/tiny_complex.yaml) has a 20-residue protein, explicit empty MSA and an ethanol ligand with an affinity request. The native parser generates ligand features from SMILES. No MSA server is contacted. The canonical subset is sufficient for this fixture; it is not a complete CCD database for arbitrary named noncanonical components. For another input, supply the complete appropriate upstream molecule assets and validate separately.

## 4. Build after checking complete native outputs

```sh
python examples/reproduce/boltz2.py build \
  --safe-state work/boltz-safe-state --artifact work/boltz2-shared \
  --molecules work/boltz-chemistry/mols --work work/boltz-build-fixture \
  --device cpu --report work/boltz-build.json
```

This creates both pinned original models and follows the original strict tensor load order, including original registered aliases. It freshly parses the YAML, creates a seeded native confidence batch and uses the original prediction writer to create the structure needed by the affinity batch. The same prepared batches are used for original self-repeat and candidate checks. The original native schedules remain:

- Confidence/structure: 3 recycles, 200 diffusion steps, 1 sample.
- Affinity: 5 recycles, 200 diffusion steps, 3 samples.

Every returned tensor and non-tensor output is compared. The expected fixture has **48 tensor outputs across both models**. Sharing is accepted only with finite, byte-identical complete outputs and a byte-identical original self-repeat on that backend. Only then is the portable artifact written. Existing artifacts, reports and work directories are refused; choose new paths for a rerun.

The historical registered storage result was **4,087,121,944 → 2,061,868,568 bytes (49.55% less)**. Logical parameter objects/counts and arithmetic are unchanged. These are inference-state denominators, not a comparison against optimizer-state or EMA-heavy training checkpoint sizes. The output gate covers this fixture and backend; it does not prove biological quality on a dataset or support arbitrary modifications to the upstream API.

## 5. Verify a fresh artifact reload, including on Metal

```sh
python examples/reproduce/boltz2.py verify \
  --safe-state work/boltz-safe-state --artifact work/boltz2-shared \
  --molecules work/boltz-chemistry/mols --work work/boltz-reload-cpu \
  --device cpu --report work/boltz-reload-cpu.json

python examples/reproduce/boltz2.py verify \
  --safe-state work/boltz-safe-state --artifact work/boltz2-shared \
  --molecules work/boltz-chemistry/mols --work work/boltz-reload-mps \
  --device mps --report work/boltz-reload-mps.json
```

Each command independently reconstructs and runs the original reference, then releases it before loading the compressed artifact and comparing all outputs on the same backend. It regenerates preprocessing rather than loading an uncommitted historical batch. MPS is checked explicitly; CUDA may be tested with `--device cuda` but the Apple evidence does not imply NVIDIA verification.

To predict from a YAML without needing the original neural checkpoints or safe-state directory:

```sh
python examples/boltz2.py examples/reproduce/tiny_complex.yaml \
  --artifact work/boltz2-shared --molecules work/boltz-chemistry/mols \
  --device mps --output work/boltz-prediction
```

The artifact loader still needs the pinned installed Boltz source and chemistry assets. It shares weights after transfer to the requested final device. The returned `bundle['confidence']` and `bundle['affinity']` preserve their original native batch APIs. Do not mutate their frozen shared parameters; a later `.to(...)` can split storage again.

## 6. Optionally pack the tensor file for transport

```sh
compressme pack work/boltz2-shared/model.safetensors work/boltz2.cmprpack \
  --report work/boltz-pack.json
compressme unpack work/boltz2.cmprpack work/boltz2-restored.safetensors \
  --max-output-bytes 3000000000 --report work/boltz-unpack.json
cmp work/boltz2-shared/model.safetensors work/boltz2-restored.safetensors
```

Keep the remaining architecture/manifest/report files with the artifact. The packed file is not loaded directly by the standard Boltz loader; unpack it back to the original tensor filename when restoring an artifact. Historical streaming packing saved a further **14.696% disk/transport bytes** with complete restored-byte verification; serializer/codec versions can affect the result. This gives no additional resident memory or inference-speed reduction.

Storage sharing is not an active-compute speed optimisation, so this tutorial makes no speedup claim. The optional request-invariant runtime in [the Boltz notes](../boltz2.md) retained outputs but showed neutral/inconsistent small-fixture timings on Metal; it remains opt-in. Measure a separate timed workload only after output validation, with no concurrent CPU/GPU jobs.

The [tutorial smoke record](../../examples/reproduce/reproduction-smoke.json) records fresh CPU builds/reloads from the pinned original weights, including all48 Boltz outputs and all requested Mol-JEPA output modes. The new benchmark command also passed a one-round execution smoke; the full historical timing study and these new tutorial entry points on MPS were not rerun in that smoke.
