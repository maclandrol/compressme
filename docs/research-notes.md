# Research overview, 7 September 2026

compressme changes how a trained function is represented by removing algebraic
redundancy, without training a student, collecting teacher targets or reducing
weight precision. This overview retains the research details and measurements
from this stage. Use the [README](../README.md) for installation and the package
interface, and the [technical report](technical-report.md) for the consolidated
results and their limits.

The first validated checkpoints include Mol-JEPA, STATE ST, STATE SE on CPU, and the complete Boltz-2
confidence/affinity pair on CPU and Apple MPS. The rewrite primitives also work
independently of them.

## Mol-JEPA results

| Mol-JEPA artifact | Parameters | Reduction | Input contract |
|---|---:|---:|---|
| Original | 45,406,721 | n/a | All original modalities |
| `artifacts/moljepa-full` | 35,225,561 | **22.42%** | All original forward inputs and outputs |
| `artifacts/moljepa-smiles` | 19,763,160 | **56.47%** | SMILES; `embeddings_data=None` |

Both compressed Mol-JEPA artifacts
retain float32 weights and all prediction outputs: 12 predicted modality
embeddings, the CLS vector, 13 latent embeddings and optional attention maps.
The SMILES artifact rejects additional modality inputs explicitly. Removing
already inactive encoders saves storage and resident weights, not their already
absent compute. The source and Python dependencies remain necessary.

On an Apple M5 MacBook Air with 16 GB unified memory, the SMILES runtime was
about 2× faster than the original in the recorded warmed, interleaved MPS run.
Each call included fresh parsing, graph creation, device transfers and every
prediction head:

| Molecules per call | Original | Earlier compressed runtime | Accelerated runtime | Speedup vs original |
|---|---:|---:|---:|---:|
| 1 | 14.47 ms | 12.59 ms | **6.56 ms** | **2.21×** |
| 4 | 22.07 ms | 18.57 ms | **11.27 ms** | **1.96×** |
| 32 | 64.59 ms | 48.74 ms | **32.42 ms** | **1.99×** |

These are same-run comparisons over 20 shuffled rounds after 10 warmups.
Background load affected absolute timings; raw samples and p10/p90 are in
[the benchmark](../benchmarks/moljepa_runtime_macos.json). First-call shader
compilation is outside these warmed timings. No molecule or output cache is used.

The speedup comes from sparse bond-only preprocessing, graph kernels that avoid
materializing large per-edge messages, known batch sizes that avoid a GPU sync,
and batched execution of all 13 readouts. Packed parameters have one owner;
there is no persistent duplicate weight cache. Runtime parameter count remains
19,763,160. CPU uses the sparse preprocessing and readout path, which was
1.35–1.59× faster than the original in the same final run.

All 64 verification SMILES passed complete output and attention checks for the
accelerated runtime: maximum absolute difference **2.15e-6 on MPS** and **4.05e-6 on CPU**
against the original trained checkpoint. Sparse graph features were independently
bitwise equal on 97 valid SMILES; invalid-input exceptions also matched.

The independent CPU check used 64 additional SMILES, including salts, isolated
ions and stereochemistry, plus mixed inputs spanning all 11 optional modalities.
Predictions, CLS vectors, latent embeddings and returned attention maps passed
`atol=rtol=1e-5`; the maximum prediction difference was **3.19e-6**. These inputs
were not used to fit anything. This measures agreement with the checkpoint, not
biological accuracy, and does not establish whether examples occurred in pretraining.

The Mol-JEPA rewrite identities hold in real arithmetic. Composing matrices
changes floating-point operation order, so these results establish neither
bit-identical outputs nor a machine-verified floating-point error certificate.
For a byte-identity requirement, retain the original arithmetic; removing only
unreachable modality encoders still saves 34.05% on the SMILES-only route.

## Run the compressed model

The Hugging Face entry point accepts other model architectures, and a dated
biology-model registry records their audit status.
`inspect_huggingface(repo_id)` pins and inspects weights without executing model
code. `compress_huggingface(repo_id, model_factory, validation=...)` applies
an exact pass to a caller-owned architecture and rejects proposals that fail the
complete-output checks. See the [Hub workflow](../docs/huggingface.md) and
[general design](../docs/general-workflow.md). It does not infer a computation graph
or promise a compression ratio from tensor names alone.

With the prepared project environment and artifacts:

```bash
cd ~/Code/compressme
.venv/bin/python examples/moljepa.py predict --device mps CCO 'c1ccccc1'
```

The Python loader preserves the callable API:

```python
from compressme import load_moljepa

model = load_moljepa("artifacts/moljepa-smiles", device="mps")
output = model(["CCO", "c1ccccc1"], return_attn=True)
print(output.predictions.shape)  # [2, 12, 512]
print(output.cls.shape)          # [2, 512]
print(output.embeddings.shape)   # [2, 13, 512]
```

The loader enables the validated runtime automatically for SMILES artifacts.
Use `accelerate=False` to retain the portable compressed execution. Runtime
layouts are rebuilt from the same stored weights after loading; export the
portable model before acceleration. Metal kernels require a PyTorch build with
`torch.mps.compile_shader` (tested with 2.14); older builds use PyTorch execution.
Use `device="cpu"` on machines without Apple Metal. These exports contain the
compressed tensors, rewrite manifest and a local copy of the audited architecture.
They do not need the original 182 MB checkpoint, an NVIDIA GPU or remote code
downloads. Loading executes the included Python architecture, so use trusted
artifacts. The initial loader constructs the original architecture before
replacing its operators; load-time peak memory is higher than final weight size.

For a new environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[molecules,test,packing]'
.venv/bin/python -m pytest -q
```

Exact development dependency versions are recorded in
`benchmarks/environment.txt`. The generic affine and low-rank package does not
require the optional molecular dependencies.

## Store the parameter combinations used by the computation

Each reduction follows from a specific restriction on the input domain or the
way a later operation uses an intermediate value.

**Affine reachability.** If a narrow input is expanded and immediately projected
again, the downstream operation can only observe that narrow affine image:

\[
h=Ax+a,\qquad y=Wh+b
\quad\Longrightarrow\quad y=(WA)x+(Wa+b).
\]

The preceding computation proves the restricted domain, so no estimated low
rank is needed. An expansion from 82 to 512 followed by a 4096-output linear
map can be stored as a 4096-by-82 map. Residual branches that need the expansion
are retained. This removes 5,504,000 parameters from Mol-JEPA's first graph layer.

**Attention as a bilinear form.** A query and key matter through their dot product:

\[
(Qx_i)^\top(Kx_j)=x_j^\top(K^\top Q)x_i.
\]

If the head projection is wider than its inputs, storing the interaction can be
smaller than storing both factors. The implementation includes query biases and
edge conditioning, removes key-bias terms that are constant within a softmax row,
and preserves each head, scaling, dropout, isolated-node behaviour and returned
attention. It removes another 4,677,160 parameters from this checkpoint.

These classical identities turn the compiler problem into locating observable
parameter combinations in a trained program. Exporting those combinations
produces checkpoint savings here without an approximation hypothesis; the
identities themselves are not new linear algebra theorems.
Query/key composition is also related to prior compression work such as
[KQ-SVD](https://arxiv.org/abs/2512.05916); this implementation performs no SVD
truncation on that path. See [the derivations and prior art](../docs/theory.md).

**Contract around the normalization statistic.** For
`Linear(d,n) → LayerNorm(n) → Linear(n,m)`, center the first affine map to form
`T`, compute full reduced QR `T=UR`, and retain:

\[
y=\frac{D[x;1]}{\sqrt{\|R[x;1]\|^2/n+\epsilon}}+c.
\]

The wide intermediate is represented by a contracted numerator and a small
denominator statistic. There is no rank truncation. An 82→512→LayerNorm→512
test block drops from **306,176 to 49,897 parameters (83.7%)**, with float32
relative output error about 4.8e-7. This result comes from a constructed
general-operator test. Mol-JEPA gains no additional reduction from it because
GELU interrupts the relevant encoder paths. FX discovers eligible sandwiches
automatically. See the
[proof and limitations](../docs/normalization-statistic.md).

**Store identical frozen rows once.** A frozen embedding table whose rows have
identical bits can retain one row and expose an expanded weight view. Ordinary
token lookup, vocabulary size and output values remain available. The general
`deduplicate_embeddings` pass checks equality, trainability, hooks and aliases;
it performs no model fitting. Custom consumers of weight strides need separate
validation. This pass reduces the published STATE ST-HVG-Replogle K562 model
from **49,396,728 to 38,901,056 parameters (21.25%)**. All tested expression/count
outputs were **bitwise identical on CPU and MPS**, using constructed numerical
batches with the actual checkpoint. Its zero table was already skipped during
expression prediction, so this saves resident weights, not active computation.

The portable artifact is `artifacts/state-st-hvg-k562`; use
`load_state_st(directory, device="mps")` in `.venv-state`. That separate environment
preserves the published model's Transformers 4.52.3 dependency. The original
checkpoint is not required to reload. See [STATE usage and scope](../docs/state.md).

**Evaluate the whole finite token domain.** A fixed vocabulary can be passed
through its original row-local encoder once, including LayerNorm and nonlinear
activations. Those smaller outputs form an inference lookup produced by partial
evaluation. `compile_finite_lookup` checks every token and
additional tensor layouts; it rejects a proposal that fails the local numerical
gate or does not save bytes. `compile_finite_blocks` discovers or selects such
blocks inside a model and requires complete-output checks with rollback. The
same pass is available through `compress_huggingface(..., method="finite_lookup")`.
`FiniteTokenFanout` also declares several consumers of one token encoder, such
as query/key/value projections and the original residual embedding. The compiler
packs all declared outputs, preserves their separate dtypes, and accounts for
shared backing storage. Portable recipes retain the compiled representation.
See [the finite-domain API and limits](../docs/finite-domains.md).

STATE SE uses two such tables for its raw and normalized gene branches, plus
shape-matched constant tokens. It retains the original 5,120-feature encoder for
custom raw-vector inputs. Parameters fall from **212,038,824 to 151,243,944
(28.67%)**. Extended CPU tests include 2,048-gene inputs and every numerical head;
maximum observed error is **6.68e-6**. Short CPU timings improved **1.39×** for
32 tokens and **1.11×** for 2,048 tokens (five interleaved rounds). The MPS
candidate failed the complete-output gate, restricting this finite-table
adapter to CPU float32. Seven constructed AnnData variants reproduce
original preprocessing and all 1,034 exported features bitwise. Use
`load_state_se_encoder(...).encode_adata(...)` for that workflow. The original
model gene-name helper still requires its protein dictionary separately.
The portable artifact is `artifacts/state-se-100m-cpu`; load it with
`load_state_se(...)` in `.venv-state`. See [STATE scope](../docs/state.md) and
[the general derivation](../docs/finite-domains.md).

## Apply the passes to other models

For an ordinary PyTorch forward, discover affine chains and fan-outs automatically:

```python
import torch
from torch import nn
from compressme import Example, compile_affine, load

model = nn.Sequential(nn.Linear(16, 256), nn.Linear(256, 64), nn.GELU()).eval()
result = compile_affine(model, validation=[Example((torch.randn(8, 16),))])
result.save("my-compressed-model")
print(result.report)

# Reload needs the original architecture constructor, not the original weights.
restored = load(lambda: nn.Sequential(nn.Linear(16, 256), nn.Linear(256, 64), nn.GELU()),
                "my-compressed-model").model
```

`compile_affine` uses PyTorch FX. It preserves the traced forward's arguments and
outputs, crosses eligible LayerNorm sandwiches, stops at other nonlinearities,
and retains shared parameters and directly read
weights conservatively. Data-dependent Python control flow and custom hooks need
an explicit adapter. A GraphModule does not preserve arbitrary helper methods or
the original parameter names. Compilation assumes pure inference forwards;
arbitrary Python side effects cannot be certified by a weight-only tool.

For compatible graph attention inside any model:

```python
from compressme import contract_attention

result = contract_attention(model, input_contract="homogeneous_coo")
```

This backend supports PyG `TransformerConv` with homogeneous dense node
features, COO edges, `concat=False`, `beta=False`, `root_weight=True`, and ordinary
source-to-target sum aggregation. Unsupported or unprofitable operators stay
unchanged. The explicit input contract matters: PyG's broader sparse and bipartite
APIs are not implemented by the replacement. `compose_affine` is the reusable
primitive for architecture adapters; Mol-JEPA uses it across the graph-kernel
boundary that FX cannot trace directly.

The exact bilinear and normalization-statistic backends require explicit model
and input dtypes; they reject autocast rather than silently change output dtype.
The delivered Mol-JEPA artifacts run in float32.

## Optional approximate compression

`compress(model, ...)` is a separate, opt-in approximate route: spectral SVD,
activation-covariance SVD using forward-only input statistics, and a LayerNorm
plus Linear factorization. No distillation or optimizer steps are used.

```python
from compressme import compress

proposal = compress(model, ratio=0.5, max_relative_operator_error=0.01,
                    validation=validation_examples, output_tolerance=1e-4)
```

This route may retain every layer if the requested approximation error cannot
be met economically. It reports local residual bounds and can roll back the
whole proposal if final-output validation fails. The Mol-JEPA results above use
none of these approximations.

For fixed LayerNorm followed by an affine map, the exact local minimax formula
is `sqrt(d) * sigma[r+1](../W diag(gamma) P)`, where `P` removes the mean direction.
The implementation evaluates residuals numerically for the stored factors and
biases. These bounds compare a single block at the same input; they do not
certify the output of an entire recurrent or diffusion model. Parameter, anchor
or dtype changes invalidate the stored bound. Bypassing PyTorch tracking through
`.data` or externally modifying shared storage is unsupported.

## Additional lossless storage compression

`result.save(directory, packing=True)` applies reversible byte shuffling and
Zstandard to the exported tensor file. Reload restores **identical tensor bytes**;
packing introduces no numerical error. The prepared SMILES weights occupy about
**67.51 MB**, versus **79.06 MB** unpacked and **181.65 MB** for the original
checkpoint. This is **62.8% less weight-file storage overall**, while resident
parameters remain 56.5% lower than the original.

The default loader recognises packed artifacts automatically. Decoding the
SMILES file took about 0.09 seconds in the local codec benchmark; architecture
construction and tensor loading add their own time. Packing does not reduce
resident tensor memory or inference work, and temporarily needs both compressed
and decompressed buffers. `load(..., max_unpack_bytes=...)` allows an explicit
larger bound for artifacts above the default 1 GiB limit. See
[format and evidence](../docs/packing.md).

## Checkpoints, reproduction and remaining work

The pinned upstream checkpoint is
[`Flogrammer/Mol-JEPA` at `4c912b4`](https://huggingface.co/Flogrammer/Mol-JEPA/tree/4c912b450175f31b5ba913a5dc921c03b27b985a).
Its source, original attribution, full CC BY-NC 4.0 licence and checksums are in
`vendor/moljepa`; those terms also apply to the derived Mol-JEPA artifacts. The
new compressor code is separate from that upstream material.

```bash
.venv/bin/python examples/moljepa.py build --checkpoint /path/to/model.safetensors --packing
.venv/bin/python examples/moljepa.py verify --checkpoint /path/to/model.safetensors \
  --device mps --report benchmarks/mol_jepa_export_mps.json
```

`artifacts/export_verification.json` records checks after a fresh export reload.
`benchmarks/` contains raw timings, errors and fixed verification SMILES. The
optional-modality audit is reproducible with `examples/audit_moljepa_api.py`.

The STATE ST and numerical STATE SE validation scopes are given above.
Boltz-2 has a [portable shared bundle](../docs/boltz2.md), described below and
in the `list_targets()` registry. NovoMolGen 32M AtomWise has a [token-only research
experiment](../docs/novomolgen.md): 12,599 complete-output tensor comparisons per
device are bitwise identical on CPU and MPS after generic lookup save/reload.
The weight reductions are modest (1.676% CPU, 1.267% MPS), with no material
speedup established. This is not a production adapter or arbitrary
`inputs_embeds` support. Other NovoMolGen variants are outside the tested scope.
X-Cell and OmniCell were removed from the work plan, and stFormer is set aside.
Bioptimus remains gated after an authenticated weight-access check and is deferred.
Earlier surveys remain available with `list_targets(include_inactive=True)`;
an archived entry is not a support claim.

Boltz-2 saves 49.55% of joint weight storage while retaining the original
arithmetic. Its released confidence and affinity checkpoints share 5,019
byte-identical named tensors. The general frozen-storage pass reduces registered
weights for both loaded models from **4.09 GB to 2.06 GB** while retaining every Parameter object,
input and output. All 48 returned native tensors match the original bytes on
CPU and MPS, both immediately and after a fresh portable reload, on one small
protein–ligand fixture using the standard sampling schedules. This reduces
joint weight storage; it does not reduce one model's arithmetic or establish a
speedup. Large-complex activation memory remains separate.

```python
from compressme import load_boltz2

bundle = load_boltz2("artifacts/boltz2-shared", device="mps")
# Use native Boltz batches and original prediction methods.
# result = bundle["confidence"].predict_step(batch, 0)
```

The loader requires the pinned local Boltz source and uses safetensors plus JSON.
It needs neither original checkpoint to reload. Native input preparation and
chemistry assets remain separate; use the [Boltz-2 setup and evidence](../docs/boltz2.md).

The shared 2.063 GB tensor file also packs losslessly to **1.760 GB**, another
14.70% disk/transport reduction. The general streaming file codec verified every
restored byte while using 53.75 MiB peak process RAM in a standalone run.
Use [`pack_file` and `unpack_file`](../docs/packing.md) for large files; this does
not further shrink the loaded tensors. The native YAML/FASTA example and a
prepared `.venv-boltz` environment are available in the project.

An optional Boltz adapter also reuses diffusion-conditioning branches that stay
constant within a request. All outputs still match the tested original bytes,
but three warmed timing pairs show no reliable MPS improvement and only a small
CPU affinity benefit. It stays off by default; [all timing rows](../benchmarks/boltz2_request_timing.json)
and [the runtime contract](../docs/boltz2.md#optional-reuse-within-a-prediction) remain available.

[Immutable parameter sharing](../docs/sharing.md) applies to related frozen
models, while [lossless table differences](../docs/frozen-tables.md) encode
families of compiled float32 lookups. Both account for all retained storage
and preserve dtype. Storage sharing also preserves Parameter objects and
enumeration. Their storage reductions do not establish faster execution.

An experimental [general FX request compiler](../experiments/request_fx/README.md)
finds branches depending only on explicitly fixed request inputs and frozen
weights. It reuses those values across dynamic calls without changing the
operators. It retains all weights and counts the extra temporary tensors;
complete-model performance has not been established for this generic path.

Affine and bilinear compressed operators support autograd; finite lookup exports
are frozen inference representations. The published Mol-JEPA wrapper itself
uses `no_grad`, so fine-tuning requires its inner graph/core route. Fewer parameters
can reduce optimizer state; faster fine-tuning has not been benchmarked. Updating
fused parameters changes the optimizer trajectory, and freely updating contracted
attention maps can change the original factor coupling. Inference equivalence
at conversion is the established result.

A separate [restricted SGD experiment](../experiments/moljepa_subspace_sgd/note.md)
uses fixed orthonormal subspaces to preserve first-layer Q/K updates with
243,024 trainable coefficients instead of 4,202,496. It requires frozen input
and shared-edge projections, is not an additional inference-artifact reduction,
and has no measured training speedup. Its failed arbitrary-input float32
raw-logit stress check remains in the evidence.

The [optimization record](../docs/optimization-record.md) keeps the accepted,
rejected and optional candidates, including unsupported cases and failed
numerical gates. The search is bounded by those tests; it cannot establish that
every possible optimization has been exhausted.
