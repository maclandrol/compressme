# From a checkpoint to a tested model rewrite

To rewrite a model, compressme needs its weights, a reproducible architecture
constructor and representative inference calls. The weights may come from
Hugging Face or a local file. The code tells us which tensors the model uses,
and the example calls let us check whether a rewrite preserves its outputs.

The workflow has four steps:

1. **Inspect.** Pin a repository revision, identify the selected weight files,
   list tensor storage and architecture requirements. Multiple checkpoint
   variants need explicit selection. Inspection does not execute downloaded
   Python code or infer an exact low rank from tensor shapes.
2. **Prove an applicable local rewrite.** Examples include composing affine
   maps, retaining a LayerNorm denominator statistic instead of its wide
   activation, and contracting oversized query/key projections into their
   bilinear interaction. Identical frozen embedding rows can be represented
   once because their equality is directly checkable in the weights. A fixed
   token vocabulary also permits complete evaluation through row-local nonlinear
   encoders, storing their smaller outputs as a lookup table.
3. **Validate the complete output contract.** Compare every tensor, discrete
   result, shape and metadata field requested by the caller, with controlled
   random seeds. A proposal that fails its tolerance is rolled back. These
   checks do not replace a downstream biological evaluation.
4. **Export and choose an execution backend.** Store portable tensor weights,
   provenance and a replayable transformation recipe. Lossless byte packing
   reduces disk space. Hardware execution can fuse operations and eliminate
   intermediate tensors without further parameter reduction.

The useful reduction depends on how the tensors are consumed. Two consecutive
affine maps may be stored as their composition. Query/key projections may be
replaced by their bilinear interaction. A frozen table of identical rows needs
only one stored row. For a fixed vocabulary, a deterministic row encoder can
sometimes be evaluated once per token and replaced by its outputs. Each case
has algebraic or structural conditions that must hold before a rewrite applies;
none requires fitting a student or changing precision.

There cannot be a guaranteed large compression ratio for every trained tensor
under an exact-output requirement. A model may contain no removable algebraic
redundancy, and a matrix with a small-looking spectrum can still matter at a
sensitive nonlinear boundary. The package returns an unchanged model and an explanation when none of its
supported reductions applies. Pruning or approximation would need a different
error contract, and biological quality still needs its own evaluation.

The model examples exercise different conditions:

| Target | What it tests |
|---|---|
| Mol-JEPA | Affine reachability, bilinear graph attention, sparse execution, full embedding outputs |
| STATE ST | Frozen redundant tables and expression/count prediction API preservation |
| STATE SE | Fixed gene lookup paths, aliases, normalization and large gene-level tables |
| Boltz-2 | Iterative structure/affinity computations and large pair activations |
| Nesso-1 | Repeated pair contractions, Python bookkeeping and full affinity/metadata outputs |
| NovoMolGen | Small token vocabularies, first-layer projection fanout, autoregressive masks/cache and sampling |

Target inclusion is not a claim that every architecture has a validated adapter.
X-Cell and OmniCell are removed from active scope, stFormer is set aside, and
Bioptimus is deferred because weight access remains gated. Their earlier audits
remain historical evidence. Only the selected NovoMolGen 32M AtomWise variant
is in scope.
Use `compressme.list_targets()` for dated, explicit support status and checkpoint
locations. Some official weights are hosted outside Hugging Face. A local
checkpoint plus the same architecture/validation interface remains necessary
for those models.

Finite-domain compilation is available through `compile_finite_lookup`;
`build_projected_lookup` additionally shares raw and normalized affine branches.
Both return a replacement for a declared row-local block. They do not infer the
safety of discarding an embedding that arbitrary Python helpers also read. See
[finite-domain derivations and API](finite-domains.md).

`compile_finite_blocks` adds model-level discovery/selection and mandatory complete
output checks. `compress_huggingface(..., method="finite_lookup",
finite_input_contract="token_indices_only")` uses that same implementation after
strict pinned-checkpoint loading. An explicit module path is optional; an explicit
contract is required. The same operator pattern can apply in different architectures. Metadata for prior transformations
must survive composition so the exported model still reloads correctly.

Repeated affine compilation keeps an existing graph when new filters would need
another replay recipe. Apply the intended filters to the original model instead.
No-op passes and rolled-back proposals preserve previous recipes and original
tensor dtypes, so composition does not silently break portable reload.
