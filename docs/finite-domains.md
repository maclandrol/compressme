# Precompute a finite token domain

A frozen token encoder has a finite set of possible inputs. If it applies the
same deterministic function independently to each token, we can evaluate every
token once and store the results in a lookup table. This requires no labels,
optimization, student model or distributional fitting.

`compile_finite_lookup` supports explicit sequential blocks: an embedding,
affine maps, normalization over the final feature axis and pointwise activations.
Token mixing would break independence, while training or mutating embedding
options would change the table's source computation. These cases, custom hooks
and known external parameter consumers are refused.

```python
from compressme import compile_finite_lookup

result = compile_finite_lookup(
    token_encoder,
    input_contract="token_indices_only",
    owner_model=model,  # checks aliases outside this block
)
print(result.report)
```

The result contains a replacement for the selected block. Acceptance requires
savings in both logical tensor bytes and unique registered backing storage,
plus full-vocabulary numerical comparisons across batch layouts, scalar IDs,
empty inputs and multidimensional indices. Repeated state keys and shared
modules count once toward resident storage; counting checkpoint entries alone
can overstate the saving.

The enclosing model still needs a complete-output comparison. Compilation uses
the learned parameters, but the exported table is an inference representation
and cannot preserve their original training parameterization.

This is established finite-domain tabulation: nonlinear operations can be
precomputed just as affine operations can when every input is known and the
function is fixed. Program specialization has a classical treatment in
[Jones, Gomard and Sestoft's *Partial Evaluation and Automatic Program Generation*](https://studwww.itu.dk/people/sestoft/pebook/).
A familiar deployment example is Google's
[two-tower retrieval workflow](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture),
which precomputes a trained candidate network's embeddings for every candidate.
Here the same principle is applied inside a model, with explicit consumer,
storage and complete-output checks. Continuous inputs or interactions between
tokens fall outside this finite-token derivation.

## Preserve several consumers of the same token

Precomputing attention projections must also preserve any live residual that
uses the original embedding. `FiniteTokenFanout` describes these uses together:
an embedding, a shared row-local prefix, named branches and an optional residual
embedding output. `compile_finite_fanout` compiles the whole declaration, which
can describe the same computation in any model.

```python
from torch import nn
from compressme import FiniteTokenFanout, Float32RMSNorm, compile_finite_fanout

source = FiniteTokenFanout(
    embedding,
    nn.Sequential(Float32RMSNorm(rms_scale, eps=1e-6)),
    {"q": query_projection, "k": key_projection, "v": value_projection},
    residual_key="residual",
).eval()
result = compile_finite_fanout(source, input_contract="token_indices_only")
outputs = result.model(token_ids)  # q, k, v, residual
result.save("compiled-token-branches", packing=True)
```

The declaration specifies the source computation. A custom module needs to
match that computation, not merely its name or attributes. `Float32RMSNorm`
explicitly
accumulates variance in float32, restores the input dtype before multiplying by
the scale, and is distinct from ordinary `nn.RMSNorm`. The compiler refuses
unproved operations and in-place activations that could couple branches.

Different dtypes use different packed tables. Bit-identical output tables share
their stored coefficients, and constant tables need only one row. Each returned
tensor is independent, so mutating one output
does not change another. Residual embeddings remain included in storage counts.
The source weights are removed only within the declared block; an enclosing
model retaining other consumers must count those weights separately.

For each named output, the compiler compares every token across several index
layouts. Both logical and unique backing bytes must decrease. JSON recipes and
strict tensor reload preserve the packed representation without source weights.
After reload or tracked mutation, the compilation comparison describes the
earlier state. Export refuses custom hooks and forward overrides that would
otherwise disappear.

### Match a known numerical execution shape

The same row-local formula can round differently in a matrix-vector kernel and
a larger matrix-multiplication kernel. Optional `row_count_profiles` build an
additional full-vocabulary table for each declared execution shape:

```python
result = compile_finite_fanout(
    source,
    input_contract="token_indices_only",
    row_count_profiles=[
        {"max_rows": 1, "evaluation_rows": 1},
        {"max_rows": 15, "evaluation_rows": 2},
    ],
)
```

Here a call with one ID selects the singleton table, 2–15 IDs select the second
table, and larger calls select the bulk table. Construction evaluates each
token in `(1, evaluation_rows)` identical positions and refuses a profile if
those positions disagree bitwise. These thresholds came from the tested
NovoMolGen MPS runtime. They are empirical
choices, not defaults for other models or devices.
The compiler tests every token, interval endpoints, mixed-token boundaries,
composite/strided layouts and empty inputs without relaxing the error limits.
The measurements cover those executions; they cannot prove agreement for every
future floating-point execution.

Every additional table counts in both storage comparisons. Shared residuals and
identical tables are deduplicated across profiles; inference gathers only the
columns needed for the selected profile. Compilation requires extra work and
temporary storage proportional to the proposed tables and evaluation shapes.
Version 2 recipes retain profiles and tensors; ordinary lookups keep version 1.
Changing a threshold invalidates the previous numerical comparison.

The enclosing-model pass accepts the same settings as
`fanout_row_count_profiles=...`; the Hugging Face workflow calls this option
`finite_row_count_profiles=...`. It applies only to declared fanout graphs and
still requires complete-model output comparisons. The caller must declare the
native token API; the pass neither discovers an arbitrary API nor executes
repository code.

## Apply the pass inside another model

`compile_finite_blocks` can select registered Sequential token encoders or
explicit `FiniteTokenFanout` graphs, or discover possible candidates. When it
rewrites a child, it retains the enclosing model class and checks aliases against
that model. Unsupported blocks and those that save no storage remain in place.
Complete-output examples are required; a failed comparison rolls back the
proposed rewrites.

```python
from compressme import Example, compile_finite_blocks

result = compile_finite_blocks(
    model.eval(),
    paths=["token_encoder"],  # omit to discover declared finite blocks
    input_contract="token_indices_only",
    validation=[Example((token_ids,), kwargs={"return_all": True})],
)
```

The contract applies to each selected block: callers pass valid integer IDs
through its forward method, without reading internal weights or depending on
side effects. The enclosing model can still accept other inputs. A checkpoint
alone cannot establish how callers use an arbitrary Python API.

The Hugging Face entry point provides the same pass after strict loading into
a trusted local architecture with matching tensor names and dtypes:

```python
from compressme import compress_huggingface

result = compress_huggingface(
    repo_id, local_model_factory,
    revision=pinned_revision,
    method="finite_lookup",
    finite_paths=["token_encoder"],
    finite_input_contract="token_indices_only",
    validation=whole_model_examples,
)
result.save("compiled-model", packing=True)
```

Supply the repository ID and examples for the actual model being compiled.
The interface loads weights into the trusted local architecture without
downloading and executing arbitrary Hub Python code. Stored recipes preserve
strict replay through the original architecture factory.

## Small local errors can change a routing decision

A rewrite can be exact in real arithmetic and still change which discrete
expert a float32 calculation selects. An actual OmniCell router table passed
all-token local comparisons with
maximum error 1.30e-6. Yet for a gene near a top-k tie, a singleton call selected
a different expert and changed the final embedding by 0.2503. We rejected the
full candidate. It also increased resident storage because another live consumer
still needed the original gene table.

Complete-output examples should include selected indices, relevant public state
and cases with small decision margins. The package compares what those examples
return; a wrapper can expose an otherwise hidden observable. Enumerating tokens
covers every ID at the tested local layouts. It does not cover every possible
downstream floating-point execution. See [the actual counterexample](omnicell-stformer-audit.md).

## Share raw and normalized branches

When an embedding feeds both raw and L2-normalized projections, those branches
can share a smaller representation. For row \(e_i\), projection
\(W\), bias \(b\), and normalization floor \(\epsilon\), store

\[
z_i = W e_i,\qquad r_i = \max(\lVert e_i\rVert_2,\epsilon).
\]

Then both outputs follow from the same projected table:

\[
W e_i+b = z_i+b,\qquad
W(e_i/r_i)+b = z_i/r_i+b.
\]

The bias stays outside the division. Subsequent LayerNorm and activations can
remain online and shared. `build_projected_lookup` implements this identity and
checks both complete token domains before acceptance. If an application also
accepts arbitrary raw vectors, it can retain the original projection for that
API while using the lookup on token inputs. Whole-model accounting must include
that retained projection.

The opportunity comes from an embedding dimension much wider than its
projection, independently of whether the tokens represent genes or how the
model was trained or distributed. STATE SE provides a real-checkpoint experiment;
complete-output validation determines whether that application is accepted.

## Real arithmetic and numerical agreement

The identities are exact in real arithmetic. Stored float32 results can differ
because GEMM reductions depend on batch shape and backend. PyTorch documents
that [batched and sliced computations need not agree bitwise](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
Even identical input rows can receive slightly different last bits in a batched
operation. The compiler therefore reports full-domain numerical measurements alongside
the algebraic identity. Those measurements are not a universal floating-point
certificate.

Portable shape recipes and strict tensor reload preserve the compiled operators
without an unreported copy of the source embedding or projection. Their tables
store the trained computation over every token, independently of previously
seen cells or molecules.

## Preserve constant rows at the original operation shape

A learned CLS or dataset token is constant, yet evaluating it alone can round
differently from evaluating it inside a large matrix multiplication.
`ShapeMatchedConstantRows` evaluates those known rows inside a zero-filled tensor
with the original leading shape and positions, using an audited row-local encoder.
It retains only the selected constant outputs in a bounded cache. The first
call may need a large temporary allocation and substantial encoder
work. Later calls of the same shape reuse the small constant result. Cell and
molecule outputs are never stored.

The helper checks tracked tensor versions, identities, dtype, device, scalar
module settings and precision settings. Mutation invalidates cached rows, and
unversioned inference tensors are evaluated without caching. Calls that need
gradients through the encoder or constants also bypass the
cache. Fully frozen encoders can reuse constants in ordinary calls without a
special grad context. Matching the execution shape is a numerical strategy that
needs target-backend validation; it cannot establish a theorem for every opaque
kernel implementation.

Finite lookup exports record whether the local compilation comparison still
applies to the tracked tensors. After weights change or an unvalidated reload,
the original measurements remain historical evidence. Direct unsafe
storage writes that bypass PyTorch version tracking are outside this contract.

Whole-model comparison reports distinguish equal numeric values (`exact`) from
equal tensor value bytes (`bitwise`). Positive and negative zero compare equal
numerically but have different bytes. The configured numerical acceptance
thresholds stay the same.
`bitwise_identical` summarises byte agreement on the supplied examples only.
