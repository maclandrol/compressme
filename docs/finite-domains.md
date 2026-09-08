# Compile the finite domain instead of approximating it

If a frozen inference path starts with a token lookup, its input domain is finite.
For a deterministic function applied independently to each token, enumerate
every token's result once and store the resulting table. No labels, optimization,
student model or distributional fitting are involved.

`compile_finite_lookup` recognizes an explicit sequential block containing an
embedding followed by supported affine maps, normalization over the final feature
axis, and pointwise activations. It refuses token mixing, training, custom hooks,
mutating embedding options and known external parameter consumers.

```python
from compressme import compile_finite_lookup

result = compile_finite_lookup(
    token_encoder,
    input_contract="token_indices_only",
    owner_model=model,  # checks aliases outside this block
)
print(result.report)
```

The result replaces that block, not its enclosing model. The compiler accepts it
only when it saves both logical tensor bytes and unique registered backing
storage, and passes full-vocabulary numerical checks using
different batch layouts, scalar IDs, empty inputs and multidimensional indices.
Repeated state keys and shared modules are counted once for resident storage;
logical checkpoint entries alone can falsely suggest a saving.
The enclosing model still needs a complete-output comparison. Learned parameters
can have been trained originally; the exported lookup is an inference representation
and cannot preserve their original training parameterization.

The smaller table can cross nonlinearities because every possible token is
enumerated. This is partial evaluation of a known program over a finite domain.
It does not justify compiling arbitrary continuous inputs or token interactions.

## Preserve several consumers of the same token

`FiniteTokenFanout` declares an embedding, a shared row-local prefix and named
branches. `compile_finite_fanout` compiles all branches together, including an
optional residual embedding output. This avoids dropping a live residual when
precomputing attention projections, and applies to any model with the same
declared computation.

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

This declaration defines the source computation; matching a custom module's
name or attributes does not establish equivalence. `Float32RMSNorm` explicitly
accumulates variance in float32, restores the input dtype before multiplying by
the scale, and is distinct from ordinary `nn.RMSNorm`. The compiler refuses
unproved operations and in-place activations that could couple branches.

Outputs share packed storage only when safe. Different dtypes use different
tables; bit-identical output tables share retained coefficients, and constant
tables use one row. Each returned tensor is independent, so mutating one output
does not change another. Residual embeddings remain included in storage counts.
The source weights are removed only within the declared block; an enclosing
model retaining other consumers must count those weights separately.

Both logical and unique backing bytes must decrease. Every token and several
index layouts are checked separately for every named output. JSON recipes and
strict tensor reload retain the packed representation without source weights.
The compilation comparison is historical after reload or tracked mutation.
Custom hooks and forward overrides cannot silently disappear during export.

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
those positions disagree bitwise. These thresholds are empirical choices from
the tested NovoMolGen MPS runtime, **not defaults for other models or devices**.
The compiler tests every token, interval endpoints, mixed-token boundaries,
composite/strided layouts and empty inputs without relaxing the error limits.
This still does not prove every future floating-point execution equivalent.

Every additional table counts toward both storage gates. Shared residuals and
identical tables are deduplicated across profiles; inference gathers only the
columns needed for the selected profile. Compilation requires extra work and
temporary storage proportional to the proposed tables and evaluation shapes.
Version 2 recipes retain profiles and tensors; ordinary lookups keep version 1.
Changing a threshold invalidates the previous numerical comparison.

The enclosing-model pass accepts the same settings as
`fanout_row_count_profiles=...`; the Hugging Face workflow calls this option
`finite_row_count_profiles=...`. It applies only to declared fanout graphs and
still requires the full-model output gate. It does not automatically discover
an arbitrary native model's token API or execute repository code.

## Apply the pass inside another model

Use `compile_finite_blocks` to select registered Sequential token encoders or
explicit `FiniteTokenFanout` graphs, or discover eligible-looking ones. It retains the enclosing model class when a
child is rewritten, verifies aliases against that model, and requires complete
output examples. Unsupported or unprofitable blocks remain in place. Any failed
complete-output comparison rolls back the proposed rewrites.

```python
from compressme import Example, compile_finite_blocks

result = compile_finite_blocks(
    model.eval(),
    paths=["token_encoder"],  # omit to discover declared finite blocks
    input_contract="token_indices_only",
    validation=[Example((token_ids,), kwargs={"return_all": True})],
)
```

Here the declared contract applies to each selected block: valid integer IDs go
through its forward method, and no external caller reads the block's internal
weights or relies on side effects. The enclosing model can still accept other
inputs. Arbitrary Python API access cannot be inferred safely from a checkpoint.

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

Repository IDs and example calls must refer to the actual model being compiled;
this interface does not download and execute arbitrary Hub Python code. Stored
recipes preserve strict replay through the original architecture factory.

## Small local errors can change a routing decision

An exact real-arithmetic identity plus close local float32 outputs is insufficient
when a later operation chooses a discrete expert. An actual OmniCell router table
passed all-token local checks (maximum error 1.30e-6). For a gene near a top-k tie,
a singleton call selected a different expert and changed the final embedding by
0.2503. The full candidate was rejected, and it also increased resident storage
because the original gene table had another live consumer.

Validate all exposed outputs, including selected indices and relevant public
state, and include small decision margins in representative probes. The package
only checks what the caller's examples return; examples can wrap a method to
include an otherwise hidden observable. Finite enumeration certifies coverage
of token IDs for the tested local layouts, not every possible downstream
floating-point execution. See [the actual counterexample](omnicell-stformer-audit.md).

## Share raw and normalized branches

A related operation retains even less information when an embedding table is
projected both with and without L2 normalization. For row \(e_i\), projection
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

This is useful when the original embedding dimension is much wider than the
observed projection. It is not restricted to genes, a particular training
algorithm or a Hugging Face model family. The State SE adapter is the current
real-checkpoint experiment; its complete-output validation determines whether
that particular application is accepted.

## What is and is not guaranteed

The identities are exact in real arithmetic. Stored float32 results can differ
because GEMM reductions depend on batch shape and backend. Even identical input
rows can receive slightly different last bits in a batched operation. Therefore
the compiler reports its full-domain numerical measurements separately from
the algebraic identity, and does not issue a universal floating-point certificate.

The generic operators have portable shape recipes and strict tensor reload.
They do not keep an unreported copy of the source embedding or projection.
Parameter-derived lookup tables contain the trained computation itself; they
are not a cache of predictions for previously seen cells or molecules.

## Preserve constant rows at the original operation shape

A learned CLS or dataset token is constant, yet evaluating it alone can round
differently from evaluating it inside a large matrix multiplication.
`ShapeMatchedConstantRows` evaluates those known rows inside a zero-filled tensor
with the original leading shape and positions, using an audited row-local encoder.
It retains only the selected constant outputs in a bounded cache. The temporary
first-call allocation and encoder work can be large; steady calls of the same
shape reuse the small constant result. It never stores cell or molecule outputs.

The helper checks tracked tensor versions, identities, dtype, device, scalar
module settings and precision settings. Mutation invalidates cached rows, and
unversioned inference tensors are evaluated without caching. Calls that need
gradients through the encoder or constants also avoid caching; fully frozen
encoders can reuse constants in ordinary calls without a special grad context. Exact-shape execution is still a numerical strategy requiring target
backend validation, not a theorem about every opaque kernel implementation.

Finite lookup exports additionally label whether their local compilation gate
still refers to the current tracked tensors. Changed weights or unvalidated
replay retain the original measurements as historical evidence. Direct unsafe
storage writes that bypass PyTorch version tracking are outside this contract.

Whole-model comparison reports distinguish equal numeric values (`exact`) from
equal tensor value bytes (`bitwise`). Positive and negative zero compare equal
numerically but have different bytes. This extra evidence field does not change
the configured numerical acceptance thresholds; `bitwise_identical` summarises
the byte result on the supplied examples only.
