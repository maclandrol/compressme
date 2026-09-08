# Request-local FX partial evaluation — experiment

This is a general implementation of loop-invariant partial evaluation, not a
Boltz-name recogniser or a renamed dictionary cache. A caller supplies an
already audited `torch.fx.GraphModule` and explicitly declares static argument
tensors. The implementation propagates that declaration through a narrow pure
operator grammar, executes every reachable static node using its original
operation and full original tensor layout, and retains only values crossing
from the static region into dynamic computation. The remaining FX graph keeps
the original dynamic operators and their order.

```python
with torch.no_grad():
    with partial_evaluate(audited_graph, {"s": conditioning}) as prepared:
        first = prepared(a=coordinates_1)
        second = prepared(a=coordinates_2)
```

The dynamic arguments keep their FX placeholder names/order. Static arguments
are fixed by the request and omitted from calls; an optional `s=conditioning`
keyword must be the identical bound tensor. Replacing it, even by equal values,
is rejected. An adapter that sees newly repeated tensors must prove the fixed
conditioning relationship and bind the actual original repeated layout once;
the general primitive does not infer equality from matching shapes.

For AdaLN, the compiler discovers `LayerNorm(s)`, both conditioning projections
and the sigmoid. It retains only scale and bias and leaves dynamic activation
normalisation/multiply/add unchanged. For a Linear–Sigmoid gate, the entire
graph becomes a bound value. For a concatenate–LayerNorm–Linear prefix followed
by dynamic time addition, it retains the projected prefix. No numerical fitting,
operator folding, rank truncation or precision change is involved.

The real-arithmetic proof is induction over a pure DAG: any node depending only
on immutable static inputs/state has the same value at every evaluation. Replace
its outgoing boundary value by that value; all dynamic successors receive the
same operands. Floating-point byte equality additionally depends on the backend
and actual execution shape, so numerical gates remain required. This is existing
partial-evaluation/compiler reasoning, not a new algebraic theorem.

## Enforced scope

The experiment accepts ordinary Linear, LayerNorm, Sigmoid, ReLU, SiLU and
Identity modules, selected functional arithmetic/normalisation, concatenation,
indexing and nonmutating shape operations. It rejects unknown/stateful/random
nodes, in-place activations including positional flags, hooks, gradients,
training, autocast, compiled wrappers, unsupported tensor subclasses and custom
dynamic containers. It never traces arbitrary Python and is not an arbitrary-code
security sandbox: the supplied graph and local PyTorch runtime remain trusted.

Source/constant/input signatures track object identity, storage, metadata and
available tensor versions, plus relevant normalisation attributes and graph
structure. Ordinary mutation, parameter rebinding, changed storage, changed
settings and stale cached outputs are refused. Unsafe in-place `.data`/foreign
alias writes can bypass these counters and violate the explicit immutable-owner
contract. Unversioned inference tensors are rejected by default; the optional
`allow_unversioned_tensors=True` requires a caller-proved immutable request owner.

Only request-local computation is reused. Every original source weight remains
explicitly registered and counted; static input references and boundary tensors
are also registered and counted using unique backing allocations. There is no
parameter compression claim. Preparation keeps static intermediate tensors until
the boundary graph is built; its report counts that retained tensor storage but
excludes allocator and kernel workspaces. Context exit explicitly removes cached
buffer slots, avoiding delayed GPU release from FX's reference cycles, and closes
the graph even on exceptions. Caller-retained returned tensors can of course
outlive the context. Serialization and device conversion of the prepared object
are refused. This is a temporary execution plan, not a saved model.

## Evidence and practical boundary

The focused suite covers 25 cases, including float32/float64, empty and strided
inputs, multiple dynamic inputs/outputs, gate/prefix derivation, mutation and
autocast rejection, shared-storage accounting, immediate cache release and
exception cleanup. `probe_boltz_adaln.py` loads only the first actual confidence
checkpoint AdaLN tensors and uses the audited native conditioning fixture. Its
eight CPU comparisons, at multiplicities 1 and 3 with four varying activations,
were byte-identical to the original module. It does not construct the complete
Boltz model or validate a sampler.

The small CPU probe separately times original execution, the guarded prepared
object and the generated dynamic graph without guards. The guard checks are
material at this scale: inspect `boltz-adaln-cpu.json` for the measured values.
The unguarded path is a diagnostic, not a safe public execution route. Full-model
speed must include preparation, dispatch, guards and extra request memory.

The primitive is reliable enough for restricted derivation and correctness
experiments now. It should not replace the separately validated native sampler
adapter without new complete-output and timing gates. A faster reusable runtime
would need an owner-checked request lease or compiled guard mechanism; removing
checks or calling the unguarded graph silently is not an established speedup.
