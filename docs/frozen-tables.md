# Lossless storage for related lookup tables

`pack_lookup_tables` stores a family of equally shaped float32 lookup tables
with one explicit base table and exact integer XOR differences. This is useful
when a compiler needs several slightly different tables to reproduce different
floating-point execution shapes.

```python
from compressme import pack_lookup_tables

result = pack_lookup_tables([bulk_table, singleton_table], base_route=0)
if result.report["storage_reduced"]:
    lookup = result.model.to("mps")
    values = lookup(token_ids, route=1)
```

All reconstructed outputs remain float32. For base bits `B` and target bits
`T`, store `D = B XOR T`; reconstruction is `B XOR D = T`. Low-bit integer codes
plus explicitly stored high-bit exceptions preserve every bit, including signed
zeros and special floating-point encodings. Identical routes need no additional
data. Incompressible routes retain a complete table.

The compiler verifies every supplied table byte and reports logical and unique
resident bytes separately, including integer indices, exception values and
padding. A saving must exist against both source measures. It never calls a
zero parameter count a model compression ratio: this representation uses frozen
buffers. The optional package recipe is replayable with safe tensor files.

Only selected rows are decoded. Keeping the common bulk route as the explicit
base avoids XOR decoding for that route. Alternate routes add integer gathers
and reconstruction work, so this is a memory option, not a speed claim. The
earlier research implementation measured 26.46% CPU and 35.74% MPS table-storage
savings on NovoMolGen's already compiled tables, with roughly 18 and 29
microseconds of extra tiny-call decoding respectively. Those timings belong to
the archived research implementation, not the public implementation or the
whole model.

Exact reconstruction of supplied table values does not prove that a table
matches its original neural network on every input shape. Finite-domain
compilation, shape selection and complete-model validation remain separate.
Device moves preserve the representation; precision changes, training, hooks
and modified codec implementations are refused. Ordinary payload mutation makes
the original byte verification historical, and malformed serialized payloads
cannot reload as valid encoded data.
