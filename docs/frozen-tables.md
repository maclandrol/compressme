# Lossless storage for related lookup tables

A compiler may need several slightly different lookup tables to reproduce the
rounding of different floating-point execution shapes. `pack_lookup_tables`
stores equally shaped float32 tables as one explicit base plus exact integer
XOR differences.

XOR coding is an established way to compress related floating-point values.
For example, [Gorilla's value codec](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)
encodes the XOR of successive time-series values. This layout uses the same
reversible identity between table routes, with indexed exceptions and row lookup
instead of Gorilla's time-series encoding.

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

The compiler verifies every supplied table byte and counts integer indices,
exception values and padding in the stored result. Both logical bytes and
unique resident bytes must decrease. Because this representation uses frozen
buffers, a zero parameter count would conceal its memory cost; compression is
reported in bytes. The optional package recipe reloads from safe tensor files.

Only selected rows are decoded. Keeping the common bulk route as the explicit
base avoids XOR decoding for that route. Alternate routes require integer
gathers and reconstruction. In the archived
research implementation, NovoMolGen's already compiled tables used 26.46% less
CPU storage and 35.74% less MPS storage, at roughly 18 and 29 microseconds of extra
tiny-call decoding respectively. Those measurements describe the research
implementation. They establish neither the public implementation's latency nor
a whole-model speedup.

XOR reconstruction preserves the supplied table values exactly. Whether those
values reproduce the original network at a given input shape still depends on
finite-domain compilation, shape selection and complete-model validation.
Device moves preserve the representation. Precision changes, training, hooks
and modified codec implementations are refused. After ordinary payload mutation,
the original byte comparison remains historical evidence; malformed serialized
payloads fail validation on reload.
