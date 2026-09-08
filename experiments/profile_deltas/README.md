# Lossless XOR storage for shape-profile lookup tables

This bounded study supports an optional **memory reduction**, with a measured decoding cost. It does not change the finite-fanout source or its archived validation evidence.

The standalone `XorProfileLookup(tables, base_route=0, low_bits='auto')` accepts matching float32 lookup matrices. `lookup(ids, route=i)` returns the selected table's original float32 value bytes. It retains one base matrix and exact integer residual codes; it does not retain the input table list or source weight storage. Identical routes cost no additional tensor storage. An unprofitable residual falls back to a full float32 table.

## Actual saved NovoMolGen tables

The study reads the existing safetensors lookup artifacts, verifies their manifest SHA256, and uses the same native CPU and MPS tables previously validated inside NovoMolGen. Each table contains 84×1,536 float32 values.

| Existing lookup | Full tables | Bulk base + exact codes | Reduction |
|---|---:|---:|---:|
| CPU, 2 profiles | 1,032,192 bytes | 759,024 bytes | 26.46% |
| MPS, 3 profiles | 1,548,288 bytes | 994,896 bytes | 35.74% |

These are actual **unique registered buffer bytes**, including low-byte codes, every padded exception slot, exception column indices and exception values. They are not parameter-only counts, compressed disk sizes, or process peak memory. No floating-point precision changes.

Against the bulk base, only 59 of 129,024 CPU values need XOR bits above bit 15. The MPS alternate tables need 59 and 57 such exceptions. No pairwise XOR changes the highest byte in these particular artifacts. That observation is not assumed by the decoder: sign, exponent and arbitrary high-bit changes are retained explicitly.

A dense low byte plus padded high-bit exceptions is slightly smaller here than dense uint16 codes. The source width fits 16-bit column indices, stored losslessly as signed int16 with an unsigned mask during decoding. One CPU residual costs 242,928 bytes; each MPS residual costs 239,400 bytes. `initial_statistics.json`, `cpu.json` and `mps.json` retain distributions and all alternative base costs.

Choosing a singleton table as MPS base can reach 924,840 bytes, but then common bulk calls require decoding. The prototype deliberately keeps the bulk table direct. The pairwise statistics also show that CSR exception layouts could reduce padding further; CSR execution has not been implemented or timed and those estimates are not counted as achieved savings.

## Exact reconstruction identity

For each float32 value, let `B` be the base's 32-bit representation and `T` the target's representation. Store `X = B XOR T`. Reconstruction is `B XOR X = T`, including signed zero and NaN payload bits. No floating subtraction, rounding or multiplication is involved.

The residual encoding stores its low 8 or 16 bits densely. Every nonzero upper component is stored at its exact feature column, with fixed per-token padding. Columns are unique within a token row. Padding contributes integer zero at column 0. At runtime, selected token rows are gathered, aligned upper bits are added to the disjoint low-bit positions, and integer XOR reconstructs the float32 representation. Bit reinterpretation returns the original dtype.

The mathematical guarantee is byte identity of the represented values, provided the integer operations and reinterpretation implement the specified 32-bit operations. This differs from a claim that independently recomputing a neural layer under a different shape yields the same floating-point answer. The original empirically selected profile still determines which target table is requested.

## Validation

Eight CPU tests pass. They cover arbitrary float32 bit patterns, signed zeros, exponent carries, infinities, NaN payloads, repeated/scalar/noncontiguous/empty IDs, source-storage ownership, incompressible fallback, columns above 65,535, safe recipe/safetensors replay, and refusal of training or float-dtype changes.

Actual CPU artifacts pass 176 selected-row byte comparisons. Actual MPS artifacts pass 264. An additional MPS check passes 42 synthetic special-bit comparisons for both 8-bit and 16-bit residual encodings, with zero differing bytes. The study does not rerun complete-model inference; it verifies exact replacement of the already-validated profile table values.

`recipe()` and `from_recipe()` provide a closed research JSON description plus ordinary safetensors state loading. No remote code or unsafe pickle loading is used. A production wrapper still needs to preserve the original fanout output descriptors and profile dispatch metadata; this prototype exposes route selection explicitly.

## Runtime cost

Twenty-one randomized interleaved rounds use 50 calls per round, ten warmup pairs, preloaded IDs/state and synchronization around each timed MPS group. Raw samples, means and sample standard deviations remain in the JSON reports.

| Lookup-only call | Full table | XOR prototype | Added time |
|---|---:|---:|---:|
| CPU singleton alternate | 0.892µs | 18.613µs | 17.721µs |
| MPS singleton alternate | 6.202µs | 35.270µs | 29.068µs |
| MPS 2-ID alternate | 6.377µs | 35.590µs | 29.213µs |
| CPU direct bulk, 32 IDs | 15.663µs | 23.697µs | 8.034µs |
| MPS direct bulk, 32 IDs | 6.310µs | 11.013µs | 4.703µs |

The bulk route performs the same single float32 gather; the prototype adds Python module/guard overhead. Alternate routes add low-code and exception gathers, integer casts, a scatter addition and XOR. Outputs allocate selected rows only, not a reconstructed full table. These timings do not establish a whole-model speedup. The result is a useful lossless memory tradeoff, particularly when full alternate tables would otherwise erase compression savings.

## Reproduction

```sh
rtk proxy python -m pytest -q test_profile_deltas.py
rtk proxy python audit.py --artifact /path/to/cpu-lookup-artifact --device cpu --output cpu.json
rtk proxy python audit.py --artifact /path/to/mps-lookup-artifact --device mps --output mps.json
```

`audit.py` records exact artifact hashes, prototype hash, Torch version, platform and every retained buffer. Checkpoint/table payloads are not included in this study directory. No production files were changed.
