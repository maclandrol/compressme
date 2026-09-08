# Native YAML composition checks

`final/` contains the final complete MPS run with the public shared-weight loader,
original Boltz parser, prediction methods and native writers, plus optional
request-invariant conditioning. All 21 confidence and 27 affinity output tensors
match every value byte of the saved original native MPS references. The input is
the 20-residue protein/ethanol fixture with standard sampling schedules and empty
MSA. These are numerical preservation tests, not a biological quality benchmark.

The exact final source files are included. Runtime SHA256:
`25cedfc5a6769ff3893832a962c618a31edd5a9e1ed3c412e706885cf6ff1c02`.
`final/verification.json` records both source hashes and byte-comparison counts;
`final/FILES.json` inventories the outputs and source copies.

The older root source copies and `mps/` outputs are retained as history. That
earlier run used runtime SHA256
`5a2ee9a0ad7738217ddd0cff9c5be7a2fb39e1110dcd5df6c6ae85e2e85fc26a`.
Its `compressme.json` mistakenly hashes the `contextmanager` wrapper in Python's
`contextlib.py` (`8b7a477...`) rather than the actual runtime module. The original
report is preserved unchanged; the actual module was separately hashed and the
correction is recorded in `benchmarks/boltz2_yaml_mps.json`. The final example
unwraps the decorator before hashing and the final run verifies the corrected
hash. Both runs passed complete output byte comparisons.
