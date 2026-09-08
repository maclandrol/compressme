# State SE MPS: exact original-row storage

The earlier finite-encoder MPS candidate failed the unchanged 1e-5 complete-output
gate because floating-point kernels produce shape-dependent results. Its sampled
profile history remains in `../state_mps_profiles`; it is not duplicated here.

Lossless XOR profile storage was estimated from the same 16 real genes at 466.17 MB
for ten full-vocabulary tables, exceeding the original 405.30 MB raw table. Signed
integer bit-pattern deltas improved the estimate to 402.49 MB; a layered signed
byte plus sparse high corrections gave 364.67 MB. These are extrapolations from
16 genes, not a full-vocabulary measurement or an accepted model. Even if storage
improved, those sampled shape thresholds would not establish all-shape API
preservation. No finite-profile MPS guard was removed.

The accepted route instead preserves the original gene vectors exactly in
independently compressed row blocks, then calls the original GPU operations.
The same transfer principle applies to other frozen embedding tables. It trades
CPU decode work for lower registered storage, without guessing kernel regimes.

Final portable reports are `portable-reload-cpu.json` and
`portable-reload-mps.json`: 27 cases, 377 tensor leaves each, all byte-identical,
including signed zero, full raw weight readback, every native numerical output
and the original gene-name helper. The unchanged CPU finite artifact is separate.

Reproducible source is in `examples/state_se/build_lossless.py`,
`lossless_checks.py`, `lossless_backend.py`, and `benchmark_lossless.py`.
Use the existing State environment and explicit local checkpoint/artifact paths;
no weights, decoded tables or temporary model caches are stored in this evidence
directory. Actual execution options from this machine are in the reports.
The portable backend runner now distinguishes the typed CPU codec payload from
ordinary device state, and includes all-device registered storage in its report.

Registered state falls from 848,155,296 to 794,613,047 bytes (-6.31%),
including all 351,756,951 bytes of encoded CPU buffers. There is no persistent decoded row cache. Peak request
memory and biological accuracy were not measured. CUDA was not tested because
no NVIDIA device is available. Timings, including any slowdown, are recorded
separately in the latency reports; storage reduction is not an inference gain.
