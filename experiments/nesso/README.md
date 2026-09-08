# Nesso-1 evidence

Original work: [Shenoy et al.](https://doi.org/10.64898/2026.08.01.742196), [source](https://github.com/recursionpharma/nesso), [weights](https://huggingface.co/recursionpharma/nesso).

Use the [Nesso guide](../../docs/nesso.md) and the portable scripts in [examples/nesso](../../examples/nesso/) for new runs. These reports contain numerical preservation checks on actual pretrained weights, not labelled binding-affinity evaluations. No CUDA result is available.

## Recorded evidence

- `reports/final_onepass_tiny_mps.json` and `reports/final_onepass_fragment_mps.json`: final complete MPS output gates. Packed contractions pass byte checks but are slower on both fixtures; the small case also records guards-only, copy, conditioning and combined candidates.

- `reports/final_onepass_cpu.json`: final package and benchmark source, complete native CPU checks and three timing pairs on each 23/143-token input. Median paired ratios 1.0475 and 1.1079; all checked tensors match bytes.

- `reports/python_checks.json`: 40 alternating CPU metadata-only pairs. The complete state-signature contents match; this does not time inference.
- `reports/final_cpu.json`: initial packed-contraction whole-model CPU benchmark, 23 and 143 tokens, before the one-pass Python audit. The matching runtime is archived in `prior-runtime/nesso_runtime.py`; contraction and chunk helpers match the package sources.
- `reports/prepared_tiny_mps.json`: initial single-chunk and conditioning candidates on 23 tokens, before the one-pass audit. Same prior runtime; no reliable gain.
- `reports/candidates_fragment_mps.json`: earlier dispatcher prototype on 143 tokens, before prepared owner checks. Its source hashes identify that exploratory version; it is not presented as a reproducible timing of the final adapter.
- `reports/old_guards_mps.json`: prior-runtime context with all numerical optimisations disabled; isolates the total wrapper overhead against unmodified inference.
- `reports/esm_cpu.json`: separate pretrained ESM final-embedding experiment; no useful speed gain and no downstream affinity validation in this report.
- `reports/transport_roundtrip.json`: reversible file packing, verified SHA256 and complete restored-byte comparison. No resident-memory or inference-speed claim.
- `reports/sdpa_probe_tiny20_mps.json`: rejected fused-attention diagnostic; complete native output bytes differ. Cold diagnostic latencies are not an accepted speed result. `sdpa_probe.py` preserves the exploratory script; it uses the documented local work paths and is excluded from the package.
- `reports/preparation.json` and `reports/download.json`: pinned source/assets, input hashes and preprocessing records. The 384-residue fixture was prepared, but its full Nesso prediction has not been benchmarked.

Every whole-model candidate is compared with the original on the same backend, with the same saved features and RNG seed. All 11 raw-forward and 21 predict-step tensor leaves must match bytes, alongside structure, dtypes and scalar metadata. Timing includes adapter construction/checks/cleanup and original pocket-selection transfers. Checkpoint loading, ESM extraction and external validation are excluded.

Downloaded weights, CCD pickle, generated feature tensors, environments and vendor code stay in ignored local storage. Reports preserve paths from the measured machine as provenance; those paths are not required by the portable tutorial. Earlier and final source revisions must not be pooled as one timing experiment.
