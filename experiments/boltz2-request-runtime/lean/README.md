# Native Boltz-2 invariant conditioning: lean immutable-request experiment

Final candidate source SHA256: `25cedfc5a6769ff3893832a962c618a31edd5a9e1ed3c412e706885cf6ff1c02`. The prior `5a2ee9...` source is retained only as pre-cleanup history. The final change hardened exceptional cleanup; it did not change normal arithmetic.

## Reusable principle and exactness scope

In a recurrent computation, if a subtree depends only on request inputs and frozen parameters, its value is invariant across iterations. It can be evaluated once, provided the evaluation retains the same dtype, tensor shape/layout, original operations and RNG effects. In this pinned Boltz implementation the selected producers are deterministic Linear/LayerNorm/Sigmoid chains, so no RNG draw is removed. We keep the time-varying branches, native attention, transitions, residuals, score function and complete diffusion schedule.

This is source-specific invariant analysis, not a general discovery pass. Atom conditioning is expanded using the actual original sample multiplicity and window shape before evaluating original producers. SingleConditioning retains the original repeated batch shape. The finite-precision choice matters: narrower representative rows can trigger different matrix kernels. The cache is destroyed at every sampler exit. A new prediction computes fresh conditioning values; no input-to-output memoization occurs.

## Runtime contract

The public adapter requires explicit `immutable_request=True`, an unmodified pinned Boltz source, frozen eval descendants, no hooks or per-instance overrides, no-grad execution, exclusive model ownership and immutable state/settings/conditioning throughout its scope. Full hierarchy/state audits occur at request boundaries; static input identity/layout/tracked versions are checked each score call. Mutations that bypass tensor versions or are undone before a boundary remain prohibited by the contract rather than claimed detected. Unsupported layouts and autocast calls use the native path. This is not a training adapter.

## Test procedure

`evaluate.py` compares the standard shared artifact's ordinary native forward to the scoped candidate, on exactly the same native features. It also checks original self-repeat, original-after-restoration, module/parameter identities, finite outputs and cache cleanup. All tensor leaves are compared by shape/dtype and contiguous CPU bytes, including integer/bool tensors and signed zeros. The prototype is hashed by its module file directly, independent of decorators.

Confidence keeps 200 diffusion steps, 3 recycles, 1 sample. Affinity keeps 200 steps, 5 recycles, 3 samples. Native preprocessing and diffusion are separately seeded. This fixture is a 20-residue protein plus ethanol; it is a complete execution/output-preservation check, not an assessment of biological quality or scalability.

Timings require an accepted matching full-output gate and include all context setup, preparation, native `predict_step` and cleanup. They exclude unchanged native featurization and artifact loading. Warmups precede alternating execution orders; fresh paired diffusion seeds are used. Every pair is byte-checked outside the timer and written immediately, preserving incomplete runs if interrupted.

## Results on this Mac

CPU and MPS both pass all 48 original output tensor leaves (639,024 logical bytes per backend), original self-repeat and after-restoration controls. All 200 score calls use the optimized route; no fallback occurs. Each request installs prepared wrappers once and restores the original module/parameter identities and state. Every one of the 12 timed original/candidate pairs is byte-identical.

Three warmed interleaved pairs per backend/model:

| Backend | Model | Original median seconds | Candidate median seconds | Ratio of medians | Median paired speed ratio | Paired range |
|---|---|---:|---:|---:|---:|---:|
| CPU, four threads | Confidence | 11.365 | 10.888 | 1.044 | 0.980 | 0.957–1.044 |
| CPU, four threads | Affinity | 21.742 | 20.995 | 1.036 | 1.031 | 1.014–1.064 |
| MPS | Confidence | 8.177 | 8.340 | 0.980 | 1.067 | 0.830–1.094 |
| MPS | Affinity | 10.880 | 10.965 | 0.992 | 1.000 | 0.992–1.007 |

A ratio above one means faster. Separate-median and paired summaries can disagree when runtimes drift; the raw paired rows are retained. There is no useful GPU gain on this fixture. CPU affinity has a small positive result in three pairs; confidence is inconsistent. This is insufficient for a general/default acceleration claim. No wider protein-size distribution or quality benchmark was run. The guarded historical implementation had similarly inconclusive performance.

Request caches retain 3,019,776 bytes for confidence and 9,059,328 bytes for affinity on this fixture, proportional to padded atoms, tokens and sample multiplicity. These are retained cache tensor bytes, not total peak memory. Transient preparation allocations and backend allocator reservations are additional. All caches are cleared at request exit. This compute adapter changes neither the saved checkpoint nor parameter storage; the separate shared-weight storage result is unaffected.

## Reproduction

The tested environment is Python 3.12.14, PyTorch 2.14.0, Boltz 2.2.1 at the pinned source revision, macOS 26.6.1 on Apple M5. Use the native runtime evidence package's complete original fixture data with `--native-dir`; its 21 canonical molecules support this fixture, not every arbitrary CCD identifier. Use the normal shared Boltz bundle with `--artifact`. The root evidence package includes the small support scripts required by this harness. No original checkpoint pickle loading is used.

```sh
PYTHONPATH=/path/to/compressme/src python lean/evaluate.py --device mps --mode gate --artifact /path/to/compiled-bundle --native-dir /path/to/native-runtime-evidence --output gate-mps.json
PYTHONPATH=/path/to/compressme/src python lean/evaluate.py --device mps --mode timing --artifact /path/to/compiled-bundle --native-dir /path/to/native-runtime-evidence --accepted-gate gate-mps.json --rounds 3 --warmups 2 --output timing-mps.json
```

Repeat with `--device cpu` and CPU report paths. The gate must pass for the exact prototype hash and backend before timing. The architecture loader verifies the pinned native source. CPU and MPS results are same-backend comparisons; no cross-backend byte-equivalence claim is made.
