# Boltz-2 request-constant conditioning: guarded prototype

This is a separate compute experiment, after byte-exact shared weight storage was validated. It preserves native FP32 score arithmetic, schedules, input tensors and every returned tensor. It computes atom AdaLN conditioning, atom output gates and the fixed SingleConditioning prefix once per diffusion request and actual multiplicity. It does not cache a prediction or reuse an intermediate between requests.

## Complete-output evidence

The tested guarded source SHA256 is `0b48242b6a0062ad652912a094e08035fd256a489b7d3b0c98ce7eaae26985de`. Both `gate-cpu.json` and `gate-mps.json` pass explicit byte comparisons for all 48 tensor leaves across confidence and affinity, original self-repeat and original-after-restoration controls. The native source is pinned at b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc. Confidence uses 200 steps / 3 recycles / 1 sample; affinity uses 200 steps / 5 recycles / 3 samples. Every one of 200 score calls per prediction used the prepared path.

The fixture is a 20-residue protein plus ethanol, 23 tokens and 160 padded atoms. This demonstrates execution and output preservation, not predictive quality on a structural biology benchmark. Both featurization and diffusion use explicit seeds. Original and candidate receive identical native feature tensors. No mixed precision is used.

## Timing result: not a reliable speed gain

`timing-mps.json` contains five warmed interleaved pairs per model. Confidence ratio of medians is 1.125, but paired ratios range 0.678 to 1.656: the variance prevents a reliable speed claim. Affinity ratio is 1.022, effectively neutral, paired 0.955 to 1.052. All timed outputs are byte-identical. The cause of variance was not established. A post-run host observation cannot establish causes during the measurements.

`timing-cpu.json` is explicitly incomplete. Five confidence pairs gave a 1.033 ratio of medians. Two completed affinity pairs gave 1.003 and 0.923. The run was intentionally stopped before the remaining affinity pairs to test a reviewed variant with less repeated bookkeeping. Completed affinity rows were recovered exactly from process output; they had passed the byte gate before being printed. No complete affinity aggregate or overall acceptance is claimed.

These are synchronized complete `predict_step` timings, including cache preparation and cleanup and runtime contract checks. They exclude unchanged preprocessing and model loading. CPU uses four threads. Every pair uses the same seed for original/candidate, varying across pairs, with alternating execution order.

## Memory and scope

Persistent request caches add 3,019,776 bytes for confidence multiplicity 1 and 9,059,328 bytes for affinity multiplicity 3 in this fixture. Cache memory grows with padded atom count, tokens and sample multiplicity. This is the sum of retained cached tensors, not peak process memory; transient allocation, allocator reservations, and original model state remain additional. All caches are cleared and original module/parameter identities restored. Weight storage and model parameter counts are unchanged by this compute experiment.

The guarded implementation and its evidence remain historical even if a faster variant succeeds. `request_constants-provisional.py` and `gate-cpu-provisional.json` preserve the earlier guard revision; they must not be confused with the final guarded source above.
