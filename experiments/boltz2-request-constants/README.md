# Request-constant Boltz-2 computation

This prototype applies ordinary partial evaluation / loop-invariant code motion to actual Boltz-2 inference. It retains all source weights, precision, sampling steps and dynamic attention/transition calculations. It does not cache predictions or reuse values across requests. The generic opportunity is to mark explicitly immutable inputs of pure subgraphs and evaluate those subgraphs once per request and exact tensor layout.

## Dependency proof

`Boltz2.forward` computes `diffusion_conditioning` once before `structure_module.sample`. Its `c` depends on the request's fixed reference-atom features and completed trunk features. It does not depend on denoising time or current noisy coordinates. The sampler keeps this dictionary and its trunk/input tensors unchanged through its denoising loop.

`DiffusionModule.forward` passes `c.float()` to `AtomAttentionEncoder`. The encoder repeats c with the actual score-call multiplicity and returns that same repeated c as `c_skip`. Neither atom transformer writes to c: it only views c into windows and passes it as condition s. `AtomAttentionDecoder` passes c_skip unchanged into its transformer. Noisy coordinates change q/a; those dynamic paths remain live.

For each of the six atom layers, the condition-only results are attention AdaLN scale and bias, attention output gate, transition AdaLN scale and bias, and transition output gate. These are pure functions of c and frozen parameters. Preparation executes each original LayerNorm, Linear and sigmoid operation at the exact original `[B*m*(N/32),32,128]` layout. At use, original activation LayerNorm, multiplication/addition, attention, transition, residual and masking operations still execute. No matrices are folded or reassociated.

`SingleConditioning` first evaluates `single_embed(norm_single(cat(s_trunk,s_inputs)))`. That prefix is also request-constant. Preparation uses the exact original repeated `[B*m,T,768]` layout. Fourier time embedding, its normalization/projection and both nonlinear residual transitions remain dynamic and run with original code/ordering on every score call.

The source chunks samples with `sample_ids.chunk(multiplicity % max_parallel_samples + 1)`. This prototype preserves that expression rather than replacing it with intended or more conventional chunking. For the tested default affinity configuration, m=3 and max_parallel_samples=1 produce an actual score multiplicity of 3. Cache entries follow observed score-call multiplicity, never inferred chunk sizes. Particle resampling changes coordinates/sample indices; it does not rewrite c, trunk or input features. No activation reuse between the separate structure and affinity requests is implied.

## Local evidence and cost

The local probe uses actual pretrained weights and a native-derived conditioning fixture from a 20-residue protein plus ethanol. Both complete original atom-transformer submodules retain their attention implementations, masks and to_keys mapping. Dynamic q varies independently of c. All 28 CPU comparisons are bitwise equal, covering both atom transformers, multiplicities 1 and 3, restoration, and full SingleConditioning at six time values. The 1e-5 mixed gate is unchanged; byte equality is the stronger observed result. This local result does not establish full predicted-coordinate equality.

For 200 score calls and one encountered layout, preparation removes 7,363 Linear calls, 2,587 LayerNorm calls and 4,776 sigmoid calls. On the 23-token/160-atom fixture it stores 3.02 MB of intermediate values at m=1 or 9.06 MB at m=3. These are operation/memory counts, not a measured speedup. There is a first-call preparation cost; weights and checkpoint size are unchanged. Whole native CPU/MPS parity and timings are evaluated independently in the runtime audit.

## Scope and implementation

`constant_conditioning_sampling_scope(model)` installs a temporary adapter around each original sampler invocation. Each request creates its own memo. Prepared modules are activated only while the original score forward executes, then restored. `finally` removes all instance overrides and clears each memo on success or failure. `stats['requests']` reports actual score counts, optimized/fallback calls, prepared layouts and cache bytes. There is no persistent output/conditioning cache.

This is a prototype for pinned, uncompiled, frozen eval modules under no-grad execution and exclusive model ownership. Every descendant must be eval, and hooks/instance forwards/compiled wrappers are refused. Producer binding identities, storage/layout metadata, LayerNorm attributes and tracked Tensor versions are checked without reading GPU values. Detected mutation or an autocast-enabled score call uses the original path. Unsafe `.data` in-place writes can bypass PyTorch version counters; every external mutation during a sampling request remains outside the explicitly immutable-input/module contract. Changes between requests are allowed because no request cache survives. Direct Prepared* helpers require their declared static-input contract; shape agreement alone is not a same-value proof.

A production integration should lower the same proven graph cut into explicit prepared payloads passed to original transformer consumers, or use the general request-scoped FX partial evaluator. It should avoid exposing the temporary module substitutions as an arbitrary-input standalone compressed model. This is inference optimization, not a fine-tuning-trajectory claim.

## Other conditioning work

The 24 token-transformer layers share the same timestep-conditioned s within one score call, making 48 condition normalizations candidates for shared statistics. However s changes after time addition and nonlinear conditioning transitions, so its AdaLN values cannot be cached across denoising steps. A full schedule determines Fourier-only values, but that small branch offers much less arithmetic. Computing each time row at the original sample shape preserves rounding; batching all times can change GEMM rounding and would need fresh gates. Precomputing every token layer's condition outputs across all 200 times would also require roughly gigabytes of extra intermediate storage on this tiny fixture, so it is not the current approach.

The earlier 24 pair-bias projections are already hoisted before sampling; their normalization-bank optimization is separate and must not be counted as a per-denoising-step saving.
