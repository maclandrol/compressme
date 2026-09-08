# What the optimizations changed

**Original models:** [Mol-JEPA, Rottach et al.](https://arxiv.org/abs/2608.22642) · [Boltz-2, Passaro et al.](https://doi.org/10.1101/2025.06.14.659707) · [STATE, Arc Institute](https://arcinstitute.org/manuscripts/State).
{ .original-work }

The tested changes reduce resident weights, file size or execution time under
different conditions. The tables separate those effects from algebraic
correctness and agreement with the original checkpoint. None of the delivered
reductions uses distillation, quantization or truncated low-rank approximation.

## Validated reductions

| General operation | Real checkpoint evidence | Practical effect |
|---|---|---|
| Compose narrow affine inputs with their consumers | Mol-JEPA first graph block | 5,504,000 fewer parameters |
| Store bilinear query/key interactions | Mol-JEPA graph attention | 4,677,160 fewer parameters |
| Remove branches excluded by a declared input contract | Mol-JEPA SMILES only | 15,462,401 fewer inactive parameters; all embedding heads retained |
| Sparse graph features, fused graph operations and batched affine heads | Mol-JEPA complete CPU/MPS calls | About 2× original warmed MPS speed in the recorded run |
| Store bit-identical frozen embedding rows once | STATE ST K562 actual weights | 21.25% fewer resident parameters; tested outputs bitwise equal |
| Lossless byte packing | Portable Mol-JEPA artifact | 67.51 MB weight file; no added numerical error |
| Store shared state entries once | Generic tied-parameter export/reload tests | Smaller file when multiple state keys share the same storage view |
| Share byte-identical immutable parameters across models | Complete Boltz-2 confidence/affinity pair, CPU and MPS | 49.55% lower joint registered storage; original Parameter objects and arithmetic retained |
| Stream reversible byte shuffling and compression | Complete 2.063 GB shared Boltz tensor file | 1.760 GB packed file; every restored byte equal; 53.75 MiB standalone peak RSS |
| Full-vocabulary partial evaluation through nonlinear row encoders | STATE SE CPU, including 2,048-gene numerical inputs | 28.67% fewer parameters while retaining raw-vector forward; maximum observed error 6.68e-6 |
| Contract a LayerNorm sandwich using its denominator statistic | Constructed eligible operator | 83.7% fewer parameters in an 82→512→512 example; not an extra biology-model saving |

The general compiler discovers closed finite-token blocks inside ordinary
PyTorch models, including through the Hugging Face workflow. It compares complete
requested outputs and rolls back proposals that fail. For STATE SE, a portable
h5ad loader also preserves preprocessing and all 1,034 exported features bitwise
on seven constructed variants, without the original protein dictionary.

The NovoMolGen token-only 32M experiment uses tables matched to numerical
execution shapes. CPU uses two tables and saves 528,896 parameters
(1.676%); MPS uses three and saves 399,872 (1.267%). All 12,599 CPU and 12,599 MPS
tensor comparisons were bitwise identical, including hidden states, attention
and generation/cache outputs. The research adapter requires token-only inputs;
arbitrary native embedding inputs remain outside its contract.
The reusable fanout compiler, including ordinary table save/reload, reproduces
the same complete results. Earlier manual-layout timings are essentially neutral;
generic-layout timings are recorded separately. See [the detailed audit](novomolgen.md).

## Rejected and optional candidates

- **Nesso-1 runtime layout and bookkeeping:** contiguous channelwise contraction
  gives modest CPU gains on the recorded native inputs. One ownership walk
  replaces three during state checks, preserving the same checked metadata.
  Attention-copy removal and repeated-conditioning reuse offer no dependable MPS
  gain; a fused-attention probe changes output bytes and remains rejected for
  the strict contract. Lossless checkpoint packing separately saves 14.91% of
  disk/transport bytes. [Methods, timings and reproduction](nesso.md).
- **Boltz request-invariant conditioning:** all 48 native output tensors match
  bytes on CPU and MPS, including all 12 timed pairs. MPS timing is neutral or
  inconsistent; CPU affinity improves slightly in three pairs. The adapter is
  optional because it adds request-cache memory and requires explicit immutable
  ownership.
  [Complete results](../experiments/boltz2-request-runtime/lean/README.md).
- **STATE SE lossless original-table storage:** a portable representation restores
  all original float32 row bytes before inference and preserves the broader
  original API. Fresh CPU and MPS reloads each pass 377 tensor byte comparisons.
  Registered model state falls by 6.31%, but the small MPS timing check is
  4.89–5.44 times slower. It is an opt-in storage mode; the faster 28.67% CPU
  parameter reduction and its MPS guard remain separate. [STATE guide](state.md).
- **STATE SE scalar piecewise count encoder:** a real-arithmetic hinge
  representation saves 502 values, but failed 4 of 105 complete output comparisons
  (maximum absolute error 2.66e-5). The candidate was not exported.
- **OmniCell finite router:** actual weights give a larger resident model, and a
  targeted near-tie changes expert selection with 0.2503 embedding error despite
  passing local logit checks. Full-output sensitivity overrides the local gate.
- **OmniCell shared scalar experts:** saves only 2,048 values in real arithmetic;
  large continuous inputs fail the float32 output gate. No whole-model gain claimed.
- **stFormer GeneEncoder:** a source definition outside the active construction
  cannot establish a reachable compression opportunity. Trained equality of its
  actual copied position tables remains unverified.
- **STATE SE shared projected table plus norms:** 38.2% parameter reduction, but
  the actual dataset classifier amplified rounding errors above the chosen gate.
  CPU and MPS diagnostics remain rejected. Local agreement did not suffice.
- **STATE SE final tables on MPS:** CPU-built tables changed some complete outputs
  beyond tolerance. The artifact is restricted to CPU float32; raw-vector-only
  MPS agreement does not qualify the complete adapter.
- **Higher-precision offline STATE projection:** rounding closer to exact real
  arithmetic did not consistently match the original float32 program better.
- **Attention gauge fixing:** smaller Mol-JEPA files in an experiment, but dense
  reconstruction caches increased resident memory and CPU timing was neutral.
- **Expanded optional UMA attention:** useful full-modality experiment, excluded
  from the SMILES artifact because that route never visits UMA.
- **Coalesced tensor transfers:** much faster transport alone, mixed complete-model
  timing; optional and off by default. Values and requested attentions match.
- **Residual/LayerNorm kernel fusion:** no dependable additional gain in the
  measured complete workload; not enabled.
- **STATE ST packed size:** its removed all-zero table already compressed very
  well. The earlier equally packed comparison was 10,392 bytes larger after the
  rewrite. The claim is resident/raw weight saving, not smaller packed files.

## Fine-tuning experiments

An isolated experiment preserves simultaneous SGD on two consecutive bias-free
linear maps using their product and two Gram matrices. Exact rational checks and
12 twenty-step CPU trajectories pass; the worst observed float32 output difference
is 2.15e-6. A fixed hidden subspace gives a simpler alternative: four trajectories,
including momentum and weight decay, retain the same effective computation while
reducing a constructed width from 64 to 12.

These experiments test conditional mathematical identities. They do not
establish a production optimizer, a new theorem or a biological-model speedup.
Intervening nonlinearities, observable hidden states and coordinatewise Adam
updates fall outside the tested contract. The required Gram matrices can also
be larger than the source weights. See [proof, code and complete results](../experiments/gram_sgd/gram-sgd-note.md).

The Mol-JEPA component experiment trains only the first graph layer's query/key
weights and biases. The input and shared edge projections stay frozen. Two fixed
orthonormal subspaces reduce trainable Q/K coefficients from 4,202,496 to 243,024. The full tested component,
including retained value, edge, skip and input weights, falls from 6,678,528 to
2,750,833 values. Twenty molecular SGD steps pass in float32 and float64; the
worst molecular convolution-output difference is 1.43e-6.

This does not further shrink the bilinear inference artifact. One
arbitrary-input float32 raw-logit stress check fails, and unfreezing either
projection introduces directions outside the retained subspace. Adam and
unrestricted whole-model fine-tuning are not covered. The [derivation and
reproduction](../experiments/moljepa_subspace_sgd/note.md) retain both positive
and negative evidence; no fine-tuning speed claim is made.

## Evidence boundaries

Whole-model probes compare every requested numerical output from trained
checkpoints, including dtype and shape. They establish agreement on those cases,
not biological quality or equality for every possible input. Reassociated
float32 arithmetic can depend on tensor shape and backend, so each artifact
records its validated device, inputs, tolerance and helper-API limits.

For Boltz-2, the official checkpoint hashes and byte equality of all 5,019
common tensors are verified. Immutable parameter-storage sharing reduces joint registered state from
4,087,121,944 to 2,061,868,568 bytes while retaining both native models' logical
parameters and operations. All 48 tested tensor outputs match byte for byte on
CPU and MPS at standard sampling schedules, including after fresh portable
reload. The [Boltz-2 report](boltz2.md) gives the full checks. This reduces joint
memory/storage; it does not speed up a single model.

A separate [general request compiler](../experiments/request_fx/README.md)
propagates declared static inputs through an audited pure FX graph and evaluates
those branches once with their original operations. Its 25 focused tests and
8 actual Boltz AdaLN CPU byte checks pass. All original weights and temporary
constants are counted. Guard overhead is material, so this remains a derivation
and correctness experiment, with no complete-model speed claim.

The optional [lossless table codec](frozen-tables.md) also preserves every stored
float32 bit. Its earlier prototype reduced NovoMolGen's extra profile-table
storage by 26.46% CPU and 35.74% MPS, adding a small decode cost; it is not enabled
automatically. [STATE MPS sampling](../experiments/state_mps_profiles/README.md)
found five encoder row-count regimes and a separate normalized singleton case.
Dense complete tables for those regimes would increase storage by 47.8%; no
full-domain MPS adapter was built or promoted from that sample.

X-Cell and OmniCell are removed from the work plan, and stFormer is set aside.
Bioptimus remains gated after the supplied-token access check and is deferred.
Other NovoMolGen variants are out of scope. Their older audits remain historical
evidence; they are not outstanding tasks or compressed-model support claims.

This search cannot prove that no further optimization exists. It stops when
the applicable candidates have been tested and each remaining idea has a specific
obstacle: a missing hypothesis, failed output check, unavailable asset or
unmeasured tradeoff. Unsupported cases retain their original execution. A new
checkpoint or backend can justify testing a rejected candidate again.
