# NovoMolGen first-QKV finite compilation: bounded real-weight result

The 32M AtomWise native Llama checkpoint has a small finite token vocabulary,
so its first RMSNorm and Q/K/V projections can be evaluated ahead of inference
for every token. This removes those original projection weights and retains
the downstream native model. No quantization, calibration data, training,
distillation or stored molecule outputs are used.

The result is a research adapter for **two-dimensional token IDs in frozen
float32 evaluation**. It explicitly rejects `inputs_embeds` and training. This
does not establish equivalence with the separate custom NovoMolGen API, nor
support for all options of the complete native API. The original native model's
broader API smoke is separately preserved in `README.md` and four JSON reports.

## Numerical discovery and retained failures

An ordinary 84-row QKV table failed the complete CPU gate for all singleton
tokens. Mathematically the row function is unchanged, but a single-row native
matrix-vector operation rounds differently from a large matrix multiplication.
The discrepancy is amplified by later layers.

The CPU candidate therefore stores one bulk table and a separate singleton
table. Each singleton entry is computed by the original RMSNorm and projection
on a `(1,1,512)` tensor. On the tested MPS implementation a third table is needed
for 2–15 total tokens; each entry is obtained from the original operations on
two identical rows. The MPS policy selects singleton, small or bulk rows for
counts 1, 2–15 or at least 16 respectively.

This threshold is an empirical result for the pinned model, dtype and installed
backend. It is not a theorem about every MPS version or hardware generation.
Backend/shape-specific compilation must be validated again when those change.

| Candidate | Backend | Complete gate | Largest absolute error |
|---|---|---|---:|
| One bulk table | CPU | Failed 84 of 104 cases | 1.1635e-4 |
| Bulk + singleton, CPU-built | CPU | Passed 123 of 123 | 0 |
| Bulk + singleton, CPU-built | MPS | Failed 123 of 123 | 1.2970e-4 |
| Bulk + singleton, MPS-built | MPS | Failed 33 of 123 | 1.8311e-4 |
| Bulk + singleton + small, MPS-built | MPS | Passed 123 of 123 | 0 |

All failed reports are retained. The acceptance rule remains separate
`max_abs <= 1e-5` and `relative_l2 <= 1e-5` for every tensor, plus exact agreement
for discrete outputs. No tolerance was widened.

For each successful backend, 134 additional cases test all factor pairs of
selected token counts 2–32, including exact boundaries 13, 14, 15 and 16,
with contiguous and strided token IDs and position IDs. These also pass
bitwise. Combined, **257 cases and 12,599 tensor comparisons per backend** have
zero observed error. This is bounded empirical verification, not exhaustive
enumeration of all possible complete sequences.

The main gate includes all 84 single tokens, lengths through 2,048, labels and
loss, dictionary and tuple results, layer hidden states, attention maps,
legacy/Dynamic/Static caches, B1/B2 decoding, and B1/B3 greedy, beam and seeded
sampling. Attention maps are requested through length 128; longer probes keep
all hidden states but omit attention maps. Generation prompts are explicitly
left-padded and include a single BOS token. Sampling finite values and infinity
masks are both checked, along with generated token IDs and decoded strings.

## Honest storage accounting

Each packed QKV table has `84 × 1,536 = 129,024` float32 values: 516,096 bytes.
All copies count toward resident state; the original token embedding remains.
The model retains 416 buffer values before and after.

| Model | Parameter values | Parameter bytes | Reduction |
|---|---:|---:|---:|
| Original | 31,556,096 | 126,224,384 | — |
| CPU, two tables | 31,027,200 | 124,108,800 | 528,896 values / 1.676% |
| MPS, three tables | 31,156,224 | 124,624,896 | 399,872 values / 1.267% |

These are resident parameter values and raw parameter bytes, not compressed
checkpoint sizes, peak process RAM or allocator usage. This experiment does
not build or claim a portable production checkpoint.

## Runtime result

Nine randomized interleaved rounds follow three warmup pairs. CPU uses four
threads; MPS is synchronized before and after each measurement. Original and
candidate use the same input, cache route, dtype and requested outputs. Every
benchmark has a bitwise output gate before timing. Inputs are already on the
target device; tokenization and device transfer are excluded.

The observed benefit is small and mixed: CPU median speed ratios span
0.979–1.029×, MPS 0.997–1.052×. These runs do not show a material acceleration.

| Native API call | CPU original → candidate (ms) | MPS original → candidate (ms) |
|---|---:|---:|
| Prefill B1 × 32 | 5.586 → 5.703 | 4.076 → 3.966 |
| Prefill B1 × 32, all hidden states/attentions | 5.587 → 5.511 | 4.384 → 4.345 |
| Prefill B4 × 32 | 14.378 → 14.269 | 8.761 → 8.331 |
| Prefill B1 × 512 | 51.146 → 50.327 | 23.751 → 23.418 |
| Decode B1, prefix 32 | 2.762 → 2.810 | 4.310 → 4.251 |
| Decode B1, all hidden states/attentions | 2.806 → 2.743 | 4.250 → 4.265 |
| Greedy B1, up to 12 new tokens | 35.658 → 35.381 | 101.284 → 100.458 |
| Greedy B1, all hidden states/attentions | 36.052 → 35.020 | 100.202 → 99.544 |

The table reports medians, not uncertainty intervals. Full samples, means and
sample standard deviations are retained in `benchmark-cpu.json` and
`benchmark-mps.json`. For example, MPS B4×32 prefill sample SD is 0.270 ms for
the original and 0.519 ms for the candidate; plain greedy generation SD is
2.468 ms versus 3.689 ms. Small median differences should not be interpreted as
established improvements across runs or machines.

## Reproduction and files

Use the isolated environment described in `README.md`. `prototype.py` loaders
take checkpoint/config paths and verify checkpoint, config and installed native
model source SHA256 values. Hashing is streamed and compatible with Python 3.10;
the actual executions used Python 3.12.14, PyTorch 2.14.0 and Transformers 4.46.2.

- `single-table/`: original prototype and failed CPU report.
- `singleton/`: CPU two-table prototype; successful CPU main/boundary reports;
  failed MPS reports with CPU-built and MPS-built tables.
- `regimes/`: MPS three-table prototype and successful main/boundary reports.
- `benchmark.py`: selects the CPU two-table or MPS three-table implementation.
- `benchmark-{cpu,mps}.json`: all timing samples and correctness gates.

The main probe and benchmark require `--checkpoint`, `--config`, `--device` and
`--output`; the probe also takes `--backend eager --table-device cpu|mps`.
Boundary probes require the same paths and target device. No model weights are
copied into this evidence directory. Both successful implementations are
research-only and preserve their explicit token-ID contract.

