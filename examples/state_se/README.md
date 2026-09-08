# Reproduce State SE numerical artifacts

Use the separate State environment with the pinned SE-100M safetensors checkpoint.
The portable artifact includes hash-verified architecture, configuration,
vocabulary and licenses; production source is not needed for normal reload.

## Lossless original model: CPU and Apple MPS

```bash
.venv-state/bin/python examples/state_se/build_lossless.py \
  --checkpoint /path/to/SE-100M/model.safetensors \
  --architecture /path/to/pinned-state-architecture \
  --output /path/to/new-lossless-artifact
```

All three paths are explicit. The builder packs original frozen gene rows and
checks every reconstructed row. It does not change normalization, encoder,
transformer, heads, raw 5,120-dimensional vector inputs or original gene-name behavior. The
result is smaller in registered storage but slower to run because requested
rows must be decoded. See [State results](../../docs/state.md) and
[the general storage operator](../../docs/packed-embeddings.md).

Create an options file with local paths:

```json
{"artifact": "/path/to/new-lossless-artifact", "checkpoint": "/path/to/SE-100M/model.safetensors"}
```

Then compare every native output and the full original embedding tensor after
portable reload, including a source self-repeat and explicit byte comparison:

```bash
.venv-state/bin/python examples/validate_backends.py \
  --provider examples/state_se/lossless_backend.py:provider \
  --options-json /path/to/options.json --device mps --require-bitwise \
  --output /path/to/new-mps-report.json
```

Repeat with `--device cpu` and a different output path for CPU regression.
Existing reports are never overwritten. `benchmark_lossless.py` measures the
separate latency tradeoff with fresh inputs and paired output checks.

## Historical CPU finite-table artifact

`build.py` recreates the smaller CPU-only finite-table representation. Supply the
architecture and tokenizer directories explicitly; do not rely on its legacy
`vendor/` defaults:

```bash
.venv-state/bin/python examples/state_se/build.py \
  --checkpoint /path/to/SE-100M/model.safetensors \
  --architecture /path/to/pinned-state-architecture \
  --tokenizer-source /path/to/pinned-state-tokenizer \
  --output /path/to/new-cpu-finite-artifact
```

The existing artifact contains these directories as `architecture/` and
`tokenizer_source/`. This path enumerates both finite encoder branches and checks
numerical outputs plus reload. `--benchmark` adds a short CPU timing comparison.
Its MPS numerical gate failed and remains enforced.

The public loader for both representations is `compressme.load_state_se`;
representation is read from the manifest. CPU AnnData ingestion/export is
validated separately for the finite-table artifact through
`compressme.load_state_se_encoder`. The new lossless MPS numerical evidence does
not extend that AnnData pipeline to GPU. See [the State guide](../../docs/state.md)
for its exact scope and reproduction instructions.
