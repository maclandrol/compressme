# NovoMolGen original native API smoke

This is a portable copy of the original 32M AtomWise model's API smoke test.
It loads native `LlamaForCausalLM` from the installed Transformers 4.46.2
package. It executes no model repository Python code. It is **not** a
compression result, speed benchmark or molecular-quality evaluation.

## Environment and local inputs

Use a separate Python 3.12 environment. The recorded runs used Python 3.12.14.

```bash
python3.12 -m venv .venv-novo-native
. .venv-novo-native/bin/activate
python -m pip install -r requirements.txt

python native_smoke.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer-source /path/to/hf-checkpoint \
  --check-inputs-only

python native_smoke.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer-source /path/to/hf-checkpoint \
  --device cpu --attention eager \
  --output /path/to/new-native-cpu-eager.json
```

`--tokenizer-source` must contain `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` and `generation_config.json`. Supply the native
`hf-checkpoint` files from
[`chandar-lab/NovoMolGen_32M_SMILES_AtomWise`](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/tree/dcd3f59261bebf84142c13617d4e129b4b0d0fdc)
at revision `dcd3f59261bebf84142c13617d4e129b4b0d0fdc`.
The script verifies pinned checkpoint, config and tokenizer/generation file
hashes before model loading. It also refuses a Transformers version other
than 4.46.2 or tokenizers other than 0.20.3, and verifies the installed
`modeling_llama.py` source hash before constructing a model. All hashes appear
in `provenance.json` and the script. Inputs are not downloaded automatically.

The process allocates its own temporary Hugging Face cache before runtime
imports. It never uses or migrates a shared/default cache. All model and
tokenizer reads are local-only. Temporary cache files are cleaned up on normal
process exit. Use a fresh process before importing other Hugging Face libraries.

Use `--device mps` for an available Apple GPU, and `--attention sdpa` for that
native attention backend. `--threads` defaults to 4. `--output` is required for
a smoke run; existing output files are refused unless `--overwrite` is explicit.
A completed report is written even if an individual stage fails, and the CLI
then exits with a nonzero status.

The reusable helper is
`load_native(checkpoint, config, tokenizer_source, *, device='cpu', attn_implementation='eager')`.
It follows the same local-input, runtime-version and private-cache checks.

## Exact scope of the smoke

Twelve stages exercise labels/loss, logits, every hidden state and attention
map, tuple outputs, equivalent and perturbed continuous `inputs_embeds`, legacy
and dynamic/static caches, one-token continuation, last-logit selection,
plain generation, generation from embeddings, greedy and two-beam generation,
and seeded sampling with a repeated seed. Inputs are two unequal-length,
left-padded SMILES prompts; generation requests eight new tokens.

The model has 31,556,096 unique parameters, loaded as float32. This bounded test
does not establish behavior for every sequence length, processor, cache type,
device, or training operation. Stage completion records API calls and their
explicit assertions; cached-versus-uncached floating comparisons are reported
separately and are not all asserted equal.

Relevant original-model behavior is preserved:

- The tokenizer call suppresses `token_type_ids`, which native generation
  rejects as unused arguments.
- In this Transformers version, requesting attention maps selects eager
  fallback even when SDPA is configured. Plain generation exercises native
  SDPA when selected.
- Cached/full and trimmed/full outputs can differ numerically within the
  unmodified model. Candidate validation must match the original backend,
  shapes, cache route and requested outputs. These observations do not justify
  relaxing a compression gate.
- Seeded sampling is checked within each configuration; cross-device sequence
  identity is not claimed.

## Historical evidence

`historical/` contains the four original CPU/MPS × eager/SDPA reports, copied
byte-for-byte with `historical/SHA256.json`. All twelve stages completed in
those recorded runs. The reports retain their original environment paths as
historical metadata. They were not regenerated or relabeled as new executions
of this portable copy. New reports use portable class-source metadata.

The refactor preserves all numerical stage function bodies. It changes local
input discovery, provenance checks, cache isolation and CLI/report handling.
There are no model weights or Transformers implementation files in this archive.

`portable-cpu-eager.json` records one fresh execution of this portable copy.
All twelve stages completed, and every stage report matches the original
CPU-eager report exactly. `portable-verification.json` records that comparison
and the successful input/version checks. The other three backend/device
combinations were not rerun as part of the portability refactor.
