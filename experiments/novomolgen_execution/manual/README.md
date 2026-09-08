# NovoMolGen native runtime evidence

The official 32M AtomWise native checkpoint executes locally on CPU and Apple
MPS using the native `LlamaForCausalLM` implementation from Transformers 4.46.2.
No Hub Python code is executed. This is an original-model API smoke, not a
compression result, speed benchmark, or molecular-quality evaluation.

## Reproducible environment

The existing `.venv-state` environment is unchanged. An isolated overlay contains
only `transformers==4.46.2` and `tokenizers==0.20.3`, installed without dependencies:

```sh
rtk proxy /Users/manu/.local/bin/uv pip install --no-deps \
  --python /Users/manu/Code/compressme/.venv-state/bin/python \
  --target /private/tmp/compressme-agent-novomolgen-runtime/deps \
  --cache-dir /private/tmp/compressme-uv-cache \
  'transformers==4.46.2' 'tokenizers==0.20.3'

rtk proxy env \
  PYTHONPATH=/private/tmp/compressme-agent-novomolgen-runtime/deps \
  HF_HOME=/private/tmp/compressme-agent-novomolgen-runtime/hf-cache \
  /Users/manu/Code/compressme/.venv-state/bin/python \
  /private/tmp/compressme-agent-novomolgen-runtime/native_smoke.py \
  --device cpu --attention eager
```

Use `--device mps` for the GPU and `--attention sdpa` for the alternative native
backend. MPS requires host GPU access; it is unavailable inside this task's
filesystem sandbox. `HF_HOME` avoids the older Transformers release attempting
to migrate the user's default Hugging Face cache.

Verified versions: Python 3.12.14, PyTorch 2.14.0, Transformers 4.46.2,
tokenizers 0.20.3. The script's `load_native(device, attn_implementation)` helper
strictly loads the original weights and returns the model and tokenizer.

## Provenance

- Model: `chandar-lab/NovoMolGen_32M_SMILES_AtomWise`, native `hf-checkpoint`
  revision `dcd3f59261bebf84142c13617d4e129b4b0d0fdc`.
- Local checkpoint: `/private/tmp/compressme-models/novomolgen32-atomwise/model.safetensors`.
- Checkpoint SHA256: `3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699`.
- 31,556,096 unique parameters, all loaded as float32. Native config: 84 tokens,
  512 hidden dimensions, 12 layers, 8 attention heads, 1,024 intermediate
  dimensions, 2,048 maximum positions, no tied input/output embedding weights.
- The installed `transformers/models/llama/modeling_llama.py` is byte-identical
  to the pinned source in `vendor/novomolgen_audit/sources`:
  SHA256 `733e7625fec5abcfa416ab9b866cb91875d41007f2e31c3e90d2ae0111ae98f7`.

## Executed checks

All twelve stages completed on all four combinations: CPU eager, MPS eager,
CPU SDPA and MPS SDPA. Reports are `native-{cpu,mps}-{eager,sdpa}.json`.

1. Forward with labels/loss, logits, all thirteen hidden states, all twelve
   attention maps and cache.
2. Tuple outputs (`return_dict=False`) compared with dictionary outputs.
3. `inputs_embeds`, including a continuous perturbation of token embeddings.
4. Legacy tuple-cache prefill and one-token decoding.
5. `DynamicCache` prefill and one-token decoding.
6. `StaticCache` prefill and one-token decoding.
7. `num_logits_to_keep=1`.
8. Plain greedy generation, exercising native SDPA when selected.
9. Generation from `inputs_embeds` without `input_ids`.
10. Greedy generation with scores, logits, hidden states and attention maps.
11. Two-beam generation with the same observable outputs.
12. Seeded sampling (`top_k=10`, temperature 0.8) and an identical-seed repeat.

The inputs are two unequal-length, left-padded SMILES prompts (`<bos>CCO` and
`<bos>c1ccccc1`). Generation requests eight new tokens. This bounded smoke does
not cover every sequence length, custom generation processor, cache type, or
training operation. Passing means that the API calls and explicit assertions
completed; the separately reported floating-point comparisons below do not
assert cross-route equality.

## API and numerical boundaries

- The released generic fast tokenizer emits `token_type_ids` by default.
  Native `LlamaForCausalLM.generate` rejects these unused keyword arguments.
  Request `return_token_type_ids=False` or pass only the supported fields.
- SDPA does not return attention maps in this version. Requesting
  `output_attentions=True` invokes the original model's eager fallback. Eager
  is the appropriate reference when validating all attention outputs.
- Forward receives `past_key_values`; a decoder layer/attention receives the
  singular `past_key_value`. Layers also accept `cache_position` and precomputed
  rotary `position_embeddings`. A replacement must preserve these paths.
- Original CPU eager cached versus uncached final logits differ by at most
  `1.6570e-5` on this probe (relative L2 `5.3816e-7`). Original MPS eager
  legacy/dynamic cached logits differ by at most `3.0041e-5` (relative L2
  `8.7019e-7`); static-cache drift is `2.8610e-5`. The MPS cached/full comparison
  fails `torch.allclose(atol=1e-5, rtol=1e-5)` despite using the same unmodified
  model. Compare a candidate with the original using the same backend, shapes,
  cache route and requested outputs; do not use this observation to loosen its
  accuracy gate.
- On MPS eager, selecting only the final logit row changes original-model
  logits by at most `2.0981e-5`; the full-versus-trimmed comparison passes the
  elementwise `1e-5` absolute-plus-relative test on this probe. CPU eager is
  bitwise equal for that comparison.
- Equivalent token embeddings produce bitwise-equal logits to `input_ids`
  within each tested backend. Seeded sampling repeats exactly within each
  tested configuration. No universal cross-device generation claim is made.

