"""Complete-output research gate for the token-only first-QKV prototype."""
import argparse
from collections.abc import Mapping
import json
import inspect, platform, transformers
from pathlib import Path
import torch
from transformers import AutoTokenizer, DynamicCache, StaticCache
from compressme.validation import compare_outputs, seeded
from prototype import load_reference, make_candidate, _CURRENT, file_sha256


def observable(value):
    if isinstance(value, (DynamicCache, StaticCache)):
        layers = (value.to_legacy_cache() if isinstance(value, DynamicCache)
                  else tuple(zip(value.key_cache, value.value_cache)))
        return {"cache_type": type(value).__name__, "sequence_length": value.get_seq_length(),
                "layers": observable(layers)}
    if isinstance(value, Mapping):
        return {key: observable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(observable(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="eager")
    parser.add_argument("--table-device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = load_reference(args.checkpoint, args.config, device=args.device, backend=args.backend)
    candidate = make_candidate(reference, table_device=args.table_device)
    records = []

    def record(name, left, right):
        try:
            metrics = compare_outputs(observable(left), observable(right))
            failed = {path: values for path, values in metrics.items()
                      if values["max_abs"] > 1e-5 or values["relative_l2"] > 1e-5}
            records.append({"case": name, "accepted": not failed, "tensor_outputs": len(metrics),
                "max_abs": max((v["max_abs"] for v in metrics.values()), default=0),
                "max_relative_l2": max((v["relative_l2"] for v in metrics.values()), default=0),
                "failed_outputs": failed, "metrics": metrics})
        except (ValueError, TypeError, RuntimeError) as error:
            records.append({"case": name, "accepted": False, "error": str(error)})
        assert _CURRENT.get() is None
        print(name, "passed" if records[-1]["accepted"] else "FAILED", flush=True)

    def pair(name, **kwargs):
        with torch.inference_mode(), seeded(791):
            left = reference(**kwargs)
        with torch.inference_mode(), seeded(791):
            right = candidate(**kwargs)
        record(name, left, right)

    device = torch.device(args.device)
    cpu_random = torch.Generator().manual_seed(537)
    with torch.inference_mode():
        # Full vocabulary and singleton kernels, including every optional layer.
        for token in range(84):
            pair(f"singleton_{token}", input_ids=torch.tensor([[token]], device=device),
                 use_cache=False, output_attentions=True, output_hidden_states=True)
        for batch, length in [(1, 84), (2, 42), (4, 17), (1, 128), (1, 512), (1, 2048)]:
            ids = (torch.arange(84).reshape(batch, length) if batch*length == 84 else
                   torch.randint(0, 84, (batch, length), generator=cpu_random)).to(device)
            pair(f"all_outputs_{batch}x{length}", input_ids=ids, labels=ids,
                 use_cache=False, output_attentions=length <= 128, output_hidden_states=True)
        ids = torch.tensor([[1, 1, 2, 16, 19, 8], [2, 19, 10, 16, 19, 8]], device=device)
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], device=device)
        pair("left_padding", input_ids=ids, attention_mask=mask, labels=ids,
             output_attentions=True, output_hidden_states=True, use_cache=True)
        pair("explicit_positions", input_ids=ids, attention_mask=mask,
             position_ids=torch.arange(37, 43, device=device).expand(2, -1),
             output_attentions=True, output_hidden_states=True, use_cache=False)
        pair("tuple_last_two_logits", input_ids=ids, num_logits_to_keep=2,
             output_attentions=True, output_hidden_states=True, use_cache=True, return_dict=False)
        for batch in (1, 2):
            for mode in ("legacy", "dynamic", "static"):
                caches = ([None, None] if mode == "legacy" else
                    ([DynamicCache(), DynamicCache()] if mode == "dynamic" else
                     [StaticCache(config=reference.config, batch_size=batch, max_cache_len=10,
                                  device=device, dtype=torch.float32) for _ in range(2)]))
                for step in range(4):
                    step_ids = ids[:batch] if step == 0 else torch.full((batch, 1), 8+step, device=device)
                    step_mask = torch.cat([mask[:batch], torch.ones((batch, step), dtype=mask.dtype, device=device)], dim=-1)
                    positions = (step_mask.cumsum(-1)-1).clamp_min(0)
                    cache_position = torch.arange(ids.shape[-1], device=device) if step == 0 else torch.tensor([ids.shape[-1]+step-1], device=device)
                    outputs = []
                    for index, model in enumerate((reference, candidate)):
                        output = model(input_ids=step_ids, past_key_values=caches[index],
                            attention_mask=step_mask, position_ids=positions if step == 0 else positions[:, -1:],
                            cache_position=cache_position, use_cache=True,
                            output_attentions=True, output_hidden_states=True)
                        outputs.append(output)
                        caches[index] = output.past_key_values
                    record(f"{mode}_cache_b{batch}_step_{step}", *outputs)
        tokenizer = AutoTokenizer.from_pretrained(Path(args.config).parent, local_files_only=True)
        tokenizer.padding_side = "left"
        prompts = [tokenizer.bos_token+x for x in ["C", "CCO", "c1ccccc1"]]
        for generation_batch in (1, 3):
          inputs = tokenizer(prompts[:generation_batch], add_special_tokens=False, padding=True,
              return_tensors="pt", return_token_type_ids=False).to(device)
          for name, options in [("greedy", {"do_sample": False}),
                              ("beam", {"do_sample": False, "num_beams": 3}),
                              ("sample", {"do_sample": True, "top_k": 12, "top_p": 0.92, "temperature": 0.85})]:
            outputs = []
            for model in (reference, candidate):
                with seeded(815):
                    outputs.append(model.generate(**inputs, **options, max_new_tokens=12,
                        return_dict_in_generate=True, output_scores=True, output_logits=True,
                        output_attentions=True, output_hidden_states=True))
            # Sampling processors intentionally emit -inf in filtered scores.
            # Keep them observable by comparing finite values and exact masks.
            def generation_observable(output):
                result = observable(output)
                if "scores" in result:
                    scores = result["scores"]
                    result["score_finite_masks"] = tuple(torch.isfinite(x) for x in scores)
                    result["score_negative_infinity_masks"] = tuple(torch.isneginf(x) for x in scores)
                    result["scores"] = tuple(torch.where(torch.isfinite(x), x, torch.zeros_like(x)) for x in scores)
                result["decoded_sequences"] = tokenizer.batch_decode(output.sequences)
                return result
            record(f"generation_b{generation_batch}_{name}", *(generation_observable(x) for x in outputs))
    result = {"method": "first_layer_finite_qkv_research", "device": args.device,
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "checkpoint_sha256": file_sha256(args.checkpoint), "config_sha256": file_sha256(args.config),
        "model_source_sha256": file_sha256(inspect.getfile(type(reference))),
        "backend": args.backend, "table_device": args.table_device,
        "reference": "Native Hugging Face LlamaForCausalLM, Transformers4.46.2",
        "original_custom_novomolgen_api_equivalence_claim": False,
        "contract": "Two-dimensional token IDs only; no inputs_embeds fallback or production promotion",
        "absolute_tolerance": 1e-5, "relative_l2_tolerance": 1e-5,
        "parameters_before": sum(p.numel() for p in reference.parameters()),
        "parameters_after": sum(p.numel() for p in candidate.parameters()),
        "table_policy": "Separate original (1,1,512) per-token QKV table for input_ids.numel()==1; original bulk table otherwise",
        "bulk_table_values": candidate._first_qkv_rows.numel(),
        "singleton_table_values": candidate._first_qkv_singleton_rows.numel(),
        "buffers_after": sum(b.numel() for b in candidate.buffers()),
        "accepted": all(row["accepted"] for row in records),
        "failed_cases": [row["case"] for row in records if not row["accepted"]],
        "case_count": len(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
