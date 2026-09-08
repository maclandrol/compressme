"""Fresh-process full-output checks without opening original model weights."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import load_file
from compressme.state_io import load_state_st
from compressme.validation import compare_outputs


def main(artifact, references, output):
    torch.set_num_threads(4)
    saved = load_file(references)
    with patch("torch.load", side_effect=AssertionError("Original checkpoint forbidden")):
        model = load_state_st(artifact)
    report = {"fresh_process": True, "original_torch_load_forbidden": True,
              "device": "cpu", "dtype": "float32", "cases": []}
    for i, (cells, padded, onehot) in enumerate([(1, False, False), (7, False, True),
                                               (64, True, False), (128, True, True)]):
        batch = {name: saved[f"case{i}.input.{name}"]
                 for name in ("ctrl_cell_emb", "pert_emb", "batch")}
        batch["pert_name"] = ["numerical_probe"] * cells
        with torch.inference_mode():
            result = model.predict_step(batch, 0, padded=padded)
        errors = {}
        for name, value in result.items():
            if isinstance(value, torch.Tensor):
                reference = saved[f"case{i}.output.{name}"]
                assert compare_outputs(reference, value)['output']['bitwise'], name
                errors[name] = float((value - reference).abs().max())
            elif name == "pert_name":
                assert value == batch[name]
        report["cases"].append({"cells": cells, "padded": padded, "onehot": onehot,
                                "all_outputs_bitwise_equal": True,
                                "maximum_absolute_errors": errors})
    layer = model.transformer_backbone.embed_tokens
    assert layer.weight.shape == (32000, 328)
    assert layer._row.numel() == 328 and not layer._row.requires_grad
    assert compare_outputs(torch.zeros(4, 328), layer(torch.tensor([0, 1, 123, 31999])))['output']['bitwise']
    report["token_index_api_preserved"] = True
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["stored_embedding_values"] = layer._row.numel()
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.artifact, args.references, args.output)
