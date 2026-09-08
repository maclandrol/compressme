"""Build a portable compressed real State ST checkpoint, then check its API.

Run with .venv-state/bin/python (Transformers 4.52.3). The output
contains audited architecture source, JSON hyperparameters and compressed
tensor-only weights. The original Lightning checkpoint is needed only here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from compressme import CompressionResult, deduplicate_embeddings
from compressme.state_io import state_factory, load_state_st
from safetensors.torch import save_file
from unittest.mock import patch
from compressme.validation import compare_outputs


def probes(model):
    with torch.random.fork_rng():
        torch.manual_seed(7923)
        for count, padded, onehot in [(1, False, False), (7, False, True),
                                      (64, True, False), (128, True, True)]:
            counts = torch.randint(0, 8, (count, model.input_dim)).float()
            basal = torch.log1p(counts / counts.sum(-1, keepdim=True) * 10000)
            perturbations = torch.nn.functional.one_hot(
                torch.randint(model.pert_dim, (count,)), model.pert_dim).float()
            labels = torch.randint(model.batch_dim, (count,))
            batch = {"ctrl_cell_emb": basal, "pert_emb": perturbations,
                     "batch": torch.nn.functional.one_hot(labels, model.batch_dim).float()
                     if onehot else labels, "pert_name": ["numerical_probe"] * count}
            yield batch, {"cells": count, "padded": padded,
                          "batch_labels_onehot": onehot}


def check_outputs(before, after):
    assert before.keys() == after.keys()
    metrics = compare_outputs(before, after)
    assert metrics and all(item['bitwise'] for item in metrics.values()), 'Output bytes changed'
    errors = {}
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after[name]), name
            errors[name] = float((value - after[name]).abs().max())
        else:
            assert value == after[name], name
    return errors


def build(checkpoint, architecture, output, *, baseline_output=None):
    checkpoint, architecture, output = map(Path, (checkpoint, architecture, output))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(architecture, output / "architecture", dirs_exist_ok=True)
    model = state_factory(output / "architecture")
    original = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    model.load_state_dict(original["state_dict"], strict=True)
    del original
    candidate, report = deduplicate_embeddings(model)
    report.update({
        "target": "State ST-HVG-Replogle K562",
        "checkpoint_repository": "arcinstitute/ST-HVG-Replogle",
        "checkpoint_revision": "bb6a9562cbbf1fd152df14cc53b4cc7517c77175",
        "checkpoint_file": "fewshot/k562/checkpoints/final.ckpt",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
        "source_inference_tensor_bytes": sum(t.numel() * t.element_size()
                                             for t in model.state_dict().values()),
        "compressed_inference_tensor_bytes": sum(t.numel() * t.element_size()
                                                 for t in candidate.state_dict().values()),
        "input_contract": "Original State HVG predict_step and token embedding IDs preserved; frozen weights",
        "validation_scope": "Synthetic numerical API checks on actual pretrained weights; no biological benchmark",
        "performance_scope": "Stored/resident weight reduction; this table is bypassed by predict_step so no active-FLOP saving claimed",
        "validation_dtype": "float32", "validation_device": "cpu", "cases": []})
    stored_cases = {}
    for i, (batch, case) in enumerate(probes(model)):
        with torch.inference_mode():
            before = model.predict_step(batch, 0, padded=case["padded"])
            after = candidate.predict_step(batch, 0, padded=case["padded"])
        case["maximum_absolute_errors"] = check_outputs(before, after)
        case["all_outputs_bitwise_equal"] = True
        report["cases"].append(case)
        for name, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                stored_cases[f"case{i}.input.{name}"] = tensor.contiguous()
        for name, tensor in before.items():
            if isinstance(tensor, torch.Tensor):
                stored_cases[f"case{i}.output.{name}"] = tensor.contiguous()
    # This fixture file is validation evidence, not part of the portable artifact.
    references = output.parent / "compressme-state-reload-references.safetensors"
    save_file({name: tensor.clone() for name, tensor in stored_cases.items()}, str(references))
    CompressionResult(candidate, report).save(output, packing=True)
    if baseline_output is not None:
        CompressionResult(model, {"target": report["target"], "method": "unchanged baseline"}).save(
            baseline_output, packing=True)
        report["baseline_packed_weights_bytes"] = (Path(baseline_output) / "model.cmppack").stat().st_size
    with patch("torch.load", side_effect=AssertionError("Original checkpoint access is forbidden at reload")):
        reloaded = load_state_st(output)
    for batch, case in probes(model):
        with torch.inference_mode():
            expected = model.predict_step(batch, 0, padded=case["padded"])
            actual = reloaded.predict_step(batch, 0, padded=case["padded"])
        check_outputs(expected, actual)
    ids = torch.tensor([0, 1, 123, 31999])
    assert torch.equal(model.transformer_backbone.embed_tokens(ids),
                       reloaded.transformer_backbone.embed_tokens(ids))
    assert reloaded.transformer_backbone.embed_tokens.weight.shape == (32000, 328)
    report["reload_all_outputs_bitwise_equal"] = True
    report["reload_forbids_torch_load"] = True
    report["token_index_api_preserved"] = True
    report["packed_weights_bytes"] = (output / "model.cmppack").stat().st_size
    report["portable_directory_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report["validation_references"] = str(references)
    report["portable_directory"] = str(output)
    (output.parent / "compressme-state-artifact-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    build(args.checkpoint, args.architecture, args.output, baseline_output=args.baseline_output)
