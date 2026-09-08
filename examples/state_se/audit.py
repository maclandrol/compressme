"""Compare every numerical State SE output on real pretrained weights.

Inputs are synthetic gene-ID/count probes, not biological quality evaluation.
Projection compilation enumerates the complete published token vocabulary.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from original import load_state_se
from adapter import StateSETokenAdapter, FinalBranchLookup

STRICT = True


def difference(reference, candidate):
    if reference is None:
        assert candidate is None
        return None
    close = torch.isclose(reference, candidate, atol=1e-5, rtol=1e-5)
    if STRICT:
        torch.testing.assert_close(reference, candidate, atol=1e-5, rtol=1e-5)
    reference_cpu = reference.detach().cpu().double()
    diff = reference_cpu - candidate.detach().cpu().double()
    return {"shape": list(reference.shape), "gate_1e5_passed": bool(close.all()),
            "gate_1e5_mismatches": int((~close).sum()), "maximum_absolute_error": float(diff.abs().max()),
            "relative_l2_error": float(torch.linalg.vector_norm(diff) /
                torch.linalg.vector_norm(reference_cpu).clamp_min(1e-30))}


def baseline_forward_ids(model, ids, mask, counts, ds):
    raw = F.normalize(model.pe_embedding(ids), dim=2)
    raw[:, 0, :] = model.cls_token.expand(raw.size(0), -1)
    if model.dataset_token is not None:
        raw = torch.cat((raw, model.dataset_token.expand(raw.size(0), -1).unsqueeze(1)), dim=1)
        mask = torch.cat((mask, torch.zeros(mask.size(0), 1, device=mask.device).bool()), dim=1)
    return model(raw, mask, counts=counts, dataset_nums=ds)


def make_batches(model, extended=False):
    with torch.random.fork_rng():
        torch.manual_seed(19083)
        cases = [(1, 1, 3, False), (1, 8, 7, True), (4, 32, 13, True), (2, 128, 31, False)]
        if extended:
            cases.extend([(1, 2, 1, True), (2, 512, 31, True), (1, 2048, 31, True)])
        for B, T, Q, use_counts in cases:
            ids = torch.randint(model.pe_embedding.num_embeddings, (B, T))
            ids[:, 0] = model.cfg.dataset.cls_token_idx
            if T > 2:
                ids[:, -1] = 0
            X = torch.randint(model.pe_embedding.num_embeddings, (B, Q))
            mask = torch.rand(B, T) > 0.6
            counts = torch.rand(B, T) * 12 if use_counts else None
            yield (ids, X, torch.rand(B, Q), torch.arange(B), torch.rand(B, Q),
                   mask, torch.full((B,), 4.0), counts, torch.arange(B, dtype=torch.int32))


def audit(model, candidate, report, device="cpu", extended=False):
    model = model.to(device)
    candidate = candidate.to(device)
    records = []
    with torch.inference_mode():
        for batch in make_batches(model, extended=extended):
            batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
            original_batch = model._compute_embedding_for_batch(batch)
            compressed_batch = candidate._compute_embedding_for_batch(batch)
            record = {"cells": batch[0].shape[0], "sentence_length": batch[0].shape[1],
                      "task_genes": batch[1].shape[1], "counts": batch[7] is not None,
                      "batch_outputs": [difference(a, b) for a, b in zip(original_batch, compressed_batch)]}
            original_all = baseline_forward_ids(model, batch[0], batch[5], batch[7], batch[8])
            compressed_all = candidate.forward_gene_ids(batch[0], batch[5], batch[7], batch[8])
            record["all_token_cell_dataset_outputs"] = [difference(a, b) for a, b in zip(original_all, compressed_all)]
            record["dataset_class_logits"] = difference(model.dataset_encoder(original_all[2]),
                                                        candidate.dataset_encoder(compressed_all[2]))
            record["dataset_latents"] = difference(model.dataset_embedder(original_all[2]),
                                                   candidate.dataset_embedder(compressed_all[2]))
            # Preserve the complete pretrained decoder, including read-depth and dataset context.
            original_ds = model.dataset_embedder(original_all[2])
            compressed_ds = candidate.dataset_embedder(compressed_all[2])
            original_merge = model.resize_batch(original_all[1], original_batch[0][0], task_counts=batch[6], ds_emb=original_ds)
            compressed_merge = candidate.resize_batch(compressed_all[1], compressed_batch[0][0], task_counts=batch[6], ds_emb=compressed_ds)
            record["gene_decoder_logits"] = difference(model.binary_decoder(original_merge), candidate.binary_decoder(compressed_merge))
            if candidate.preserve_raw_forward:
                raw = torch.randn(batch[0].shape[0], batch[0].shape[1] + 1, model.cfg.tokenizer.token_dim, device=device)
                raw_mask = torch.zeros(raw.shape[:2], dtype=torch.bool, device=device)
                raw_before = model(raw, raw_mask, counts=None)
                raw_after = candidate(raw, raw_mask, counts=None)
                record["arbitrary_raw_vector_outputs"] = [difference(a, b) for a, b in zip(raw_before, raw_after)]
                record["arbitrary_raw_gene_encoder"] = difference(model.gene_embedding_layer(raw), candidate.gene_embedding_layer(raw))
            records.append(record)
    return {"device": device, "dtype": "float32", "all_outputs_atol": 1e-5, "all_outputs_rtol": 1e-5,
            "actual_pretrained_weights": True, "input_kind": "Synthetic numerical gene-ID/count probes",
            "biological_quality_benchmark": False, "lookup_report": report,
            "parameters_before": sum(p.numel() for p in model.parameters()),
            "parameters_after": sum(p.numel() for p in candidate.parameters()), "cases": records}

