"""Reproducible complete-output reload fingerprints and short CPU timings."""
import hashlib
import random
import statistics
import time
from types import SimpleNamespace
import torch


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
            tasks = torch.randint(model.pe_embedding.num_embeddings, (B, Q))
            mask = torch.rand(B, T) > 0.6
            counts = torch.rand(B, T) * 12 if use_counts else None
            yield (ids, tasks, torch.rand(B, Q), torch.arange(B), torch.rand(B, Q), mask,
                   torch.full((B,), 4.0), counts, torch.arange(B, dtype=torch.int32))


def fingerprint(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def output_fingerprints(model, *, extended=True):
    metadata = SimpleNamespace(pe_embedding=SimpleNamespace(num_embeddings=19790), cfg=model.cfg)
    cases = []
    with torch.inference_mode():
        for batch in make_batches(metadata, extended=extended):
            computed = model._compute_embedding_for_batch(batch)
            full = model.forward_gene_ids(batch[0], batch[5], batch[7], batch[8])
            values = {f"batch.{i}": value for i, value in enumerate(computed) if value is not None}
            values.update({f"full.{i}": value for i, value in enumerate(full) if value is not None})
            values["dataset_classifier"] = model.dataset_encoder(full[2])
            values["dataset_latents"] = model.dataset_embedder(full[2])
            merged = model.resize_batch(full[1], computed[0][0], task_counts=batch[6], ds_emb=values["dataset_latents"])
            values["gene_decoder"] = model.binary_decoder(merged)
            raw = torch.randn(batch[0].shape[0], batch[0].shape[1] + 1, 5120)
            raw_result = model(raw, torch.zeros(raw.shape[:2], dtype=torch.bool), counts=None)
            values.update({f"raw.{i}": value for i, value in enumerate(raw_result) if value is not None})
            values["raw_gene_encoder"] = model.gene_embedding_layer(raw)
            cases.append({"cells": batch[0].shape[0], "tokens": batch[0].shape[1],
                          "outputs": {name: fingerprint(value) for name, value in values.items()}})
    return cases


def benchmark(original, candidate, *, rounds=7):
    def new_batch(tokens):
        ids = torch.randint(19790, (1, tokens)); ids[:, 0] = 3
        tasks = torch.randint(19790, (1, 31))
        return (ids, tasks, torch.rand(1, 31), torch.zeros(1, dtype=torch.long), torch.rand(1, 31),
                torch.zeros(1, tokens, dtype=torch.bool), torch.full((1,), 4.0),
                torch.rand(1, tokens)*12, torch.zeros(1, dtype=torch.int32))
    result = []
    ordering = random.Random(1708)
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(2035)
        for tokens in (32, 2048):
            # Library/model initialization is already warm. This isolates the
            # first call at a shape with no parameter-only constant cache entry.
            candidate._shape_constant_cache = None
            batch = new_batch(tokens)
            start = time.perf_counter(); original._compute_embedding_for_batch(batch)
            baseline_first = (time.perf_counter()-start)*1000
            start = time.perf_counter(); candidate._compute_embedding_for_batch(batch)
            compressed_first = (time.perf_counter()-start)*1000
            for _ in range(2):
                original._compute_embedding_for_batch(new_batch(tokens))
                candidate._compute_embedding_for_batch(new_batch(tokens))
            samples = {"original": [], "compiled": []}
            for _ in range(rounds):
                batch = new_batch(tokens)  # Fresh genes/counts each round.
                names = ["original", "compiled"]
                ordering.shuffle(names)
                for name in names:
                    model = original if name == "original" else candidate
                    start = time.perf_counter(); model._compute_embedding_for_batch(batch)
                    samples[name].append((time.perf_counter()-start)*1000)
            before, after = (statistics.median(samples[name]) for name in ("original", "compiled"))
            result.append({"cells": 1, "tokens": tokens, "rounds": rounds,
                           "first_shape_call_ms": {"original": baseline_first, "compiled": compressed_first},
                           "median_ms": {"original": before, "compiled": after},
                           "speed_ratio": before/after, "samples_ms": samples,
                           "constant_cache_hits": candidate._shape_constant_cache.hits,
                           "constant_cache_misses": candidate._shape_constant_cache.misses})
    return {"device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "timing_scope": "Original complete numerical batch API; interleaved order; fresh gene/count inputs; excludes imports/model loading/compilation",
            "first_call_scope": "First call at an uncached shape after library/model warmup; constant cache stores parameters only, never cell outputs",
            "cases": result}
