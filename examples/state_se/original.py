"""Minimal loader of pinned State SE safetensors into unmodified source.

The public CLI expects Lightning .ckpt but this release contains safetensors.
This bridge instantiates the same architecture from the published config, uses
the upstream inference override dropout=0, validates tied encoder keys, and
loads every tensor strictly. It avoids importing data/training CLI packages.
"""
from __future__ import annotations
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import warnings
import torch
from torch import nn
from omegaconf import OmegaConf
from safetensors.torch import load_file


def state_se_class(source_root=None):
    root = Path(source_root)
    if (root / "src/state/emb").is_dir():
        root = root / "src/state/emb"
    files = [root / "utils.py", root / "nn/model.py", root / "nn/loss.py",
             root / "nn/flash_transformer.py"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    name = "compressme_audited_state_se_" + identity
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(root)]
        sys.modules[name] = pkg
        nn_pkg = types.ModuleType(name + ".nn")
        nn_pkg.__path__ = [str(root / "nn")]
        sys.modules[name + ".nn"] = nn_pkg
        old_path = list(sys.path)
        try:
            with warnings.catch_warnings():
                importlib.import_module(name + ".nn.model")
        finally:
            sys.path[:] = old_path
    return sys.modules[name + ".nn.model"].StateEmbeddingModel


def load_state_se(weights, config=None,
                  source_root=None, protein_embeddings=None):
    if config is None or source_root is None:
        raise ValueError("Provide the pinned architecture directory and config path")
    cfg = OmegaConf.load(config)
    Model = state_se_class(source_root)
    tensors = load_file(str(weights), device="cpu")
    # The published YAML says 14420 datasets; the released head contains 14418.
    # Restore the checkpoint's actual shape rather than dropping its head or
    # inventing two output rows. Every checkpoint tensor is still strict-loaded.
    if "dataset_encoder.4.weight" in tensors:
        cfg.dataset[cfg.dataset.current].num_datasets = tensors["dataset_encoder.4.weight"].shape[0]
    for name, value in tensors.items():
        if name.startswith("gene_embedding_layer."):
            original = "encoder." + name.removeprefix("gene_embedding_layer.")
            assert original in tensors and torch.equal(value, tensors[original]), name
    with torch.random.fork_rng():
        model = Model(token_dim=cfg.tokenizer.token_dim, d_model=cfg.model.emsize,
                      nhead=cfg.model.nhead, d_hid=cfg.model.d_hid, nlayers=cfg.model.nlayers,
                      output_dim=cfg.model.output_dim, dropout=0.0, compiled=False,
                      max_lr=cfg.optimizer.max_lr,
                      emb_cnt=tensors["pe_embedding.weight"].shape[0],
                      emb_size=tensors["pe_embedding.weight"].shape[1], cfg=cfg)
        model.pe_embedding = nn.Embedding.from_pretrained(tensors["pe_embedding.weight"], freeze=True)
    model.load_state_dict(tensors, strict=True)
    assert model.encoder is model.gene_embedding_layer
    model.eval()
    if protein_embeddings is not None:
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(protein_embeddings)
        if unsafe:
            raise ValueError(f"Protein dictionary needs unaudited pickle globals: {unsafe}")
        proteins = torch.load(protein_embeddings, weights_only=True, map_location="cpu", mmap=True)
        if not isinstance(proteins, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                   for k, v in proteins.items()):
            raise ValueError("Expected a plain string-to-tensor protein dictionary")
        stacked = torch.vstack(list(proteins.values()))
        if not torch.equal(stacked, model.pe_embedding.weight):
            raise ValueError("Published dictionary row order/content differs from the stored model lookup")
        model.protein_embeds = proteins
    return model

