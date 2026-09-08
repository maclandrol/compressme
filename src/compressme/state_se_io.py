"""Portable State SE hybrid artifacts with local, hash-verified architecture.

The original 869MB checkpoint is not needed to reload. Artifacts execute their
vendored Python source; use trusted artifacts. Hashes detect accidental edits,
not coordinated changes to code and its manifest.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import warnings
import torch
from torch import nn


def state_se_factory(architecture_directory, *, representation="finite_lookup"):
    from omegaconf import OmegaConf
    root = Path(architecture_directory).resolve()
    manifest = json.loads((root / "SOURCE.json").read_text())
    for name, expected in manifest["sha256"].items():
        file = (root / name).resolve()
        if root not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"State SE architecture hash mismatch: {name}")
    identity = hashlib.sha256(json.dumps(manifest["sha256"], sort_keys=True).encode()).hexdigest()[:16]
    package_name = "compressme_local_state_se_" + identity
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(root)]
        sys.modules[package_name] = pkg
        nn_pkg = types.ModuleType(package_name + ".nn")
        nn_pkg.__path__ = [str(root / "nn")]
        sys.modules[nn_pkg.__name__] = nn_pkg
        saved_path = list(sys.path)
        try:
            with warnings.catch_warnings():
                importlib.import_module(package_name + ".nn.model")
                importlib.import_module(package_name + ".adapter")
        except Exception:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    del sys.modules[name]
            raise
        finally:
            sys.path[:] = saved_path
    Model = sys.modules[package_name + ".nn.model"].StateEmbeddingModel
    adapter = sys.modules[package_name + ".adapter"]
    cfg = OmegaConf.create(json.loads((root / "config.json").read_text()))
    with torch.random.fork_rng(devices=[]):
        source = Model(token_dim=cfg.tokenizer.token_dim, d_model=cfg.model.emsize,
                       nhead=cfg.model.nhead, d_hid=cfg.model.d_hid, nlayers=cfg.model.nlayers,
                       output_dim=cfg.model.output_dim, dropout=0.0, compiled=False,
                       max_lr=cfg.optimizer.max_lr, emb_cnt=19790, emb_size=5120, cfg=cfg).eval()
        if representation == "lossless_original_embedding":
            # The general packed-embedding recipe replaces this placeholder
            # before strict loading. All original encoder/forward code remains.
            source.pe_embedding = nn.Identity().eval()
            return source.requires_grad_(False)
        if representation != "finite_lookup":
            raise ValueError("Unknown State SE artifact representation")
        # The generic serialization recipe replaces these two placeholders
        # BEFORE strict weight loading. No original protein table is allocated.
        lookup = adapter.FinalBranchLookup(nn.Identity(), nn.Identity()).eval()
        model = adapter.StateSETokenAdapter(source, lookup, copy_downstream=False,
                                           preserve_raw_forward=True, retain_protein_dictionary=False)
    model.post_projection = nn.Identity().eval()
    model.constant_mode = "source_shape"
    model.set_gene_vocabulary(json.loads((root / "vocabulary.json").read_text()))
    return model


def load_state_se(directory, *, device="cpu", protein_embeddings=None):
    from .serialization import load
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    validated_devices = manifest.get("report", {}).get("validated_devices", ["cpu"])
    destination = torch.device(device)
    if destination.type not in validated_devices:
        raise ValueError(f"State SE artifact is validated for {validated_devices}; {destination.type} did not pass its numerical gate")
    representation = manifest.get("report", {}).get("representation", "finite_lookup")
    model = load(lambda: state_se_factory(root / "architecture", representation=representation), root).model.to(destination).eval()
    if representation == "lossless_original_embedding":
        names = tuple(json.loads((root / "architecture" / "vocabulary.json").read_text()))
        if len(names) != model.pe_embedding.num_embeddings or len(set(names)) != len(names):
            raise ValueError("State SE vocabulary does not match the lossless embedding")
        model.gene_names = names
        model.gene_to_index = _GeneIndex(names)
        model.protein_embeds = _PackedProteinRows(model.pe_embedding, model.gene_to_index)
    if protein_embeddings is not None:
        if isinstance(protein_embeddings, (str, Path)):
            proteins = torch.load(protein_embeddings, weights_only=True, mmap=True, map_location="cpu")
        else:
            proteins = protein_embeddings
        if not isinstance(proteins, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                   and tuple(v.shape) == (5120,) for k, v in proteins.items()):
            raise ValueError("Expected an original string-to-5120-vector protein dictionary")
        model.protein_embeds = proteins
    return model


from collections.abc import Mapping


class _GeneIndex(Mapping):
    """Immutable ordered metadata that remains safe to deepcopy with a model."""
    def __init__(self, names):
        self._indices = {name: i for i, name in enumerate(names)}
    def __len__(self):
        return len(self._indices)
    def __iter__(self):
        return iter(self._indices)
    def __getitem__(self, name):
        return self._indices[name]
    def __deepcopy__(self, memo):
        return self


class _PackedProteinRows(Mapping):
    """Original protein vectors reconstructed on demand, without a second table."""
    def __init__(self, embedding, indices):
        self._embedding = embedding
        self._indices = indices
    def __len__(self):
        return len(self._indices)
    def __iter__(self):
        return iter(self._indices)
    def __contains__(self, name):
        return name in self._indices
    def __getitem__(self, name):
        index = self._indices[name]
        embedding = self._embedding
        # The original name helper stacks CPU dictionary rows then transfers
        # the resulting tensor. Preserve that route and its zero-sum guard.
        block = embedding._block(index // embedding.block_rows)
        return block[index % embedding.block_rows].clone()


def load_state_se_encoder(directory, *, tokenizer_source=None):
    """Load reusable CPU AnnData encoding from a trusted portable SE artifact.

    The tokenizer source is resolved inside the artifact by default. State's
    optional dependencies are imported only when this loader is called. No
    original checkpoint or raw protein dictionary is required.
    """
    from .state_anndata import StateSEAnnDataEncoder
    root = Path(directory)
    source = Path(tokenizer_source) if tokenizer_source is not None else root / "tokenizer_source"
    if not (source / "SOURCE.json").is_file():
        raise FileNotFoundError("State SE AnnData encoding needs its pinned tokenizer_source directory")
    model = load_state_se(root, device="cpu")
    return StateSEAnnDataEncoder(model, source_directory=source)
