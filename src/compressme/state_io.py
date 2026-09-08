"""Local portable State ST artifacts with verified architecture source files.

Use a compatible Transformers4 release for the published Replogle checkpoint.
Source hashes detect accidental changes, not coordinated manifest tampering.
Loading an artifact executes its vendored Python; use trusted artifacts.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import types
import torch
from torch import nn


def state_factory(architecture_directory):
    """Construct the unmodified published architecture from local JSON/source."""
    import transformers
    if transformers.__version__.split(".")[0] != "4":
        raise RuntimeError("This State checkpoint requires Transformers 4 (validated with 4.52.3); use the separate .venv-state environment or install compressme[state] there.")
    root=Path(architecture_directory).resolve()
    manifest=json.loads((root/"SOURCE.json").read_text())
    for name,expected in manifest["sha256"].items():
        file=(root/name).resolve()
        if file.parent!=root or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"State architecture source hash mismatch: {name}")
    hp=json.loads((root/"hparams.json").read_text())
    if hp.get("finetune_vci_decoder",False) or hp.get("embed_key")!="X_hvg":
        raise ValueError("This audited State loader supports HVG inference without the legacy VCI decoder")
    identity=hashlib.sha256(json.dumps(manifest["sha256"],sort_keys=True).encode()).hexdigest()[:16]
    package_name=f"compressme_local_state_{identity}"
    if package_name not in sys.modules:
        package=types.ModuleType(package_name)
        package.__path__=[str(root)]
        sys.modules[package_name]=package
        # The upstream optional decoder imports an obsolete 'vci' package.
        # This checkpoint never instantiates it; retain an explicit failure for
        # that branch while importing every used class from unmodified source.
        bridge=types.ModuleType(package_name+".decoders")
        class UnavailableFinetuneVCICountsDecoder(nn.Module):
            def __init__(self,*args,**kwargs):
                raise ValueError("Legacy VCI decoder is outside this audited State HVG loader")
        bridge.FinetuneVCICountsDecoder=UnavailableFinetuneVCICountsDecoder
        sys.modules[bridge.__name__]=bridge
        try:
            importlib.import_module(package_name+".state_transition")
        except Exception:
            for key in tuple(sys.modules):
                if key==package_name or key.startswith(package_name+"."):
                    del sys.modules[key]
            raise
    module=sys.modules[package_name+".state_transition"]
    with torch.random.fork_rng(),contextlib.redirect_stdout(io.StringIO()):
        model=module.StateTransitionPerturbationModel(**copy.deepcopy(hp)).eval()
    return model


def load_state_st(directory, *, device="cpu"):
    """Load compressed weights and recipe; never fetch the original checkpoint."""
    from .serialization import load
    root=Path(directory)
    result=load(lambda:state_factory(root/"architecture"),root)
    return result.model.to(device).eval()
