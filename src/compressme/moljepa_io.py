"""Load an exported Mol-JEPA artifact without its original large checkpoint.

An artifact includes a local copy of the audited upstream Python architecture,
with its attribution/license. Loading executes that Python code; use trusted
artifacts. Hashes detect accidental edits, not malicious replacement of both
code and manifest. No remote code or weights are fetched by this loader.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def moljepa_factory(source_directory):
    """Build the original architecture from locally vendored, hashed sources."""
    root = Path(source_directory).resolve()
    provenance = json.loads((root / "SOURCE.json").read_text())
    for name, expected in provenance["sha256"].items():
        path = (root / name).resolve()
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Architecture source hash mismatch: {name}")
    identity = hashlib.sha256(json.dumps(provenance["sha256"],sort_keys=True).encode()).hexdigest()[:16]
    package_name = f"compressme_local_moljepa_{identity}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, root / "__init__.py",
                                                      submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(package_name, None)
            raise
    package = sys.modules[package_name]
    config = package.MolJEPAConfig.from_json_file(str(root / "config.json"))
    return package.MolJEPAModel(config).eval()


def load_moljepa(directory, *, device="cpu", accelerate=True):
    """Load portable weights and enable the validated SMILES runtime.

    Runtime layouts are built after loading and never duplicate packed weights.
    Set accelerate=False to use the portable compressed execution unchanged.
    Full-modality artifacts keep their original routing.
    """
    from .serialization import load
    root = Path(directory)
    result = load(lambda: moljepa_factory(root / "architecture"), root)
    model = result.model.to(device).eval()
    from .moljepa import SmilesOnlyMolJEPA
    if accelerate and isinstance(model,SmilesOnlyMolJEPA):
        from .smiles_runtime import accelerate_smiles
        model = accelerate_smiles(model,inplace=True,metal=model.device.type == "mps")
    return model
