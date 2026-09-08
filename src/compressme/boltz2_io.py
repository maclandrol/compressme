"""Reload the complete pinned Boltz-2 confidence/affinity pair without pickle.

The artifact contains ordinary compressme safetensors and pure JSON constructor
arguments. Boltz itself must be installed locally at the exact audited source
revision. This module does not download or execute code supplied by an artifact.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil

import torch
from torch import nn

PINNED_BOLTZ_REVISION = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
PINNED_BOLTZ_VERSION = "2.2.1"
PINNED_SOURCE_SHA256 = "5036f6ffd799f875173492446e59fce99358e9fa98469ef9cf576e3830c98f96"
ARCHITECTURE_FILES = ("boltz-source.json", "confidence.json", "affinity.json", "BOLTZ_LICENSE")
ARCHITECTURE_FORMAT = "compressme-boltz2-original-pair-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Architecture JSON is unexpectedly large")
    value = json.loads(path.read_text(), object_pairs_hook=_no_duplicate_keys,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))
    def audit(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Architecture numbers must be finite")
        if isinstance(item, dict):
            for child in item.values():
                audit(child)
        elif isinstance(item, list):
            for child in item:
                audit(child)
    audit(value)
    return value


def _architecture(directory):
    root = Path(directory).resolve(strict=True)
    manifest = _read_json(root / "ARCHITECTURE.json")
    if (not isinstance(manifest, dict) or manifest.get("format") != ARCHITECTURE_FORMAT
            or set(manifest.get("files", {})) != set(ARCHITECTURE_FILES)):
        raise ValueError("Unsupported Boltz-2 architecture manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.resolve(strict=True).parent != root or _sha256(path) != expected:
            raise ValueError(f"Boltz-2 architecture integrity failure: {name}")
    source = _read_json(root / "boltz-source.json")
    if (source.get("revision") != PINNED_BOLTZ_REVISION
            or source.get("distribution_version") != PINNED_BOLTZ_VERSION
            or source.get("python_file_set_sha256") != PINNED_SOURCE_SHA256):
        raise ValueError("Boltz-2 architecture is not the audited source revision")
    canonical = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError("Boltz-2 source inventory hash mismatch")
    kwargs = {name: _read_json(root / f"{name}.json") for name in ("confidence", "affinity")}
    for name, values in kwargs.items():
        if not isinstance(values, dict) or values.get("use_kernels") is not False or values.get("ema") is not False:
            raise ValueError(f"Invalid native inference configuration for {name}")
    return source, kwargs


def _verify_local_boltz(source):
    """Inspect source bytes before importing Boltz's model or its dependencies."""
    spec = importlib.util.find_spec("boltz")
    if spec is None or not spec.origin or not spec.submodule_search_locations:
        raise ImportError("Install the pinned local Boltz 2.2.1 dependency before loading this artifact")
    root = Path(spec.origin).resolve(strict=True).parent
    if importlib.metadata.version("boltz") != PINNED_BOLTZ_VERSION:
        raise ImportError(f"Boltz {PINNED_BOLTZ_VERSION} is required")
    files = {}
    for path in sorted(root.rglob("*.py")):
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Boltz source contains an external source-file link")
        files[path.relative_to(root).as_posix()] = _sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != PINNED_SOURCE_SHA256 or files != source["files"]:
        raise ImportError(f"Installed Boltz source differs from audited revision {PINNED_BOLTZ_REVISION}")
    return root


def boltz2_factory(architecture_directory) -> nn.ModuleDict:
    """Construct both original model classes; no checkpoint is read here.

    This ordinary factory initially allocates both complete model states. Its
    cold-load peak is therefore larger than the final shared storage. It does
    not promise a meta-device or zero-copy load.
    """
    source, kwargs = _architecture(architecture_directory)
    _verify_local_boltz(source)
    Boltz2 = importlib.import_module("boltz.model.models.boltz2").Boltz2
    accepted = set(inspect.signature(Boltz2).parameters)
    for name, values in kwargs.items():
        extra = set(values) - accepted
        if extra:
            raise ValueError(f"Unsupported constructor fields for {name}: {sorted(extra)}")
    from .validation import seeded
    with seeded(0):
        models = nn.ModuleDict({name: Boltz2(**values) for name, values in kwargs.items()})
    return models.eval().requires_grad_(False)


def load_boltz2(directory, device="cpu") -> nn.ModuleDict:
    """Load both full original APIs, then deduplicate frozen storage on device.

    Use ``bundle['confidence'].predict_step(batch, 0)`` or the affinity member's
    original API with native Boltz batches. Native preprocessing and chemistry
    assets remain separate. Weights are frozen: changing either member's shared
    parameters would also change the other member. Move to the final device via
    this loader; a subsequent ``.to(...)`` may split shared storage again.
    """
    from .serialization import load
    from .sharing import share_frozen_parameters
    root = Path(directory).resolve(strict=True)
    loaded = load(lambda: boltz2_factory(root / "architecture"), root)
    model = loaded.model
    if type(model) is not nn.ModuleDict or tuple(model.keys()) != ("confidence", "affinity"):
        raise ValueError("Boltz-2 artifact does not contain the complete confidence/affinity pair")
    model.eval().requires_grad_(False)
    model.to(torch.device(device))
    shared = share_frozen_parameters(model, inplace=True)
    model = shared.model
    model._compressme_boltz2_artifact_report = loaded.report
    model._compressme_boltz2_storage_report = shared.report
    model._compressme_boltz2_revision = PINNED_BOLTZ_REVISION
    return model


def install_architecture(artifact_directory, *, source_directory=None) -> Path:
    """Copy the small audited data files beside a separately saved artifact.

    This never reads or writes weights and refuses to replace different existing
    architecture files. ``source_directory`` is a trusted local data directory.
    """
    source = Path(source_directory) if source_directory is not None else Path(__file__).parent / "data" / "boltz2"
    _architecture(source)
    target = Path(artifact_directory) / "architecture"
    target.mkdir(parents=True, exist_ok=True)
    for name in (*ARCHITECTURE_FILES, "ARCHITECTURE.json"):
        destination = target / name
        if destination.exists() and _sha256(destination) != _sha256(source / name):
            raise FileExistsError(f"Refusing to replace different architecture file: {destination}")
        if not destination.exists():
            shutil.copyfile(source / name, destination)
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Copy the pinned Boltz-2 JSON architecture beside an existing compressme artifact")
    parser.add_argument("artifact_directory")
    parser.add_argument("--source-directory")
    args = parser.parse_args()
    print(install_architecture(args.artifact_directory, source_directory=args.source_directory))
