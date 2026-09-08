from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import random
import shutil

import pytest
import torch
from torch import nn

import compressme.boltz2_io as portable


def test_public_loader_is_available_without_boltz_import():
    import os
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import sys; from compressme import load_boltz2; assert callable(load_boltz2); assert 'boltz' not in sys.modules"],
                   env={**os.environ, "PYTHONPATH": str(Path(portable.__file__).parent.parent)},
                   check=True, capture_output=True, text=True)


@pytest.fixture
def architecture(tmp_path):
    target = tmp_path / "architecture"
    shutil.copytree(Path(portable.__file__).parent / "data" / "boltz2", target)
    return target


def test_architecture_hashes_and_pure_json(architecture):
    source, kwargs = portable._architecture(architecture)
    assert source["revision"] == portable.PINNED_BOLTZ_REVISION
    assert kwargs["confidence"]["predict_args"]["recycling_steps"] == 3
    assert kwargs["affinity"]["predict_args"]["diffusion_samples"] == 3
    with (architecture / "confidence.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="integrity"):
        portable._architecture(architecture)


@pytest.mark.parametrize("text", ['{"key":1,"key":2}', '{"key":NaN}', '{"key":1e999}'])
def test_json_rejects_ambiguous_or_nonfinite_data(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        portable._read_json(path)


def test_source_guard_covers_entire_installed_tree(tmp_path, monkeypatch):
    root = tmp_path / "boltz"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "part.py").write_text("VALUE = 3\n")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(portable, "PINNED_SOURCE_SHA256", digest)
    monkeypatch.setattr(portable.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(root / "__init__.py"), submodule_search_locations=[str(root)]))
    monkeypatch.setattr(portable.importlib.metadata, "version", lambda _: portable.PINNED_BOLTZ_VERSION)
    assert portable._verify_local_boltz({"files": files}) == root
    (root / "extra.py").write_text("# An unreviewed new module\n")
    with pytest.raises(ImportError, match="differs"):
        portable._verify_local_boltz({"files": files})


def test_factory_preserves_classes_keys_rng_and_freezes(architecture, monkeypatch):
    _, kwargs = portable._architecture(architecture)
    events = []

    class FakeBoltz(nn.Module):
        def __init__(self, **arguments):
            super().__init__()
            events.append(arguments)
            self.weight = nn.Parameter(torch.randn(3))
            self.random_value = random.random()
    names = sorted(set(kwargs["confidence"]) | set(kwargs["affinity"]))
    FakeBoltz.__signature__ = inspect.Signature([inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in names])
    monkeypatch.setattr(portable, "_verify_local_boltz", lambda _: None)
    original_import = portable.importlib.import_module
    monkeypatch.setattr(portable.importlib, "import_module", lambda name: SimpleNamespace(Boltz2=FakeBoltz) if name == "boltz.model.models.boltz2" else original_import(name))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle loader used"))
    torch_state, python_state = torch.random.get_rng_state().clone(), random.getstate()
    result = portable.boltz2_factory(architecture)
    assert type(result) is nn.ModuleDict and list(result) == ["confidence", "affinity"]
    assert all(type(child) is FakeBoltz for child in result.values())
    assert not any(module.training for module in result.modules())
    assert not any(parameter.requires_grad for parameter in result.parameters())
    assert events == list(kwargs.values())
    assert torch.equal(torch_state, torch.random.get_rng_state()) and python_state == random.getstate()


def test_load_places_then_shares_and_retains_report(tmp_path, monkeypatch):
    import compressme.serialization
    import compressme.sharing
    events = []
    bundle = nn.ModuleDict({"confidence": nn.Linear(2, 2), "affinity": nn.Linear(2, 2)})
    parameter_ids = [id(p) for p in bundle.parameters()]
    real_to = bundle.to
    def tracked_to(device):
        events.append(("to", str(device)))
        return real_to(device)
    bundle.to = tracked_to
    monkeypatch.setattr(portable, "boltz2_factory", lambda path: bundle)
    def load(factory, directory):
        events.append(("load", str(directory)))
        return SimpleNamespace(model=factory(), report={"loaded": True})
    def share(model, *, inplace):
        assert inplace and model is bundle
        assert not any(p.requires_grad for p in model.parameters())
        assert not any(m.training for m in model.modules())
        events.append(("share", "cpu"))
        return SimpleNamespace(model=model, report={"shared": True})
    monkeypatch.setattr(compressme.serialization, "load", load)
    monkeypatch.setattr(compressme.sharing, "share_frozen_parameters", share)
    result = portable.load_boltz2(tmp_path)
    assert result is bundle
    assert [event[0] for event in events] == ["load", "to", "share"]
    assert parameter_ids == [id(p) for p in result.parameters()]
    assert result._compressme_boltz2_artifact_report == {"loaded": True}
    assert result._compressme_boltz2_storage_report == {"shared": True}


def test_install_does_not_touch_weights_or_replace_different_data(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"sentinel")
    target = portable.install_architecture(tmp_path)
    assert portable.install_architecture(tmp_path) == target
    assert weights.read_bytes() == b"sentinel"
    (target / "confidence.json").write_text("{}")
    with pytest.raises(FileExistsError):
        portable.install_architecture(tmp_path)


def test_symlink_architecture_escape_rejected(architecture, tmp_path):
    external = tmp_path / "outside.json"
    path = architecture / "confidence.json"
    shutil.copyfile(path, external)
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="integrity"):
        portable._architecture(architecture)
