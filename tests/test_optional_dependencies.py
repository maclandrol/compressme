"""Import boundaries are checked in new processes, not an already-loaded Torch test run."""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
BIOLOGY = {"boltz", "state", "moljepa", "rdkit", "torch_geometric", "transformers",
           "molfeat", "anndata", "geomloss", "lightning", "omegaconf"}
BLOCKER = """
import importlib.abc
class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, ForbiddenImport())
"""


def run_isolated(code, *, no_site=False, blocked=()):
    prefix = f"import sys\nsys.path.insert(0, {str(SOURCE)!r})\nblocked = {set(blocked)!r}\n"
    command = [sys.executable, "-I"] + (["-S"] if no_site else [])
    result = subprocess.run(command + ["-c", prefix + BLOCKER + textwrap.dedent(code)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_base_import_and_public_metadata_need_only_standard_library():
    run_isolated("""
        import compressme
        assert 'compile_affine' in dir(compressme)
        assert 'compile_affine' not in vars(compressme)
        from compressme import list_targets, get_target, pack_file, unpack_file
        from compressme import pack_bytes, unpack_bytes, inspect_huggingface
        from compressme import accelerate_smiles, contract_attention
        assert get_target('boltz2')['name'] == 'Boltz-2'
        assert list_targets()
        assert compressme.pack_bytes is pack_bytes
        assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
        try:
            compressme.nonexistent_public_api
        except AttributeError:
            pass
        else:
            raise AssertionError('Unknown attribute must raise AttributeError')
    """, no_site=True, blocked=BIOLOGY | {"torch", "safetensors", "numpy", "zstandard", "huggingface_hub"})


def test_missing_torch_has_an_actionable_extra_message():
    run_isolated("""
        import compressme
        try:
            compressme.compile_affine
        except ModuleNotFoundError as exc:
            assert exc.name == 'torch'
            assert 'compressme[torch]' in str(exc)
        else:
            raise AssertionError('Torch is absent under -S')
    """, no_site=True)


def test_missing_packing_dependency_is_deferred_until_call():
    run_isolated("""
        from compressme import pack_bytes
        try:
            pack_bytes(b'payload')
        except ImportError as exc:
            assert 'compressme[packing]' in str(exc)
        else:
            raise AssertionError('Packing dependencies are absent under -S')
    """, no_site=True)


def test_hub_metadata_needs_no_torch_or_remote_architecture():
    run_isolated("""
        import types
        from compressme import inspect_huggingface
        class API:
            def model_info(self, repo_id, **kwargs):
                assert kwargs['files_metadata'] is True
                return {'sha': 'a' * 40, 'siblings': [
                    {'rfilename': 'model.safetensors', 'size': 200},
                    {'rfilename': 'modeling_custom.py', 'size': 100},
                ]}
        sys.modules['huggingface_hub'] = types.SimpleNamespace(HfApi=API)
        report = inspect_huggingface('example/model')
        assert report['revision'] == 'a' * 40
        assert report['weight_bytes'] == 200
        assert report['remote_code_executed'] is False
    """, no_site=True, blocked=BIOLOGY | {"torch", "safetensors", "numpy", "zstandard"})


def test_all_existing_public_exports_resolve_without_biology_packages():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    run_isolated("""
        import compressme
        for name in compressme.__all__:
            assert getattr(compressme, name) is getattr(compressme, name), name
        from compressme.affine import compile_affine
        assert compressme.compile_affine is compile_affine
    """, blocked=BIOLOGY)


def test_core_torch_rewrites_can_run_without_numpy():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    run_isolated(f"""
        class NoNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'numpy':
                    raise ModuleNotFoundError('NumPy deliberately absent', name='numpy')
        sys.meta_path.insert(0, NoNumpy())
        import torch
        from compressme import compile_affine, share_frozen_parameters
        def factory():
            return torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Linear(16, 4)).eval()
        result = compile_affine(factory())
        values = torch.randn(2, 3)
        assert result.model(values).shape == (2, 4)
        left = torch.nn.Linear(3, 4).eval().requires_grad_(False)
        import copy
        bundle = torch.nn.ModuleList([left, copy.deepcopy(left)]).eval()
        shared = share_frozen_parameters(bundle)
        assert shared.model[0].weight.untyped_storage().data_ptr() == shared.model[1].weight.untyped_storage().data_ptr()
        assert 'numpy' not in sys.modules
    """, blocked=BIOLOGY)


def test_file_packing_works_without_torch_or_model_dependencies(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("zstandard")
    run_isolated(f"""
        from pathlib import Path
        from compressme import pack_file, unpack_file
        root = Path({str(tmp_path)!r})
        payload = bytes(range(256)) * 4097 + b'odd'
        (root / 'source.bin').write_bytes(payload)
        packed = pack_file(root / 'source.bin', root / 'packed.cmppack')
        restored = unpack_file(root / 'packed.cmppack', root / 'restored.bin')
        assert (root / 'restored.bin').read_bytes() == payload
        assert restored['sha256'] == packed['sha256']
    """, blocked=BIOLOGY | {"torch", "safetensors", "huggingface_hub"})


def test_cli_targets_needs_no_third_party_packages():
    run_isolated("""
        import runpy
        sys.argv = ['compressme', 'targets', 'boltz2']
        runpy.run_module('compressme', run_name='__main__')
    """, no_site=True, blocked=BIOLOGY | {"torch", "safetensors", "numpy", "zstandard", "huggingface_hub"})


def test_cli_local_tensor_metadata_needs_no_tensor_runtime(tmp_path):
    run_isolated(f"""
        import json, runpy, struct
        from pathlib import Path
        path = Path({str(tmp_path / 'local.safetensors')!r})
        header = json.dumps({{'x': {{'dtype': 'F32', 'shape': [2],
                                   'data_offsets': [0, 8]}}}}).encode()
        header += b' ' * (-len(header) % 8)
        path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
        sys.argv = ['compressme', 'inspect', str(path), '--details']
        runpy.run_module('compressme', run_name='__main__')
    """, no_site=True, blocked=BIOLOGY | {"torch", "safetensors", "numpy", "zstandard", "huggingface_hub"})
